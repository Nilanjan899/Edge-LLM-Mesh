#!/usr/bin/env python3
"""
main.py — Central Telemetry Ingestion API (FastAPI)
=====================================================

Receives batched telemetry payloads from edge devices and publishes
each record to a RabbitMQ exchange for downstream processing by
``process_metrics.py``.

Endpoints:
    - ``GET  /health`` — Liveness check (used by edge circuit breaker).
    - ``POST /ingest/telemetry`` — Accept a batch of inference metrics.
    - ``GET  /metrics`` — Prometheus scrape endpoint (via ``prometheus_client``).

Run with::

    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

import pika
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("central.api")

# ---------------------------------------------------------------------------
# Configuration (env vars with sane defaults)
# ---------------------------------------------------------------------------
RABBITMQ_HOST: str = os.environ.get("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT: int = int(os.environ.get("RABBITMQ_PORT", "5672"))
RABBITMQ_USER: str = os.environ.get("RABBITMQ_USER", "guest")
RABBITMQ_PASS: str = os.environ.get("RABBITMQ_PASS", "guest")
RABBITMQ_EXCHANGE: str = os.environ.get("RABBITMQ_EXCHANGE", "edge_telemetry")
RABBITMQ_ROUTING_KEY: str = os.environ.get("RABBITMQ_ROUTING_KEY", "telemetry.ingest")

# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------
registry = CollectorRegistry()

ingest_counter = Counter(
    "edge_telemetry_ingested_total",
    "Total number of telemetry records ingested",
    ["device_id"],
    registry=registry,
)
ingest_errors = Counter(
    "edge_telemetry_ingest_errors_total",
    "Total number of ingest errors",
    registry=registry,
)
ingest_latency = Histogram(
    "edge_telemetry_ingest_latency_seconds",
    "Latency of the /ingest/telemetry endpoint",
    registry=registry,
)
generation_speed_gauge = Gauge(
    "edge_generation_speed_tokens_per_sec",
    "Latest generation speed reported by an edge device",
    ["device_id", "model_name"],
    registry=registry,
)
memory_usage_gauge = Gauge(
    "edge_memory_usage_mb",
    "Latest RAM usage reported by an edge device",
    ["device_id"],
    registry=registry,
)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------
class TelemetryRecord(BaseModel):
    """A single inference telemetry record from an edge device."""

    id: str
    timestamp: float
    device_id: str
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_time_ms: float = 0.0
    ttft_ms: Optional[float] = None
    tokens_per_sec: float = 0.0
    ram_usage_mb: float = 0.0
    vram_usage_mb: Optional[float] = None
    prompt_text: Optional[str] = None
    synced: Optional[int] = None  # Ignored server-side


class TelemetryBatch(BaseModel):
    """Batch payload sent by an edge device."""

    device_id: str = Field(..., description="Originating edge device identifier")
    batch: list[TelemetryRecord] = Field(..., description="List of telemetry records")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    timestamp: float = Field(default_factory=time.time)
    rabbitmq_connected: bool = False


# ---------------------------------------------------------------------------
# RabbitMQ Connection Manager
# ---------------------------------------------------------------------------
class RabbitMQPublisher:
    """Manages a persistent connection to RabbitMQ for publishing messages.

    Reconnects automatically on channel/connection errors.
    """

    def __init__(self) -> None:
        self._connection: Optional[pika.BlockingConnection] = None
        self._channel: Optional[pika.adapters.blocking_connection.BlockingChannel] = None

    def connect(self) -> None:
        """Establish connection and declare the exchange."""
        try:
            credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
            params = pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300,
            )
            self._connection = pika.BlockingConnection(params)
            self._channel = self._connection.channel()
            self._channel.exchange_declare(
                exchange=RABBITMQ_EXCHANGE,
                exchange_type="topic",
                durable=True,
            )
            self._channel.queue_declare(queue="telemetry_queue", durable=True)
            self._channel.queue_bind(
                queue="telemetry_queue",
                exchange=RABBITMQ_EXCHANGE,
                routing_key=RABBITMQ_ROUTING_KEY,
            )
            logger.info("Connected to RabbitMQ at %s:%d", RABBITMQ_HOST, RABBITMQ_PORT)
        except Exception as exc:
            logger.error("Failed to connect to RabbitMQ: %s", exc)
            self._connection = None
            self._channel = None

    def publish(self, message: dict[str, Any]) -> bool:
        """Publish a JSON message to the telemetry exchange.

        Args:
            message: Dictionary to serialize and publish.

        Returns:
            ``True`` if published successfully.
        """
        if not self._channel or (self._connection and self._connection.is_closed):
            self.connect()
        if not self._channel:
            return False
        try:
            self._channel.basic_publish(
                exchange=RABBITMQ_EXCHANGE,
                routing_key=RABBITMQ_ROUTING_KEY,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=pika.DeliveryMode.Persistent,
                    content_type="application/json",
                ),
            )
            return True
        except (pika.exceptions.AMQPError, Exception) as exc:
            logger.error("Publish failed: %s — will reconnect on next call.", exc)
            self._channel = None
            return False

    @property
    def is_connected(self) -> bool:
        """Check if the RabbitMQ connection is alive."""
        return (
            self._connection is not None
            and self._connection.is_open
            and self._channel is not None
            and self._channel.is_open
        )

    def close(self) -> None:
        """Gracefully close the connection."""
        if self._connection and self._connection.is_open:
            self._connection.close()
            logger.info("RabbitMQ connection closed.")


# ---------------------------------------------------------------------------
# Application Lifecycle
# ---------------------------------------------------------------------------
publisher = RabbitMQPublisher()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Connect to RabbitMQ on startup, disconnect on shutdown."""
    publisher.connect()
    yield
    publisher.close()


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Edge-LLM Telemetry Mesh — Central API",
    description="Ingests inference telemetry from edge devices and routes it to RabbitMQ.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health_check() -> HealthResponse:
    """Liveness probe — returns 200 even if RabbitMQ is down (degraded mode)."""
    return HealthResponse(
        status="ok",
        rabbitmq_connected=publisher.is_connected,
    )


@app.post("/ingest/telemetry", status_code=status.HTTP_202_ACCEPTED, tags=["ingest"])
async def ingest_telemetry(payload: TelemetryBatch) -> JSONResponse:
    """Ingest a batch of telemetry records from an edge device.

    Each record is published individually to the RabbitMQ exchange so that
    downstream consumers can process them independently.

    Args:
        payload: :class:`TelemetryBatch` containing device_id and list of records.

    Returns:
        202 Accepted with count of published messages.

    Raises:
        HTTPException: If RabbitMQ is unreachable and no records could be published.
    """
    with ingest_latency.time():
        published = 0
        failed = 0

        for record in payload.batch:
            msg = record.model_dump()
            msg.pop("synced", None)  # Strip local-only field

            if publisher.publish(msg):
                published += 1
                ingest_counter.labels(device_id=record.device_id).inc()

                # Update Prometheus gauges with the latest values
                generation_speed_gauge.labels(
                    device_id=record.device_id,
                    model_name=record.model_name,
                ).set(record.tokens_per_sec)
                memory_usage_gauge.labels(device_id=record.device_id).set(
                    record.ram_usage_mb
                )
            else:
                failed += 1
                ingest_errors.inc()

        if published == 0 and failed > 0:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RabbitMQ unreachable — 0 records published.",
            )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={
            "status": "accepted",
            "published": published,
            "failed": failed,
            "device_id": payload.device_id,
        },
    )


@app.get("/metrics", tags=["ops"])
async def prometheus_metrics() -> Response:
    """Expose Prometheus-compatible metrics for scraping."""
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST,
    )
