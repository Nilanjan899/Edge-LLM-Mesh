#!/usr/bin/env python3
"""
process_metrics.py — RabbitMQ Consumer → Prometheus Exporter
==============================================================

Long-running worker that:
1. Connects to RabbitMQ and consumes from the ``telemetry_queue``.
2. Parses each message (a JSON telemetry record from an edge device).
3. Updates Prometheus gauges and counters so Grafana dashboards stay current.

The worker exposes its own ``/metrics`` HTTP endpoint on port 9091
so Prometheus can scrape the consumer-side metrics independently
from the FastAPI ingest endpoint.

Usage::

    python process_metrics.py

Environment Variables:
    RABBITMQ_HOST       — default ``rabbitmq``
    RABBITMQ_PORT       — default ``5672``
    RABBITMQ_USER       — default ``guest``
    RABBITMQ_PASS       — default ``guest``
    METRICS_PORT        — default ``9091``
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Optional

import pika
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    start_http_server,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("central.consumer")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RABBITMQ_HOST: str = os.environ.get("RABBITMQ_HOST", "rabbitmq")
RABBITMQ_PORT: int = int(os.environ.get("RABBITMQ_PORT", "5672"))
RABBITMQ_USER: str = os.environ.get("RABBITMQ_USER", "guest")
RABBITMQ_PASS: str = os.environ.get("RABBITMQ_PASS", "guest")
RABBITMQ_EXCHANGE: str = os.environ.get("RABBITMQ_EXCHANGE", "edge_telemetry")
RABBITMQ_ROUTING_KEY: str = os.environ.get("RABBITMQ_ROUTING_KEY", "telemetry.ingest")
RABBITMQ_QUEUE: str = "telemetry_queue"
METRICS_PORT: int = int(os.environ.get("METRICS_PORT", "9091"))

# ---------------------------------------------------------------------------
# Prometheus Metrics (consumer-side)
# ---------------------------------------------------------------------------
registry = CollectorRegistry()

records_consumed = Counter(
    "consumer_records_consumed_total",
    "Total telemetry records consumed from RabbitMQ",
    ["device_id"],
    registry=registry,
)
consume_errors = Counter(
    "consumer_errors_total",
    "Total errors while processing consumed messages",
    registry=registry,
)

# ---- Edge device gauges (set per device/model) ----
edge_generation_speed_gauge = Gauge(
    "edge_generation_speed_tokens_per_sec",
    "Latest generation speed (tokens/sec) from an edge device",
    ["device_id", "model_name"],
    registry=registry,
)
edge_memory_usage_gauge = Gauge(
    "edge_memory_usage_mb",
    "Latest RAM usage (MB) from an edge device",
    ["device_id"],
    registry=registry,
)
edge_vram_usage_gauge = Gauge(
    "edge_vram_usage_mb",
    "Latest VRAM usage (MB) from an edge device (0 if unavailable)",
    ["device_id"],
    registry=registry,
)
edge_ttft_gauge = Gauge(
    "edge_ttft_ms",
    "Latest Time-To-First-Token (ms) from an edge device",
    ["device_id", "model_name"],
    registry=registry,
)
edge_total_time_histogram = Histogram(
    "edge_inference_total_time_ms",
    "Distribution of total inference time (ms)",
    ["device_id"],
    buckets=[50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000],
    registry=registry,
)
edge_prompt_tokens_gauge = Gauge(
    "edge_prompt_tokens",
    "Latest prompt token count from an edge device",
    ["device_id", "model_name"],
    registry=registry,
)
edge_completion_tokens_gauge = Gauge(
    "edge_completion_tokens",
    "Latest completion token count from an edge device",
    ["device_id", "model_name"],
    registry=registry,
)


# ---------------------------------------------------------------------------
# Message Processing
# ---------------------------------------------------------------------------
def process_record(record: dict[str, Any]) -> None:
    """Update Prometheus metrics from a single telemetry record.

    Args:
        record: Deserialized JSON telemetry record.
    """
    device_id: str = record.get("device_id", "unknown")
    model_name: str = record.get("model_name", "unknown")

    records_consumed.labels(device_id=device_id).inc()

    # Generation speed
    tokens_per_sec = record.get("tokens_per_sec", 0.0)
    edge_generation_speed_gauge.labels(
        device_id=device_id, model_name=model_name
    ).set(tokens_per_sec)

    # Memory
    ram_mb = record.get("ram_usage_mb", 0.0)
    edge_memory_usage_gauge.labels(device_id=device_id).set(ram_mb)

    vram_mb = record.get("vram_usage_mb")
    if vram_mb is not None:
        edge_vram_usage_gauge.labels(device_id=device_id).set(vram_mb)

    # TTFT
    ttft = record.get("ttft_ms")
    if ttft is not None:
        edge_ttft_gauge.labels(device_id=device_id, model_name=model_name).set(ttft)

    # Inference time histogram
    total_time = record.get("total_time_ms", 0.0)
    edge_total_time_histogram.labels(device_id=device_id).observe(total_time)

    # Token counts
    edge_prompt_tokens_gauge.labels(
        device_id=device_id, model_name=model_name
    ).set(record.get("prompt_tokens", 0))
    edge_completion_tokens_gauge.labels(
        device_id=device_id, model_name=model_name
    ).set(record.get("completion_tokens", 0))

    logger.info(
        "Processed record from %s/%s — %.1f tok/s, RAM=%.0f MB",
        device_id, model_name, tokens_per_sec, ram_mb,
    )


def on_message(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    method: pika.spec.Basic.Deliver,
    properties: pika.spec.BasicProperties,
    body: bytes,
) -> None:
    """RabbitMQ message callback.

    Parses the JSON body, processes the telemetry record, and acknowledges
    the message. Malformed messages are rejected (nack without requeue).
    """
    try:
        record = json.loads(body)
        process_record(record)
        channel.basic_ack(delivery_tag=method.delivery_tag)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in message: %s", exc)
        consume_errors.inc()
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    except Exception as exc:
        logger.exception("Error processing message: %s", exc)
        consume_errors.inc()
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)


# ---------------------------------------------------------------------------
# Consumer Loop
# ---------------------------------------------------------------------------
def run_consumer() -> None:
    """Connect to RabbitMQ, declare queue, and start consuming.

    Reconnects automatically on connection loss with exponential backoff.
    """
    # Start Prometheus metrics HTTP server on a background thread
    logger.info("Starting Prometheus metrics server on port %d", METRICS_PORT)
    start_http_server(METRICS_PORT, registry=registry)

    backoff: float = 1.0
    max_backoff: float = 60.0

    while True:
        try:
            credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
            params = pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300,
            )

            connection = pika.BlockingConnection(params)
            channel = connection.channel()

            # Ensure exchange and queue exist
            channel.exchange_declare(
                exchange=RABBITMQ_EXCHANGE,
                exchange_type="topic",
                durable=True,
            )
            channel.queue_declare(queue=RABBITMQ_QUEUE, durable=True)
            channel.queue_bind(
                queue=RABBITMQ_QUEUE,
                exchange=RABBITMQ_EXCHANGE,
                routing_key=RABBITMQ_ROUTING_KEY,
            )

            # Fair dispatch: prefetch 1 message at a time
            channel.basic_qos(prefetch_count=1)

            channel.basic_consume(
                queue=RABBITMQ_QUEUE,
                on_message_callback=on_message,
            )

            logger.info(
                "Connected to RabbitMQ at %s:%d — consuming from '%s'",
                RABBITMQ_HOST, RABBITMQ_PORT, RABBITMQ_QUEUE,
            )
            backoff = 1.0  # Reset backoff on successful connection
            channel.start_consuming()

        except pika.exceptions.AMQPConnectionError as exc:
            logger.error(
                "RabbitMQ connection lost: %s — retrying in %.0fs", exc, backoff
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)
        except KeyboardInterrupt:
            logger.info("Consumer interrupted — shutting down.")
            break
        except Exception:
            logger.exception("Unexpected error in consumer loop — retrying in %.0fs", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)


if __name__ == "__main__":
    run_consumer()
