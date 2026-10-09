#!/usr/bin/env python3
"""
telemetry_client.py — Circuit-Breaker / Offline-First Telemetry Daemon
========================================================================

Background daemon that runs alongside the edge inference engine.
Implements the **Offline-First** pattern:

1. All inference metrics are written to a local SQLite DB first (by ``inference.py``).
2. This daemon periodically scans the DB for un-synced rows.
3. If the central server is reachable, it batches those rows into a JSON payload
   and POSTs them to the central FastAPI ``/ingest/telemetry`` endpoint.
4. Successfully synced rows are marked ``synced=1`` in the local DB.
5. If the network is down, the daemon backs off exponentially (circuit breaker)
   and retries later — no data is ever lost.

Usage::

    # Start the daemon (runs forever in the background)
    python telemetry_client.py --server-url http://central-server:8000

    # Or import and start programmatically
    from telemetry_client import TelemetryDaemon
    daemon = TelemetryDaemon("http://central-server:8000")
    daemon.start()
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("edge.telemetry")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DB_PATH = os.environ.get(
    "EDGE_TELEMETRY_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_cache.db"),
)
_DB_LOCK = threading.Lock()

DEFAULT_SERVER_URL = "http://localhost:8000"
DEFAULT_FLUSH_INTERVAL_SEC: int = 30
DEFAULT_BATCH_SIZE: int = 50
DEFAULT_MAX_RETRIES: int = 5
DEFAULT_HEALTH_ENDPOINT: str = "/health"
DEFAULT_INGEST_ENDPOINT: str = "/ingest/telemetry"


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------
class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()      # Normal operation — requests flow through.
    OPEN = auto()        # Tripped — skip requests, wait for cooldown.
    HALF_OPEN = auto()   # Trial — allow a single probe request.


@dataclass
class CircuitBreaker:
    """Simple circuit breaker with exponential back-off.

    Attributes:
        failure_threshold: Consecutive failures before tripping to OPEN.
        recovery_timeout: Base seconds to wait before transitioning to HALF_OPEN.
        max_backoff: Maximum back-off time in seconds.
    """

    failure_threshold: int = 3
    recovery_timeout: float = 10.0
    max_backoff: float = 300.0  # 5 minutes

    # Internal state
    _state: CircuitState = CircuitState.CLOSED
    _failure_count: int = 0
    _last_failure_time: float = 0.0
    _current_backoff: float = 10.0

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state, auto-transitioning from OPEN → HALF_OPEN."""
        if self._state == CircuitState.OPEN:
            elapsed = time.time() - self._last_failure_time
            if elapsed >= self._current_backoff:
                logger.info(
                    "Circuit breaker → HALF_OPEN (after %.0fs cooldown)", self._current_backoff
                )
                self._state = CircuitState.HALF_OPEN
        return self._state

    def record_success(self) -> None:
        """Record a successful request — reset breaker to CLOSED."""
        if self._state != CircuitState.CLOSED:
            logger.info("Circuit breaker → CLOSED (connection restored)")
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._current_backoff = self.recovery_timeout

    def record_failure(self) -> None:
        """Record a failed request — increment counter, potentially trip to OPEN."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._current_backoff = min(self._current_backoff * 2, self.max_backoff)
            logger.warning(
                "Circuit breaker → OPEN (failures=%d, backoff=%.0fs)",
                self._failure_count,
                self._current_backoff,
            )


# ---------------------------------------------------------------------------
# Telemetry Daemon
# ---------------------------------------------------------------------------
class TelemetryDaemon:
    """Background daemon that flushes local telemetry to the central server.

    Args:
        server_url: Base URL of the central FastAPI server.
        db_path: Path to the local SQLite telemetry database.
        flush_interval: Seconds between flush attempts.
        batch_size: Maximum rows to send per batch.
    """

    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        db_path: str = _DB_PATH,
        flush_interval: int = DEFAULT_FLUSH_INTERVAL_SEC,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.db_path = db_path
        self.flush_interval = flush_interval
        self.batch_size = batch_size

        self._breaker = CircuitBreaker()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Database helpers
    # ------------------------------------------------------------------
    def _get_unsynced(self) -> list[dict[str, Any]]:
        """Fetch un-synced telemetry rows from the local cache.

        Returns:
            List of row dicts ready for JSON serialization.
        """
        with _DB_LOCK:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.execute(
                    "SELECT * FROM telemetry WHERE synced = 0 ORDER BY timestamp ASC LIMIT ?",
                    (self.batch_size,),
                )
                rows = [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()
        return rows

    def _mark_synced(self, ids: list[str]) -> None:
        """Mark rows as synced in the local DB.

        Args:
            ids: List of telemetry record IDs to mark.
        """
        if not ids:
            return
        with _DB_LOCK:
            conn = sqlite3.connect(self.db_path)
            try:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"UPDATE telemetry SET synced = 1 WHERE id IN ({placeholders})",
                    ids,
                )
                conn.commit()
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------
    def _health_check(self) -> bool:
        """Ping the central server health endpoint.

        Returns:
            ``True`` if the server responds with 2xx.
        """
        try:
            resp = requests.get(
                f"{self.server_url}{DEFAULT_HEALTH_ENDPOINT}",
                timeout=5,
            )
            return resp.status_code < 300
        except requests.RequestException:
            return False

    def _send_batch(self, records: list[dict[str, Any]]) -> bool:
        """POST a batch of telemetry records to the central server.

        Args:
            records: List of telemetry row dicts.

        Returns:
            ``True`` if the server accepted the batch.
        """
        payload = {
            "device_id": records[0].get("device_id", "unknown"),
            "batch": records,
        }
        try:
            resp = requests.post(
                f"{self.server_url}{DEFAULT_INGEST_ENDPOINT}",
                json=payload,
                timeout=15,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code < 300:
                return True
            logger.warning("Server responded with %d: %s", resp.status_code, resp.text[:200])
            return False
        except requests.RequestException as exc:
            logger.warning("Failed to send batch: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Flush loop
    # ------------------------------------------------------------------
    def _flush_once(self) -> None:
        """Perform a single flush attempt (called from the background loop)."""
        # Check circuit breaker
        state = self._breaker.state
        if state == CircuitState.OPEN:
            logger.debug("Circuit OPEN — skipping flush.")
            return

        # Health check (acts as the HALF_OPEN probe)
        if not self._health_check():
            self._breaker.record_failure()
            return

        # Fetch un-synced records
        records = self._get_unsynced()
        if not records:
            logger.debug("No un-synced records to flush.")
            self._breaker.record_success()
            return

        logger.info("Flushing %d un-synced telemetry records…", len(records))

        # Send batch
        if self._send_batch(records):
            ids = [r["id"] for r in records]
            self._mark_synced(ids)
            self._breaker.record_success()
            logger.info("✅  Successfully synced %d records.", len(ids))
        else:
            self._breaker.record_failure()

    def _run_loop(self) -> None:
        """Main daemon loop — runs in a background thread."""
        logger.info(
            "Telemetry daemon started (server=%s, interval=%ds, batch=%d)",
            self.server_url,
            self.flush_interval,
            self.batch_size,
        )
        while not self._stop_event.is_set():
            try:
                self._flush_once()
            except Exception:
                logger.exception("Unexpected error in flush loop")
            self._stop_event.wait(timeout=self.flush_interval)
        logger.info("Telemetry daemon stopped.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background flush thread (daemon thread — dies with main)."""
        if self._thread and self._thread.is_alive():
            logger.warning("Daemon already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="telemetry-flush")
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        """Signal the daemon to stop and wait for it to finish.

        Args:
            timeout: Maximum seconds to wait for the thread to join.
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def flush_now(self) -> None:
        """Trigger an immediate flush (useful for testing)."""
        self._flush_once()

    def get_stats(self) -> dict[str, Any]:
        """Return daemon diagnostics.

        Returns:
            Dictionary with circuit breaker state, pending count, etc.
        """
        with _DB_LOCK:
            conn = sqlite3.connect(self.db_path)
            try:
                (total,) = conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()
                (pending,) = conn.execute("SELECT COUNT(*) FROM telemetry WHERE synced = 0").fetchone()
            finally:
                conn.close()
        return {
            "circuit_state": self._breaker.state.name,
            "total_records": total,
            "pending_sync": pending,
            "server_url": self.server_url,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the telemetry daemon as a standalone process."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Edge telemetry daemon — flushes local metrics to the central server.",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default=DEFAULT_SERVER_URL,
        help=f"Central server base URL (default: {DEFAULT_SERVER_URL})",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=_DB_PATH,
        help="Path to the local SQLite telemetry DB",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_FLUSH_INTERVAL_SEC,
        help=f"Flush interval in seconds (default: {DEFAULT_FLUSH_INTERVAL_SEC})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Max records per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    args = parser.parse_args()

    daemon = TelemetryDaemon(
        server_url=args.server_url,
        db_path=args.db_path,
        flush_interval=args.interval,
        batch_size=args.batch_size,
    )

    daemon.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
            stats = daemon.get_stats()
            logger.info(
                "Daemon heartbeat — circuit=%s, pending=%d, total=%d",
                stats["circuit_state"],
                stats["pending_sync"],
                stats["total_records"],
            )
    except KeyboardInterrupt:
        logger.info("Shutting down…")
        daemon.stop()


if __name__ == "__main__":
    main()
