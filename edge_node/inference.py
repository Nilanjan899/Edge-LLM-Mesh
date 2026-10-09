#!/usr/bin/env python3
"""
inference.py — Edge LLM Inference Engine
==========================================

Lightweight CLI / importable module for running GGUF-quantized LLMs
on edge hardware via ``llama-cpp-python``.

Every inference call is automatically instrumented: execution time,
prompt/completion token counts, and system memory usage are recorded
into the local SQLite telemetry cache for later upload by the
``telemetry_client`` daemon.

Usage (CLI)::

    python inference.py --model ./models/phi-3-q4_k_m.gguf --prompt "Explain MLOps in 3 sentences."

Usage (Interactive REPL)::

    python inference.py --model ./models/phi-3-q4_k_m.gguf --interactive

Usage (as library)::

    from inference import EdgeLLM
    llm = EdgeLLM("./models/phi-3-q4_k_m.gguf")
    response = llm.generate("Hello, world!")
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import psutil

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None  # type: ignore[assignment,misc]
    print(
        "[WARN] llama-cpp-python not installed. "
        "Install with: pip install llama-cpp-python",
        file=sys.stderr,
    )

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("edge.inference")

# ---------------------------------------------------------------------------
# Telemetry DB setup (shared across threads)
# ---------------------------------------------------------------------------
_DB_PATH = os.environ.get(
    "EDGE_TELEMETRY_DB",
    str(Path(__file__).resolve().parent / "local_cache.db"),
)
_DB_LOCK = threading.Lock()


def _init_db(db_path: str = _DB_PATH) -> None:
    """Create the telemetry table if it doesn't exist.

    Uses WAL journal mode for better concurrent read/write performance
    and is safe against thread collisions when combined with ``_DB_LOCK``.

    Args:
        db_path: Absolute path to the SQLite database file.
    """
    with _DB_LOCK:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS telemetry (
                id            TEXT PRIMARY KEY,
                timestamp     REAL NOT NULL,
                device_id     TEXT NOT NULL,
                model_name    TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_time_ms REAL NOT NULL,
                ttft_ms       REAL,
                tokens_per_sec REAL NOT NULL,
                ram_usage_mb  REAL NOT NULL,
                vram_usage_mb REAL,
                prompt_text   TEXT,
                synced        INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        conn.commit()
        conn.close()


def _log_telemetry(record: "InferenceRecord", db_path: str = _DB_PATH) -> None:
    """Insert a single inference record into the local SQLite cache.

    Thread-safe: acquires ``_DB_LOCK`` before writing.

    Args:
        record: Populated :class:`InferenceRecord` dataclass.
        db_path: Path to the SQLite database.
    """
    with _DB_LOCK:
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                """
                INSERT INTO telemetry
                    (id, timestamp, device_id, model_name,
                     prompt_tokens, completion_tokens, total_time_ms,
                     ttft_ms, tokens_per_sec, ram_usage_mb, vram_usage_mb,
                     prompt_text, synced)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    record.id,
                    record.timestamp,
                    record.device_id,
                    record.model_name,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_time_ms,
                    record.ttft_ms,
                    record.tokens_per_sec,
                    record.ram_usage_mb,
                    record.vram_usage_mb,
                    record.prompt_text,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    logger.debug("Telemetry logged: %s  (%.1f tok/s)", record.id[:8], record.tokens_per_sec)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class InferenceRecord:
    """Immutable snapshot of a single inference invocation's metrics."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)
    device_id: str = field(default_factory=lambda: os.environ.get("EDGE_DEVICE_ID", "edge-default"))
    model_name: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_time_ms: float = 0.0
    ttft_ms: Optional[float] = None
    tokens_per_sec: float = 0.0
    ram_usage_mb: float = 0.0
    vram_usage_mb: Optional[float] = None
    prompt_text: Optional[str] = None


# ---------------------------------------------------------------------------
# Core LLM wrapper
# ---------------------------------------------------------------------------
class EdgeLLM:
    """Thin wrapper around ``llama-cpp-python`` that instruments every call.

    Args:
        model_path: Path to a ``.gguf`` model file.
        n_ctx: Context window size in tokens.
        n_gpu_layers: Number of layers to offload to GPU (``-1`` = all).
        device_id: Identifier for this edge node (shows up in telemetry).
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        device_id: Optional[str] = None,
    ) -> None:
        if Llama is None:
            raise ImportError("llama-cpp-python is required but not installed.")

        self.model_path = model_path
        self.model_name = Path(model_path).stem
        self.device_id = device_id or os.environ.get("EDGE_DEVICE_ID", "edge-default")

        logger.info("Loading model: %s  (n_ctx=%d, n_gpu_layers=%d)", model_path, n_ctx, n_gpu_layers)
        self._llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )
        _init_db()
        logger.info("Model loaded successfully.")

    # ------------------------------------------------------------------
    # System metrics helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _get_ram_usage_mb() -> float:
        """Return current process RSS in megabytes."""
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)

    @staticmethod
    def _get_vram_usage_mb() -> Optional[float]:
        """Attempt to read VRAM usage via nvidia-smi. Returns ``None`` on failure."""
        try:
            import subprocess

            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            # Sum all GPUs
            total = sum(float(x.strip()) for x in result.stdout.strip().split("\n") if x.strip())
            return total
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate a completion and log telemetry.

        Args:
            prompt: The input text.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            stop: Optional stop sequences.

        Returns:
            The generated completion text.
        """
        ram_before = self._get_ram_usage_mb()
        vram_before = self._get_vram_usage_mb()

        t_start = time.perf_counter()
        ttft: Optional[float] = None

        # --- Run inference ---
        output = self._llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop or [],
            echo=False,
        )

        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0

        # --- Extract token counts from llama.cpp usage dict ---
        usage = output.get("usage", {})
        prompt_tokens: int = usage.get("prompt_tokens", 0)
        completion_tokens: int = usage.get("completion_tokens", 0)
        generated_text: str = output["choices"][0]["text"] if output.get("choices") else ""

        # Approximate TTFT as total_time * (prompt_tokens / total_tokens) if not streaming
        total_tokens = prompt_tokens + completion_tokens
        if total_tokens > 0 and prompt_tokens > 0:
            ttft = total_ms * (prompt_tokens / total_tokens)

        tokens_per_sec = (completion_tokens / (total_ms / 1000.0)) if total_ms > 0 else 0.0

        ram_after = self._get_ram_usage_mb()
        vram_after = self._get_vram_usage_mb()

        record = InferenceRecord(
            device_id=self.device_id,
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_time_ms=total_ms,
            ttft_ms=ttft,
            tokens_per_sec=tokens_per_sec,
            ram_usage_mb=max(ram_before, ram_after),
            vram_usage_mb=max(vram_before, vram_after) if vram_before and vram_after else vram_after or vram_before,
            prompt_text=prompt[:500],  # Truncate for storage
        )
        _log_telemetry(record)

        logger.info(
            "Generated %d tokens in %.0f ms (%.1f tok/s) | RAM %.0f MB",
            completion_tokens, total_ms, tokens_per_sec, record.ram_usage_mb,
        )
        return generated_text

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Chat-style inference using a list of ``{"role": ..., "content": ...}`` messages.

        Converts the messages into a single prompt string and delegates to
        :meth:`generate`.

        Args:
            messages: List of chat messages.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The assistant's reply text.
        """
        # Simple template — works for most instruct models
        prompt_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}\n")
        prompt_parts.append("<|assistant|>\n")
        full_prompt = "".join(prompt_parts)

        return self.generate(full_prompt, max_tokens=max_tokens, temperature=temperature)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Edge LLM Inference — run a GGUF model locally with telemetry.",
    )
    parser.add_argument("--model", type=str, required=True, help="Path to .gguf model file")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (non-interactive)")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive chat REPL")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context window size")
    parser.add_argument("--n-gpu-layers", type=int, default=0, help="GPU layers to offload (-1=all)")
    parser.add_argument("--device-id", type=str, default=None, help="Edge device identifier")
    return parser


def main() -> None:
    """CLI entry point."""
    args = _build_parser().parse_args()

    llm = EdgeLLM(
        model_path=args.model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        device_id=args.device_id,
    )

    if args.interactive:
        print("\n🤖 Edge LLM Interactive Chat  (type 'quit' to exit)\n" + "─" * 50)
        history: list[dict[str, str]] = []
        while True:
            try:
                user_input = input("\n📝 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not user_input:
                continue
            history.append({"role": "user", "content": user_input})
            response = llm.chat(history, max_tokens=args.max_tokens, temperature=args.temperature)
            history.append({"role": "assistant", "content": response})
            print(f"\n🤖 Assistant: {response}")
    elif args.prompt:
        response = llm.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temperature)
        print(response)
    else:
        print("Error: Provide --prompt or --interactive", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
