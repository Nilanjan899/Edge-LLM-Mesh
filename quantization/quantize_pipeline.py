#!/usr/bin/env python3
"""
quantize_pipeline.py — Kaggle-Ready GGUF Quantization Pipeline
================================================================

Designed to run inside a **Kaggle Notebook** with Dual T4 GPUs.
Downloads a HuggingFace model, converts it to GGUF format using
llama.cpp tooling, and writes the quantized artifact to /kaggle/working/.

Notebook Setup (paste these into sequential cells before running):
----------------------------------------------------------------------
Cell 1 — Clone repository & install deps:
    !git clone https://github.com/Nilanjan899/Edge-LLM-Mesh.git /kaggle/working/Edge-LLM-Mesh
    %cd /kaggle/working/Edge-LLM-Mesh/quantization
    !pip install -q -r requirements.txt

Cell 2 — Build llama.cpp with CMake (the old Makefile has been removed):
    !git clone https://github.com/ggerganov/llama.cpp /kaggle/working/llama.cpp
    %cd /kaggle/working/llama.cpp
    !pip install -q -r requirements.txt
    !cmake -B build -DGGML_CUDA=ON -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs
    !cmake --build build --config Release -j$(nproc)

Cell 3 — Run the pipeline:
    %cd /kaggle/working/Edge-LLM-Mesh/quantization
    !python quantize_pipeline.py \\
        --model-id "microsoft/Phi-3-mini-4k-instruct" \\
        --quant-type "q4_k_m" \\
        --output-dir "/kaggle/working/quantized_models"

Cell 4 — Verify & download artifact:
    import os
    output_dir = "/kaggle/working/quantized_models"
    for f in os.listdir(output_dir):
        size_gb = os.path.getsize(os.path.join(output_dir, f)) / 1e9
        print(f"{f} → {size_gb:.2f} GB")
----------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants – sensible defaults for a Kaggle T4 environment
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID: str = "microsoft/Phi-3-mini-4k-instruct"
DEFAULT_QUANT_TYPE: str = "q4_k_m"
DEFAULT_OUTPUT_DIR: str = "/kaggle/working/quantized_models"
DEFAULT_LLAMA_CPP_DIR: str = "/kaggle/working/llama.cpp"

# Supported quantization types (llama.cpp naming convention)
SUPPORTED_QUANT_TYPES: list[str] = [
    "q2_k", "q3_k_s", "q3_k_m", "q3_k_l",
    "q4_0", "q4_1", "q4_k_s", "q4_k_m",
    "q5_0", "q5_1", "q5_k_s", "q5_k_m",
    "q6_k", "q8_0", "f16",
]


def _run(cmd: list[str], cwd: Optional[str] = None) -> None:
    """Run a subprocess and stream output; raise on failure."""
    logger.info("$ %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            logger.info("  %s", line)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd)}"
        )


def download_model(model_id: str, cache_dir: str) -> Path:
    """Download a HuggingFace model snapshot to *cache_dir*.

    Args:
        model_id: HuggingFace repository identifier (e.g. ``microsoft/Phi-3-mini-4k-instruct``).
        cache_dir: Local directory to store the downloaded model files.

    Returns:
        Path to the downloaded model directory.
    """
    logger.info("Downloading model '%s' → %s", model_id, cache_dir)
    local_dir = os.path.join(cache_dir, model_id.replace("/", "_"))
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        ignore_patterns=["*.md", "*.txt", ".gitattributes"],
    )
    logger.info("Download complete: %s", local_dir)
    return Path(local_dir)


def _find_convert_script(llama_cpp_dir: str) -> str:
    """Locate the ``convert_hf_to_gguf.py`` script inside the llama.cpp tree.

    The script has moved between llama.cpp versions, so we check multiple
    known locations and return the first match.

    Args:
        llama_cpp_dir: Root directory of the cloned llama.cpp repository.

    Returns:
        Absolute path to the convert script.

    Raises:
        FileNotFoundError: If the script cannot be found at any known location.
    """
    candidates = [
        os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py"),
        os.path.join(llama_cpp_dir, "scripts", "convert_hf_to_gguf.py"),
        os.path.join(llama_cpp_dir, "gguf-py", "scripts", "convert_hf_to_gguf.py"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"convert_hf_to_gguf.py not found in {llama_cpp_dir}. "
        f"Searched: {candidates}"
    )


def _find_quantize_binary(llama_cpp_dir: str) -> str:
    """Locate the ``llama-quantize`` binary built by CMake.

    CMake places binaries in ``build/bin/`` by default, but we also check
    the repo root for legacy Makefile builds.

    Args:
        llama_cpp_dir: Root directory of the cloned llama.cpp repository.

    Returns:
        Absolute path to the llama-quantize binary.

    Raises:
        FileNotFoundError: If the binary cannot be found.
    """
    candidates = [
        os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize"),
        os.path.join(llama_cpp_dir, "build", "llama-quantize"),
        os.path.join(llama_cpp_dir, "llama-quantize"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"llama-quantize binary not found. "
        f"Searched: {candidates}. "
        "Ensure llama.cpp was built with: "
        "cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j$(nproc)"
    )


def convert_to_gguf(
    model_dir: Path,
    output_dir: Path,
    llama_cpp_dir: str,
) -> Path:
    """Convert a HuggingFace model directory to F16 GGUF using llama.cpp.

    Args:
        model_dir: Path to the downloaded HuggingFace model.
        output_dir: Directory where the intermediate F16 GGUF will be written.
        llama_cpp_dir: Root directory of the compiled llama.cpp repository.

    Returns:
        Path to the generated F16 GGUF file.
    """
    convert_script = _find_convert_script(llama_cpp_dir)
    logger.info("Using convert script: %s", convert_script)

    f16_path = output_dir / "model-f16.gguf"
    _run(
        [
            sys.executable, convert_script,
            str(model_dir),
            "--outfile", str(f16_path),
            "--outtype", "f16",
        ]
    )
    if not f16_path.exists():
        raise FileNotFoundError(f"Expected F16 GGUF not found at {f16_path}")
    logger.info("F16 GGUF created: %s (%.2f GB)", f16_path, f16_path.stat().st_size / 1e9)
    return f16_path


def quantize_gguf(
    f16_path: Path,
    quant_type: str,
    output_dir: Path,
    llama_cpp_dir: str,
) -> Path:
    """Quantize an F16 GGUF to a smaller representation.

    Args:
        f16_path: Path to the F16 GGUF file.
        quant_type: llama.cpp quantization type string (e.g. ``q4_k_m``).
        output_dir: Directory for the final quantized GGUF file.
        llama_cpp_dir: Root directory of the compiled llama.cpp repository.

    Returns:
        Path to the quantized GGUF artifact.
    """
    if quant_type not in SUPPORTED_QUANT_TYPES:
        raise ValueError(
            f"Unsupported quant type '{quant_type}'. Choose from: {SUPPORTED_QUANT_TYPES}"
        )

    quantize_bin = _find_quantize_binary(llama_cpp_dir)
    logger.info("Using quantize binary: %s", quantize_bin)

    quant_path = output_dir / f"model-{quant_type}.gguf"
    _run([quantize_bin, str(f16_path), str(quant_path), quant_type.upper()])

    if not quant_path.exists():
        raise FileNotFoundError(f"Quantized GGUF not found at {quant_path}")
    logger.info(
        "Quantized GGUF created: %s (%.2f GB)",
        quant_path,
        quant_path.stat().st_size / 1e9,
    )
    return quant_path


def cleanup_intermediate(f16_path: Path) -> None:
    """Remove intermediate F16 GGUF to free disk space on Kaggle.

    Args:
        f16_path: Path to the intermediate F16 file.
    """
    if f16_path.exists():
        logger.info("Cleaning up intermediate file: %s", f16_path)
        f16_path.unlink()


def main() -> None:
    """Entry point for the quantization pipeline."""
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace model and quantize it to GGUF format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--quant-type",
        type=str,
        default=DEFAULT_QUANT_TYPE,
        choices=SUPPORTED_QUANT_TYPES,
        help=f"Quantization type (default: {DEFAULT_QUANT_TYPE})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for GGUF artifacts (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--keep-f16",
        action="store_true",
        help="Keep the intermediate F16 GGUF file (default: delete to save space)",
    )
    parser.add_argument(
        "--llama-cpp-dir",
        type=str,
        default=DEFAULT_LLAMA_CPP_DIR,
        help=f"Path to compiled llama.cpp repo (default: {DEFAULT_LLAMA_CPP_DIR})",
    )
    args = parser.parse_args()

    llama_cpp_dir: str = args.llama_cpp_dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = str(output_dir / "_hf_cache")

    # ---- Step 1: Download ----
    logger.info("=" * 60)
    logger.info("STEP 1/3  ·  Downloading model from HuggingFace")
    logger.info("=" * 60)
    model_dir = download_model(args.model_id, cache_dir)

    # ---- Step 2: Convert to F16 GGUF ----
    logger.info("=" * 60)
    logger.info("STEP 2/3  ·  Converting to F16 GGUF")
    logger.info("=" * 60)
    f16_path = convert_to_gguf(model_dir, output_dir, llama_cpp_dir)

    # ---- Step 3: Quantize ----
    logger.info("=" * 60)
    logger.info("STEP 3/3  ·  Quantizing to %s", args.quant_type)
    logger.info("=" * 60)
    quant_path = quantize_gguf(f16_path, args.quant_type, output_dir, llama_cpp_dir)

    # ---- Cleanup ----
    if not args.keep_f16:
        cleanup_intermediate(f16_path)

    # ---- Optionally remove HF cache to reclaim ~10+ GB ----
    if os.path.isdir(cache_dir):
        logger.info("Removing HF cache at %s", cache_dir)
        shutil.rmtree(cache_dir, ignore_errors=True)

    logger.info("=" * 60)
    logger.info("✅  Done!  Quantized model artifact:")
    logger.info("    %s  (%.2f GB)", quant_path, quant_path.stat().st_size / 1e9)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
