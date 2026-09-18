#!/usr/bin/env python3
"""
scripts/validate_lora_adapter.py
================================
LegalMind AI — LoRA Adapter Integrity & NaN Detection CLI

Validates:
1. adapter_config.json existence and parameter validity.
2. adapter_model.safetensors or pytorch_model.bin presence.
3. Tensor weights inspection for NaN and Inf values.
4. Norm and magnitude statistics across all adapter layers.
5. Automated fallback recommendation:
   - If adapter is VALID -> Eligible for inference.
   - If adapter contains NaNs -> Mandatory fallback to Base Qwen3.6-35B + RAG.

Usage:
    python scripts/validate_lora_adapter.py --adapter fine_tuning/adapters/lora-legal-v1
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "fine_tuning"))

from validate_adapter import validate_adapter_directory, inspect_safetensors_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate_lora_cli")


def main():
    parser = argparse.ArgumentParser(description="LegalMind AI LoRA Adapter Validator")
    parser.add_argument(
        "--adapter",
        "-a",
        default="fine_tuning/adapters/lora-legal-v1",
        help="Path to adapter directory",
    )
    args = parser.parse_args()

    adapter_path = Path(args.adapter).resolve()
    log.info("=" * 70)
    log.info(f"LEGALMINDAI — LORA ADAPTER SAFETY AUDIT: {adapter_path}")
    log.info("=" * 70)

    is_valid, report = validate_adapter_directory(adapter_path)

    log.info(f"Status        : {report.get('status', 'UNKNOWN')}")
    log.info(f"Is Valid      : {is_valid}")
    if "total_tensors" in report:
        log.info(f"Total Tensors : {report['total_tensors']}")
        log.info(f"Total Params  : {report['total_params']:,}")
        log.info(f"NaN Tensors   : {report['nan_tensors_count']}")
        log.info(f"Inf Tensors   : {report['inf_tensors_count']}")

    if not is_valid:
        log.warning("\n" + "!" * 70)
        log.warning("SAFETY AUDIT FAILED: ADAPTER CONTAINS CORRUPT OR INVALID TENSORS.")
        log.warning("ACTION REQUIRED: AUTOMATED FALLBACK TO BASE QWEN3.6-35B + RAG ENFORCED.")
        log.warning("!" * 70)
        sys.exit(1)
    else:
        log.info("\n" + "=" * 70)
        log.info("SAFETY AUDIT PASSED: ADAPTER IS 100% HEALTHY AND CLEAN.")
        log.info("=" * 70)
        sys.exit(0)


if __name__ == "__main__":
    main()
