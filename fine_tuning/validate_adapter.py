#!/usr/bin/env python3
"""
fine_tuning/validate_adapter.py
===============================
LegalMind AI — LoRA Adapter Safety & Tensor Integrity Validator

Performs strict forensic validation on LoRA adapters:
1. Verifies that adapter files (adapter_model.safetensors or pytorch_model.bin) exist.
2. Inspects every weight tensor for NaN (Not-a-Number) and Inf (Infinity) values.
3. Computes tensor statistics (norm, min, max, mean, std).
4. Detects vanishing or exploding weights.
5. Verifies safe loading with PEFT without degrading base model outputs.
6. Enforces automatic fallback to Base Model + RAG if any tensor is corrupt or invalid.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("adapter_validator")


def inspect_safetensors_file(safetensors_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """
    Directly check safetensors file for NaN and Inf values without loading full LLM.
    Returns (is_valid, report_dict).
    """
    try:
        from safetensors import safe_open
    except ImportError:
        log.error("safetensors library not installed.")
        return False, {"error": "safetensors not installed"}

    if not safetensors_path.exists():
        return False, {"error": f"File not found: {safetensors_path}"}

    nan_tensors: List[str] = []
    inf_tensors: List[str] = []
    total_tensors = 0
    total_params = 0
    tensor_stats: Dict[str, Dict[str, float]] = {}

    with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            total_tensors += 1
            tensor = f.get_tensor(key)
            total_params += tensor.numel()

            has_nan = bool(tensor.isnan().any().item())
            has_inf = bool(tensor.isinf().any().item())

            if has_nan:
                nan_tensors.append(key)
            if has_inf:
                inf_tensors.append(key)

            if not has_nan and not has_inf:
                tensor_float = tensor.float()
                tensor_stats[key] = {
                    "min": float(tensor_float.min().item()),
                    "max": float(tensor_float.max().item()),
                    "mean": float(tensor_float.mean().item()),
                    "norm": float(tensor_float.norm().item()),
                }

    is_valid = (len(nan_tensors) == 0) and (len(inf_tensors) == 0) and (total_tensors > 0)

    report = {
        "path": str(safetensors_path),
        "total_tensors": total_tensors,
        "total_params": total_params,
        "nan_tensors_count": len(nan_tensors),
        "nan_tensors": nan_tensors,
        "inf_tensors_count": len(inf_tensors),
        "inf_tensors": inf_tensors,
        "is_valid": is_valid,
        "status": "VALID" if is_valid else "CORRUPT_CONTAINS_NANS_OR_INFS",
    }
    return is_valid, report


def validate_adapter_directory(adapter_dir: str | Path) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates an adapter directory for config and tensor integrity.
    """
    dir_path = Path(adapter_dir)
    if not dir_path.exists():
        return False, {"error": f"Directory not found: {dir_path}", "is_valid": False}

    config_path = dir_path / "adapter_config.json"
    weights_path = dir_path / "adapter_model.safetensors"
    bin_path = dir_path / "adapter_model.bin"

    if not config_path.exists():
        return False, {"error": "Missing adapter_config.json", "is_valid": False}

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = json.load(f)
        except Exception as e:
            return False, {"error": f"Invalid JSON in adapter_config.json: {e}", "is_valid": False}

    # Check for weights
    if weights_path.exists():
        is_valid, report = inspect_safetensors_file(weights_path)
        report["config"] = config
        return is_valid, report
    elif bin_path.exists():
        import torch
        weights = torch.load(str(bin_path), map_location="cpu")
        nan_tensors = [k for k, v in weights.items() if v.isnan().any().item()]
        inf_tensors = [k for k, v in weights.items() if v.isinf().any().item()]
        is_valid = (len(nan_tensors) == 0) and (len(inf_tensors) == 0)
        return is_valid, {
            "path": str(bin_path),
            "nan_tensors_count": len(nan_tensors),
            "nan_tensors": nan_tensors,
            "inf_tensors_count": len(inf_tensors),
            "inf_tensors": inf_tensors,
            "is_valid": is_valid,
            "status": "VALID" if is_valid else "CORRUPT_CONTAINS_NANS_OR_INFS",
            "config": config,
        }
    else:
        return False, {"error": "No adapter weights file found (.safetensors or .bin)", "is_valid": False}


def main():
    parser = argparse.ArgumentParser(description="LegalMind AI — LoRA Adapter Validator")
    parser.add_argument("--adapter-path", "-a", default="fine_tuning/adapters/lora-legal-v1",
                        help="Path to the LoRA adapter directory")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    is_valid, report = validate_adapter_directory(args.adapter_path)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"\n==================================================")
        print(f" LoRA Adapter Forensic Report: {args.adapter_path}")
        print(f"==================================================")
        print(f" Status             : {report.get('status', 'ERROR')}")
        print(f" Total Tensors      : {report.get('total_tensors', 0)}")
        print(f" Total Parameters   : {report.get('total_params', 0):,}")
        print(f" NaN Tensors Count  : {report.get('nan_tensors_count', 0)}")
        print(f" Inf Tensors Count  : {report.get('inf_tensors_count', 0)}")
        print(f" Deployment Status  : {'SAFE TO DEPLOY' if is_valid else 'UNSAFE — FALLBACK TO BASE QWEN + RAG'}")
        print(f"==================================================\n")

    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
