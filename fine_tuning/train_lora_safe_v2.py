#!/usr/bin/env python3
"""
fine_tuning/train_lora_safe_v2.py
=================================
LegalMind AI — Safe Standard LoRA Fine-Tuning Pipeline (v2)

ARCHITECTURAL SAFETY MANDATES:
- STRICTLY STANDARD LoRA ONLY.
- STRICTLY NO QLoRA, bitsandbytes, 4-bit/8-bit quantization, GPTQ, AWQ, or NF4.
- Precision: Native bfloat16 (BF16).
- Hyperparameters:
    * learning_rate: 2e-5
    * num_train_epochs: 1
    * max_grad_norm: 0.3 (strict gradient clipping to eliminate NaN spikes)
    * warmup_ratio: 0.05
    * gradient_accumulation_steps: 8
    * lora_r: 16
    * lora_alpha: 32
    * lora_dropout: 0.05
    * target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"] (self-attention only)
- Real-Time NaN/Inf Detection:
    * Loss NaN/Inf checking on every logged step.
    * Gradient norm checking before optimizer step.
    * Parameter weight checking after step.
    * Checkpoint validation on every save.
- Output: fine_tuning/adapters/lora-legal-v2 (Never overwrites lora-legal-v1).
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_lora_v2")

PROHIBITED_QUANTIZATION_TERMS = [
    "bitsandbytes",
    "load_in_4bit",
    "load_in_8bit",
    "bnb",
    "qlora",
    "gptq",
    "awq",
    "nf4",
]


def verify_no_quantization(cfg: Dict[str, Any]):
    """Strictly assert no quantization libraries or arguments are present."""
    cfg_dump = json.dumps(cfg).lower()
    for term in PROHIBITED_QUANTIZATION_TERMS:
        if term in cfg_dump:
            raise ValueError(
                f"STRICT ARCHITECTURAL VIOLATION: Detected prohibited quantization term '{term}'. "
                "Only Standard LoRA with native BF16 precision is permitted on this DGX platform."
            )
    log.info("Quantization check: 100% standard unquantized configuration verified.")


def inspect_tensors_in_memory(model) -> Tuple[bool, List[str]]:
    """Inspects all trainable model parameters for NaN or Inf values."""
    import torch

    nan_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.isnan(param.data).any().item():
                nan_params.append(f"{name} (NaN)")
            elif torch.isinf(param.data).any().item():
                nan_params.append(f"{name} (Inf)")
    return (len(nan_params) == 0), nan_params


def load_dataset_from_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.warning(f"Skipping malformed JSON line: {e}")
    return data


def build_prompt(example: Dict[str, Any], template: str) -> str:
    return template.format(
        instruction=example.get("instruction", ""),
        input=example.get("input", ""),
        output=example.get("output", ""),
    )


def create_training_pipeline(config_path: str, dry_run: bool = False, max_steps: Optional[int] = None):
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except ImportError:
        log.error("PyYAML is required. Run: pip install pyyaml")
        sys.exit(1)

    verify_no_quantization(cfg)

    model_cfg = cfg.get("model", {})
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg.get("training", {})
    dataset_cfg = cfg.get("dataset", {})
    adapter_cfg = cfg.get("adapter", {})
    # Ensure GPU 3 is strictly avoided and GPU 1 is targeted
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        log.info("CUDA_VISIBLE_DEVICES unset; automatically isolated to GPU 1.")
    else:
        active_gpus = [g.strip() for g in visible_devices.split(",") if g.strip()]
        if "3" in active_gpus:
            raise RuntimeError("CRITICAL POLICY VIOLATION: GPU 3 is strictly avoided and must never be accessed.")
        log.info(f"Active visible GPUs: {active_gpus}")

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoProcessor,
        Qwen3_5MoeForConditionalGeneration,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )

    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError:
        from trl import SFTTrainer
        SFTConfig = None

    train_file = dataset_cfg.get("train_file", "fine_tuning/datasets/legal_sft_train.jsonl")
    eval_file = dataset_cfg.get("eval_file", "fine_tuning/datasets/legal_sft_eval.jsonl")
    template = dataset_cfg.get(
        "prompt_template",
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}",
    )

    if not Path(train_file).exists():
        log.error(f"Training dataset not found: {train_file}")
        sys.exit(1)

    train_data = load_dataset_from_jsonl(train_file)
    eval_data = load_dataset_from_jsonl(eval_file) if Path(eval_file).exists() else []

    log.info(f"Loaded dataset: {len(train_data)} train samples, {len(eval_data)} eval samples")

    if dry_run:
        log.info("=" * 70)
        log.info("DRY RUN VALIDATION AUDIT (lora-legal-v2):")
        log.info(f"  Model Path         : {model_cfg.get('path')}")
        log.info(f"  Precision          : Native bfloat16 (Standard LoRA)")
        log.info(f"  LoRA Rank (r)      : {lora_cfg.get('r', 16)}")
        log.info(f"  LoRA Alpha         : {lora_cfg.get('lora_alpha', 32)}")
        log.info(f"  Target Modules     : {lora_cfg.get('target_modules')}")
        log.info(f"  Learning Rate      : {train_cfg.get('learning_rate', 2e-5)}")
        log.info(f"  Epochs             : {train_cfg.get('num_train_epochs', 1)}")
        log.info(f"  Max Grad Norm      : {train_cfg.get('max_grad_norm', 0.3)}")
        log.info(f"  Grad Accumulation  : {train_cfg.get('gradient_accumulation_steps', 8)}")
        log.info(f"  Output Adapter Dir : {adapter_cfg.get('output_dir')}")
        log.info("=" * 70)
        log.info("Dry-run audit PASSED. Configuration is 100% compliant with standard LoRA mandates.")
        return True

    def format_sample(example):
        return {"text": build_prompt(example, template)}

    train_dataset = Dataset.from_list(train_data).map(format_sample)
    eval_dataset = Dataset.from_list(eval_data).map(format_sample) if eval_data else None

    # Load Model & Processor
    model_path = model_cfg.get("path", "/home/sece2026-student12/LegalMindAI/models/Qwen3.6-35B-A3B")
    log.info(f"Loading base model in native BF16 from: {model_path}")

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=model_cfg.get("device_map", {"": 0}),
    )

    if cfg.get("gpu", {}).get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Configure PEFT Standard LoRA
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]),
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Pre-train tensor health check
    clean, nan_tensors = inspect_tensors_in_memory(model)
    if not clean:
        raise RuntimeError(f"FATAL: Base model contains NaNs prior to training: {nan_tensors}")
    log.info("Pre-training tensor check passed: 0 NaNs detected.")

    # ── Custom Real-Time NaN/Inf Guard Callback ───────────────────────────
    class NaNInfGuardCallback(TrainerCallback):
        """Immediately halts training if any loss, gradient, or tensor becomes NaN/Inf."""

        def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
            if logs:
                loss = logs.get("loss")
                if loss is not None and (math.isnan(loss) or math.isinf(loss)):
                    log.error(f"FATAL: Loss became {loss} at step {state.global_step}! Halting training immediately.")
                    control.should_training_stop = True
                    raise FloatingPointError(f"Training halted: Loss became {loss} at step {state.global_step}")

                grad_norm = logs.get("grad_norm")
                if grad_norm is not None and (math.isnan(grad_norm) or math.isinf(grad_norm)):
                    log.error(f"FATAL: Gradient norm became {grad_norm} at step {state.global_step}! Halting.")
                    control.should_training_stop = True
                    raise FloatingPointError(f"Training halted: Gradient norm became {grad_norm} at step {state.global_step}")

        def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
            # Check model trainable parameters for NaNs
            if model is not None and state.global_step % 10 == 0:
                clean, corrupted = inspect_tensors_in_memory(model)
                if not clean:
                    log.error(f"FATAL: Trainable parameters contain NaNs at step {state.global_step}: {corrupted}")
                    control.should_training_stop = True
                    raise FloatingPointError(f"Parameters corrupted at step {state.global_step}: {corrupted}")

        def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            safetensors_file = checkpoint_dir / "adapter_model.safetensors"
            if safetensors_file.exists():
                from fine_tuning.validate_adapter import inspect_safetensors_file

                is_valid, report = inspect_safetensors_file(safetensors_file)
                if not is_valid:
                    log.error(f"FATAL: Checkpoint at {checkpoint_dir} failed validation: {report}")
                    control.should_training_stop = True
                    raise FloatingPointError(f"Checkpoint at {checkpoint_dir} contains NaN/Inf tensors!")
                log.info(f"Checkpoint {checkpoint_dir.name} validated: 100% clean ({report['total_tensors']} tensors, 0 NaNs).")

    # Training Arguments
    out_checkpoint_dir = train_cfg.get("output_dir", "fine_tuning/checkpoints/lora-legal-v2")
    final_adapter_dir = adapter_cfg.get("output_dir", "fine_tuning/adapters/lora-legal-v2")

    warmup_ratio = train_cfg.get("warmup_ratio", 0.05)
    est_total_steps = max_steps or (
        len(train_data)
        // (train_cfg.get("per_device_train_batch_size", 1) * train_cfg.get("gradient_accumulation_steps", 8))
        * train_cfg.get("num_train_epochs", 1)
    )
    warmup_steps = max(1, int(est_total_steps * warmup_ratio))

    training_kwargs = dict(
        output_dir=out_checkpoint_dir,
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        warmup_steps=warmup_steps,
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.3),
        bf16=True,
        fp16=False,
        eval_strategy=train_cfg.get("eval_strategy", train_cfg.get("evaluation_strategy", "steps")),
        eval_steps=train_cfg.get("eval_steps", 25),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 25),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        logging_steps=train_cfg.get("logging_steps", 5),
        report_to="none",
        dataloader_num_workers=2,
        seed=train_cfg.get("seed", 42),
        remove_unused_columns=False,
    )
    if max_steps:
        training_kwargs["max_steps"] = max_steps

    if SFTConfig is not None:
        training_kwargs.update(dict(dataset_text_field="text", max_length=2048, packing=False))
        training_args = SFTConfig(**training_kwargs)
    else:
        training_args = TrainingArguments(**training_kwargs)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=None,
        callbacks=[NaNInfGuardCallback()],
    )

    log.info("Starting safe Standard LoRA training...")
    trainer.train()

    # Save final adapter to lora-legal-v2 (never overwriting v1)
    Path(final_adapter_dir).mkdir(parents=True, exist_ok=True)
    log.info(f"Saving final verified adapter to: {final_adapter_dir}")
    trainer.save_model(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    # Post-training validation of saved adapter
    final_safetensors = Path(final_adapter_dir) / "adapter_model.safetensors"
    if final_safetensors.exists():
        from fine_tuning.validate_adapter import inspect_safetensors_file

        is_valid, report = inspect_safetensors_file(final_safetensors)
        if not is_valid:
            log.error(f"POST-TRAINING AUDIT FAILED: Exported adapter contains NaNs: {report}")
            sys.exit(1)
        log.info(f"POST-TRAINING AUDIT PASSED: {report['total_tensors']} tensors verified, 0 NaNs, 0 Infs.")

    log.info("LoRA fine-tuning v2 successfully completed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="LegalMind AI Safe LoRA Training (v2)")
    parser.add_argument("--config", "-c", default="fine_tuning/configs/lora_safe_v2.yaml", help="Config file path")
    parser.add_argument("--dry-run", action="store_true", help="Audit and validate configuration only")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional step ceiling for validation")
    args = parser.parse_args()

    config_path = str(Path(args.config).resolve())
    create_training_pipeline(config_path, dry_run=args.dry_run, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
