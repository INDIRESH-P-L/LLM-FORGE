#!/usr/bin/env python3
"""
fine_tuning/train_lora.py
=========================
LegalMind AI — LoRA Fine-Tuning Script

Fine-tunes Qwen3.6-35B-A3B with PEFT LoRA using the SFT Trainer from TRL.

Training teaches the model:
  - Legal response structure (Answer / Provisions / Judgments / Reasoning)
  - Citation formatting
  - Evidence-grounded answering
  - Legal issue extraction
  - Case summarization
  - Uncertainty / abstention

IMPORTANT: Run ONLY after the RAG system is working.
This does NOT fine-tune legal knowledge into the model — only reasoning style.

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=1 python fine_tuning/train_lora.py --config fine_tuning/configs/lora_config.yaml

    # Multi-GPU (GPUs 1 and 2)
    CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 fine_tuning/train_lora.py --config fine_tuning/configs/lora_config.yaml
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_lora")


def load_config(config_path: str) -> dict:
    try:
        import yaml
        with open(config_path) as f:
            return yaml.safe_load(f)
    except ImportError:
        log.error("PyYAML not installed. Run: pip install pyyaml")
        sys.exit(1)


def build_prompt(example: dict, template: str) -> str:
    """Format a single training example using the prompt template."""
    return template.format(
        instruction = example.get("instruction", ""),
        input       = example.get("input", ""),
        output      = example.get("output", ""),
    )


def load_dataset_from_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    log.warning(f"Skipping malformed JSON: {e}")
    return data


def main():
    parser = argparse.ArgumentParser(description="LegalMind AI — LoRA Fine-Tuning")
    parser.add_argument("--config", "-c", default="fine_tuning/configs/lora_config.yaml",
                        help="Path to lora_config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load config and dataset only, do not train")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Load config ──────────────────────────────────────────────────────────
    log.info(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    model_cfg    = cfg.get("model", {})
    lora_cfg     = cfg.get("lora", {})
    train_cfg    = cfg.get("training", {})
    dataset_cfg  = cfg.get("dataset", {})
    adapter_cfg  = cfg.get("adapter", {})

    # ── Check required packages ──────────────────────────────────────────────
    missing = []
    for pkg in ["torch", "transformers", "peft", "trl", "datasets"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error(f"Missing packages: {', '.join(missing)}")
        log.error("Run: pip install " + " ".join(missing))
        sys.exit(1)

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (AutoProcessor,
                              Qwen3_5MoeForConditionalGeneration)
    try:
        from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
    except ImportError:
        from trl import SFTTrainer
        DataCollatorForCompletionOnlyLM = None

    log.info(f"PyTorch version : {torch.__version__}")
    log.info(f"CUDA available  : {torch.cuda.is_available()}")
    log.info(f"Visible GPUs    : {torch.cuda.device_count()}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_file = dataset_cfg.get("train_file", "fine_tuning/datasets/legal_sft_train.jsonl")
    eval_file  = dataset_cfg.get("eval_file",  "fine_tuning/datasets/legal_sft_eval.jsonl")
    template   = dataset_cfg.get("prompt_template", "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}")

    if not Path(train_file).exists():
        log.error(f"Training dataset not found: {train_file}")
        log.error("Build the dataset using fine_tuning/datasets/ first.")
        sys.exit(1)

    log.info(f"Loading training dataset: {train_file}")
    train_data = load_dataset_from_jsonl(train_file)
    log.info(f"  Training examples: {len(train_data)}")

    eval_data = []
    if Path(eval_file).exists():
        eval_data = load_dataset_from_jsonl(eval_file)
        log.info(f"  Eval examples: {len(eval_data)}")

    if args.dry_run:
        log.info("Dry run audit summary:")
        log.info(f"  Base Model Path  : {model_cfg.get('path')}")
        log.info(f"  Precision        : {'bfloat16' if train_cfg.get('bf16') else 'float32'} (Standard LoRA, unquantized)")
        log.info(f"  LoRA Rank (r)    : {lora_cfg.get('r', 16)}")
        log.info(f"  LoRA Alpha       : {lora_cfg.get('lora_alpha', 32)}")
        log.info(f"  LoRA Dropout     : {lora_cfg.get('lora_dropout', 0.05)}")
        log.info(f"  Target Modules   : {lora_cfg.get('target_modules', [])}")
        log.info(f"  Learning Rate    : {train_cfg.get('learning_rate', 2e-5)}")
        log.info(f"  Max Grad Norm    : {train_cfg.get('max_grad_norm', 0.3)}")
        log.info(f"  Batch Size / Acc : {train_cfg.get('per_device_train_batch_size', 1)} / {train_cfg.get('gradient_accumulation_steps', 8)}")
        log.info(f"  Train Samples    : {len(train_data)}")
        log.info(f"  Eval Samples     : {len(eval_data)}")
        log.info("✅ Dry run audit successful. Configuration is valid for Standard LoRA.")
        return

    # ── Build HF Dataset with formatted prompts ───────────────────────────
    def format_sample(example):
        return {"text": build_prompt(example, template)}

    train_dataset = Dataset.from_list(train_data).map(format_sample)
    eval_dataset  = Dataset.from_list(eval_data).map(format_sample) if eval_data else None

    # ── Load model + processor ────────────────────────────────────────────
    model_path = model_cfg.get("path", "/home/sece2026-student12/LegalMindAI/models/Qwen3.6-35B-A3B")
    log.info(f"Loading base model: {model_path}")

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if model_cfg.get("dtype") == "bfloat16" else torch.float16,
        device_map=model_cfg.get("device_map", "auto"),
    )

    if train_cfg.get("gradient_checkpointing") or cfg.get("gpu", {}).get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # ── LoRA config ───────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r              = lora_cfg.get("r", 16),
        lora_alpha     = lora_cfg.get("lora_alpha", 32),
        lora_dropout   = lora_cfg.get("lora_dropout", 0.05),
        bias           = lora_cfg.get("bias", "none"),
        task_type      = TaskType.CAUSAL_LM,
        target_modules = lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
    )

    # ── Training arguments ────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir                  = train_cfg.get("output_dir", "fine_tuning/checkpoints/lora-legal-v1"),
        num_train_epochs            = train_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size = train_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size  = train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 8),
        learning_rate               = train_cfg.get("learning_rate", 2e-4),
        warmup_steps                = 100,
        lr_scheduler_type           = train_cfg.get("lr_scheduler_type", "cosine"),
        weight_decay                = train_cfg.get("weight_decay", 0.01),
        max_grad_norm               = train_cfg.get("max_grad_norm", 1.0),
        bf16                        = train_cfg.get("bf16", True),
        fp16                        = train_cfg.get("fp16", False),
        eval_strategy               = train_cfg.get("evaluation_strategy", "steps"),
        eval_steps                  = train_cfg.get("eval_steps", 100),
        save_strategy               = train_cfg.get("save_strategy", "steps"),
        save_steps                  = train_cfg.get("save_steps", 100),
        save_total_limit            = train_cfg.get("save_total_limit", 3),
        load_best_model_at_end      = train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model       = train_cfg.get("metric_for_best_model", "eval_loss"),
        logging_steps               = train_cfg.get("logging_steps", 10),
        report_to                   = train_cfg.get("report_to", "none"),
        dataloader_num_workers      = train_cfg.get("dataloader_num_workers", 4),
        seed                        = train_cfg.get("seed", 42),
        remove_unused_columns       = False,
        dataset_text_field          = "text",
        max_length                  = model_cfg.get("max_seq_length", 4096),
        packing                     = False,
    )

    # ── SFT Trainer ───────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_dataset,
        eval_dataset    = eval_dataset,
        processing_class = tokenizer,
        peft_config     = peft_config,
    )

    log.info("Starting LoRA fine-tuning…")
    log.info(f"  Training examples : {len(train_dataset)}")
    log.info(f"  Eval examples     : {len(eval_dataset) if eval_dataset else 0}")
    log.info(f"  LoRA rank         : {peft_config.r}")
    log.info(f"  Target modules    : {peft_config.target_modules}")
    log.info(f"  Epochs            : {training_args.num_train_epochs}")
    log.info(f"  Effective batch   : {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────
    adapter_dir = adapter_cfg.get("output_dir", "fine_tuning/adapters/lora-legal-v1")
    log.info(f"Saving LoRA adapter to {adapter_dir}")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    log.info("✅ LoRA fine-tuning complete!")
    log.info(f"Adapter saved: {adapter_dir}")
    log.info("To use the adapter: load base model + load_adapter(adapter_dir)")


if __name__ == "__main__":
    main()
