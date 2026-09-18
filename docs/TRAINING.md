# LegalMindAI — Safe Standard LoRA Fine-Tuning Guide

This guide details the fine-tuning architecture and execution protocols for specializing **Qwen3.6-35B-A3B** on Indian legal reasoning.

---

## 1. Architectural Principles

### Strictly Standard LoRA Only (No Quantization):
To protect numerical precision and prevent NaN divergence on the DGX B200 architecture, the following are **strictly prohibited**:
- ❌ No QLoRA (`bitsandbytes`)
- ❌ No 4-bit or 8-bit quantization (`load_in_4bit`, `load_in_8bit`)
- ❌ No post-training quantization schemes (GPTQ, AWQ, NF4)
- ✅ **Native `torch.bfloat16` precision only**.

---

## 2. Hyperparameter Specifications

| Hyperparameter | Value | Rationale |
|---|---|---|
| **Precision** | `bfloat16` | Dynamic range matches FP32, eliminating underflow. |
| **Learning Rate** | `2e-5` | Conservative learning rate preventing gradient explosion. |
| **Max Gradient Norm** | `0.3` | Strict gradient clipping to avoid numerical overflow. |
| **Gradient Accumulation** | `8` | Simulates effective batch size of 8 or 16 without VRAM pressure. |
| **Warmup Ratio** | `0.05` | Linear warmup for early layer stability. |
| **Weight Decay** | `0.01` | Regularization preventing parameter drift. |
| **LoRA Rank ($r$)** | `16` | Sufficient expressiveness for legal formatting. |
| **LoRA Alpha ($\alpha$)** | `32` | Standard $2\times$ scaling factor. |
| **LoRA Dropout** | `0.05` | Dropout regularization on adapter matrices. |
| **Target Modules** | `["q_proj", "v_proj", "k_proj", "o_proj"]` | Standard self-attention projection layers. |

---

## 3. Dataset Preparation

Instruction pairs are structured in standard SFT format:
```json
{
  "instruction": "Explain the essential elements of anticipatory bail under Indian criminal procedure.",
  "input": "User inquiry concerning pre-arrest bail under CrPC / BNSS.",
  "output": "## 1. Direct Answer\nAnticipatory bail is governed by Section 438 CrPC (now Section 482 BNSS)..."
}
```

To build or validate the dataset:
```bash
.venv/bin/python fine_tuning/datasets/build_sft_dataset.py
```

---

## 4. Execution Protocols

### Dry Run (Validation Only):
Verify configuration, dataset integrity, and prohibited quantization checks without running training:
```bash
.venv/bin/python scripts/train_lora_safe.py --dry-run
```

### Training Execution:
Execute safe training on GPU 1:
```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/train_lora_safe.py \
    --model-path models/Qwen3.6-35B-A3B \
    --dataset fine_tuning/datasets/sft_train.jsonl \
    --output-dir fine_tuning/adapters/lora-legal-v2 \
    --lr 2e-5 \
    --grad-accum 8 \
    --max-grad-norm 0.3
```

---

## 5. Checkpoint Inspection

Every checkpoint must be validated before deployment:
```bash
.venv/bin/python scripts/validate_lora_adapter.py --adapter fine_tuning/adapters/lora-legal-v2
```
If exit code is 0, the adapter is verified clean and safe for production inference.
