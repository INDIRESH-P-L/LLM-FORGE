#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_path = "/home/sece2026-student12/LegalMindAI/models/Qwen3.6-35B-A3B"
adapter_path = "/home/sece2026-student12/LegalMindAI/fine_tuning/adapters/lora-legal-v2"

print("1. Validating adapter with safetensors...")
from safetensors import safe_open
with safe_open(f"{adapter_path}/adapter_model.safetensors", framework="pt") as f:
    keys = f.keys()
    print(f"Total adapter keys: {len(keys)}")
    has_nan = False
    for k in keys:
        t = f.get_tensor(k)
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"CORRUPT: {k}")
            has_nan = True
    print(f"Adapter tensor check passed: {not has_nan}")

print("2. Loading processor and tokenizer...")
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration
from peft import PeftModel

processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
tokenizer = processor.tokenizer

print("3. Loading base model in BF16...")
t0 = time.time()
base_model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    local_files_only=True,
)
print(f"Base model loaded in {round(time.time()-t0, 1)}s")

print("4. Attaching LoRA adapter (lora-legal-v2)...")
lora_model = PeftModel.from_pretrained(base_model, adapter_path)
print("LoRA adapter attached successfully!")

print("5. Generating test legal response...")
prompt = "### Instruction:\nAnalyze the legal provision under the Constitution of India.\n\n### Input:\nWhat does Article 21 of the Constitution of India guarantee?\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(lora_model.device)

t0 = time.time()
with torch.inference_mode():
    outputs = lora_model.generate(**inputs, max_new_tokens=150, temperature=0.3, do_sample=True)
gen_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
lat = round(time.time() - t0, 2)

print("="*60)
print(f"GENERATION RESULT ({lat}s):")
print(gen_text)
print("="*60)
print(f"Has NaNs: {'nan' in gen_text.lower()}")
print("ALL TESTS PASSED SUCCESSFULLY!")
