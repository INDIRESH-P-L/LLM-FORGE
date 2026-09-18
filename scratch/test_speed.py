import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/home/sece2026-student12/LegalMindAI/scripts")))
import torch

from scripts_05_rag import LegalMindModel

t0 = time.time()
print("Getting model instance...")
model = LegalMindModel.get_instance(gpu=1)
print(f"Model ready in {time.time() - t0:.2f}s")

# Test 1: Generate with chat completion
messages = [
    {"role": "system", "content": "You are a senior Supreme Court judge. Answer concisely in 2-3 sentences."},
    {"role": "user", "content": "Counsel, what is your primary submission regarding Section 45 PMLA?"}
]

print("\n--- Testing generate_chat_completion ---")
t_start = time.time()
resp = model.generate_chat_completion(messages, max_new_tokens=256, temperature=0.3)
t_gen = time.time() - t_start

inputs = model.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt")
prompt_len = inputs.shape[1] if hasattr(inputs, "shape") else inputs["input_ids"].shape[1]
output_tokens = len(model.processor.tokenizer.encode(resp))

print(f"Time: {t_gen:.2f}s")
print(f"Prompt tokens: {prompt_len}, Output tokens: {output_tokens}")
print(f"Tokens/sec: {output_tokens / t_gen:.2f}")
print("Response preview:\n", resp[:300])
