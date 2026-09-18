import torch
from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

MODEL_PATH = "/home/sece2026-student12/LegalMindAI/models/Qwen3.6-35B-A3B"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("Loading Qwen3.6-35B-A3B...")
model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
)

print("\nModel loaded successfully!")
print("Model device:", model.device)
print("\nLegalMind AI - Qwen3.6")
print("Type 'exit' to quit.\n")

while True:
    question = input("You: ")

    if question.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    messages = [
        {
            "role": "user",
            "content": question
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        k: v.to(model.device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, input_length:]

    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    print("\nAI:", response)
    print("\n" + "-" * 80 + "\n")
