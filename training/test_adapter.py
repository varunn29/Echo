
import torch

from peft import PeftModel
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


BASE_MODEL = "Qwen/Qwen3.5-0.8B-Base"
ADAPTER_PATH = "training-output/qwen3.5-0.8b-lora"

PROMPT = "I recently started learning PostgreSQL and"


def generate_text(model, tokenizer):
    inputs = tokenizer(PROMPT, return_tensors="pt")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )


def main():
    torch.set_num_threads(4)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading original base model...")
    base_model = Qwen3_5ForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        device_map="cpu",
    )

    base_model.config.use_cache = True
    base_model.eval()

    print("\nGenerating with original model...")
    base_output = generate_text(base_model, tokenizer)

    print("\nLoading personal LoRA adapter...")
    personal_model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    personal_model.eval()

    print("\nGenerating with personalized model...")
    personal_output = generate_text(personal_model, tokenizer)

    print("\n" + "=" * 60)
    print("ORIGINAL BASE MODEL")
    print("=" * 60)
    print(PROMPT + base_output)

    print("\n" + "=" * 60)
    print("PERSONALIZED MODEL")
    print("=" * 60)
    print(PROMPT + personal_output)


if __name__ == "__main__":
    main()
