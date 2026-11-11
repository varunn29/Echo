import json
import random
import time
from pathlib import Path

import psutil
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Qwen3_5ForCausalLM


# Configuration for Experiment V2
BASE_MODEL = "Qwen/Qwen3.5-0.8B-Base"
DATASET_PATH = Path("training-data/personal_dataset_v1_train.jsonl")
OUTPUT_DIR = Path("training-output/qwen3.5-0.8b-lora-v2")

EPOCHS = 3
LEARNING_RATE = 5e-5
RANDOM_SEED = 42


def load_dataset(tokenizer):
    rows = []

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    if not rows:
        raise ValueError("The training dataset is empty.")

    samples = []

    for row in rows:
        text = row.get("text", "").strip()

        if not text:
            continue

        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=256,
        )

        input_ids = encoded["input_ids"]

        if tokenizer.eos_token_id is not None:
            if input_ids[-1] != tokenizer.eos_token_id:
                input_ids.append(tokenizer.eos_token_id)

        samples.append(input_ids)

    if not samples:
        raise ValueError("No valid writing samples were found.")

    return samples


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.set_num_threads(4)

    start_time = time.time()
    process = psutil.Process()

    print("====================================================")
    print("Echo — Personal LoRA Training Experiment V2")
    print("Base Model:", BASE_MODEL)
    print("Dataset Path:", DATASET_PATH)
    print("Output Dir:", OUTPUT_DIR)
    print("====================================================\n")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading base model...")
    model = Qwen3_5ForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        device_map="cpu",
    )
    model.config.use_cache = False

    print("Loading writing samples...")
    samples = load_dataset(tokenizer)

    total_tokens = sum(len(sample) for sample in samples)

    print(f"Loaded {len(samples)} training samples ({total_tokens} total tokens with EOS).")

    # Configure personal LoRA adapter (identical configuration)
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("Attaching LoRA adapter...")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.train()

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters()
         if parameter.requires_grad),
        lr=LEARNING_RATE,
    )

    epoch_losses = []
    total_optimizer_steps = 0
    step_losses = []

    print("\nStarting fine-tuning...\n")

    for epoch in range(EPOCHS):
        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)

        total_loss = 0.0

        for sample_idx, sample in enumerate(shuffled_samples):
            input_ids = torch.tensor(
                [sample],
                dtype=torch.long,
            )

            optimizer.zero_grad()

            output = model(
                input_ids=input_ids,
                labels=input_ids,
                use_cache=False,
            )

            loss = output.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()
            total_optimizer_steps += 1

            loss_val = loss.item()
            total_loss += loss_val
            step_losses.append(loss_val)

        average_loss = total_loss / len(shuffled_samples)
        epoch_losses.append(average_loss)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| Average loss: {average_loss:.4f} "
            f"| Cumulative steps: {total_optimizer_steps}"
        )

    # Save only the personal adapter, not the full base model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSaving personal adapter V2...")
    model.save_pretrained(OUTPUT_DIR)

    elapsed_time = time.time() - start_time
    memory_usage = process.memory_info().rss / (1024 ** 3)

    summary = {
        "experiment": "Personal LoRA Training V2",
        "base_model": BASE_MODEL,
        "dataset_path": str(DATASET_PATH),
        "samples": len(samples),
        "total_tokens_with_eos": total_tokens,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "total_optimizer_steps": total_optimizer_steps,
        "epoch_losses": epoch_losses,
        "training_time_seconds": round(elapsed_time, 2),
        "process_ram_gb": round(memory_usage, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    summary_path = OUTPUT_DIR / "training_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    print("\nFine-tuning completed successfully!")
    print("Adapter location:", OUTPUT_DIR)
    print("Training time:", round(elapsed_time, 2), "seconds")
    print("Process RAM:", round(memory_usage, 2), "GB")
    print("Total optimizer steps:", total_optimizer_steps)
    print("Epoch losses:", epoch_losses)


if __name__ == "__main__":
    main()

