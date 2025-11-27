import os
import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="../rails-app/export_sft.jsonl",
        help="Path to SFT JSONL dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./sft-llama-3-8b",
        help="Where to save the SFT model",
    )
    args = parser.parse_args()

    dataset_path = args.dataset
    output_dir = args.output_dir

    print(f"Using dataset: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    device = get_device()
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # LOAD DATASET
    # ---------------------------------------------------------

    dataset = load_dataset("json", data_files=dataset_path)

    # Expected fields: { "instruction": "...", "output": "..." }

    def format_example(example):
        return {
            "text": f"<s>[INST] {example['instruction']} [/INST]\n{example['output']}</s>"
        }

    dataset = dataset.map(format_example)

    # ---------------------------------------------------------
    # LOAD MODEL + TOKENIZER
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )

    if device == "mps":
        model.to("mps")
    elif device == "cuda":
        model.to("cuda")

    # ---------------------------------------------------------
    # APPLY LORA
    # ---------------------------------------------------------

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ---------------------------------------------------------
    # TRAINING
    # ---------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        warmup_ratio=0.1,
        optim="adamw_torch",
        fp16=False,  # MPS handles precision internally
        bf16=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        dataset_text_field="text",
    )

    trainer.train()

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n🎉 DONE: SFT model saved to:", output_dir)


if __name__ == "__main__":
    main()
