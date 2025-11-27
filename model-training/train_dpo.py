import os
import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer
import torch

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
SFT_CHECKPOINT = "./sft-llama-3-8b"
DEFAULT_DATASET_PATH = "../rails-app/export_dpo.jsonl"
DEFAULT_OUTPUT_DIR = "./dpo-llama-3-8b"
BETA = 0.1
MAX_LENGTH = 1024


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
        default=DEFAULT_DATASET_PATH,
        help="Path to DPO JSONL dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save the DPO model",
    )
    args = parser.parse_args()

    dataset_path = args.dataset
    output_dir = args.output_dir

    print("Using dataset:", dataset_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    device = get_device()
    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # LOAD DATASET
    # ---------------------------------------------------------
    # export_dpo.jsonl:
    # { "instruction": "...", "chosen": "...", "rejected": "..." }

    dataset = load_dataset("json", data_files=dataset_path)

    def format_example(example):
        return {
            "prompt": example["instruction"],
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }

    dataset = dataset.map(format_example)

    # ---------------------------------------------------------
    # LOAD MODEL + TOKENIZER
    # ---------------------------------------------------------

    if os.path.isdir(SFT_CHECKPOINT):
        model_name_or_path = SFT_CHECKPOINT
        print(f"Loading SFT checkpoint from: {SFT_CHECKPOINT}")
    else:
        model_name_or_path = BASE_MODEL
        print(f"SFT checkpoint not found. Using base model: {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
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
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ---------------------------------------------------------
    # TRAINING ARGS
    # ---------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_train_epochs=1,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        bf16=False,
        fp16=False,
        warmup_ratio=0.1,
        optim="adamw_torch",
    )

    # ---------------------------------------------------------
    # DPO TRAINER
    # ---------------------------------------------------------

    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        beta=BETA,
        train_dataset=dataset["train"],
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        max_target_length=MAX_LENGTH,
        prompt_column="prompt",
        chosen_column="chosen",
        rejected_column="rejected",
    )

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    dpo_trainer.train()

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n🎉 DONE: DPO-tuned model saved to:", output_dir)


if __name__ == "__main__":
    main()
