import os
import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from trl import RewardTrainer
import torch

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_DATASET_PATH = "../rails-app/export_rm.jsonl"
DEFAULT_OUTPUT_DIR = "./reward-model-llama-3-8b"

NUM_LABELS = 1  # scalar reward
BATCH_SIZE = 1
GRADIENT_ACCUM = 4
LR = 1e-5
NUM_EPOCHS = 2
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
        help="Path to RM JSONL dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to save the reward model",
    )
    args = parser.parse_args()

    dataset_path = args.dataset
    output_dir = args.output_dir

    print("Using dataset:", dataset_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    device = get_device()
    print("Using device:", device)

    # ---------------------------------------------------------
    # LOAD DATASET
    # ---------------------------------------------------------
    # export_rm.jsonl: { instruction, response, reward }

    dataset = load_dataset("json", data_files=dataset_path)

    def preprocess(example):
        text = f"<s>[INST] {example['instruction']} [/INST]\n{example['response']}</s>"
        return {"text": text, "label": float(example["reward"])}

    dataset = dataset.map(preprocess)

    # ---------------------------------------------------------
    # LOAD MODEL + TOKENIZER
    # ---------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        problem_type="regression",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )

    if device == "mps":
        model.to("mps")
    elif device == "cuda":
        model.to("cuda")

    # ---------------------------------------------------------
    # APPLY LORA TO REWARD MODEL
    # ---------------------------------------------------------

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="SEQ_CLS",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ---------------------------------------------------------
    # TRAINING ARGUMENTS
    # ---------------------------------------------------------

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        warmup_ratio=0.1,
        optim="adamw_torch",
        fp16=False,
        bf16=False,
    )

    # ---------------------------------------------------------
    # REWARD TRAINER
    # ---------------------------------------------------------

    trainer = RewardTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        max_length=MAX_LENGTH,
        label_column="label",
        dataset_text_field="text",
    )

    trainer.train()

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n🎉 DONE: Reward model saved to:", output_dir)


if __name__ == "__main__":
    main()
