import os
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

BASE_MODEL = "meta-llama/Meta-Llama-3.1-13B-Instruct"
SFT_CHECKPOINT = "./sft-llama-3.1-13b"  # from train_sft.py (if you ran it)
DATASET_PATH = "../rails-app/export_dpo.jsonl"
OUTPUT_DIR = "./dpo-llama-3.1-13b"

BATCH_SIZE = 1
GRADIENT_ACCUM = 4
LR = 5e-6
NUM_EPOCHS = 1
BETA = 0.1          # DPO beta (strength of preference signal)
MAX_LENGTH = 1024   # prompt + response max length


# ---------------------------------------------------------
# HELPER: DEVICE INFO
# ---------------------------------------------------------

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------
# Rails export_dpo.jsonl has:
# { "instruction": "...", "chosen": "...", "rejected": "..." }

dataset = load_dataset("json", data_files=DATASET_PATH)

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

# If SFT checkpoint exists, start DPO from there, else base model
if os.path.isdir(SFT_CHECKPOINT):
    model_name_or_path = SFT_CHECKPOINT
    print(f"Loading SFT checkpoint from: {SFT_CHECKPOINT}")
else:
    model_name_or_path = BASE_MODEL
    print(f"SFT checkpoint not found. Using base model: {BASE_MODEL}")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
# LLaMA models usually have no pad token; set it to eos
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
)

# Move manually to MPS/CPU if needed (DPOTrainer will also help)
if device == "mps":
    model.to("mps")
elif device == "cuda":
    model.to("cuda")
else:
    model.to("cpu")

# ---------------------------------------------------------
# APPLY LORA (no 4-bit on Mac, just standard LoRA)
# ---------------------------------------------------------

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
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

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ---------------------------------------------------------
# TRAINING ARGS
# ---------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUM,
    learning_rate=LR,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    bf16=False,   # bf16 not on Mac; MPS uses its own dtype
    fp16=False,   # leave False for MPS; it handles precision internally
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

dpo_trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n🎉 DONE: DPO-tuned model saved to:", OUTPUT_DIR)

