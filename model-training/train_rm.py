import os
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

BASE_MODEL = "meta-llama/Meta-Llama-3.1-13B-Instruct"
DATASET_PATH = "../rails-app/export_rm.jsonl"
OUTPUT_DIR = "./reward-model-llama-3.1-13b"

NUM_LABELS = 1  # scalar reward
BATCH_SIZE = 1
GRADIENT_ACCUM = 4
LR = 1e-5
NUM_EPOCHS = 2
MAX_LENGTH = 1024


# ---------------------------------------------------------
# DEVICE SETUP
# ---------------------------------------------------------

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("Using device:", device)


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------
# export_rm.jsonl contains:
# { instruction, response, reward }

dataset = load_dataset("json", data_files=DATASET_PATH)


def preprocess(example):
    # simple concatenation; reward model sees both prompt and response
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
    problem_type="regression",   # critical: reward is a continuous value
    torch_dtype=torch.float16 if device != "cpu" else torch.float32,
)

if device == "mps":
    model.to("mps")
elif device == "cuda":
    model.to("cuda")


# ---------------------------------------------------------
# APPLY LoRA TO REWARD MODEL
# ---------------------------------------------------------

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
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
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    warmup_ratio=0.1,
    optim="adamw_torch",
    fp16=False,  # MPS controls mixed precision automatically
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

# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------

trainer.train()

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n🎉 DONE: Reward model saved to:", OUTPUT_DIR)

