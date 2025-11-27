import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

MODEL_NAME = "meta-llama/Meta-Llama-3.1-13B-Instruct"
DATASET_PATH = "../rails-app/export_sft.jsonl"
OUTPUT_DIR = "./sft-llama-3.1-13b"

BATCH_SIZE = 1
GRADIENT_ACCUM = 8
LR = 2e-5
NUM_EPOCHS = 2

# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

dataset = load_dataset("json", data_files=DATASET_PATH)

# dataset['train'][0] looks like:
# { "instruction": "...", "output": "..." }


def format_example(example):
    return {
        "text": f"<s>[INST] {example['instruction']} [/INST]\n{example['output']}</s>"
    }


dataset = dataset.map(format_example)

# ---------------------------------------------------------
# LOAD MODEL + TOKENIZER
# ---------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    load_in_4bit=True,  # QLoRA compression
    device_map="auto"
)

# ---------------------------------------------------------
# APPLY LORA
# ---------------------------------------------------------

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUM,
    learning_rate=LR,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=20,
    save_steps=200,
    save_total_limit=3,
    fp16=True,
    warmup_ratio=0.1,
    optim="paged_adamw_32bit",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
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
# SAVE MODEL
# ---------------------------------------------------------

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n🎉 DONE: SFT model saved to:", OUTPUT_DIR)

