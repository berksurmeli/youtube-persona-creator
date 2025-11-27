import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ---------------------------------------------------------
# merge_lora.py
#
# Usage example:
#   python merge_lora.py \
#       --base-model meta-llama/Meta-Llama-3.1-13B-Instruct \
#       --adapter-path ./sft-llama-3.1-13b \
#       --output-path ./merged-sft-llama-3.1-13b
#
#   python merge_lora.py \
#       --base-model meta-llama/Meta-Llama-3.1-13B-Instruct \
#       --adapter-path ./dpo-llama-3.1-13b \
#       --output-path ./merged-dpo-llama-3.1-13b
# ---------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Base HF model name or path (e.g. meta-llama/Meta-Llama-3.1-13B-Instruct)",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to LoRA/PEFT adapter directory (e.g. ./sft-llama-3.1-13b)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Where to save the merged full model",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",  # we'll move to MPS/GPU only if needed
    )

    print(f"Loading LoRA adapter from: {args.adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print("Merging LoRA weights into base model (this may take a while)...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to: {args.output_path}")
    merged_model.save_pretrained(args.output_path)

    print("Copying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.output_path)

    print("\n🎉 Done! Merged model is saved at:", args.output_path)


if __name__ == "__main__":
    main()

