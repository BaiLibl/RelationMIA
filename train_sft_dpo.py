import os
import warnings

import torch
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

from common import MODEL_DICT, default_split_sizes, model_paths
from _args import parse_args
from explore_datasets import DATASET_LOADERS


warnings.filterwarnings("ignore", message="fan_in_fan_out")
warnings.filterwarnings("ignore", message="`position_ids`")
set_seed(42)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


TARGET_MODULES = {
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "gpt2-medium": ["c_attn", "c_proj", "c_fc"],
    "gpt2-large": ["c_attn", "c_proj", "c_fc"],
    "llama-3b": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "qwen-0.5b": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "qwen-1.5b": ["q_proj", "v_proj", "k_proj", "o_proj"],
}


def build_lora_config(model_name):
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES[model_name],
    )


def build_quant_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def run_sft(args, paths):
    model_name = MODEL_DICT[args.model_name]
    n_train, n_eval = default_split_sizes(args.sft_data)
    tokenizer = load_tokenizer(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if args.use_quant else None,
        quantization_config=build_quant_config() if args.use_quant else None,
    )
    model.config.use_cache = False

    train_ds, eval_ds = DATASET_LOADERS[args.sft_data](n_train, n_eval, for_dpo=False)
    cfg = SFTConfig(
        output_dir=os.path.dirname(paths["sft"]),
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=8e-5,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.05,
        optim="adamw_torch",
        report_to="none",
        fp16=torch.cuda.is_available(),
        max_length=1024,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        run_name=f"{args.model_name}-{args.sft_data}-sft",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=None if args.no_lora else build_lora_config(args.model_name),
    )
    trainer.train()
    trainer.save_model(paths["sft"])
    tokenizer.save_pretrained(paths["sft"])
    return paths["sft"]


def run_dpo(args, paths):
    model_name = MODEL_DICT[args.model_name]
    n_train, n_eval = default_split_sizes(args.dpo_data)
    tokenizer = load_tokenizer(paths["sft"])
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=build_quant_config() if args.use_quant else None,
    )
    model = PeftModel.from_pretrained(base_model, paths["sft"], is_trainable=True)
    model.config.use_cache = False

    train_ds, eval_ds = DATASET_LOADERS[args.dpo_data](n_train, n_eval, for_dpo=True)
    cfg = DPOConfig(
        output_dir=os.path.dirname(paths["dpo"]),
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-4,
        warmup_steps=100,
        beta=0.4,
        optim="adamw_torch",
        # max_prompt_length=256,
        max_length=1024,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        run_name=f"{args.model_name}-{args.dpo_data}-dpo",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(paths["dpo"])


def main():
    args = parse_args()
    paths = model_paths(args.model_name, args.sft_data, args.dpo_data)
    os.makedirs(os.path.dirname(paths["sft"]), exist_ok=True)
    os.makedirs(os.path.dirname(paths["dpo"]), exist_ok=True)

    print("=== Stage 1: SFT Training ===")
    run_sft(args, paths)
    print("=== Stage 2: DPO Training ===")
    run_dpo(args, paths)
    print("All training finished successfully.")


if __name__ == "__main__":
    main()
