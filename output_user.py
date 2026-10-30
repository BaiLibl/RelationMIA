import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from _args import parse_args
from common import MODEL_DICT, default_split_sizes, model_paths
from explore_datasets import DATASET_LOADERS


def load_tokenizer(path, padding_side="right"):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_lora_model(base_model, adapter_path):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def load_plain_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    model.eval()
    return model


def build_dpo_trainer(model, tokenizer, train_ds, eval_ds):
    cfg = DPOConfig(
        output_dir="saved/dummy_dpo_trainer",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        fp16=torch.cuda.is_available(),
        max_length=1024,
        max_prompt_length=256,
        truncation_mode="keep_end",
        report_to="none",
        remove_unused_columns=False,
    )




    return DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )


def collect_logprobs(trainer, model, ref_model=None, max_samples=2000):
    rows = []
    print('here')
    print("Train dataset columns:", trainer.train_dataset.column_names)
    original_collator = trainer.data_collator

    user_fields = [
        "user_id", "age", "gender",
        "employment_status", "education", "marital_status"
    ]

    def metadata_collator(features):
        batch = original_collator(features)

        for field in user_fields:
            batch[field] = [f[field] for f in features]

        return batch

    trainer.data_collator = metadata_collator
    def process(dataloader, label):
        seen = 0
        for batch in tqdm(dataloader, desc=f"label={label}"):
        # for batch in dataloader:
            # print(batch.keys())
            if seen >= max_samples:
                break 
            # 1. 提取元数据 (在送入模型前，从 batch 中直接拿)
            # 注意：TRL collator 通常会把无法转 tensor 的字段保留为 list
            # print(batch.keys())
            meta = {field: batch[field] for field in user_fields if field in batch}
            
            # print(len(meta))
            # print(meta)
            # 2. 送模型 (去掉额外字段，防止报错)
            model_inputs = trainer._prepare_inputs(batch)
            with torch.no_grad():
                out = trainer.concatenated_forward(model, model_inputs)
                row = {
                    "m_chosen": out["chosen_logps"].detach().cpu().numpy(),
                    "m_rejected": out["rejected_logps"].detach().cpu().numpy(),
                }
                if ref_model is not None:
                    ref = trainer.concatenated_forward(ref_model, model_inputs)
                    row["r_chosen"] = ref["chosen_logps"].detach().cpu().numpy()
                    row["r_rejected"] = ref["rejected_logps"].detach().cpu().numpy()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 3. 组装数据
            batch_size = len(row["m_chosen"])
            limit = min(batch_size, max_samples - seen)
            for i in range(limit):
                sample = {key: value[i] for key, value in row.items()}
                sample["label"] = label
                # 把元数据填进去
                for field, values in meta.items():
                    sample[field] = values[i]
                rows.append(sample)
                
            seen += batch_size

    process(trainer.get_train_dataloader(), label=1)
    process(trainer.get_eval_dataloader(), label=0)
    return rows


def load_mia_datasets(args):
    n_train, n_eval = default_split_sizes(args.dpo_data)
    loader = DATASET_LOADERS[args.dpo_data]
    return loader(
        n_train,
        n_eval,
        for_dpo=True,
        n_members=args.n_member,
        n_nonmembers=args.n_nonmember,
    )


def main():
    args = parse_args()
    train_ds, eval_ds = load_mia_datasets(args)

    base_model = MODEL_DICT[args.model_name]
    paths = model_paths(args.model_name, args.sft_data, args.dpo_data)
    tokenizer = load_tokenizer(base_model)
    ref_model = load_lora_model(base_model, paths["sft"])
    target_model = load_lora_model(base_model, paths["dpo"])
    save_path = args.output or f"premia_{args.sft_data}_{args.dpo_data}_{args.model_name}.npy"
    trainer = build_dpo_trainer(target_model, tokenizer, train_ds, eval_ds)
    rows = collect_logprobs(trainer, target_model, ref_model, max_samples=max(args.n_member, args.n_nonmember))

    np.save(save_path, rows)
    print(f"Saved {len(rows)} samples to {save_path}")


if __name__ == "__main__":
    main()
