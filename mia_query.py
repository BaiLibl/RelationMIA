import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from _args import parse_args
from common import MODEL_DICT, default_split_sizes, model_paths
from explore_datasets import DATASET_LOADERS
from prompt_file import build_prompt_queries


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
    )
    return DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )


def decode_batch_text(tokenizer, batch, key, index):
    ids = batch[key][index]
    if torch.is_tensor(ids):
        ids = ids.detach().cpu().tolist()
    if tokenizer.pad_token_id is not None:
        ids = [token_id for token_id in ids if token_id != tokenizer.pad_token_id]
    return tokenizer.decode(ids, skip_special_tokens=True)


def generate_query_outputs(model, tokenizer, queries, max_new_tokens=3, batch_size=8, desc="query model"):
    device = next(model.parameters()).device
    outputs_all = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for start in tqdm(range(0, len(queries), batch_size), desc=desc):
        batch_queries = queries[start: start + batch_size]
        inputs = tokenizer(
            batch_queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for output in outputs:
            generated_ids = output[prompt_len:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            outputs_all.append(text)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tokenizer.padding_side = old_padding_side
    return outputs_all


def collect_queries(args, tokenizer, trainer, target_model, ref_model, max_samples=2000):
    rows = []
    records = []
    n_queries = 10

    def process(dataloader, label):
        seen = 0
        for batch in tqdm(dataloader, desc=f"build queries label={label}"):
            if seen >= max_samples:
                break

            batch_size = len(batch["prompt_input_ids"])
            keep = min(batch_size, max_samples - seen)
            for i in range(keep):
                sample_id = len(rows)
                prompt = decode_batch_text(tokenizer, batch, "prompt_input_ids", i)
                chosen = decode_batch_text(tokenizer, batch, "chosen_input_ids", i)
                rejected = decode_batch_text(tokenizer, batch, "rejected_input_ids", i)
                rows.append({
                    "label": label,
                    "m_outputs": [],
                    "r_outputs": [],
                })

                query_prompts = build_prompt_queries(prompt, chosen, rejected)
                for prompt_name, query_text in query_prompts.items():
                    records.append({
                        "sample_id": sample_id,
                        "prompt_name": prompt_name,
                        "query": query_text,
                    })
            seen += batch_size

    process(trainer.get_train_dataloader(), label=1)
    process(trainer.get_eval_dataloader(), label=0)

    expanded_queries = []
    expanded_records = []
    for record in records:
        for _ in range(n_queries):
            expanded_queries.append(record["query"])
            expanded_records.append(record)

    m_decoded_outputs = generate_query_outputs(
        target_model,
        tokenizer,
        expanded_queries,
        max_new_tokens=3,
        batch_size=args.sem_generation_batch_size,
        desc="query target model",
    )
    r_decoded_outputs = generate_query_outputs(
        ref_model,
        tokenizer,
        expanded_queries,
        max_new_tokens=3,
        batch_size=args.sem_generation_batch_size,
        desc="query reference model",
    )

    for idx, record in enumerate(expanded_records):
        rows[record["sample_id"]]["m_outputs"].append(m_decoded_outputs[idx])
        rows[record["sample_id"]]["r_outputs"].append(r_decoded_outputs[idx])

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
    save_path = args.output or f"query/query_{args.sft_data}_{args.dpo_data}_{args.model_name}.npy"
    trainer = build_dpo_trainer(target_model, tokenizer, train_ds, eval_ds)
    rows = collect_queries(
        args,
        tokenizer,
        trainer,
        target_model,
        ref_model,
        max_samples=max(args.n_member, args.n_nonmember),
    )

    np.save(save_path, rows)
    print(f"Saved {len(rows)} samples to {save_path}")


if __name__ == "__main__":
    main()
