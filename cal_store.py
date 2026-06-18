import argparse
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from common import MODEL_DICT, model_paths
from output_store import load_lora_model, load_mia_datasets, load_plain_model, load_tokenizer

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def parse_args():
    parser = argparse.ArgumentParser(description="Store target and calibration model tuple logprobs")
    parser.add_argument("--model_name", type=str, default="qwen-0.5b", choices=MODEL_DICT.keys())
    parser.add_argument("--sft_data", type=str, default="uf")
    parser.add_argument("--dpo_data", type=str, default="uf")
    parser.add_argument("--n_member", type=int, default=1000)
    parser.add_argument("--n_nonmember", type=int, default=1000)
    parser.add_argument("--cal_model", type=str, required=True, help="HF path/name of calibration model")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def check_target_path(args):
    paths = model_paths(args.model_name, args.sft_data, args.dpo_data)
    dpo_path = os.path.abspath(paths["dpo"])
    if not os.path.exists(dpo_path):
        raise FileNotFoundError(f"DPO/target path does not exist: {dpo_path}")
    return paths


def dataset_to_samples(dataset, label):
    return [
        {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "label": label,
        }
        for row in dataset
    ]


def make_labeled_input(tokenizer, prompt, response, max_length):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id]
    if not response_ids:
        response_ids = [tokenizer.eos_token_id]

    if len(response_ids) >= max_length:
        response_ids = response_ids[: max_length - 1]
        prompt_ids = prompt_ids[-1:]
    else:
        max_prompt_len = max_length - len(response_ids)
        prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = prompt_ids + response_ids
    answer_start = len(prompt_ids)
    answer_len = len(response_ids)
    return input_ids, answer_start, answer_len


def pad_batch(tokenizer, encoded):
    max_len = max(len(x[0]) for x in encoded)
    pad_id = tokenizer.pad_token_id
    input_ids = []
    attention_mask = []
    answer_starts = []
    answer_lens = []

    for ids, answer_start, answer_len in encoded:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        answer_starts.append(answer_start)
        answer_lens.append(answer_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "answer_starts": answer_starts,
        "answer_lens": answer_lens,
    }


def build_response_records(samples):
    records = []
    for sample_idx, sample in enumerate(samples):
        records.append({
            "sample_idx": sample_idx,
            "side": "chosen",
            "prompt": sample["prompt"],
            "response": sample["chosen"],
        })
        records.append({
            "sample_idx": sample_idx,
            "side": "rejected",
            "prompt": sample["prompt"],
            "response": sample["rejected"],
        })
    return records


def compute_logprobs(records, model, tokenizer, max_length=1024, batch_size=4, desc="logprobs"):
    encoded = [
        make_labeled_input(tokenizer, record["prompt"], record["response"], max_length)
        for record in records
    ]
    scored = []
    device = model.device

    for start in tqdm(range(0, len(records), batch_size), desc=desc):
        batch_records = records[start: start + batch_size]
        batch_encoded = encoded[start: start + batch_size]
        batch = pad_batch(tokenizer, batch_encoded)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits, dim=-1)

        for i, record in enumerate(batch_records):
            answer_start = batch["answer_starts"][i]
            answer_len = batch["answer_lens"][i]
            total = 0.0
            count = 0
            for pos in range(answer_start, answer_start + answer_len):
                if pos == 0:
                    continue
                token_id = input_ids[i, pos]
                total += float(log_probs[i, pos - 1, token_id].detach().cpu())
                count += 1
            scored.append({
                "sample_idx": record["sample_idx"],
                "side": record["side"],
                "logp": total,
                "avg_logp": total / max(count, 1),
                "num_tokens": count,
            })

        del logits, log_probs

    return scored


def attach_rows(samples, target_scored, cal_scored, cal_model_name):
    target_grouped = [{} for _ in samples]
    cal_grouped = [{} for _ in samples]

    for record in target_scored:
        target_grouped[record["sample_idx"]][record["side"]] = record
    for record in cal_scored:
        cal_grouped[record["sample_idx"]][record["side"]] = record

    rows = []
    for idx, sample in enumerate(samples):
        tc = target_grouped[idx]["chosen"]
        tr = target_grouped[idx]["rejected"]
        cc = cal_grouped[idx]["chosen"]
        cr = cal_grouped[idx]["rejected"]

        target_margin = tc["logp"] - tr["logp"]
        cal_margin = cc["logp"] - cr["logp"]
        target_avg_margin = tc["avg_logp"] - tr["avg_logp"]
        cal_avg_margin = cc["avg_logp"] - cr["avg_logp"]

        rows.append({
            "label": int(sample["label"]),
            "cal_model": cal_model_name,
            "target_chosen_logp": tc["logp"],
            "target_rejected_logp": tr["logp"],
            "target_chosen_avg_logp": tc["avg_logp"],
            "target_rejected_avg_logp": tr["avg_logp"],
            "target_margin": target_margin,
            "target_avg_margin": target_avg_margin,
            "cal_chosen_logp": cc["logp"],
            "cal_rejected_logp": cr["logp"],
            "cal_chosen_avg_logp": cc["avg_logp"],
            "cal_rejected_avg_logp": cr["avg_logp"],
            "cal_margin": cal_margin,
            "cal_avg_margin": cal_avg_margin,
            "chosen_target_tokens": tc["num_tokens"],
            "rejected_target_tokens": tr["num_tokens"],
            "chosen_cal_tokens": cc["num_tokens"],
            "rejected_cal_tokens": cr["num_tokens"],
        })
    return rows


def safe_name(text):
    return text.replace("/", "-").replace(" ", "_")


def default_output_path(args):
    return f"cal_score_{args.sft_data}_{args.dpo_data}_{args.model_name}_{safe_name(args.cal_model)}.pkl"


def main():
    args = parse_args()
    train_ds, eval_ds = load_mia_datasets(args)
    samples = dataset_to_samples(train_ds, label=1) + dataset_to_samples(eval_ds, label=0)
    records = build_response_records(samples)

    base_model = MODEL_DICT[args.model_name]
    paths = check_target_path(args)

    print(f"Loaded {len(samples)} samples: member={len(train_ds)} nonmember={len(eval_ds)}")
    print(f"Target base: {base_model}")
    print(f"Target DPO adapter: {paths['dpo']}")
    print(f"Calibration model: {args.cal_model}")

    target_tokenizer = load_tokenizer(base_model)
    target_model = load_lora_model(base_model, paths["dpo"])
    target_scored = compute_logprobs(
        records,
        target_model,
        target_tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        desc="target logprobs",
    )

    del target_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cal_tokenizer = load_tokenizer(args.cal_model)
    cal_model = load_plain_model(args.cal_model)
    cal_scored = compute_logprobs(
        records,
        cal_model,
        cal_tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        desc="calibration logprobs",
    )

    rows = attach_rows(samples, target_scored, cal_scored, args.cal_model)
    save_path = args.output or default_output_path(args)
    with open(save_path, "wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(rows)} calibration scored samples to {save_path}")


if __name__ == "__main__":
    main()
