import argparse
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from common import MODEL_DICT, model_paths
from explore_datasets import DATASET_LOADERS
from output_store import load_lora_model, load_mia_datasets, load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate x0 prompt-anchor logprob scores for PMI-style MIA")
    parser.add_argument("--model_name", type=str, default="gpt2", choices=MODEL_DICT.keys())
    parser.add_argument("--sft_data", type=str, default="imdb", choices=DATASET_LOADERS.keys())
    parser.add_argument("--dpo_data", type=str, default="imdb", choices=DATASET_LOADERS.keys())
    parser.add_argument("--n_member", type=int, default=1000)
    parser.add_argument("--n_nonmember", type=int, default=1000)
    parser.add_argument("--prompt_type", type=str, default="random", choices=["random", "length", "neighbor"])
    parser.add_argument("--n_prompt_anchors", type=int, default=10)
    parser.add_argument("--length_pool", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embed_batch_size", type=int, default=64)
    parser.add_argument("--sim_min", type=float, default=0.40)
    parser.add_argument("--sim_max", type=float, default=0.80)
    parser.add_argument("--neighbor_pool", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def dataset_to_samples(dataset, label):
    return [
        {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "label": int(label),
        }
        for row in dataset
    ]


def prompt_length(prompt):
    return len(str(prompt).split())


def sample_without_self(rng, candidates, self_idx, k):
    candidates = [int(idx) for idx in candidates if int(idx) != int(self_idx)]
    if not candidates:
        return []
    replace = len(candidates) < k
    return [int(x) for x in rng.choice(candidates, size=k, replace=replace)]


def random_anchor_indices(samples, n_prompt_anchors, seed):
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(samples))
    return [sample_without_self(rng, all_indices, idx, n_prompt_anchors) for idx in range(len(samples))]


def length_anchor_indices(samples, n_prompt_anchors, length_pool, seed):
    rng = np.random.default_rng(seed)
    lengths = np.asarray([prompt_length(sample["prompt"]) for sample in samples], dtype=np.int64)
    out = []
    for idx, length in enumerate(lengths):
        dist = np.abs(lengths - length)
        order = np.argsort(dist, kind="stable")
        pool = [int(x) for x in order if int(x) != idx][: max(length_pool, n_prompt_anchors)]
        out.append(sample_without_self(rng, pool, idx, n_prompt_anchors))
    return out


def load_sentence_transformer(model_name):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise ImportError(
            "prompt_type=neighbor requires sentence_transformers. "
            "Install it or use --prompt_type random/length."
        ) from exc
    return SentenceTransformer(model_name)


def neighbor_anchor_indices(samples, args):
    rng = np.random.default_rng(args.seed)
    prompts = [sample["prompt"] for sample in samples]
    model = load_sentence_transformer(args.embed_model)
    embeddings = model.encode(
        prompts,
        batch_size=args.embed_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    sims = embeddings @ embeddings.T

    out = []
    sim_info = []
    for idx in tqdm(range(len(samples)), desc="select prompt neighbors"):
        row = sims[idx].copy()
        row[idx] = -np.inf
        valid = np.where((row >= args.sim_min) & (row <= args.sim_max))[0]
        if len(valid) == 0:
            valid = np.where(np.isfinite(row))[0]
        valid = valid[np.argsort(row[valid])[::-1]]
        pool = valid[: max(args.neighbor_pool, args.n_prompt_anchors)]
        selected = sample_without_self(rng, pool, idx, args.n_prompt_anchors)
        out.append(selected)
        sim_info.append([float(row[j]) for j in selected])
    return out, sim_info


def check_model_paths(paths):
    dpo_path = os.path.abspath(paths["dpo"])
    if not os.path.exists(dpo_path):
        raise FileNotFoundError(f"DPO/target path does not exist: {dpo_path}")


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


def build_score_records(samples, anchor_indices):
    records = []
    for sample_idx, sample in enumerate(samples):
        for side in ["chosen", "rejected"]:
            records.append({
                "sample_idx": sample_idx,
                "side": side,
                "prompt_kind": "x",
                "prompt_idx": -1,
                "prompt": sample["prompt"],
                "response": sample[side],
            })
        for anchor_pos, x0_idx in enumerate(anchor_indices[sample_idx]):
            x0_prompt = samples[x0_idx]["prompt"]
            for side in ["chosen", "rejected"]:
                records.append({
                    "sample_idx": sample_idx,
                    "side": side,
                    "prompt_kind": "x0",
                    "prompt_idx": anchor_pos,
                    "prompt": x0_prompt,
                    "response": sample[side],
                })
    return records


def compute_logprobs(records, model, tokenizer, max_length=1024, batch_size=4):
    encoded = [make_labeled_input(tokenizer, r["prompt"], r["response"], max_length) for r in records]
    scored = []
    device = model.device
    for start in tqdm(range(0, len(records), batch_size), desc="prompt-anchor logprobs"):
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
            for pos in range(answer_start, answer_start + answer_len):
                if pos == 0:
                    continue
                token_id = input_ids[i, pos]
                total += float(log_probs[i, pos - 1, token_id].detach().cpu())
            scored.append({
                "sample_idx": int(record["sample_idx"]),
                "side": record["side"],
                "prompt_kind": record["prompt_kind"],
                "prompt_idx": int(record["prompt_idx"]),
                "logp": float(total),
                "avg_logp": float(total / max(1, answer_len)),
                "answer_len": int(answer_len),
            })
        del log_probs
    return scored


def build_rows(samples, anchor_indices, prompt_type, scored_records, anchor_sims=None):
    rows = []
    for idx, sample in enumerate(samples):
        row = {
            "label": int(sample["label"]),
            "prompt_type": prompt_type,
            "x0_indices": [int(x) for x in anchor_indices[idx]],
            "chosen_logp_x": None,
            "rejected_logp_x": None,
            "chosen_avg_logp_x": None,
            "rejected_avg_logp_x": None,
            "chosen_logps_x0": [],
            "rejected_logps_x0": [],
            "chosen_avg_logps_x0": [],
            "rejected_avg_logps_x0": [],
        }
        rows.append(row)

    for record in scored_records:
        row = rows[record["sample_idx"]]
        side = record["side"]
        if record["prompt_kind"] == "x":
            row[f"{side}_logp_x"] = record["logp"]
            row[f"{side}_avg_logp_x"] = record["avg_logp"]
        else:
            row[f"{side}_logps_x0"].append(record["logp"])
            row[f"{side}_avg_logps_x0"].append(record["avg_logp"])
    return rows


def default_output_path(args):
    return f"prompt_score_{args.prompt_type}_{args.sft_data}_{args.dpo_data}_{args.model_name}_K{args.n_prompt_anchors}.pkl"


def main():
    args = parse_args()
    if args.n_prompt_anchors <= 0:
        raise ValueError("--n_prompt_anchors must be positive")

    train_ds, eval_ds = load_mia_datasets(args)
    samples = dataset_to_samples(train_ds, label=1) + dataset_to_samples(eval_ds, label=0)

    anchor_sims = None
    if args.prompt_type == "random":
        anchor_indices = random_anchor_indices(samples, args.n_prompt_anchors, args.seed)
    elif args.prompt_type == "length":
        anchor_indices = length_anchor_indices(samples, args.n_prompt_anchors, args.length_pool, args.seed)
    elif args.prompt_type == "neighbor":
        anchor_indices, anchor_sims = neighbor_anchor_indices(samples, args)
    else:
        raise ValueError(f"unknown prompt_type: {args.prompt_type}")

    base_model = MODEL_DICT[args.model_name]
    paths = model_paths(args.model_name, args.sft_data, args.dpo_data)
    check_model_paths(paths)
    tokenizer = load_tokenizer(base_model)
    target_model = load_lora_model(base_model, paths["dpo"])

    records = build_score_records(samples, anchor_indices)
    scored_records = compute_logprobs(
        records,
        target_model,
        tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    rows = build_rows(samples, anchor_indices, args.prompt_type, scored_records, anchor_sims=anchor_sims)

    save_path = args.output or default_output_path(args)
    with open(save_path, "wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)

    counts = np.asarray([len(row["x0_indices"]) for row in rows], dtype=np.int64)
    print(f"Saved {len(rows)} prompt-anchor score rows to {save_path}")
    print(f"prompt_type={args.prompt_type}, n_prompt_anchors={args.n_prompt_anchors}")
    print(f"member={sum(row['label'] == 1 for row in rows)}, nonmember={sum(row['label'] == 0 for row in rows)}")
    print(f"x0 counts: mean={counts.mean():.2f}, min={counts.min()}, max={counts.max()}")


if __name__ == "__main__":
    main()
