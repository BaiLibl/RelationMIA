import argparse
import os
import pickle

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm

from common import MODEL_DICT, model_paths
from eval_tools import clean_array, mia_auc_tpr, population_wrapper
from output_store import build_dpo_trainer, load_lora_model, load_mia_datasets, load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Reference-free counterfactual tuple MIA")
    parser.add_argument("--model_name", type=str, default="gpt2", choices=MODEL_DICT.keys())
    parser.add_argument("--sft_data", type=str, default="imdb")
    parser.add_argument("--dpo_data", type=str, default="imdb")
    parser.add_argument("--n_member", type=int, default=1000)
    parser.add_argument("--n_nonmember", type=int, default=1000)
    parser.add_argument("--sem_K", type=int, default=8, help="number of counterfactual response pairs per tuple")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length_match_pool", type=int, default=50, help="sample replacements from this many closest-length responses")
    parser.add_argument("--sigmoid_betas", type=str, default="0.01,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save_counter_logprobs", action="store_true")
    parser.add_argument("--popu", type=int, default=1)
    parser.add_argument("--n_popu", type=int, default=1000)
    return parser.parse_args()


def check_model_paths(paths):
    dpo_path = os.path.abspath(paths["dpo"])
    if not os.path.exists(dpo_path):
        raise FileNotFoundError(f"DPO/target path does not exist: {dpo_path}")


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


def response_length(text):
    return len(str(text).split())


def build_length_pool(texts):
    return [
        {"text": text, "length": response_length(text)}
        for text in texts
    ]


def sample_length_matched(pool, original, rng, match_pool_size=50):
    if len(pool) <= 1:
        return original

    original_len = response_length(original)
    candidates = [item for item in pool if item["text"] != original]
    if not candidates:
        return original

    candidates.sort(key=lambda item: abs(item["length"] - original_len))
    top_k = max(1, min(match_pool_size, len(candidates)))
    selected = candidates[int(rng.integers(0, top_k))]
    return selected["text"]


def build_counterfactual_records(samples, sem_k, seed=42, length_match_pool=50):
    rng = np.random.default_rng(seed)
    chosen_pool = build_length_pool([x["chosen"] for x in samples])
    rejected_pool = build_length_pool([x["rejected"] for x in samples])

    n_replace_chosen = sem_k // 2
    n_replace_rejected = sem_k - n_replace_chosen

    flat_records = []
    for sample_idx, sample in enumerate(samples):
        x = sample["prompt"]
        yw = sample["chosen"]
        yl = sample["rejected"]

        flat_records.append({
            "sample_idx": sample_idx,
            "kind": "original",
            "prompt": x,
            "chosen": yw,
            "rejected": yl,
        })

        for _ in range(n_replace_chosen):
            yw_alt = sample_length_matched(chosen_pool, yw, rng, match_pool_size=length_match_pool)
            flat_records.append({
                "sample_idx": sample_idx,
                "kind": "replace_chosen",
                "prompt": x,
                "chosen": yw_alt,
                "rejected": yl,
            })

        for _ in range(n_replace_rejected):
            yl_alt = sample_length_matched(rejected_pool, yl, rng, match_pool_size=length_match_pool)
            flat_records.append({
                "sample_idx": sample_idx,
                "kind": "replace_rejected",
                "prompt": x,
                "chosen": yw,
                "rejected": yl_alt,
            })

    return flat_records


def compute_target_logprobs(flat_records, target_model, tokenizer):
    flat_dataset = Dataset.from_list([
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
        for r in flat_records
    ])
    trainer = build_dpo_trainer(target_model, tokenizer, flat_dataset, flat_dataset)
    dataloader = trainer.get_eval_dataloader()

    scored = []
    offset = 0
    for batch in tqdm(dataloader, desc="counterfactual target logprobs"):
        with torch.no_grad():
            out = trainer.concatenated_forward(target_model, batch)

        m_chosen = out["chosen_logps"].detach().cpu().numpy()
        m_rejected = out["rejected_logps"].detach().cpu().numpy()
        batch_size = len(m_chosen)

        for i in range(batch_size):
            meta = flat_records[offset + i]
            margin = float(m_chosen[i] - m_rejected[i])
            scored.append({
                "sample_idx": meta["sample_idx"],
                "kind": meta["kind"],
                "m_chosen": float(m_chosen[i]),
                "m_rejected": float(m_rejected[i]),
                "margin": margin,
            })
        offset += batch_size

    return scored


def finite_mean(values, default=0.0):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return float(default)
    return float(values[finite].mean())


def finite_std(values, default=0.0):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() <= 1:
        return float(default)
    return float(values[finite].std())


def parse_betas(beta_text):
    betas = []
    for item in str(beta_text).split(","):
        item = item.strip()
        if item:
            betas.append(float(item))
    if not betas:
        raise ValueError("--sigmoid_betas must contain at least one value")
    return betas


def sigmoid_margin(margin, beta):
    x = np.clip(beta * np.asarray(margin, dtype=np.float64), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def attach_counter_scores(samples, scored_records, save_counter_logprobs=False, sigmoid_betas=None):
    if sigmoid_betas is None:
        sigmoid_betas = []
    grouped = [[] for _ in samples]
    for record in scored_records:
        grouped[record["sample_idx"]].append(record)

    rows = []
    for sample_idx, sample in enumerate(samples):
        records = grouped[sample_idx]
        original = next((x for x in records if x["kind"] == "original"), None)
        if original is None:
            raise ValueError(f"Missing original record for sample {sample_idx}")

        counter = [x for x in records if x["kind"] != "original"]
        replace_chosen = [x["margin"] for x in counter if x["kind"] == "replace_chosen"]
        replace_rejected = [x["margin"] for x in counter if x["kind"] == "replace_rejected"]
        all_counter = [x["margin"] for x in counter]

        base_margin = float(original["margin"])
        ref_chosen = finite_mean(replace_chosen, default=base_margin)
        ref_rejected = finite_mean(replace_rejected, default=base_margin)
        ref_all = finite_mean(all_counter, default=base_margin)
        std_all = finite_std(all_counter, default=0.0)
        rank_values = np.asarray(all_counter + [base_margin], dtype=np.float64)

        row = {
            "label": int(sample["label"]),
            "base_margin": base_margin,
            "ref_chosen": ref_chosen,
            "ref_rejected": ref_rejected,
            "ref_all": ref_all,
            "score_chosen_ref": base_margin - ref_chosen,
            "score_rejected_ref": base_margin - ref_rejected,
            "score_all_ref": base_margin - ref_all,
            "score_z": (base_margin - ref_all) / (std_all + 1e-8),
            "score_rank": float(np.mean(rank_values <= base_margin)),
            "num_replace_chosen": len(replace_chosen),
            "num_replace_rejected": len(replace_rejected),
        }

        for beta in sigmoid_betas:
            beta_name = str(beta).replace(".", "p").replace("-", "m")
            p_base = float(sigmoid_margin(base_margin, beta))

            # Probability-space reference: E[sigmoid(beta * g_counter)].
            p_chosen = finite_mean(sigmoid_margin(replace_chosen, beta), default=p_base)
            p_rejected = finite_mean(sigmoid_margin(replace_rejected, beta), default=p_base)
            p_all = finite_mean(sigmoid_margin(all_counter, beta), default=p_base)
            row[f"sigmoid_chosen_ref_b{beta_name}"] = p_base - p_chosen
            row[f"sigmoid_rejected_ref_b{beta_name}"] = p_base - p_rejected
            row[f"sigmoid_all_ref_b{beta_name}"] = p_base - p_all

            # Margin-space implicit reference: sigmoid(beta * g0) - sigmoid(beta * E[g_counter]).
            row[f"sigmoid_mean_chosen_ref_b{beta_name}"] = p_base - float(sigmoid_margin(ref_chosen, beta))
            row[f"sigmoid_mean_rejected_ref_b{beta_name}"] = p_base - float(sigmoid_margin(ref_rejected, beta))
            row[f"sigmoid_mean_all_ref_b{beta_name}"] = p_base - float(sigmoid_margin(ref_all, beta))
        if save_counter_logprobs:
            row["counter_logprobs"] = [
                {k: v for k, v in record.items() if k != "sample_idx"}
                for record in records
            ]
        rows.append(row)

    return rows


def evaluate(rows, popu=1, n_popu=1000):
    labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
    score_names = [
        "base_margin",
        "score_chosen_ref",
        "score_rejected_ref",
        "score_all_ref",
        "score_z",
        "score_rank",
    ]
    sigmoid_names = sorted([name for name in rows[0].keys() if name.startswith("sigmoid_")])
    score_names.extend(sigmoid_names)

    print("=== Counterfactual Tuple MIA ===")
    for name in score_names:
        scores = clean_array([r[name] for r in rows])
        eval_scores = scores
        eval_labels = labels
        if popu > 1:
            eval_scores, eval_labels = population_wrapper(scores, labels, popu, n_popu=n_popu)
        auc, tpr = mia_auc_tpr(eval_scores, eval_labels)
        print(f"{name:>30s}: AUROC={auc:.4f}, TPR@FPR=0.05={tpr:.4f}, n={len(eval_scores)}")


def default_output_path(args):
    return f"counter_tuple_{args.sft_data}_{args.dpo_data}_{args.model_name}_K{args.sem_K}.pkl"


def main():
    args = parse_args()
    if args.sem_K <= 0:
        raise ValueError("--sem_K must be positive")

    train_ds, eval_ds = load_mia_datasets(args)
    samples = dataset_to_samples(train_ds, label=1) + dataset_to_samples(eval_ds, label=0)

    base_model = MODEL_DICT[args.model_name]
    paths = model_paths(args.model_name, args.sft_data, args.dpo_data)
    check_model_paths(paths)

    tokenizer = load_tokenizer(base_model)
    target_model = load_lora_model(base_model, paths["dpo"])

    sigmoid_betas = parse_betas(args.sigmoid_betas)
    flat_records = build_counterfactual_records(
        samples,
        args.sem_K,
        seed=args.seed,
        length_match_pool=args.length_match_pool,
    )
    scored_records = compute_target_logprobs(flat_records, target_model, tokenizer)
    rows = attach_counter_scores(
        samples,
        scored_records,
        save_counter_logprobs=args.save_counter_logprobs,
        sigmoid_betas=sigmoid_betas,
    )

    save_path = args.output or default_output_path(args)
    with open(save_path, "wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(rows)} counterfactual scored samples to {save_path}")
    evaluate(rows, popu=args.popu, n_popu=args.n_popu)


if __name__ == "__main__":
    main()
