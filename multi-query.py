import itertools
import os
import pickle

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm

from _args import parse_args
from common import MODEL_DICT, model_paths
from eval_tools import clean_array, mia_auc_tpr, population_wrapper, premia_scores
from output_store import build_dpo_trainer, load_lora_model, load_mia_datasets, load_tokenizer
from sem_query import SemanticNeighborBuilder, SemanticQueryConfig, build_semantic_relation_queries


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


def validate_args(args):
    if args.n_member <= 0 or args.n_nonmember <= 0:
        raise ValueError("--n_member and --n_nonmember must be positive")
    if args.sem_K <= 0:
        raise ValueError("--sem_K must be positive")
    for name in ["n_prompt_neighbors", "n_chosen_neighbors", "n_rejected_neighbors"]:
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be non-negative")
    if not 0.0 <= args.semantic_threshold <= 1.0:
        raise ValueError("--semantic_threshold must be in [0, 1]")
    if not 0.0 <= args.edit_threshold <= 1.0:
        raise ValueError("--edit_threshold must be in [0, 1]")


def check_model_paths(paths):
    sft_path = os.path.abspath(paths["sft"])
    dpo_path = os.path.abspath(paths["dpo"])
    if sft_path == dpo_path:
        raise ValueError(f"Reference and target model paths are identical: {sft_path}")
    if not os.path.exists(sft_path):
        raise FileNotFoundError(f"SFT/reference path does not exist: {sft_path}")
    if not os.path.exists(dpo_path):
        raise FileNotFoundError(f"DPO/target path does not exist: {dpo_path}")


def build_semantic_config(args):
    return SemanticQueryConfig(
        num_return_sequences=args.num_return_sequences,
        n_prompt_neighbors=args.n_prompt_neighbors,
        n_chosen_neighbors=args.n_chosen_neighbors,
        n_rejected_neighbors=args.n_rejected_neighbors,
        semantic_threshold=args.semantic_threshold,
        edit_threshold=args.edit_threshold,
        max_length=args.max_query_length,
        max_probes_per_tuple=args.max_probes_per_tuple,
        paraphrase_model=args.paraphrase_model,
        embed_model=args.embed_model,
        disable_generation=args.disable_generation,
    )


def iter_semantic_combinations(query_row, sem_k):
    prompts = query_row["prompt_neighbors"]
    chosens = query_row["chosen_neighbors"]
    rejecteds = query_row["rejected_neighbors"]
    combos = itertools.product(range(len(prompts)), range(len(chosens)), range(len(rejecteds)))
    for count, (i, j, k) in enumerate(combos):
        if count >= sem_k:
            break
        yield i, j, k, prompts[i], chosens[j], rejecteds[k]


def build_query_rows(args, samples):
    cfg = build_semantic_config(args)
    neighbor_builder = SemanticNeighborBuilder(cfg)
    rows = []
    for sample in tqdm(samples, desc="semantic query generation"):
        row = build_semantic_relation_queries(sample, neighbor_builder, cfg)
        row["label"] = int(sample.get("label", -1))
        rows.append(row)
    return rows


def flatten_query_rows(query_rows, sem_k):
    flat_records = []
    for sample_idx, row in enumerate(query_rows):
        combos = list(iter_semantic_combinations(row, sem_k))
        if not combos:
            combos = [(
                0,
                0,
                0,
                row["prompt_neighbors"][0],
                row["chosen_neighbors"][0],
                row["rejected_neighbors"][0],
            )]

        for i, j, k, prompt, chosen, rejected in combos:
            flat_records.append({
                "sample_idx": sample_idx,
                "prompt_idx": i,
                "chosen_idx": j,
                "rejected_idx": k,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            })
    return flat_records


def compute_flat_logprobs(flat_records, target_model, ref_model, tokenizer):
    if not flat_records:
        return []

    flat_dataset = Dataset.from_list([
        {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
        }
        for row in flat_records
    ])
    trainer = build_dpo_trainer(target_model, tokenizer, flat_dataset, flat_dataset)
    dataloader = trainer.get_eval_dataloader()

    scored = []
    offset = 0
    for batch in tqdm(dataloader, desc="DPOTrainer logprobs"):
        with torch.no_grad():
            target_out = trainer.concatenated_forward(target_model, batch)
            ref_out = trainer.concatenated_forward(ref_model, batch)

        m_chosen = target_out["chosen_logps"].detach().cpu().numpy()
        m_rejected = target_out["rejected_logps"].detach().cpu().numpy()
        r_chosen = ref_out["chosen_logps"].detach().cpu().numpy()
        r_rejected = ref_out["rejected_logps"].detach().cpu().numpy()

        batch_size = len(m_chosen)
        for i in range(batch_size):
            meta = flat_records[offset + i]
            scored.append({
                "sample_idx": meta["sample_idx"],
                "prompt_idx": meta["prompt_idx"],
                "chosen_idx": meta["chosen_idx"],
                "rejected_idx": meta["rejected_idx"],
                "m_chosen": float(m_chosen[i]),
                "m_rejected": float(m_rejected[i]),
                "r_chosen": float(r_chosen[i]),
                "r_rejected": float(r_rejected[i]),
            })
        offset += batch_size

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return scored


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return 0.0
    return float(values[finite].mean())


def attach_scores(query_rows, scored_records, save_query_logprobs=False):
    grouped = [[] for _ in query_rows]
    for record in scored_records:
        grouped[record["sample_idx"]].append(record)

    rows = []
    for sample_idx, row in enumerate(query_rows):
        records = grouped[sample_idx]
        out = dict(row)
        out.update({
            "m_chosen": finite_mean([x["m_chosen"] for x in records]),
            "m_rejected": finite_mean([x["m_rejected"] for x in records]),
            "r_chosen": finite_mean([x["r_chosen"] for x in records]),
            "r_rejected": finite_mean([x["r_rejected"] for x in records]),
            "num_query_combinations": len(records),
            "num_valid_combinations": int(sum(
                np.isfinite(x["m_chosen"])
                and np.isfinite(x["m_rejected"])
                and np.isfinite(x["r_chosen"])
                and np.isfinite(x["r_rejected"])
                for x in records
            )),
        })
        if save_query_logprobs:
            out["query_logprobs"] = [
                {k: v for k, v in x.items() if k != "sample_idx"}
                for x in records
            ]
        rows.append(out)
    return rows


def run_semantic_method(args, samples, target_model, ref_model, tokenizer):
    query_rows = build_query_rows(args, samples)
    flat_records = flatten_query_rows(query_rows, args.sem_K)
    scored_records = compute_flat_logprobs(flat_records, target_model, ref_model, tokenizer)
    return attach_scores(query_rows, scored_records, args.save_query_logprobs)


QUERY_METHODS = {
    "semantic": run_semantic_method,
}


def rows_to_arrays(rows):
    return {
        "m_chosen": clean_array([row["m_chosen"] for row in rows]),
        "m_rejected": clean_array([row["m_rejected"] for row in rows]),
        "r_chosen": clean_array([row["r_chosen"] for row in rows]),
        "r_rejected": clean_array([row["r_rejected"] for row in rows]),
        "labels": np.asarray([row["label"] for row in rows], dtype=np.int64),
    }


def evaluate_scores(scores, labels, popu=1, n_popu=1000):
    results = []
    for name, score in scores.items():
        eval_score = score
        eval_labels = labels
        if popu > 1:
            eval_score, eval_labels = population_wrapper(score, labels, popu, n_popu=n_popu)
        auc, tpr = mia_auc_tpr(eval_score, eval_labels)
        results.append((name, auc, tpr, len(eval_score)))
    return results


def default_output_path(args):
    return f"sem{args.sem_K}_{args.query_method}_{args.sft_data}_{args.dpo_data}_{args.model_name}.pkl"


def save_rows(rows, path):
    with open(path, "wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    args = parse_args()
    validate_args(args)

    train_ds, eval_ds = load_mia_datasets(args)
    samples = dataset_to_samples(train_ds, label=1) + dataset_to_samples(eval_ds, label=0)

    base_model = MODEL_DICT[args.model_name]
    paths = model_paths(args.model_name, args.sft_data, args.dpo_data)
    check_model_paths(paths)

    tokenizer = load_tokenizer(base_model)
    ref_model = load_lora_model(base_model, paths["sft"])
    target_model = load_lora_model(base_model, paths["dpo"])

    rows = QUERY_METHODS[args.query_method](args, samples, target_model, ref_model, tokenizer)

    save_path = args.output or default_output_path(args)
    save_rows(rows, save_path)
    print(f"Saved {len(rows)} multi-query scored samples to {save_path}")

    arrays = rows_to_arrays(rows)
    scores = premia_scores(
        arrays["m_chosen"],
        arrays["m_rejected"],
        arrays["r_chosen"],
        arrays["r_rejected"],
    )

    print(f"=== Semantic Multi-Query PreMIA: {save_path} sem_K={args.sem_K} popu={args.popu} ===")
    for name, auc, tpr, n in evaluate_scores(scores, arrays["labels"], args.popu, args.n_popu):
        print(f"{name:>8s}: AUROC={auc:.4f}, TPR@FPR=0.05={tpr:.4f}, n={n}")


if __name__ == "__main__":
    main()
