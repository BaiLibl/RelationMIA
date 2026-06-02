import csv
import os

import numpy as np

from _args import parse_args
from common import MODEL_DICT, default_split_sizes, model_paths
from eval_tools import mia_auc_tpr, population_wrapper, premia_scores
from explore_datasets import DATASET_LOADERS
from output_store import build_dpo_trainer, collect_logprobs, load_lora_model, load_tokenizer


def load_datasets(args):
    n_train, n_eval = default_split_sizes(args.dpo_data)
    return DATASET_LOADERS[args.dpo_data](
        n_train,
        n_eval,
        for_dpo=True,
        n_members=args.n_member,
        n_nonmembers=args.n_nonmember,
    )


def rows_to_arrays(rows):
    return {
        "m_chosen": np.asarray([row["m_chosen"] for row in rows], dtype=np.float64),
        "m_rejected": np.asarray([row["m_rejected"] for row in rows], dtype=np.float64),
        "r_chosen": np.asarray([row["r_chosen"] for row in rows], dtype=np.float64),
        "r_rejected": np.asarray([row["r_rejected"] for row in rows], dtype=np.float64),
        "labels": np.asarray([row["label"] for row in rows], dtype=np.int64),
    }


def evaluate(scores, labels, args):
    results = []
    for name, score in scores.items():
        eval_score = score
        eval_labels = labels
        if args.popu > 1:
            eval_score, eval_labels = population_wrapper(score, labels, args.popu, args.n_popu)
        auc, tpr = mia_auc_tpr(eval_score, eval_labels)
        results.append({"mia": name, "auc": auc, "tpr": tpr, "n": len(eval_score)})
    return results


def append_results(path, args, rows):
    fieldnames = ["model_name", "sft_data", "dpo_data", "popu", "mia", "auc", "tpr", "n"]
    exists = os.path.exists(path)
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({
                "model_name": args.model_name,
                "sft_data": args.sft_data,
                "dpo_data": args.dpo_data,
                "popu": args.popu,
                "mia": row["mia"],
                "auc": f"{row['auc']:.4f}",
                "tpr": f"{row['tpr']:.4f}",
                "n": row["n"],
            })


def main():
    args = parse_args()
    base_model = MODEL_DICT[args.model_name]
    paths = model_paths(args.model_name, args.sft_data, args.dpo_data)

    print(">>> Loading models ...")
    tokenizer = load_tokenizer(base_model)
    ref_model = load_lora_model(base_model, paths["sft"])
    target_model = load_lora_model(base_model, paths["dpo"])

    print(">>> Loading datasets ...")
    train_ds, eval_ds = load_datasets(args)
    trainer = build_dpo_trainer(target_model, tokenizer, train_ds, eval_ds)

    rows = collect_logprobs(
        trainer,
        target_model,
        ref_model,
        max_samples=max(args.n_member, args.n_nonmember),
    )
    arrays = rows_to_arrays(rows)
    scores = premia_scores(
        arrays["m_chosen"],
        arrays["m_rejected"],
        arrays["r_chosen"],
        arrays["r_rejected"],
    )
    results = evaluate(scores, arrays["labels"], args)

    print(f"=== PreMIA Baseline: popu={args.popu} ===")
    for row in results:
        print(f"{row['mia']:>8s}: AUROC={row['auc']:.4f}, TPR@FPR=0.05={row['tpr']:.4f}, n={row['n']}")

    if args.results_csv:
        append_results(args.results_csv, args, results)


if __name__ == "__main__":
    main()
