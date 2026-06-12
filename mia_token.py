import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from eval_tools import clean_array, mia_auc_tpr, population_wrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Token-level MIA from query_token.py outputs")
    parser.add_argument("--input", type=str, required=True, help="token-level pkl file")
    parser.add_argument("--ks", type=str, default="5,10,20", help="comma-separated percentages for min/max-k")
    parser.add_argument("--n_buckets", type=int, default=10)
    parser.add_argument("--fpr", type=float, default=0.05)
    parser.add_argument("--popu", type=int, default=1)
    parser.add_argument("--n_popu", type=int, default=1000)
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


def parse_float_list(text):
    if text is None:
        return None
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def value_at(mapping, temp):
    if temp in mapping:
        return mapping[temp]
    key = float(temp)
    if key in mapping:
        return mapping[key]
    text_key = str(float(temp))
    if text_key in mapping:
        return mapping[text_key]
    short_key = str(temp)
    if short_key in mapping:
        return mapping[short_key]
    raise KeyError(f"Temperature {temp} not found. keys={list(mapping.keys())[:5]}")


def available_temps(row):
    return sorted(float(k) for k in row["chosen_token_logps"].keys())


def finite_tokens(values):
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def tail_stat(values, k_percent, mode, stat):
    arr = np.sort(finite_tokens(values))
    if len(arr) == 0:
        return np.nan
    k = max(1, int(np.ceil(len(arr) * float(k_percent) / 100.0)))
    if mode == "min":
        selected = arr[:k]
    elif mode == "max":
        selected = arr[-k:]
    else:
        raise ValueError(f"unknown mode: {mode}")

    if stat == "mean":
        return float(np.mean(selected))
    if stat == "sum":
        return float(np.sum(selected))
    raise ValueError(f"unknown stat: {stat}")


def bucket_means(values, n_buckets):
    arr = np.sort(finite_tokens(values))
    if len(arr) == 0:
        return [np.nan] * n_buckets
    chunks = np.array_split(arr, n_buckets)
    return [float(np.mean(chunk)) if len(chunk) else np.nan for chunk in chunks]


def eval_score(scores, labels, fpr=0.05, popu=1, n_popu=1000):
    scores = clean_array(scores)
    eval_scores = scores
    eval_labels = labels
    if popu > 1:
        eval_scores, eval_labels = population_wrapper(scores, labels, popu, n_popu=n_popu)
    auc, tpr = mia_auc_tpr(eval_scores, eval_labels, fpr_threshold=fpr)
    return auc, tpr, len(eval_scores)


def add_result(results, source, temp, score_name, scores, labels, args):
    auc, tpr, n = eval_score(scores, labels, fpr=args.fpr, popu=args.popu, n_popu=args.n_popu)
    results.append({
        "source": source,
        "temperature": float(temp),
        "score": score_name,
        "auroc": float(auc),
        f"tpr_at_fpr_{args.fpr:g}": float(tpr),
        "n": int(n),
    })


def evaluate_rows(rows, source, temps, ks, args):
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    results = []

    for temp in temps:
        chosen = [value_at(row["chosen_token_logps"], temp) for row in rows]
        rejected = [value_at(row["rejected_token_logps"], temp) for row in rows]

        for k in ks:
            for mode in ["min", "max"]:
                for stat in ["mean", "sum"]:
                    chosen_scores = np.asarray([tail_stat(x, k, mode, stat) for x in chosen], dtype=np.float64)
                    rejected_scores = np.asarray([tail_stat(x, k, mode, stat) for x in rejected], dtype=np.float64)
                    scores = chosen_scores - rejected_scores
                    add_result(results, source, temp, f"{mode}{k:g}_{stat}_margin", scores, labels, args)

        chosen_buckets = np.asarray([bucket_means(x, args.n_buckets) for x in chosen], dtype=np.float64)
        rejected_buckets = np.asarray([bucket_means(x, args.n_buckets) for x in rejected], dtype=np.float64)
        bucket_scores = chosen_buckets - rejected_buckets
        for bucket_idx in range(args.n_buckets):
            add_result(
                results,
                source,
                temp,
                f"bucket{bucket_idx + 1:02d}_mean_margin",
                bucket_scores[:, bucket_idx],
                labels,
                args,
            )

    return results


def print_results(results, fpr):
    tpr_key = f"tpr_at_fpr_{fpr:g}"
    by_source = {}
    for row in results:
        by_source.setdefault(row["source"], []).append(row)

    for source, rows in by_source.items():
        print(f"=== Token MIA: {source} ===")
        rows = sorted(rows, key=lambda x: (x["temperature"], x["score"]))
        for row in rows:
            print(
                f"T={row['temperature']:<4g} {row['score']:>24s}: "
                f"AUROC={row['auroc']:.4f}, "
                f"TPR@FPR={fpr:g}={row[tpr_key]:.4f}, n={row['n']}"
            )
        best = max(rows, key=lambda x: x["auroc"])
        print(
            f"best: T={best['temperature']:g} {best['score']} "
            f"AUROC={best['auroc']:.4f}, TPR@FPR={fpr:g}={best[tpr_key]:.4f}"
        )


def save_csv(results, path):
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved CSV results to {path}")


def main():
    args = parse_args()
    ks = parse_float_list(args.ks)
    if not ks:
        raise ValueError("--ks must contain at least one percentage")
    path = Path(args.input)
    with open(path, "rb") as handle:
        rows = pickle.load(handle)
    if not rows:
        raise ValueError(f"empty file: {path}")

    temps = available_temps(rows[0])
    print(f"Loaded {len(rows)} samples from {path}")
    print("temps:", ",".join(f"{x:g}" for x in temps))
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    print("member:", int((labels == 1).sum()), "nonmember:", int((labels == 0).sum()))

    all_results = evaluate_rows(rows, path.name, temps, ks, args)
    print_results(all_results, args.fpr)
    if args.output_csv:
        save_csv(all_results, args.output_csv)


if __name__ == "__main__":
    main()
