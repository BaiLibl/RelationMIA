import argparse
import pickle

import numpy as np

from eval_tools import clean_array, mia_auc_tpr, population_wrapper


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(description="MIA with target minus calibration margin")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--popu", type=int, default=1)
    parser.add_argument("--n_popu", type=int, default=1000)
    return parser.parse_args()


def evaluate_score(name, scores, labels, popu=1, n_popu=1000):
    scores = clean_array(scores)
    eval_scores = scores
    eval_labels = labels
    if popu > 1:
        eval_scores, eval_labels = population_wrapper(scores, labels, popu, n_popu=n_popu)
    auc, tpr = mia_auc_tpr(eval_scores, eval_labels)
    print(f"{name:>24s}: AUROC={auc:.4f}, TPR@FPR=0.05={tpr:.4f}, n={len(eval_scores)}")


def standardize(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float64)
    if finite.sum() == 0:
        return out
    mean = values[finite].mean()
    std = values[finite].std()
    out[finite] = (values[finite] - mean) / (std + EPS)
    return out


def percentile_rank(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float64)
    if finite.sum() == 0:
        return out
    finite_values = values[finite]
    sorted_values = np.sort(finite_values)
    left = np.searchsorted(sorted_values, finite_values, side="left")
    right = np.searchsorted(sorted_values, finite_values, side="right")
    out[finite] = 0.5 * (left + right) / max(len(sorted_values), 1)
    return out


def quantile_map(source_values, target_values):
    source_values = np.asarray(source_values, dtype=np.float64)
    target_values = np.asarray(target_values, dtype=np.float64)
    finite_source = np.isfinite(source_values)
    finite_target = np.isfinite(target_values)
    mapped = np.zeros_like(source_values, dtype=np.float64)
    if finite_source.sum() == 0 or finite_target.sum() == 0:
        return mapped

    src_sorted = np.sort(source_values[finite_source])
    tgt_sorted = np.sort(target_values[finite_target])
    src_vals = source_values[finite_source]
    left = np.searchsorted(src_sorted, src_vals, side="left")
    right = np.searchsorted(src_sorted, src_vals, side="right")
    percentiles = 0.5 * (left + right) / max(len(src_sorted), 1)

    if len(tgt_sorted) == 1:
        mapped[finite_source] = tgt_sorted[0]
        return mapped

    positions = percentiles * (len(tgt_sorted) - 1)
    mapped[finite_source] = np.interp(positions, np.arange(len(tgt_sorted)), tgt_sorted)
    return mapped


def add_calibrated_scores(scores, target_margin, cal_margin, suffix=""):
    name_suffix = f"_{suffix}" if suffix else ""

    scores[f"z_target_minus_cal{name_suffix}"] = standardize(target_margin) - standardize(cal_margin)
    scores[f"rank_target_minus_cal{name_suffix}"] = percentile_rank(target_margin) - percentile_rank(cal_margin)

    mapped_cal = quantile_map(cal_margin, target_margin)
    scores[f"qmap_target_minus_cal{name_suffix}"] = target_margin - mapped_cal


def main():
    args = parse_args()
    with open(args.input, "rb") as handle:
        rows = pickle.load(handle)

    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    cal_model = rows[0].get("cal_model", "unknown") if rows else "unknown"

    target_margin = np.asarray([row["target_margin"] for row in rows], dtype=np.float64)
    cal_margin = np.asarray([row["cal_margin"] for row in rows], dtype=np.float64)
    target_avg_margin = np.asarray([row["target_avg_margin"] for row in rows], dtype=np.float64)
    cal_avg_margin = np.asarray([row["cal_avg_margin"] for row in rows], dtype=np.float64)

    scores = {
        "target_minus_cal": target_margin - cal_margin,
        "target_minus_cal_avg": target_avg_margin - cal_avg_margin,
    }
    add_calibrated_scores(scores, target_margin, cal_margin)
    add_calibrated_scores(scores, target_avg_margin, cal_avg_margin, suffix="avg")

    print(f"Loaded {len(rows)} rows from {args.input}")
    for name, score in scores.items():
        evaluate_score(name, score, labels, popu=args.popu, n_popu=args.n_popu)


if __name__ == "__main__":
    main()
