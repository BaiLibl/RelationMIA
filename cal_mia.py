import argparse
import pickle

import numpy as np

from eval_tools import clean_array, mia_auc_tpr, population_wrapper


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
        "target_margin": target_margin,
        "cal_margin": cal_margin,
        "target_minus_cal": target_margin - cal_margin,
        "target_avg_margin": target_avg_margin,
        "cal_avg_margin": cal_avg_margin,
        "target_minus_cal_avg": target_avg_margin - cal_avg_margin,
    }

    print(f"Loaded {len(rows)} rows from {args.input}")
    print(f"cal_model: {cal_model}")
    print(f"member: {int(labels.sum())} nonmember: {int((labels == 0).sum())}")
    print("=== Calibration MIA: target - cal ===")
    for name, score in scores.items():
        evaluate_score(name, score, labels, popu=args.popu, n_popu=args.n_popu)


if __name__ == "__main__":
    main()
