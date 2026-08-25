import argparse
import pickle

import numpy as np

from eval_tools import clean_array, mia_auc_tpr, population_wrapper


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(description="MIA from target-generated response similarity/logprob margins")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=2)
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
    print(f"{name:>32s}: AUROC={auc:.4f}, TPR@FPR=0.05={tpr:.4f}, n={len(eval_scores)}")


def get_scores(rows, key):
    return np.asarray([row.get(key, 0.0) for row in rows], dtype=np.float64)


def response_values(row, key):
    return np.asarray([resp.get(key, 0.0) for resp in row.get("responses", [])], dtype=np.float64)


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


def finite_min(values, default=0.0):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return float(default)
    return float(values[finite].min())


def finite_max(values, default=0.0):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return float(default)
    return float(values[finite].max())


def topk_mean(values, k, largest=True):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    k = max(1, min(int(k), len(values)))
    values = np.sort(values)
    if largest:
        values = values[-k:]
    else:
        values = values[:k]
    return float(values.mean())


def build_response_scores(rows, top_k=2):
    out = {
        "response_chosen_similarity": get_scores(rows, "mean_sim_chosen"),
        "response_rejected_similarity": get_scores(rows, "mean_sim_rejected"),
        "response_chosen_margin": get_scores(rows, "mean_chosen_margin"),
        "response_rejected_margin": get_scores(rows, "mean_rejected_margin"),
    }

    sim_margins = []
    sim_mass = []
    sim_stability = []
    sim_topk = []
    sim_worst = []

    bracket_rates = []
    interval_min_means = []
    interval_min_topk = []
    interval_soft_means = []
    chosen_margin_topk = []
    rejected_margin_topk = []
    response_logp_std = []

    for row in rows:
        sim_margin = response_values(row, "sim_margin")
        chosen_margin = response_values(row, "chosen_margin")
        rejected_margin = response_values(row, "rejected_margin")
        response_logp = response_values(row, "response_logp")

        interval_min = np.minimum(chosen_margin, rejected_margin)
        bracket = (chosen_margin > 0) & (rejected_margin > 0)
        interval_soft = np.minimum(chosen_margin, rejected_margin) / (np.abs(chosen_margin) + np.abs(rejected_margin) + EPS)

        sim_margins.append(finite_mean(sim_margin))
        sim_mass.append(float(np.mean(sim_margin > 0)) if len(sim_margin) else 0.0)
        sim_stability.append(finite_mean(sim_margin) / (finite_std(sim_margin) + EPS))
        sim_topk.append(topk_mean(sim_margin, top_k, largest=True))
        sim_worst.append(finite_min(sim_margin))

        bracket_rates.append(float(np.mean(bracket)) if len(bracket) else 0.0)
        interval_min_means.append(finite_mean(interval_min))
        interval_min_topk.append(topk_mean(interval_min, top_k, largest=True))
        interval_soft_means.append(finite_mean(interval_soft))
        chosen_margin_topk.append(topk_mean(chosen_margin, top_k, largest=True))
        rejected_margin_topk.append(topk_mean(rejected_margin, top_k, largest=True))
        response_logp_std.append(finite_std(response_logp))

    out.update({
        "sim_margin_mean": np.asarray(sim_margins, dtype=np.float64),
        "sim_chosen_side_mass": np.asarray(sim_mass, dtype=np.float64),
        "sim_margin_stability": np.asarray(sim_stability, dtype=np.float64),
        "sim_margin_topk": np.asarray(sim_topk, dtype=np.float64),
        "sim_margin_worst": np.asarray(sim_worst, dtype=np.float64),
        "logp_bracketing_rate": np.asarray(bracket_rates, dtype=np.float64),
        "logp_interval_min_mean": np.asarray(interval_min_means, dtype=np.float64),
        "logp_interval_min_topk": np.asarray(interval_min_topk, dtype=np.float64),
        "logp_interval_soft_mean": np.asarray(interval_soft_means, dtype=np.float64),
        "chosen_margin_topk": np.asarray(chosen_margin_topk, dtype=np.float64),
        "rejected_margin_topk": np.asarray(rejected_margin_topk, dtype=np.float64),
        "response_logp_std": np.asarray(response_logp_std, dtype=np.float64),
    })
    return out


def main():
    args = parse_args()
    with open(args.input, "rb") as handle:
        rows = pickle.load(handle)

    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    scores = build_response_scores(rows, top_k=args.top_k)

    print(f"Loaded {len(rows)} rows from {args.input}")
    print(f"member: {int(labels.sum())} nonmember: {int((labels == 0).sum())}")
    print(f"top_k: {args.top_k}")
    print("=== Response-Generation MIA ===")
    for name, score in scores.items():
        evaluate_score(name, score, labels, popu=args.popu, n_popu=args.n_popu)


if __name__ == "__main__":
    main()
