import argparse
import csv
import pickle

import numpy as np

from eval_tools import clean_array, mia_auc_tpr, population_wrapper


def parse_args():
    parser = argparse.ArgumentParser(description="PMI-style prompt-prior calibrated preference MIA")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--popu", type=int, default=1)
    parser.add_argument("--n_popu", type=int, default=1000)
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


def load_rows(path):
    with open(path, "rb") as handle:
        rows = pickle.load(handle)
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def finite(values):
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def finite_mean(values, default=0.0):
    arr = finite(values)
    if len(arr) == 0:
        return float(default)
    return float(arr.mean())


def finite_std(values, default=0.0):
    arr = finite(values)
    if len(arr) <= 1:
        return float(default)
    return float(arr.std())


def percentile_rank(value, values):
    arr = finite(values)
    if len(arr) == 0:
        return 0.5
    return float(np.mean(arr <= value))


def z_score(value, values, eps=1e-8):
    arr = finite(values)
    if len(arr) == 0:
        return 0.0
    return float((value - arr.mean()) / (arr.std() + eps))


def build_features(rows, eps=1e-8):
    labels = []
    bank = {}

    def add(name, value):
        bank.setdefault(name, []).append(float(value))

    for row in rows:
        labels.append(int(row["label"]))

        c_x = float(row["chosen_logp_x"])
        r_x = float(row["rejected_logp_x"])
        c0 = np.asarray(row["chosen_logps_x0"], dtype=np.float64)
        r0 = np.asarray(row["rejected_logps_x0"], dtype=np.float64)

        c_avg_x = float(row["chosen_avg_logp_x"])
        r_avg_x = float(row["rejected_avg_logp_x"])
        c_avg0 = np.asarray(row["chosen_avg_logps_x0"], dtype=np.float64)
        r_avg0 = np.asarray(row["rejected_avg_logps_x0"], dtype=np.float64)

        raw_avg = c_avg_x - r_avg_x

        c_avg_lift = c_avg_x - finite_mean(c_avg0, default=c_avg_x)
        r_avg_lift = r_avg_x - finite_mean(r_avg0, default=r_avg_x)
        pmi_avg = c_avg_lift - r_avg_lift

        anchor_avg_margins = c_avg0 - r_avg0

        # Core scores only. Use per-token averages to avoid length effects.
        # pmi_avg_margin is algebraically identical to avg_margin_minus_anchor_mean.
        add("raw_avg_margin", raw_avg)
        add("pmi_avg_margin", pmi_avg)
        add("avg_margin_z", z_score(raw_avg, anchor_avg_margins, eps=eps))
        add("avg_margin_rank", percentile_rank(raw_avg, anchor_avg_margins))

        # Side-specific diagnostics: whether chosen/rejected alone carries the signal.
        add("chosen_avg_lift", c_avg_lift)
        add("rejected_avg_lift", r_avg_lift)
        add("chosen_avg_z", z_score(c_avg_x, c_avg0, eps=eps))
        add("rejected_avg_z", z_score(r_avg_x, r_avg0, eps=eps))

        # Relative-position versions of PMI under separate chosen/rejected prompt-prior distributions.
        add("pmi_avg_zdiff", z_score(c_avg_x, c_avg0, eps=eps) - z_score(r_avg_x, r_avg0, eps=eps))
        add("pmi_avg_rankdiff", percentile_rank(c_avg_x, c_avg0) - percentile_rank(r_avg_x, r_avg0))

    return np.asarray(labels, dtype=np.int64), bank


def evaluate(bank, labels, popu=1, n_popu=1000):
    results = []
    for name, values in bank.items():
        scores = clean_array(values)
        eval_scores = scores
        eval_labels = labels
        if popu > 1:
            eval_scores, eval_labels = population_wrapper(scores, labels, popu, n_popu=n_popu)
        auc, tpr = mia_auc_tpr(eval_scores, eval_labels)
        results.append({
            "score": name,
            "auroc": float(auc),
            "tpr_at_fpr_0.05": float(tpr),
            "n": int(len(eval_scores)),
        })
    return results


def print_results(results):
    print("=== Prompt-Prior Calibrated MIA ===")
    for row in sorted(results, key=lambda x: x["score"]):
        print(
            f"{row['score']:>32s}: AUROC={row['auroc']:.4f}, "
            f"TPR@FPR=0.05={row['tpr_at_fpr_0.05']:.4f}, n={row['n']}"
        )
    best = max(results, key=lambda x: x["auroc"])
    print(
        f"best: {best['score']} AUROC={best['auroc']:.4f}, "
        f"TPR@FPR=0.05={best['tpr_at_fpr_0.05']:.4f}"
    )


def save_csv(results, path):
    if not results:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved CSV results to {path}")


def main():
    args = parse_args()
    rows = load_rows(args.input)
    labels, bank = build_features(rows, eps=args.eps)

    print(f"Loaded {len(rows)} rows from {args.input}")
    print("prompt_type:", rows[0].get("prompt_type"))
    print("member:", int((labels == 1).sum()), "nonmember:", int((labels == 0).sum()))
    print("x0 anchors:", len(rows[0].get("chosen_logps_x0", [])))

    results = evaluate(bank, labels, popu=args.popu, n_popu=args.n_popu)
    print_results(results)
    if args.output_csv:
        save_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
