import argparse
import csv
import pickle

import numpy as np

from eval_tools import clean_array, mia_auc_tpr, population_wrapper


ANCHOR_KINDS = ["all", "prompt", "chosen", "rejected", "mixed"]


def parse_args():
    parser = argparse.ArgumentParser(description="Reference-free semantic anchor-calibrated MIA")
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
    if "query_logprobs" not in rows[0]:
        raise ValueError("query_logprobs not found. Re-run multi-query.py with --save_query_logprobs.")
    return rows


def target_margin(record):
    return float(record["m_chosen"] - record["m_rejected"])


def query_indices(query):
    return query["prompt_idx"], query["chosen_idx"], query["rejected_idx"]


def split_queries(row):
    original = None
    generated = []
    for query in row["query_logprobs"]:
        if query_indices(query) == (0, 0, 0):
            original = query
        else:
            generated.append(query)
    if original is None:
        if not row["query_logprobs"]:
            raise ValueError("empty query_logprobs row")
        original = row["query_logprobs"][0]
    return original, generated


def query_kind(query):
    p, c, r = query_indices(query)
    changed = int(p != 0) + int(c != 0) + int(r != 0)
    if changed == 0:
        return "original"
    if changed > 1:
        return "mixed"
    if p != 0:
        return "prompt"
    if c != 0:
        return "chosen"
    return "rejected"


def finite_anchor_values(values):
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def rp_z(original_score, anchor_scores, eps=1e-8):
    anchors = finite_anchor_values(anchor_scores)
    if len(anchors) == 0:
        return 0.0
    if len(anchors) == 1:
        return float(original_score - anchors.mean())
    return float((original_score - anchors.mean()) / (anchors.std() + eps))


def rank_percentile(original_score, anchor_scores):
    anchors = finite_anchor_values(anchor_scores)
    if len(anchors) == 0:
        return 0.5
    return float(np.mean(anchors <= original_score))


def build_scores(rows, eps=1e-8):
    labels = []
    counts = {kind: [] for kind in ANCHOR_KINDS}
    score_bank = {}

    for kind in ANCHOR_KINDS:
        score_bank[f"{kind}_rp_z"] = []
        score_bank[f"{kind}_rank"] = []

    for row in rows:
        labels.append(int(row["label"]))
        original, generated = split_queries(row)
        original_score = target_margin(original)

        anchors_by_kind = {kind: [] for kind in ANCHOR_KINDS}
        for query in generated:
            kind = query_kind(query)
            anchors_by_kind["all"].append(query)
            anchors_by_kind[kind].append(query)

        for kind in ANCHOR_KINDS:
            anchors = [target_margin(query) for query in anchors_by_kind[kind]]
            counts[kind].append(len(anchors))
            score_bank[f"{kind}_rp_z"].append(rp_z(original_score, anchors, eps=eps))
            score_bank[f"{kind}_rank"].append(rank_percentile(original_score, anchors))

    return np.asarray(labels, dtype=np.int64), score_bank, counts


def evaluate_scores(score_bank, labels, popu=1, n_popu=1000):
    results = []
    for name, values in score_bank.items():
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


def print_counts(counts):
    print("=== Anchor Counts ===")
    for kind in ANCHOR_KINDS:
        arr = np.asarray(counts[kind], dtype=np.int64)
        print(
            f"{kind:>8s}: mean={arr.mean():.2f}, median={np.median(arr):.2f}, "
            f"min={arr.min()}, max={arr.max()}, empty={(arr == 0).sum()}"
        )


def print_results(results):
    print("=== Reference-Free Semantic Anchor-Calibrated MIA ===")
    for row in sorted(results, key=lambda x: x["score"]):
        print(
            f"{row['score']:>16s}: AUROC={row['auroc']:.4f}, "
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
    labels, score_bank, counts = build_scores(rows, eps=args.eps)

    print(f"Loaded {len(rows)} rows from {args.input}")
    print("member:", int((labels == 1).sum()), "nonmember:", int((labels == 0).sum()))
    print("score: M(q) = m_chosen - m_rejected only; no reference logprobs used")
    print_counts(counts)

    results = evaluate_scores(score_bank, labels, popu=args.popu, n_popu=args.n_popu)
    print_results(results)
    if args.output_csv:
        save_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
