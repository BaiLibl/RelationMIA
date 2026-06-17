import argparse
import csv
import pickle

import numpy as np

from eval_tools import clean_array, mia_auc_tpr, population_wrapper


DEFAULT_BETAS = "0.05,0.1,0.2"


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic preference graph MIA")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--top_frac", type=float, default=0.2)
    parser.add_argument("--bottom_frac", type=float, default=0.2)
    parser.add_argument("--betas", type=str, default=DEFAULT_BETAS)
    parser.add_argument("--popu", type=int, default=1)
    parser.add_argument("--n_popu", type=int, default=1000)
    parser.add_argument("--output_csv", type=str, default=None)
    return parser.parse_args()


def parse_float_list(text):
    values = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError("expected at least one float")
    return values


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def finite_values(values):
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def finite_mean(values, default=0.0):
    arr = finite_values(values)
    if len(arr) == 0:
        return float(default)
    return float(arr.mean())


def finite_std(values, default=0.0):
    arr = finite_values(values)
    if len(arr) <= 1:
        return float(default)
    return float(arr.std())


def tail_mean(values, frac, largest=False, default=0.0):
    arr = np.sort(finite_values(values))
    if len(arr) == 0:
        return float(default)
    k = max(1, int(np.ceil(len(arr) * float(frac))))
    selected = arr[-k:] if largest else arr[:k]
    return float(selected.mean())


def load_rows(path):
    with open(path, "rb") as handle:
        rows = pickle.load(handle)
    if not rows:
        raise ValueError(f"empty input: {path}")
    if "query_logprobs" not in rows[0]:
        raise ValueError("query_logprobs not found. Re-run multi-query.py with --save_query_logprobs.")
    return rows


def edge_margin(query):
    return float(query["m_chosen"] - query["m_rejected"])


def query_indices(query):
    return int(query["prompt_idx"]), int(query["chosen_idx"]), int(query["rejected_idx"])


def build_edges(row):
    edges = []
    original = None
    for query in row["query_logprobs"]:
        p_idx, c_idx, r_idx = query_indices(query)
        # Current semantic-anchor setting keeps prompt fixed. If prompt variants exist,
        # keep only the original-prompt graph so all edges share the same x.
        if p_idx != 0:
            continue
        margin = edge_margin(query)
        edge = {
            "chosen_idx": c_idx,
            "rejected_idx": r_idx,
            "margin": margin,
        }
        edges.append(edge)
        if (c_idx, r_idx) == (0, 0):
            original = edge

    if original is None:
        for query in row["query_logprobs"]:
            p_idx, c_idx, r_idx = query_indices(query)
            if (p_idx, c_idx, r_idx) == (0, 0, 0):
                original = {"chosen_idx": c_idx, "rejected_idx": r_idx, "margin": edge_margin(query)}
                break
    if original is None and edges:
        original = edges[0]
    return original, edges


def row_col_degrees(edges):
    rows = {}
    cols = {}
    for edge in edges:
        rows.setdefault(edge["chosen_idx"], []).append(edge["margin"])
        cols.setdefault(edge["rejected_idx"], []).append(edge["margin"])
    return rows, cols


def graph_features(row, args, betas):
    original, edges = build_edges(row)
    if original is None or not edges:
        return None

    margins = np.asarray([edge["margin"] for edge in edges], dtype=np.float64)
    other_margins = np.asarray(
        [edge["margin"] for edge in edges if not (edge["chosen_idx"] == 0 and edge["rejected_idx"] == 0)],
        dtype=np.float64,
    )
    if len(other_margins) == 0:
        other_margins = margins

    g0 = float(original["margin"])
    mean_all = finite_mean(margins, default=g0)
    std_all = finite_std(margins, default=0.0)
    median_all = float(np.median(finite_values(margins))) if len(finite_values(margins)) else g0
    max_other = float(np.max(finite_values(other_margins))) if len(finite_values(other_margins)) else g0
    min_all = float(np.min(finite_values(margins))) if len(finite_values(margins)) else g0

    rows_by_chosen, cols_by_rejected = row_col_degrees(edges)
    orig_row = rows_by_chosen.get(0, [g0])
    orig_col = cols_by_rejected.get(0, [g0])
    row_means = [finite_mean(v, default=g0) for v in rows_by_chosen.values()]
    col_means = [finite_mean(v, default=g0) for v in cols_by_rejected.values()]
    orig_row_mean = finite_mean(orig_row, default=g0)
    orig_col_mean = finite_mean(orig_col, default=g0)

    features = {}

    # 1. Original edge rank in the cross graph.
    features["orig_margin"] = g0
    features["orig_rank_all"] = float(np.mean(margins <= g0))
    features["orig_rank_other"] = float(np.mean(other_margins <= g0))

    # 2. Original edge uniqueness / gap relative to semantic edges.
    features["orig_gap_mean"] = g0 - mean_all
    features["orig_gap_median"] = g0 - median_all
    features["orig_gap_top"] = g0 - tail_mean(other_margins, args.top_frac, largest=True, default=g0)
    features["orig_gap_max"] = g0 - max_other
    features["orig_z_all"] = (g0 - mean_all) / (std_all + args.eps)

    # 3. Robust dominance / worst-case edge.
    features["worst_edge"] = min_all
    features["bottom_mean"] = tail_mean(margins, args.bottom_frac, largest=False, default=g0)
    features["orig_minus_worst"] = g0 - min_all

    # 4. Consistency of W -> L direction.
    features["consistency_pos"] = float(np.mean(margins > 0.0))
    features["consistency_orig_row"] = float(np.mean(np.asarray(orig_row, dtype=np.float64) > 0.0))
    features["consistency_orig_col"] = float(np.mean(np.asarray(orig_col, dtype=np.float64) > 0.0))

    # 5. Graph separation / node-degree style structure.
    # These are rank/relative-position variants, avoiding plain cross-cluster mean as the main signal.
    features["sep_mean_over_std"] = mean_all / (std_all + args.eps)
    features["orig_row_mean"] = orig_row_mean
    features["orig_col_mean"] = orig_col_mean
    features["orig_row_rank"] = float(np.mean(np.asarray(row_means, dtype=np.float64) <= orig_row_mean))
    features["orig_col_rank"] = float(np.mean(np.asarray(col_means, dtype=np.float64) <= orig_col_mean))
    features["orig_row_gap"] = orig_row_mean - finite_mean(row_means, default=orig_row_mean)
    features["orig_col_gap"] = orig_col_mean - finite_mean(col_means, default=orig_col_mean)

    for beta in betas:
        probs = sigmoid(beta * margins)
        p0 = float(sigmoid(beta * g0))
        features[f"sigmoid_b{beta:g}_orig"] = p0
        features[f"sigmoid_b{beta:g}_rank"] = float(np.mean(probs <= p0))
        features[f"sigmoid_b{beta:g}_gap_mean"] = p0 - finite_mean(probs, default=p0)
        features[f"sigmoid_b{beta:g}_consistency_09"] = float(np.mean(probs > 0.9))
        entropy = -(probs * np.log(probs + args.eps) + (1 - probs) * np.log(1 - probs + args.eps))
        features[f"sigmoid_b{beta:g}_entropy_mean"] = finite_mean(entropy, default=0.0)

    features["num_edges"] = float(len(edges))
    features["num_chosen_nodes"] = float(len(rows_by_chosen))
    features["num_rejected_nodes"] = float(len(cols_by_rejected))
    return features


def evaluate_feature_bank(feature_bank, labels, args):
    results = []
    for name, values in feature_bank.items():
        if name.startswith("num_"):
            continue
        scores = clean_array(values)
        eval_scores = scores
        eval_labels = labels
        if args.popu > 1:
            eval_scores, eval_labels = population_wrapper(scores, labels, args.popu, n_popu=args.n_popu)
        auc, tpr = mia_auc_tpr(eval_scores, eval_labels)
        results.append({
            "score": name,
            "auroc": float(auc),
            "tpr_at_fpr_0.05": float(tpr),
            "n": int(len(eval_scores)),
        })
    return results


def print_results(results):
    print("=== Semantic Preference Graph MIA ===")
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
    betas = parse_float_list(args.betas)
    rows = load_rows(args.input)

    labels = []
    feature_bank = {}
    skipped = 0
    for row in rows:
        feats = graph_features(row, args, betas)
        if feats is None:
            skipped += 1
            continue
        labels.append(int(row["label"]))
        for name, value in feats.items():
            feature_bank.setdefault(name, []).append(value)

    labels = np.asarray(labels, dtype=np.int64)
    print(f"Loaded {len(rows)} rows from {args.input}")
    print(f"usable rows: {len(labels)}, skipped: {skipped}")
    print("member:", int((labels == 1).sum()), "nonmember:", int((labels == 0).sum()))
    for key in ["num_edges", "num_chosen_nodes", "num_rejected_nodes"]:
        if key in feature_bank:
            arr = np.asarray(feature_bank[key], dtype=np.float64)
            print(f"{key}: mean={arr.mean():.2f}, median={np.median(arr):.2f}, min={arr.min():.0f}, max={arr.max():.0f}")

    results = evaluate_feature_bank(feature_bank, labels, args)
    print_results(results)
    if args.output_csv:
        save_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
