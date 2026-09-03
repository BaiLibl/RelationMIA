import argparse
import os

import numpy as np

from eval_tools import clean_array, mia_auc_tpr, population_wrapper


EPS = 1e-12


def softmax(logps):
    x = np.asarray(logps, dtype=np.float64)
    if x.size == 0:
        return x.reshape(0, 3)
    x = x.reshape(-1, 3)
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.maximum(np.sum(exp_x, axis=1, keepdims=True), EPS)


def safe_logps(values):
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return x.reshape(-1, 3)


def abc_stats(logps):
    logps = safe_logps(logps)
    probs = softmax(logps)

    if len(logps) == 0:
        return {
            "logp_A_sum": 0.0,
            "logp_B_sum": 0.0,
            "logp_C_sum": 0.0,
            "logp_A_win": 0.0,
            "p_A_sum": 0.0,
            "p_B_sum": 0.0,
            "p_C_sum": 0.0,
            "p_A_win": 0.0,
        }

    return {
        "logp_A_sum": float(np.sum(logps[:, 0])),
        "logp_B_sum": float(np.sum(logps[:, 1])),
        "logp_C_sum": float(np.sum(logps[:, 2])),
        "logp_A_win": float(np.sum((logps[:, 0] > logps[:, 1]) & (logps[:, 0] > logps[:, 2]))),
        "p_A_sum": float(np.sum(probs[:, 0])),
        "p_B_sum": float(np.sum(probs[:, 1])),
        "p_C_sum": float(np.sum(probs[:, 2])),
        "p_A_win": float(np.sum((probs[:, 0] > probs[:, 1]) & (probs[:, 0] > probs[:, 2]))),
    }


def row_signals(row):
    m_stats = abc_stats(row.get("m_logps", []))
    r_stats = abc_stats(row.get("r_logps", []))

    signals = {
        "m_logp_A_sum": m_stats["logp_A_sum"],
        "m_logp_B_sum": m_stats["logp_B_sum"],
        "m_logp_C_sum": m_stats["logp_C_sum"],
        "m_logp_A_win": m_stats["logp_A_win"],
        "m_p_A_sum": m_stats["p_A_sum"],
        "m_p_B_sum": m_stats["p_B_sum"],
        "m_p_C_sum": m_stats["p_C_sum"],
        "m_p_A_win": m_stats["p_A_win"],
    }

    for name in (
        "logp_A_sum",
        "logp_B_sum",
        "logp_C_sum",
        "logp_A_win",
        "p_A_sum",
        "p_B_sum",
        "p_C_sum",
        "p_A_win",
    ):
        signals[f"delta_{name}"] = m_stats[name] - r_stats[name]

    signals["delta_pref"] = (
        (m_stats["p_A_sum"] - m_stats["p_C_sum"])
        - (r_stats["p_A_sum"] - r_stats["p_C_sum"])
    )
    signals["delta_uncertainty"] = r_stats["p_B_sum"] - m_stats["p_B_sum"]
    return signals


def collect_signal_arrays(rows):
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    signal_rows = [row_signals(row) for row in rows]
    names = list(signal_rows[0].keys()) if signal_rows else []
    signals = {
        name: clean_array([signal[name] for signal in signal_rows])
        for name in names
    }
    return labels, signals


def evaluate_signals(rows, popu=1, n_popu=1000):
    labels, signals = collect_signal_arrays(rows)
    results = []
    for name, scores in signals.items():
        eval_scores = scores
        eval_labels = labels
        if popu > 1:
            eval_scores, eval_labels = population_wrapper(scores, labels, popu, n_popu=n_popu)
        auc, tpr = mia_auc_tpr(eval_scores, eval_labels)
        results.append((name, auc, tpr, len(eval_scores)))
    return results


def attach_signals(rows):
    out = []
    for row in rows:
        item = dict(row)
        item["signals"] = row_signals(row)
        out.append(item)
    return out


def default_output_path(input_path):
    root, ext = os.path.splitext(input_path)
    return f"{root}_abc{ext or '.npy'}"


def main():
    parser = argparse.ArgumentParser(description="Evaluate A/B/C logprob membership signals")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--save_signals", action="store_true")
    parser.add_argument("--popu", type=int, default=1)
    parser.add_argument("--n_popu", type=int, default=1000)
    args = parser.parse_args()

    rows = np.load(args.input, allow_pickle=True)
    print(f"Loaded {len(rows)} rows from {args.input}")

    if args.save_signals:
        output_path = args.output or default_output_path(args.input)
        processed = attach_signals(rows)
        np.save(output_path, processed)
        print(f"Saved {len(processed)} signal rows to {output_path}")

    print("=== ABC Membership Signals ===")
    for name, auc, tpr, n in evaluate_signals(rows, popu=args.popu, n_popu=args.n_popu):
        print(f"{name:>22s}: AUROC={auc:.4f}, TPR@FPR=0.05={tpr:.4f}, n={n}")


if __name__ == "__main__":
    main()
