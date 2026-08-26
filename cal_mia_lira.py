import argparse
import pickle

import numpy as np

from eval_tools import clean_array, mia_auc_tpr, population_wrapper


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(description="Offline LiRA-style MIA with multiple OUT calibration models")
    parser.add_argument("--inputs", nargs="+", required=True, help="aligned cal_score pkl files from different cal models")
    parser.add_argument("--popu", type=int, default=1)
    parser.add_argument("--n_popu", type=int, default=1000)
    parser.add_argument("--eps", type=float, default=EPS)
    return parser.parse_args()


def load_rows(path):
    with open(path, "rb") as handle:
        rows = pickle.load(handle)
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def field_array(rows, key):
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def standardize(values, eps=EPS):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float64)
    if finite.sum() == 0:
        return out
    out[finite] = (values[finite] - values[finite].mean()) / (values[finite].std() + eps)
    return out


def percentile_rank(values):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    out = np.zeros_like(values, dtype=np.float64)
    if finite.sum() == 0:
        return out
    sorted_values = np.sort(values[finite])
    left = np.searchsorted(sorted_values, values[finite], side="left")
    right = np.searchsorted(sorted_values, values[finite], side="right")
    out[finite] = 0.5 * (left + right) / max(len(sorted_values), 1)
    return out


def evaluate_score(name, scores, labels, popu=1, n_popu=1000):
    scores = clean_array(scores)
    eval_scores = scores
    eval_labels = labels
    if popu > 1:
        eval_scores, eval_labels = population_wrapper(scores, labels, popu, n_popu=n_popu)
    auc, tpr = mia_auc_tpr(eval_scores, eval_labels)
    print(f"{name:>24s}: AUROC={auc:.4f}, TPR@FPR=0.05={tpr:.4f}, n={len(eval_scores)}")


def validate_aligned_rows(reference_rows, rows, path):
    if len(reference_rows) != len(rows):
        raise ValueError(f"file length mismatch for {path}: {len(rows)} != {len(reference_rows)}")
    ref_labels = field_array(reference_rows, "label").astype(np.int64)
    labels = field_array(rows, "label").astype(np.int64)
    if not np.array_equal(ref_labels, labels):
        raise ValueError(f"label order mismatch for {path}")
    ref_target = field_array(reference_rows, "target_margin")
    target = field_array(rows, "target_margin")
    if not np.allclose(ref_target, target, equal_nan=True):
        raise ValueError(f"target_margin mismatch for {path}")


def lira_scores(target_margin, out_matrix, eps=EPS):
    out_mean = np.nanmean(out_matrix, axis=0)
    out_std = np.nanstd(out_matrix, axis=0) + eps
    offline_z = (target_margin - out_mean) / out_std
    offline_nll = 0.5 * offline_z * offline_z + np.log(out_std)
    return {
        "out_mean_centered": target_margin - out_mean,
        "offline_lira_z": offline_z,
        "offline_lira_nll": offline_nll,
    }


def main():
    args = parse_args()
    all_rows = [load_rows(path) for path in args.inputs]
    reference_rows = all_rows[0]
    for path, rows in zip(args.inputs[1:], all_rows[1:]):
        validate_aligned_rows(reference_rows, rows, path)

    labels = field_array(reference_rows, "label").astype(np.int64)
    target_margin = field_array(reference_rows, "target_margin")

    cal_margins = np.vstack([field_array(rows, "cal_margin") for rows in all_rows])
    z_cal_margins = np.vstack([standardize(row, eps=args.eps) for row in cal_margins])

    z_target_margin = standardize(target_margin, eps=args.eps)
    rank_cal_margins = np.vstack([percentile_rank(row) for row in cal_margins])
    rank_target_margin = percentile_rank(target_margin)

    scores = {}
    scores.update({f"raw_{name}": score for name, score in lira_scores(target_margin, cal_margins, args.eps).items()})
    scores.update({
        f"zscore_{name}": score
        for name, score in lira_scores(z_target_margin, z_cal_margins, args.eps).items()
    })
    scores.update({
        f"rank_{name}": score
        for name, score in lira_scores(rank_target_margin, rank_cal_margins, args.eps).items()
    })

    print(f"Loaded {len(args.inputs)} OUT calibration models")
    for path, rows in zip(args.inputs, all_rows):
        print(f"  {rows[0].get('cal_model', path)}: {path}")
    print(f"member: {int(labels.sum())} nonmember: {int((labels == 0).sum())}")
    print("=== Offline LiRA Calibration MIA ===")
    for name, score in scores.items():
        evaluate_score(name, score, labels, popu=args.popu, n_popu=args.n_popu)


if __name__ == "__main__":
    main()
