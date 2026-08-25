import argparse
import pickle

import numpy as np

from eval_tools import clean_array, mia_auc_tpr, population_wrapper


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(description="MIA with target minus calibration margin")
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--background", type=str, default=None)
    parser.add_argument("--ensemble_inputs", nargs="+", default=None)
    parser.add_argument("--ensemble_backgrounds", nargs="+", default=None)
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


def load_rows(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def margin_array(rows, key):
    return np.asarray([row[key] for row in rows], dtype=np.float64)


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


def quantile_map_apply(query_values, source_reference, target_reference):
    query_values = np.asarray(query_values, dtype=np.float64)
    source_reference = np.asarray(source_reference, dtype=np.float64)
    target_reference = np.asarray(target_reference, dtype=np.float64)
    finite_query = np.isfinite(query_values)
    finite_source = np.isfinite(source_reference)
    finite_target = np.isfinite(target_reference)
    mapped = np.zeros_like(query_values, dtype=np.float64)
    if finite_query.sum() == 0 or finite_source.sum() == 0 or finite_target.sum() == 0:
        return mapped

    src_sorted = np.sort(source_reference[finite_source])
    tgt_sorted = np.sort(target_reference[finite_target])
    query = query_values[finite_query]
    left = np.searchsorted(src_sorted, query, side="left")
    right = np.searchsorted(src_sorted, query, side="right")
    percentiles = 0.5 * (left + right) / max(len(src_sorted), 1)
    percentiles = np.clip(percentiles, 0.0, 1.0)

    if len(tgt_sorted) == 1:
        mapped[finite_query] = tgt_sorted[0]
        return mapped

    positions = percentiles * (len(tgt_sorted) - 1)
    mapped[finite_query] = np.interp(positions, np.arange(len(tgt_sorted)), tgt_sorted)
    return mapped


def quantile_map(source_values, target_values):
    return quantile_map_apply(source_values, source_values, target_values)


def build_scores(target_margin, cal_margin):
    rank_score = percentile_rank(target_margin) - percentile_rank(cal_margin)
    mapped_cal = quantile_map(cal_margin, target_margin)
    qmap_score = target_margin - mapped_cal

    return {
        "target_minus_cal": target_margin - cal_margin,
        "z_target_minus_cal": standardize(target_margin) - standardize(cal_margin),
        "rank_target_minus_cal": rank_score,
        "qmap_target_minus_cal": qmap_score,
        "qmap_rank_ens": 0.5 * standardize(qmap_score) + 0.5 * standardize(rank_score),
    }


def add_background_scores(scores, target_margin, cal_margin, background_path):
    bg_rows = load_rows(background_path)
    bg_target_margin = margin_array(bg_rows, "target_margin")
    bg_cal_margin = margin_array(bg_rows, "cal_margin")

    bg_mapped_cal = quantile_map_apply(cal_margin, bg_cal_margin, bg_target_margin)
    bg_qmap_score = target_margin - bg_mapped_cal
    bg_rank_score = percentile_rank(target_margin) - percentile_rank(bg_mapped_cal)

    scores["bg_qmap_target_minus_cal"] = bg_qmap_score
    scores["bg_qmap_rank_ens"] = 0.5 * standardize(bg_qmap_score) + 0.5 * standardize(bg_rank_score)
    return bg_rows


def validate_aligned_rows(reference_rows, rows, path):
    if len(reference_rows) != len(rows):
        raise ValueError(f"Ensemble file length mismatch for {path}: {len(rows)} != {len(reference_rows)}")
    ref_labels = np.asarray([row["label"] for row in reference_rows], dtype=np.int64)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    if not np.array_equal(ref_labels, labels):
        raise ValueError(f"Label order mismatch for ensemble file: {path}")
    ref_target = margin_array(reference_rows, "target_margin")
    target = margin_array(rows, "target_margin")
    if not np.allclose(ref_target, target, equal_nan=True):
        raise ValueError(f"Target margin mismatch for ensemble file: {path}")


def build_ensemble_scores(input_paths, background_paths=None):
    if background_paths is not None and len(background_paths) != len(input_paths):
        raise ValueError("--ensemble_backgrounds must have the same number of paths as --ensemble_inputs")

    all_rows = [load_rows(path) for path in input_paths]
    reference_rows = all_rows[0]
    for path, rows in zip(input_paths[1:], all_rows[1:]):
        validate_aligned_rows(reference_rows, rows, path)

    labels = np.asarray([row["label"] for row in reference_rows], dtype=np.int64)
    target_margin = margin_array(reference_rows, "target_margin")
    target_rank = percentile_rank(target_margin)

    rank_scores = []
    qmap_scores = []
    model_names = []

    for idx, (path, rows) in enumerate(zip(input_paths, all_rows)):
        cal_margin = margin_array(rows, "cal_margin")
        model_names.append(rows[0].get("cal_model", path) if rows else path)

        if background_paths is None:
            mapped_cal = quantile_map(cal_margin, target_margin)
            rank_score = target_rank - percentile_rank(cal_margin)
        else:
            bg_rows = load_rows(background_paths[idx])
            bg_target_margin = margin_array(bg_rows, "target_margin")
            bg_cal_margin = margin_array(bg_rows, "cal_margin")
            mapped_cal = quantile_map_apply(cal_margin, bg_cal_margin, bg_target_margin)
            rank_score = target_rank - percentile_rank(mapped_cal)

        qmap_scores.append(target_margin - mapped_cal)
        rank_scores.append(rank_score)

    rank_matrix = np.vstack(rank_scores)
    qmap_matrix = np.vstack(qmap_scores)
    ens_rank = rank_matrix.mean(axis=0)
    ens_qmap = qmap_matrix.mean(axis=0)
    ens_qmap_rank = 0.5 * standardize(ens_qmap) + 0.5 * standardize(ens_rank)

    return labels, model_names, {
        "ens_rank_target_minus_cal": ens_rank,
        "ens_qmap_target_minus_cal": ens_qmap,
        "ens_qmap_rank_ens": ens_qmap_rank,
    }


def run_single(args):
    if args.input is None:
        raise ValueError("--input is required unless --ensemble_inputs is provided")

    rows = load_rows(args.input)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    cal_model = rows[0].get("cal_model", "unknown") if rows else "unknown"

    target_margin = margin_array(rows, "target_margin")
    cal_margin = margin_array(rows, "cal_margin")
    scores = build_scores(target_margin, cal_margin)
    bg_rows = None
    if args.background:
        bg_rows = add_background_scores(scores, target_margin, cal_margin, args.background)

    print(f"Loaded {len(rows)} rows from {args.input}")
    if args.background:
        print(f"background: {args.background} n={len(bg_rows)}")
    print(f"cal_model: {cal_model}")
    print(f"member: {int(labels.sum())} nonmember: {int((labels == 0).sum())}")
    print("=== Calibration MIA ===")
    for name, score in scores.items():
        evaluate_score(name, score, labels, popu=args.popu, n_popu=args.n_popu)


def run_ensemble(args):
    labels, model_names, scores = build_ensemble_scores(args.ensemble_inputs, args.ensemble_backgrounds)
    print(f"Loaded ensemble with {len(args.ensemble_inputs)} calibration models")
    for path, model_name in zip(args.ensemble_inputs, model_names):
        print(f"  {model_name}: {path}")
    if args.ensemble_backgrounds:
        print("backgrounds:")
        for path in args.ensemble_backgrounds:
            print(f"  {path}")
    print(f"member: {int(labels.sum())} nonmember: {int((labels == 0).sum())}")
    print("=== Calibration Ensemble MIA ===")
    for name, score in scores.items():
        evaluate_score(name, score, labels, popu=args.popu, n_popu=args.n_popu)


def main():
    args = parse_args()
    if args.ensemble_inputs:
        run_ensemble(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
