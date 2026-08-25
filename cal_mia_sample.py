import argparse
import pickle

import numpy as np

from eval_tools import clean_array, mia_auc_tpr, population_wrapper


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(
        description="MIA with separately aligned chosen/rejected sample logprobs"
    )
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
    print(f"{name:>32s}: AUROC={auc:.4f}, TPR@FPR=0.05={tpr:.4f}, n={len(eval_scores)}")


def load_rows(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def field_array(rows, key):
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


def component_arrays(rows):
    return {
        "target_chosen": field_array(rows, "target_chosen_logp"),
        "target_rejected": field_array(rows, "target_rejected_logp"),
        "cal_chosen": field_array(rows, "cal_chosen_logp"),
        "cal_rejected": field_array(rows, "cal_rejected_logp"),
    }


def build_scores(rows):
    arrays = component_arrays(rows)
    chosen_residual = arrays["target_chosen"] - quantile_map(
        arrays["cal_chosen"], arrays["target_chosen"]
    )
    rejected_residual = arrays["target_rejected"] - quantile_map(
        arrays["cal_rejected"], arrays["target_rejected"]
    )
    chosen_rank = percentile_rank(arrays["target_chosen"]) - percentile_rank(arrays["cal_chosen"])
    rejected_rank = percentile_rank(arrays["target_rejected"]) - percentile_rank(arrays["cal_rejected"])

    pair = chosen_residual - rejected_residual
    rank_pair = chosen_rank - rejected_rank

    return {
        "chosen_component": chosen_residual,
        "rejected_component": -rejected_residual,
        "component_pair": pair,
        "component_rank_pair": rank_pair,
        "component_rank_ens": 0.5 * standardize(pair) + 0.5 * standardize(rank_pair),
    }


def add_background_scores(scores, rows, background_path):
    bg_rows = load_rows(background_path)
    arrays = component_arrays(rows)
    bg = component_arrays(bg_rows)

    mapped_chosen = quantile_map_apply(
        arrays["cal_chosen"], bg["cal_chosen"], bg["target_chosen"]
    )
    mapped_rejected = quantile_map_apply(
        arrays["cal_rejected"], bg["cal_rejected"], bg["target_rejected"]
    )

    chosen_residual = arrays["target_chosen"] - mapped_chosen
    rejected_residual = arrays["target_rejected"] - mapped_rejected
    chosen_rank = percentile_rank(arrays["target_chosen"]) - percentile_rank(mapped_chosen)
    rejected_rank = percentile_rank(arrays["target_rejected"]) - percentile_rank(mapped_rejected)

    pair = chosen_residual - rejected_residual
    rank_pair = chosen_rank - rejected_rank

    scores["bg_chosen_component"] = chosen_residual
    scores["bg_rejected_component"] = -rejected_residual
    scores["bg_component_pair"] = pair
    scores["bg_component_rank_pair"] = rank_pair
    scores["bg_component_rank_ens"] = 0.5 * standardize(pair) + 0.5 * standardize(rank_pair)
    return bg_rows


def validate_aligned_rows(reference_rows, rows, path):
    if len(reference_rows) != len(rows):
        raise ValueError(f"Ensemble file length mismatch for {path}: {len(rows)} != {len(reference_rows)}")
    ref_labels = np.asarray([row["label"] for row in reference_rows], dtype=np.int64)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    if not np.array_equal(ref_labels, labels):
        raise ValueError(f"Label order mismatch for ensemble file: {path}")
    for key in ["target_chosen_logp", "target_rejected_logp"]:
        if not np.allclose(field_array(reference_rows, key), field_array(rows, key), equal_nan=True):
            raise ValueError(f"Target logprob mismatch for ensemble file: {path} key={key}")


def build_ensemble_scores(input_paths, background_paths=None):
    if background_paths is not None and len(background_paths) != len(input_paths):
        raise ValueError("--ensemble_backgrounds must have the same number of paths as --ensemble_inputs")

    all_rows = [load_rows(path) for path in input_paths]
    reference_rows = all_rows[0]
    for path, rows in zip(input_paths[1:], all_rows[1:]):
        validate_aligned_rows(reference_rows, rows, path)

    labels = np.asarray([row["label"] for row in reference_rows], dtype=np.int64)
    target = component_arrays(reference_rows)
    target_chosen_rank = percentile_rank(target["target_chosen"])
    target_rejected_rank = percentile_rank(target["target_rejected"])

    qmap_scores = []
    rank_scores = []
    model_names = []

    for idx, (path, rows) in enumerate(zip(input_paths, all_rows)):
        arrays = component_arrays(rows)
        model_names.append(rows[0].get("cal_model", path) if rows else path)

        if background_paths is None:
            mapped_chosen = quantile_map(arrays["cal_chosen"], arrays["target_chosen"])
            mapped_rejected = quantile_map(arrays["cal_rejected"], arrays["target_rejected"])
            chosen_rank = target_chosen_rank - percentile_rank(arrays["cal_chosen"])
            rejected_rank = target_rejected_rank - percentile_rank(arrays["cal_rejected"])
        else:
            bg = component_arrays(load_rows(background_paths[idx]))
            mapped_chosen = quantile_map_apply(
                arrays["cal_chosen"], bg["cal_chosen"], bg["target_chosen"]
            )
            mapped_rejected = quantile_map_apply(
                arrays["cal_rejected"], bg["cal_rejected"], bg["target_rejected"]
            )
            chosen_rank = target_chosen_rank - percentile_rank(mapped_chosen)
            rejected_rank = target_rejected_rank - percentile_rank(mapped_rejected)

        qmap_scores.append(
            (arrays["target_chosen"] - mapped_chosen)
            - (arrays["target_rejected"] - mapped_rejected)
        )
        rank_scores.append(chosen_rank - rejected_rank)

    ens_qmap = np.vstack(qmap_scores).mean(axis=0)
    ens_rank = np.vstack(rank_scores).mean(axis=0)

    return labels, model_names, {
        "ens_component_pair": ens_qmap,
        "ens_component_rank_pair": ens_rank,
        "ens_component_rank_ens": 0.5 * standardize(ens_qmap) + 0.5 * standardize(ens_rank),
    }


def run_single(args):
    if args.input is None:
        raise ValueError("--input is required unless --ensemble_inputs is provided")

    rows = load_rows(args.input)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    cal_model = rows[0].get("cal_model", "unknown") if rows else "unknown"
    scores = build_scores(rows)
    bg_rows = None
    if args.background:
        bg_rows = add_background_scores(scores, rows, args.background)

    print(f"Loaded {len(rows)} rows from {args.input}")
    if args.background:
        print(f"background: {args.background} n={len(bg_rows)}")
    print(f"cal_model: {cal_model}")
    print(f"member: {int(labels.sum())} nonmember: {int((labels == 0).sum())}")
    print("=== Sample-Component Calibration MIA ===")
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
    print("=== Sample-Component Calibration Ensemble MIA ===")
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
