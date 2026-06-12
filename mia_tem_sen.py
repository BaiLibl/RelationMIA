import argparse
import csv
import pickle
from pathlib import Path

import numpy as np

from eval_tools import clean_array, mia_auc_tpr, population_wrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Temperature-sensitive token MIA")
    parser.add_argument("--inputs", type=str, default=None, help="comma-separated token pkl files from query_token.py")
    parser.add_argument("--input_dir", type=str, default=None, help="directory containing token pkl files")
    parser.add_argument("--pattern", type=str, default="*.pkl", help="glob pattern used with --input_dir")
    parser.add_argument("--ks", type=str, default="5,10,20", help="comma-separated top-std token percentages")
    parser.add_argument("--stats", type=str, default="mean", help="comma-separated aggregators: mean,sum")
    parser.add_argument("--fpr", type=float, default=0.05)
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
        raise ValueError("expected at least one numeric value")
    return values


def parse_str_list(text):
    values = [x.strip() for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("expected at least one value")
    return values


def input_paths(args):
    paths = []
    if args.inputs:
        paths.extend(Path(x.strip()) for x in args.inputs.split(",") if x.strip())
    if args.input_dir:
        paths.extend(sorted(Path(args.input_dir).glob(args.pattern)))
    if not paths:
        raise ValueError("Please provide --inputs or --input_dir")
    return sorted(dict.fromkeys(paths), key=lambda x: str(x))


def value_at(mapping, temp):
    if temp in mapping:
        return mapping[temp]
    key = float(temp)
    if key in mapping:
        return mapping[key]
    text_key = str(float(temp))
    if text_key in mapping:
        return mapping[text_key]
    short_key = str(temp)
    if short_key in mapping:
        return mapping[short_key]
    raise KeyError(f"Temperature {temp} not found. keys={list(mapping.keys())[:5]}")


def load_temperature_files(paths):
    merged = None
    labels = None
    temps = []

    for path in paths:
        with open(path, "rb") as handle:
            rows = pickle.load(handle)
        if not rows:
            print(f"[skip] empty file: {path}")
            continue

        file_temps = sorted(float(k) for k in rows[0]["chosen_token_logps"].keys())
        if len(file_temps) != 1:
            raise ValueError(f"{path} should contain exactly one temperature, got {file_temps}")
        temp = file_temps[0]

        file_labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
        if merged is None:
            labels = file_labels
            merged = [
                {
                    "chosen": {},
                    "rejected": {},
                }
                for _ in rows
            ]
        else:
            if len(rows) != len(merged):
                raise ValueError(f"sample count mismatch in {path}: {len(rows)} vs {len(merged)}")
            if not np.array_equal(labels, file_labels):
                raise ValueError(f"label order mismatch in {path}")

        for idx, row in enumerate(rows):
            merged[idx]["chosen"][temp] = np.asarray(value_at(row["chosen_token_logps"], temp), dtype=np.float64)
            merged[idx]["rejected"][temp] = np.asarray(value_at(row["rejected_token_logps"], temp), dtype=np.float64)
        temps.append(temp)
        print(f"Loaded T={temp:g} from {path}")

    if merged is None:
        raise ValueError("No non-empty input files found")

    temps = sorted(temps)
    return merged, labels, np.asarray(temps, dtype=np.float64)


def stack_side(sample, side, temps):
    arrays = [np.asarray(sample[side][float(temp)], dtype=np.float64) for temp in temps]
    min_len = min(len(x) for x in arrays)
    if min_len == 0:
        return np.empty((len(temps), 0), dtype=np.float64)
    arrays = [x[:min_len] for x in arrays]
    return np.stack(arrays, axis=0)


def select_top_std_indices(values_by_t, k_percent):
    if values_by_t.shape[1] == 0:
        return np.asarray([], dtype=np.int64)
    std = np.nanstd(values_by_t, axis=0)
    finite = np.isfinite(std)
    if not finite.any():
        return np.asarray([], dtype=np.int64)
    valid_idx = np.where(finite)[0]
    k = max(1, int(np.ceil(len(valid_idx) * float(k_percent) / 100.0)))
    order = valid_idx[np.argsort(std[valid_idx])]
    return order[-k:]


def aggregate_selected(values_by_t, indices, stat):
    if len(indices) == 0:
        return np.full(values_by_t.shape[0], np.nan, dtype=np.float64)
    selected = values_by_t[:, indices]
    if stat == "mean":
        return np.nanmean(selected, axis=1)
    if stat == "sum":
        return np.nansum(selected, axis=1)
    raise ValueError(f"unknown stat: {stat}")


def nearest_temp_index(temps, target):
    return int(np.argmin(np.abs(np.asarray(temps, dtype=np.float64) - float(target))))


def slope_feature(temps, curves):
    x = np.asarray(temps, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom == 0:
        return np.zeros(curves.shape[0], dtype=np.float64)
    y = curves - np.nanmean(curves, axis=1, keepdims=True)
    return np.nansum(y * x[None, :], axis=1) / denom


def build_sensitive_margin_curves(samples, temps, k_percent, stat):
    curves = []
    chosen_selected_counts = []
    rejected_selected_counts = []

    for sample in samples:
        chosen_t = stack_side(sample, "chosen", temps)
        rejected_t = stack_side(sample, "rejected", temps)
        chosen_idx = select_top_std_indices(chosen_t, k_percent)
        rejected_idx = select_top_std_indices(rejected_t, k_percent)

        chosen_curve = aggregate_selected(chosen_t, chosen_idx, stat)
        rejected_curve = aggregate_selected(rejected_t, rejected_idx, stat)
        curves.append(chosen_curve - rejected_curve)
        chosen_selected_counts.append(len(chosen_idx))
        rejected_selected_counts.append(len(rejected_idx))

    return (
        np.asarray(curves, dtype=np.float64),
        np.asarray(chosen_selected_counts, dtype=np.int64),
        np.asarray(rejected_selected_counts, dtype=np.int64),
    )


def curve_features(curves, temps):
    idx_low = 0
    idx_mid = nearest_temp_index(temps, 1.0)
    idx_high = len(temps) - 1
    low_mask = temps <= 0.7
    high_mask = temps >= 1.3
    if not low_mask.any():
        low_mask[idx_low] = True
    if not high_mask.any():
        high_mask[idx_high] = True

    centered = curves - curves[:, [idx_mid]]
    return {
        "slope": slope_feature(temps, curves),
        "centered_low": centered[:, idx_low],
        "centered_high": centered[:, idx_high],
    }


def eval_score(scores, labels, fpr=0.05, popu=1, n_popu=1000):
    scores = clean_array(scores)
    eval_scores = scores
    eval_labels = labels
    if popu > 1:
        eval_scores, eval_labels = population_wrapper(scores, labels, popu, n_popu=n_popu)
    auc, tpr = mia_auc_tpr(eval_scores, eval_labels, fpr_threshold=fpr)
    return auc, tpr, len(eval_scores)


def add_result(results, name, scores, labels, args):
    auc, tpr, n = eval_score(scores, labels, fpr=args.fpr, popu=args.popu, n_popu=args.n_popu)
    results.append({
        "score": name,
        "auroc": float(auc),
        f"tpr_at_fpr_{args.fpr:g}": float(tpr),
        "n": int(n),
    })


def print_results(results, fpr):
    tpr_key = f"tpr_at_fpr_{fpr:g}"
    print("=== Temperature-Sensitive Token MIA ===")
    for row in sorted(results, key=lambda x: x["score"]):
        print(
            f"{row['score']:>36s}: "
            f"AUROC={row['auroc']:.4f}, "
            f"TPR@FPR={fpr:g}={row[tpr_key]:.4f}, n={row['n']}"
        )
    best = max(results, key=lambda x: x["auroc"])
    print(
        f"best: {best['score']} AUROC={best['auroc']:.4f}, "
        f"TPR@FPR={fpr:g}={best[tpr_key]:.4f}"
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
    ks = parse_float_list(args.ks)
    stats = parse_str_list(args.stats)
    for stat in stats:
        if stat not in {"mean", "sum"}:
            raise ValueError("--stats only supports mean,sum")

    paths = input_paths(args)
    samples, labels, temps = load_temperature_files(paths)
    # print(f"Merged {len(samples)} samples from {len(paths)} files")
    # print("temps:", ",".join(f"{x:g}" for x in temps))
    # print("member:", int((labels == 1).sum()), "nonmember:", int((labels == 0).sum()))

    results = []
    for k in ks:
        for stat in stats:
            curves, chosen_counts, rejected_counts = build_sensitive_margin_curves(samples, temps, k, stat)
            # print(
            #     f"TopStd{k:g}% {stat}: "
            #     f"mean selected chosen={chosen_counts.mean():.2f}, rejected={rejected_counts.mean():.2f}"
            # )

            for temp_idx, temp in enumerate(temps):
                add_result(
                    results,
                    f"topstd{k:g}_{stat}_T{temp:g}",
                    curves[:, temp_idx],
                    labels,
                    args,
                )

            for feat_name, scores in curve_features(curves, temps).items():
                add_result(results, f"topstd{k:g}_{stat}_{feat_name}", scores, labels, args)

    print_results(results, args.fpr)
    if args.output_csv:
        save_csv(results, args.output_csv)


if __name__ == "__main__":
    main()
