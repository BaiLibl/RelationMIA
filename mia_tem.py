import argparse
import csv
import pickle

import numpy as np

from eval_tools import clean_array, mia_auc_tpr, population_wrapper


DEFAULT_SCORE_FIELDS = "margin_T,avg_margin_T"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate temperature-wise MIA scores from tem_query.py outputs")
    parser.add_argument("--input", type=str, required=True, help="pkl file produced by tem_query.py")
    parser.add_argument(
        "--score_fields",
        type=str,
        default=DEFAULT_SCORE_FIELDS,
        help="comma-separated score dict fields to evaluate, e.g. margin_T,avg_margin_T,chosen_logps_T",
    )
    parser.add_argument("--fpr", type=float, default=0.05)
    parser.add_argument("--popu", type=int, default=1)
    parser.add_argument("--n_popu", type=int, default=1000)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument(
        "--curve_field",
        type=str,
        default="avg_margin_T",
        help="score dict field used for curve-level features",
    )
    parser.add_argument("--no_curve_features", action="store_true")
    return parser.parse_args()


def parse_score_fields(text):
    fields = [x.strip() for x in str(text).split(",") if x.strip()]
    if not fields:
        raise ValueError("--score_fields must contain at least one field")
    return fields


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
    raise KeyError(f"Temperature {temp} not found in mapping keys: {list(mapping.keys())[:5]}")


def collect_scores(rows, field, temps):
    if field not in rows[0]:
        return None
    return np.asarray([
        [value_at(row[field], float(temp)) for temp in temps]
        for row in rows
    ], dtype=np.float64)


def evaluate_by_temperature(values, labels, temps, score_name, fpr=0.05, popu=1, n_popu=1000):
    results = []
    for col, temp in enumerate(temps):
        scores = clean_array(values[:, col])
        eval_scores = scores
        eval_labels = labels
        if popu > 1:
            eval_scores, eval_labels = population_wrapper(scores, labels, popu, n_popu=n_popu)
        auc, tpr = mia_auc_tpr(eval_scores, eval_labels, fpr_threshold=fpr)
        results.append({
            "score": score_name,
            "temperature": float(temp),
            "auroc": float(auc),
            f"tpr_at_fpr_{fpr:g}": float(tpr),
            "n": int(len(eval_scores)),
        })
    return results


def nearest_temp_index(temps, target):
    return int(np.argmin(np.abs(np.asarray(temps, dtype=float) - float(target))))


def temperature_slope(temps, values):
    x = np.asarray(temps, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return np.zeros(y.shape[0], dtype=np.float64)
    centered_y = y - np.nanmean(y, axis=1, keepdims=True)
    return np.nansum(centered_y * x[None, :], axis=1) / denom


def build_curve_features(values, temps):
    values = np.asarray(values, dtype=np.float64)
    temps = np.asarray(temps, dtype=np.float64)

    idx_low = nearest_temp_index(temps, 0.3)
    idx_mid = nearest_temp_index(temps, 1.0)
    idx_high = nearest_temp_index(temps, 2.0)
    low_mask = temps <= 0.7
    high_mask = temps >= 1.3

    centered = values - values[:, [idx_mid]]
    features = {
        "curve_low_T": values[:, idx_low],
        "curve_mid_T": values[:, idx_mid],
        "curve_high_T": values[:, idx_high],
        "curve_sharp_low_mid": values[:, idx_low] - values[:, idx_mid],
        "curve_low_high_gap": np.nanmean(values[:, low_mask], axis=1) - np.nanmean(values[:, high_mask], axis=1),
        "curve_slope": temperature_slope(temps, values),
        # "curve_area_mean": np.nanmean(values, axis=1),
        # "curve_area_trapz": np.trapz(values, temps, axis=1) / (temps[-1] - temps[0]),
        # "curve_curvature_low_mid_high": values[:, idx_low] - 2 * values[:, idx_mid] + values[:, idx_high],
        # "curve_centered_low": centered[:, idx_low],
        # "curve_centered_high": centered[:, idx_high],
        # "curve_centered_range": np.nanmax(centered, axis=1) - np.nanmin(centered, axis=1),
        # "curve_abs_slope": np.abs(temperature_slope(temps, values)),
    }
    return features


def evaluate_scalar_scores(feature_scores, labels, fpr=0.05, popu=1, n_popu=1000):
    results = []
    for name, scores in feature_scores.items():
        scores = clean_array(scores)
        eval_scores = scores
        eval_labels = labels
        if popu > 1:
            eval_scores, eval_labels = population_wrapper(scores, labels, popu, n_popu=n_popu)
        auc, tpr = mia_auc_tpr(eval_scores, eval_labels, fpr_threshold=fpr)
        results.append({
            "score": name,
            "temperature": "curve",
            "auroc": float(auc),
            f"tpr_at_fpr_{fpr:g}": float(tpr),
            "n": int(len(eval_scores)),
        })
    return results


def print_results(results, fpr):
    if not results:
        print("No results to print.")
        return

    grouped = {}
    for row in results:
        grouped.setdefault(row["score"], []).append(row)

    tpr_key = f"tpr_at_fpr_{fpr:g}"
    for score_name, rows in grouped.items():
        # print(f"=== Temperature MIA: {score_name} ===")
        for row in rows:
            if row["temperature"] == "curve":
                prefix = "curve"
            else:
                prefix = f"T={row['temperature']:<4g}"
            print(
                f"{score_name} {prefix} "
                f"AUROC={row['auroc']:.4f}, "
                f"TPR@FPR={fpr:g}={row[tpr_key]:.4f}, "
                f"n={row['n']}"
            )
        best = max(rows, key=lambda x: x["auroc"])
        if best["temperature"] == "curve":
            best_temp = "curve"
        else:
            best_temp = f"T={best['temperature']:g}"
        # print(
        #     f"best {score_name}: {best_temp}, "
        #     f"AUROC={best['auroc']:.4f}, TPR@FPR={fpr:g}={best[tpr_key]:.4f}"
        # )


def save_csv(results, path):
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved CSV results to {path}")


def main():
    args = parse_args()
    score_fields = parse_score_fields(args.score_fields)

    with open(args.input, "rb") as handle:
        rows = pickle.load(handle)
    if not rows:
        raise ValueError("Input pkl contains no rows")

    temps = np.asarray(rows[0]["temps"], dtype=float)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)

    print(f"Loaded {len(rows)} samples from {args.input}")
    print("temps:", ",".join(f"{x:g}" for x in temps))
    print("member:", int((labels == 1).sum()), "nonmember:", int((labels == 0).sum()))

    all_results = []
    for field in score_fields:
        values = collect_scores(rows, field, temps)
        if values is None:
            print(f"[skip] field not found: {field}")
            continue
        all_results.extend(
            evaluate_by_temperature(
                values,
                labels,
                temps,
                field,
                fpr=args.fpr,
                popu=args.popu,
                n_popu=args.n_popu,
            )
        )

    if not args.no_curve_features:
        curve_values = collect_scores(rows, args.curve_field, temps)
        if curve_values is None:
            print(f"[skip] curve field not found: {args.curve_field}")
        else:
            curve_features = build_curve_features(curve_values, temps)
            prefixed = {f"{args.curve_field}:{name}": score for name, score in curve_features.items()}
            all_results.extend(
                evaluate_scalar_scores(
                    prefixed,
                    labels,
                    fpr=args.fpr,
                    popu=args.popu,
                    n_popu=args.n_popu,
                )
            )

    print_results(all_results, args.fpr)
    if args.output_csv:
        save_csv(all_results, args.output_csv)


if __name__ == "__main__":
    main()
