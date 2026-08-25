import argparse
import pickle

import numpy as np

from eval_tools import clean_array, mia_auc_tpr, population_wrapper


FEATURE_NAMES = [
    "cal_chosen_logp",
    "cal_rejected_logp",
    "cal_margin",
    "chosen_cal_tokens",
    "rejected_cal_tokens",
    "cal_token_gap",
]


def parse_args():
    parser = argparse.ArgumentParser(description="MIA with a trained output-space mapping model")
    parser.add_argument("--input", type=str, required=True, help="cal_score pkl with member/nonmember rows")
    parser.add_argument("--mapping", type=str, required=True, help="mapping pkl produced by cal_bg_mapping.py")
    parser.add_argument("--popu", type=int, default=1)
    parser.add_argument("--n_popu", type=int, default=1000)
    return parser.parse_args()


def load_rows(path):
    with open(path, "rb") as handle:
        rows = pickle.load(handle)
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def load_mapping(path):
    with open(path, "rb") as handle:
        mapping = pickle.load(handle)
    if "model" not in mapping:
        raise ValueError(f"mapping file does not contain a trained model: {path}")
    return mapping


def field_array(rows, key):
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def build_features(rows):
    cal_chosen = field_array(rows, "cal_chosen_logp")
    cal_rejected = field_array(rows, "cal_rejected_logp")
    chosen_tokens = field_array(rows, "chosen_cal_tokens")
    rejected_tokens = field_array(rows, "rejected_cal_tokens")
    return np.column_stack([
        cal_chosen,
        cal_rejected,
        cal_chosen - cal_rejected,
        chosen_tokens,
        rejected_tokens,
        chosen_tokens - rejected_tokens,
    ])


def target_margin(rows):
    return field_array(rows, "target_margin")


def evaluate_score(name, scores, labels, popu=1, n_popu=1000):
    scores = clean_array(scores)
    eval_scores = scores
    eval_labels = labels
    if popu > 1:
        eval_scores, eval_labels = population_wrapper(scores, labels, popu, n_popu=n_popu)
    auc, tpr = mia_auc_tpr(eval_scores, eval_labels)
    print(f"{name:>24s}: AUROC={auc:.4f}, TPR@FPR=0.05={tpr:.4f}, n={len(eval_scores)}")


def main():
    args = parse_args()
    rows = load_rows(args.input)
    mapping = load_mapping(args.mapping)

    expected_features = mapping.get("feature_names", FEATURE_NAMES)
    if list(expected_features) != FEATURE_NAMES:
        raise ValueError(
            f"feature mismatch: mapping expects {expected_features}, "
            f"but this script builds {FEATURE_NAMES}"
        )

    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    x = build_features(rows)
    raw_margin = target_margin(rows)
    mapped_margin = mapping["model"].predict(x)
    mapping_residual = raw_margin - mapped_margin

    print(f"Loaded {len(rows)} eval rows from {args.input}")
    print(f"Loaded mapping model from {args.mapping}")
    print(f"mapping model: {mapping.get('model_type', 'unknown')}")
    print(f"mapping background: {mapping.get('background', 'unknown')}")
    print(f"member: {int(labels.sum())} nonmember: {int((labels == 0).sum())}")
    print("features:", ", ".join(FEATURE_NAMES))
    print("=== Output-Space Mapping Calibration MIA ===")
    evaluate_score("raw_target_margin", raw_margin, labels, args.popu, args.n_popu)
    evaluate_score("mapping_residual", mapping_residual, labels, args.popu, args.n_popu)


if __name__ == "__main__":
    main()
