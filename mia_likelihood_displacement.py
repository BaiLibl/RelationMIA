import argparse
import pickle

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import ExtraTreeClassifier

from eval_tools import clean_array, mia_auc_tpr


FEATURE_NAMES = [
    "target_minus_ref_margin",
    "chosen_displacement",
    "rejected_displacement",
    "chosen_len",
    "rejected_len",
]


CLASSIFIER_BUILDERS = {
    "lr": lambda args: make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=args.c,
            max_iter=args.max_iter,
            class_weight="balanced" if args.class_weight else None,
            random_state=args.seed,
        ),
    ),
    "rt": lambda args: ExtraTreeClassifier(
        max_depth=None if args.max_depth <= 0 else args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced" if args.class_weight else None,
        random_state=args.seed,
    ),
    "ets": lambda args: ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=None if args.max_depth <= 0 else args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced" if args.class_weight else None,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    ),
    "rf": lambda args: RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=None if args.max_depth <= 0 else args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced" if args.class_weight else None,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Likelihood displacement feature MIA")
    parser.add_argument("--input", type=str, required=True, help="cal_score pkl file")
    parser.add_argument("--clf", type=str, default="lr", choices=CLASSIFIER_BUILDERS.keys())
    parser.add_argument("--cv", type=int, default=5, help="stratified CV folds; use 0 for one holdout split")
    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--min_samples_leaf", type=int, default=10)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--class_weight", action="store_true")
    return parser.parse_args()


def load_rows(path):
    with open(path, "rb") as handle:
        rows = pickle.load(handle)
    if not rows:
        raise ValueError(f"empty input: {path}")
    return rows


def field_array(rows, key):
    return np.asarray([row[key] for row in rows], dtype=np.float64)


def build_features(rows):
    target_chosen = field_array(rows, "target_chosen_logp")
    target_rejected = field_array(rows, "target_rejected_logp")
    ref_chosen = field_array(rows, "cal_chosen_logp")
    ref_rejected = field_array(rows, "cal_rejected_logp")
    chosen_len = field_array(rows, "chosen_cal_tokens")
    rejected_len = field_array(rows, "rejected_cal_tokens")

    target_margin = target_chosen - target_rejected
    ref_margin = ref_chosen - ref_rejected
    chosen_displacement = target_chosen - ref_chosen
    rejected_displacement = target_rejected - ref_rejected

    return np.column_stack([
        target_margin - ref_margin,
        chosen_displacement,
        rejected_displacement,
        chosen_len,
        rejected_len,
    ])


def clean_features(x):
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    col_finite = np.isfinite(x)
    col_defaults = np.zeros(x.shape[1], dtype=np.float64)
    for idx in range(x.shape[1]):
        if col_finite[:, idx].any():
            col_defaults[idx] = np.nanmedian(x[col_finite[:, idx], idx])
    out = x.copy()
    row_idx, col_idx = np.where(~finite)
    out[row_idx, col_idx] = col_defaults[col_idx]
    return out


def score_classifier(model, x_train, y_train, x_test):
    model.fit(x_train, y_train)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test)
    return model.predict(x_test)


def evaluate_scores(scores, labels):
    scores = clean_array(scores)
    auc, tpr = mia_auc_tpr(scores, labels)
    return auc, tpr


def run_holdout(args, x, y):
    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    model = CLASSIFIER_BUILDERS[args.clf](args)
    scores = score_classifier(model, x[train_idx], y[train_idx], x[test_idx])
    auc, tpr = evaluate_scores(scores, y[test_idx])
    return [(auc, tpr, len(test_idx))]


def run_cv(args, x, y):
    splitter = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)
    rows = []
    for train_idx, test_idx in splitter.split(x, y):
        model = CLASSIFIER_BUILDERS[args.clf](args)
        scores = score_classifier(model, x[train_idx], y[train_idx], x[test_idx])
        auc, tpr = evaluate_scores(scores, y[test_idx])
        rows.append((auc, tpr, len(test_idx)))
    return rows


def print_feature_weights(args, x, y):
    if args.clf != "lr":
        return
    model = CLASSIFIER_BUILDERS[args.clf](args)
    model.fit(x, y)
    estimator = model[-1]
    coefs = estimator.coef_.reshape(-1)
    print("coefficients:")
    for name, coef in zip(FEATURE_NAMES, coefs):
        print(f"  {name}: {coef:.6f}")
    print(f"  intercept: {float(estimator.intercept_[0]):.6f}")


def main():
    args = parse_args()
    rows = load_rows(args.input)
    x = clean_features(build_features(rows))
    y = field_array(rows, "label").astype(np.int64)

    if args.cv > 1:
        results = run_cv(args, x, y)
    else:
        results = run_holdout(args, x, y)

    aucs = np.asarray([row[0] for row in results], dtype=np.float64)
    tprs = np.asarray([row[1] for row in results], dtype=np.float64)
    ns = np.asarray([row[2] for row in results], dtype=np.int64)

    print(f"Loaded {len(rows)} rows from {args.input}")
    print(f"clf: {args.clf}")
    print(f"member: {int(y.sum())} nonmember: {int((y == 0).sum())}")
    print("features:", ", ".join(FEATURE_NAMES))
    print("=== Likelihood Displacement Feature MIA ===")
    for idx, (auc, tpr, n) in enumerate(results, start=1):
        print(f"fold={idx:<2d}: AUROC={auc:.4f}, TPR@FPR=0.05={tpr:.4f}, n={n}")
    print(f"mean : AUROC={aucs.mean():.4f}, TPR@FPR=0.05={tprs.mean():.4f}, n={int(ns.sum())}")
    print(f"std  : AUROC={aucs.std():.4f}, TPR@FPR=0.05={tprs.std():.4f}")
    print_feature_weights(args, x, y)


if __name__ == "__main__":
    main()
