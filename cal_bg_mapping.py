import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODEL_BUILDERS = {
    "linear": lambda args: make_pipeline(StandardScaler(), LinearRegression()),
    "ridge": lambda args: make_pipeline(StandardScaler(), Ridge(alpha=args.alpha)),
    "huber": lambda args: make_pipeline(StandardScaler(), HuberRegressor(alpha=args.alpha, max_iter=args.max_iter)),
    "gbr": lambda args: GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        random_state=args.seed,
    ),
    "hgb": lambda args: HistGradientBoostingRegressor(
        max_iter=args.n_estimators,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.alpha,
        random_state=args.seed,
    ),
    "rf": lambda args: RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=None if args.max_depth <= 0 else args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    ),
}


FEATURE_NAMES = [
    "cal_chosen_logp",
    "cal_rejected_logp",
    "cal_margin",
    "chosen_cal_tokens",
    "rejected_cal_tokens",
    "cal_token_gap",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train output-space mapping from calibration outputs to target margin")
    parser.add_argument("--background", type=str, required=True, help="cal_bg pkl used to train mapping")
    parser.add_argument("--output", type=str, default=None, help="path to save trained mapping model")
    parser.add_argument("--model", type=str, default="ridge", choices=MODEL_BUILDERS.keys())
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--max_leaf_nodes", type=int, default=31)
    parser.add_argument("--min_samples_leaf", type=int, default=5)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
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


def finite_xy(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if finite.sum() == 0:
        raise ValueError("no finite training rows")
    return x[finite], y[finite]


def print_linear_coefficients(model):
    estimator = model[-1] if hasattr(model, "steps") else model
    if not hasattr(estimator, "coef_"):
        return
    coefs = np.asarray(estimator.coef_, dtype=np.float64).reshape(-1)
    print("coefficients:")
    for name, coef in zip(FEATURE_NAMES, coefs):
        print(f"  {name}: {coef:.6f}")
    if hasattr(estimator, "intercept_"):
        print(f"  intercept: {float(estimator.intercept_):.6f}")


def default_output_path(args):
    Path("mapping").mkdir(parents=True, exist_ok=True)
    bg_name = Path(args.background).stem
    return os.path.join("mapping", f"{bg_name}_{args.model}_mapping.pkl")


def main():
    args = parse_args()
    bg_rows = load_rows(args.background)

    train_x, train_y = finite_xy(build_features(bg_rows), target_margin(bg_rows))
    model = MODEL_BUILDERS[args.model](args)
    model.fit(train_x, train_y)

    save_path = args.output or default_output_path(args)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as handle:
        pickle.dump(
            {
                "model": model,
                "model_type": args.model,
                "feature_names": FEATURE_NAMES,
                "background": args.background,
                "n_train": int(len(train_y)),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print(f"Loaded {len(bg_rows)} background rows from {args.background}")
    print(f"finite training rows: {len(train_y)}")
    print(f"model: {args.model}")
    print("features:", ", ".join(FEATURE_NAMES))
    print_linear_coefficients(model)
    print(f"Saved mapping model to {save_path}")


if __name__ == "__main__":
    main()
