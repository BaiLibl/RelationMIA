"""Plot member/nonmember likelihood displacement from resp_store CSV."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUPS = ((0, "Non-member", "#3575B5"), (1, "Member", "#D76A32"))


def load_scores(path):
    records = []
    skipped = 0
    k = None
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"label", "m_chosen", "m_rejected", "r_chosen", "r_rejected",
                    "m_response", "r_response"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing fields: {required - set(reader.fieldnames or [])}")
        for line, row in enumerate(reader, 2):
            label = float(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Line {line}: label must be 0 or 1")
            target = np.asarray(json.loads(row["m_response"]), dtype=float)
            reference = np.asarray(json.loads(row["r_response"]), dtype=float)
            if target.ndim != 2 or target.shape[0] != 2 or target.shape[1] == 0 or target.shape != reference.shape:
                raise ValueError(f"Line {line}: response fields must have matching shape [2,K]")
            if k is None:
                k = target.shape[1]
            if target.shape[1] != k:
                raise ValueError(f"Line {line}: inconsistent K")
            dw = float(row["m_chosen"]) - float(row["r_chosen"])
            dl = float(row["m_rejected"]) - float(row["r_rejected"])
            dz = target - reference
            if not (np.isfinite([dw, dl]).all() and np.isfinite(target).all()
                    and np.isfinite(reference).all()):
                skipped += 1
                continue
            margin = dw - dl
            # Per-sample averages retain generation source and avoid counting
            # correlated responses as independent samples.
            records.append({
                "label": int(label), "d_chosen": dw, "d_rejected": dl,
                "margin_gap": margin,
                "d_z_target_mean": dz[0].mean(),
                "d_z_reference_mean": dz[1].mean(),
                "ld_pattern": float(margin > 0 and dw < 0),
                "joint_target_rate": float(margin > 0 and dw < 0) * np.mean(dz[0] > 0),
                "joint_reference_rate": float(margin > 0 and dw < 0) * np.mean(dz[1] > 0),
                "z_target_up_rate": np.mean(dz[0] > 0),
                "z_reference_up_rate": np.mean(dz[1] > 0),
            })
    if not records or {row["label"] for row in records} != {0, 1}:
        raise ValueError("Need finite records from both labels")
    print(f"Loaded {len(records)} samples, K={k}, skipped {skipped} nonfinite rows")
    return {key: np.asarray([row[key] for row in records]) for key in records[0]}


def distribution(ax, values, labels, title, bins):
    edges = np.histogram_bin_edges(values, bins=bins)
    for label, name, color in GROUPS:
        ax.hist(values[labels == label], bins=edges, density=True, histtype="step",
                linewidth=1.8, color=color, label=name)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set(title=title, xlabel="Log-probability difference", ylabel="Density")
    ax.legend(fontsize=8)


def save(fig, directory, name):
    fig.tight_layout()
    fig.savefig(directory / f"{name}.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="resp_uf_k2.csv")
    parser.add_argument("--outdir", default="plots/resp_displacement")
    parser.add_argument("--bins", type=int, default=40)
    args = parser.parse_args()
    if args.bins < 2:
        parser.error("bins must be at least 2")
    scores = load_scores(args.input)
    labels = scores["label"]
    directory = Path(args.outdir)
    directory.mkdir(parents=True, exist_ok=True)
    titles = {
        "d_chosen": "Chosen: target minus reference",
        "d_rejected": "Rejected: target minus reference",
        "margin_gap": "PreMIA margin gap",
        "d_z_target_mean": "Mean change: target-generated responses",
        "d_z_reference_mean": "Mean change: reference-generated responses",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, key in zip(axes.flat, titles):
        distribution(ax, scores[key], labels, titles[key], args.bins)
    ax = axes.flat[-1]
    for label, name, color in GROUPS:
        mask = labels == label
        ax.scatter(scores["d_chosen"][mask], scores["margin_gap"][mask],
                   s=10, alpha=0.25, color=color, label=name)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set(title="LD pattern: chosen < 0, margin gap > 0",
           xlabel="Chosen displacement", ylabel="Margin gap")
    ax.legend(fontsize=8)
    save(fig, directory, "ld_distributions")

    rate_keys = ("ld_pattern", "z_target_up_rate", "z_reference_up_rate",
                 "joint_target_rate", "joint_reference_rate")
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = np.arange(len(rate_keys))
    for offset, (label, name, color) in zip((-0.18, 0.18), GROUPS):
        rates = [scores[key][labels == label].mean() for key in rate_keys]
        ax.bar(positions + offset, rates, width=0.36, color=color, label=name)
    ax.set_xticks(positions, ["Margin up &\nchosen down",
                            "Target z\nup fraction", "Reference z\nup fraction",
                            "LD & target z\nup fraction", "LD & reference z\nup fraction"])
    ax.set(ylim=(0, 1), ylabel="Sample-averaged fraction")
    ax.legend()
    save(fig, directory, "pattern_rates")

    print(f"Saved PNG/PDF plots to {directory}")
    print("Changes are sequence logprobs, not length-normalized. Co-occurrence does not prove mass transfer.")


if __name__ == "__main__":
    main()
