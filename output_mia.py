import numpy as np

from _args import parse_args
from eval_tools import clean_array, mia_auc_tpr, population_wrapper, premia_scores


def default_input(args):
    return f"premia_{args.sft_data}_{args.dpo_data}_{args.model_name}.npy"


def load_cached_outputs(path):
    data = np.load(path, allow_pickle=True)
    arrays = {
        "m_chosen": clean_array([row["m_chosen"] for row in data]),
        "m_rejected": clean_array([row["m_rejected"] for row in data]),
        "r_chosen": clean_array([row["r_chosen"] for row in data]),
        "r_rejected": clean_array([row["r_rejected"] for row in data]),
        "labels": np.asarray([row["label"] for row in data], dtype=np.int64),
    }
    return arrays


def evaluate_scores(scores, labels, popu=1, n_popu=1000):
    rows = []
    for name, score in scores.items():
        eval_score = score
        eval_labels = labels
        if popu > 1:
            eval_score, eval_labels = population_wrapper(score, labels, popu, n_popu=n_popu)
        auc, tpr = mia_auc_tpr(eval_score, eval_labels)
        rows.append((name, auc, tpr, len(eval_score)))
    return rows


def main():
    args = parse_args()
    input_path = args.input or default_input(args)
    arrays = load_cached_outputs(input_path)
    scores = premia_scores(
        arrays["m_chosen"],
        arrays["m_rejected"],
        arrays["r_chosen"],
        arrays["r_rejected"],
    )

    print(f"=== PreMIA Baseline: {input_path} popu={args.popu} ===")
    for name, auc, tpr, n in evaluate_scores(scores, arrays["labels"], args.popu, args.n_popu):
        print(f"{name:>8s}: AUROC={auc:.4f}, TPR@FPR=0.05={tpr:.4f}, n={n}")


if __name__ == "__main__":
    main()
