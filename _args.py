import argparse

from common import MODEL_DICT
from explore_datasets import DATASET_LOADERS


def parse_args():
    parser = argparse.ArgumentParser(description="RelationMIA experiment arguments")

    # Shared: model and dataset selection
    parser.add_argument("--model_name", type=str, default="gpt2", choices=MODEL_DICT.keys())
    parser.add_argument("--sft_data", type=str, default="imdb", choices=DATASET_LOADERS.keys())
    parser.add_argument("--dpo_data", type=str, default="imdb", choices=DATASET_LOADERS.keys())

    # Train: train_sft_dpo.py
    parser.add_argument("--use_quant", action="store_true")
    parser.add_argument("--no_lora", action="store_true")

    # Store/cache: output_store.py
    parser.add_argument("--n_member", type=int, default=2000)
    parser.add_argument("--n_nonmember", type=int, default=2000)
    parser.add_argument("--mode", choices=["premia", "judge"], default="premia")
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--output", type=str, default=None)

    # Eval: output_mia.py
    parser.add_argument("--input", type=str, default=None)

    # Population MIA: output_mia.py and mia_population.py
    parser.add_argument("--popu", type=int, default=1)
    parser.add_argument("--n_popu", type=int, default=1000)

    # Direct online baseline: mia_population.py
    parser.add_argument("--results_csv", type=str, default=None)

    return parser.parse_args()
