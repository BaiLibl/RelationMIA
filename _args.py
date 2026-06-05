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

    # Multi-query attacks: multi-query.py
    parser.add_argument("--query_method", type=str, default="semantic", choices=["semantic"])
    parser.add_argument("--num_return_sequences", type=int, default=4)
    parser.add_argument("--n_prompt_neighbors", type=int, default=2)
    parser.add_argument("--n_chosen_neighbors", type=int, default=2)
    parser.add_argument("--n_rejected_neighbors", type=int, default=2)
    parser.add_argument("--semantic_threshold", type=float, default=0.85)
    parser.add_argument("--edit_threshold", type=float, default=0.10)
    parser.add_argument("--max_query_length", type=int, default=1024)
    parser.add_argument("--max_probes_per_tuple", type=int, default=64)
    parser.add_argument("--sem_K", type=int, default=16)
    parser.add_argument("--paraphrase_model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--rewrite_profile", type=str, default="balanced", choices=["strict", "balanced", "diverse"])
    parser.add_argument("--rewrite_temperature", type=float, default=None)
    parser.add_argument("--min_length_ratio", type=float, default=None)
    parser.add_argument("--max_length_ratio", type=float, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--disable_generation", action="store_true")
    parser.add_argument("--no_length_normalize", action="store_true")
    parser.add_argument("--save_query_logprobs", action="store_true")

    return parser.parse_args()
