import argparse
import os
import pickle

import torch

from common import MODEL_DICT, default_split_sizes, model_paths
from explore_datasets import DATASET_LOADERS
from output_store import load_lora_model, load_plain_model, load_tokenizer
from cal_store import (
    attach_rows,
    build_response_records,
    compute_logprobs,
    dataset_to_samples,
    safe_name,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Store background target/calibration tuple logprobs for qmap fitting")
    parser.add_argument("--model_name", type=str, default="qwen-0.5b", choices=MODEL_DICT.keys())
    parser.add_argument("--sft_data", type=str, default="uf")
    parser.add_argument("--dpo_data", type=str, default="uf")
    parser.add_argument("--cal_model", type=str, required=True, help="HF path/name of calibration model")
    parser.add_argument("--bg_split", type=str, default="eval", choices=["train", "eval"])
    parser.add_argument("--bg_start", type=int, default=1000)
    parser.add_argument("--n_background", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def check_target_path(args):
    paths = model_paths(args.model_name, args.sft_data, args.dpo_data)
    dpo_path = os.path.abspath(paths["dpo"])
    if not os.path.exists(dpo_path):
        raise FileNotFoundError(f"DPO/target path does not exist: {dpo_path}")
    return paths


def load_background_dataset(args):
    n_train, n_eval = default_split_sizes(args.dpo_data)
    loader = DATASET_LOADERS[args.dpo_data]
    train_ds, eval_ds = loader(n_train, n_eval, for_dpo=True)
    source = train_ds if args.bg_split == "train" else eval_ds

    if args.bg_start < 0:
        raise ValueError("--bg_start must be non-negative")
    if args.n_background <= 0:
        raise ValueError("--n_background must be positive")
    if args.bg_start >= len(source):
        raise ValueError(f"--bg_start={args.bg_start} is outside {args.bg_split} split of size {len(source)}")

    end = min(args.bg_start + args.n_background, len(source))
    return source.select(range(args.bg_start, end))


def default_output_path(args):
    return (
        f"cal_bg_{args.bg_split}{args.bg_start}_n{args.n_background}_"
        f"{args.sft_data}_{args.dpo_data}_{args.model_name}_{safe_name(args.cal_model)}.pkl"
    )


def main():
    args = parse_args()
    bg_ds = load_background_dataset(args)
    samples = dataset_to_samples(bg_ds, label=-1)
    records = build_response_records(samples)

    base_model = MODEL_DICT[args.model_name]
    paths = check_target_path(args)

    print(f"Loaded {len(samples)} background samples from {args.dpo_data}/{args.bg_split} "
          f"[{args.bg_start}:{args.bg_start + len(samples)}]")
    print(f"Target base: {base_model}")
    print(f"Target DPO adapter: {paths['dpo']}")
    print(f"Calibration model: {args.cal_model}")

    target_tokenizer = load_tokenizer(base_model)
    target_model = load_lora_model(base_model, paths["dpo"])
    target_scored = compute_logprobs(
        records,
        target_model,
        target_tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        desc="background target logprobs",
    )

    del target_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cal_tokenizer = load_tokenizer(args.cal_model)
    cal_model = load_plain_model(args.cal_model)
    cal_scored = compute_logprobs(
        records,
        cal_model,
        cal_tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
        desc="background calibration logprobs",
    )

    rows = attach_rows(samples, target_scored, cal_scored, args.cal_model)
    for row in rows:
        row["label"] = -1
        row["background_split"] = args.bg_split
        row["background_start"] = args.bg_start

    save_path = args.output or default_output_path(args)
    with open(save_path, "wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(rows)} background calibration samples to {save_path}")


if __name__ == "__main__":
    main()
