import argparse
import pickle

import numpy as np
import torch
from tqdm import tqdm

from common import MODEL_DICT, model_paths
from eval_tools import clean_array, mia_auc_tpr, population_wrapper
from output_store import load_lora_model, load_mia_datasets, load_tokenizer


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(description="MIA from generated-response semantic similarity ranking")
    parser.add_argument("--model_name", type=str, default="qwen-0.5b", choices=MODEL_DICT.keys())
    parser.add_argument("--sft_data", type=str, default="uf")
    parser.add_argument("--dpo_data", type=str, default="uf")
    parser.add_argument("--n_member", type=int, default=1000)
    parser.add_argument("--n_nonmember", type=int, default=1000)
    parser.add_argument("--num_responses", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--prompt_format", type=str, default="raw", choices=["raw", "assistant"])
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embed_batch_size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--sim_filter", type=float, default=None)
    parser.add_argument("--min_words", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--popu", type=int, default=1)
    parser.add_argument("--n_popu", type=int, default=1000)
    parser.add_argument("--input", type=str, default=None, help="optional cached pkl produced by this script")
    parser.add_argument("--output", type=str, default=None, help="optional path to cache generated rows")
    return parser.parse_args()


def dataset_to_samples(dataset, label):
    return [
        {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "label": int(label),
        }
        for row in dataset
    ]


def clean_generation(text):
    return " ".join(str(text).split())


def unique_keep_order(texts):
    seen = set()
    out = []
    for text in texts:
        key = " ".join(str(text).split())
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def build_generation_prompt(prompt, prompt_format="raw"):
    prompt = str(prompt).strip()
    if prompt_format == "raw":
        return prompt + "\n"
    return (
        "You are a helpful assistant. Provide a direct answer to the user's request.\n\n"
        f"User:\n{prompt}\n\n"
        "Assistant:"
    )


def encode_prompts(tokenizer, prompts, max_prompt_length):
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )


def generate_responses(samples, model, tokenizer, args):
    rows = []
    device = model.device

    for start in tqdm(range(0, len(samples), args.generation_batch_size), desc="generate responses"):
        batch_samples = samples[start: start + args.generation_batch_size]
        prompts = [build_generation_prompt(sample["prompt"], args.prompt_format) for sample in batch_samples]
        inputs = encode_prompts(tokenizer, prompts, args.max_prompt_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.num_responses,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = []
        for output in outputs:
            new_tokens = output[input_len:]
            decoded.append(clean_generation(tokenizer.decode(new_tokens, skip_special_tokens=True)))

        for batch_idx, sample in enumerate(batch_samples):
            left = batch_idx * args.num_responses
            right = left + args.num_responses
            responses = [
                text for text in unique_keep_order(decoded[left:right])
                if len(text.split()) >= args.min_words
            ]
            rows.append({
                "label": int(sample["label"]),
                "prompt": sample["prompt"],
                "chosen": sample["chosen"],
                "rejected": sample["rejected"],
                "responses": responses,
            })

    return rows


def load_embedder(model_name):
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def cosine_matrix(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T


def finite_mean(values, default=0.0):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.any():
        return float(default)
    return float(values[finite].mean())


def finite_std(values, default=0.0):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if finite.sum() <= 1:
        return float(default)
    return float(values[finite].std())


def topk_mean(values, k):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    k = max(1, min(int(k), len(values)))
    return float(np.sort(values)[-k:].mean())


def attach_similarity(rows, embedder, args):
    all_texts = []
    spans = []
    for row in rows:
        start = len(all_texts)
        responses = row.get("responses", [])
        all_texts.append(row["chosen"])
        all_texts.append(row["rejected"])
        all_texts.extend(responses)
        spans.append((start, len(responses)))

    embeddings = embedder.encode(
        all_texts,
        batch_size=args.embed_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for row, (start, n_responses) in zip(rows, spans):
        chosen_emb = embeddings[start: start + 1]
        rejected_emb = embeddings[start + 1: start + 2]
        response_embs = embeddings[start + 2: start + 2 + n_responses]
        chosen_rejected_sim = float(cosine_matrix(chosen_emb, rejected_emb).reshape(-1)[0])

        if n_responses == 0:
            sim_chosen = np.asarray([], dtype=np.float64)
            sim_rejected = np.asarray([], dtype=np.float64)
        else:
            sim_chosen = cosine_matrix(response_embs, chosen_emb).reshape(-1).astype(np.float64)
            sim_rejected = cosine_matrix(response_embs, rejected_emb).reshape(-1).astype(np.float64)

        if args.sim_filter is not None and n_responses > 0:
            keep = np.maximum(sim_chosen, sim_rejected) >= args.sim_filter
            sim_chosen = sim_chosen[keep]
            sim_rejected = sim_rejected[keep]

        gap = sim_chosen - sim_rejected
        row["sim_chosen"] = sim_chosen.tolist()
        row["sim_rejected"] = sim_rejected.tolist()
        row["sim_gap"] = gap.tolist()
        row["chosen_rejected_sim"] = chosen_rejected_sim

        row["sim_mean_gap"] = finite_mean(gap)
        row["sim_median_gap"] = float(np.median(gap)) if len(gap) else 0.0
        row["sim_positive_rate"] = float(np.mean(gap > 0)) if len(gap) else 0.0
        row["sim_top1_gap"] = float(np.max(gap)) if len(gap) else 0.0
        row["sim_topk_gap"] = topk_mean(gap, args.topk)
        row["sim_std_gap"] = finite_std(gap)
        row["sim_stability"] = row["sim_mean_gap"] / (row["sim_std_gap"] + EPS)
        row["sim_nn_gap"] = (
            float(np.max(sim_chosen)) - float(np.max(sim_rejected))
            if len(gap) else 0.0
        )
        row["num_valid_responses"] = int(len(gap))

    return rows


def load_or_generate_rows(args):
    if args.input:
        with open(args.input, "rb") as handle:
            rows = pickle.load(handle)
        if not rows:
            raise ValueError(f"empty input: {args.input}")
        return rows

    train_ds, eval_ds = load_mia_datasets(args)
    samples = dataset_to_samples(train_ds, label=1) + dataset_to_samples(eval_ds, label=0)

    base_model = MODEL_DICT[args.model_name]
    paths = model_paths(args.model_name, args.sft_data, args.dpo_data)
    tokenizer = load_tokenizer(base_model, padding_side="left")
    target_model = load_lora_model(base_model, paths["dpo"])
    rows = generate_responses(samples, target_model, tokenizer, args)
    del target_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def score_array(rows, key):
    return np.asarray([row.get(key, 0.0) for row in rows], dtype=np.float64)


def evaluate_scores(rows, popu=1, n_popu=1000):
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    score_names = [
        "sim_mean_gap",
        "sim_median_gap",
        "sim_positive_rate",
        "sim_top1_gap",
        "sim_topk_gap",
        "sim_stability",
        "sim_nn_gap",
    ]
    results = []
    for name in score_names:
        scores = clean_array(score_array(rows, name))
        eval_scores = scores
        eval_labels = labels
        if popu > 1:
            eval_scores, eval_labels = population_wrapper(scores, labels, popu, n_popu=n_popu)
        auc, tpr = mia_auc_tpr(eval_scores, eval_labels)
        results.append((name, auc, tpr, len(eval_scores)))
    return labels, results


def main():
    args = parse_args()
    if args.num_responses <= 0:
        raise ValueError("--num_responses must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = load_or_generate_rows(args)
    embedder = load_embedder(args.embed_model)
    rows = attach_similarity(rows, embedder, args)

    if args.output:
        with open(args.output, "wb") as handle:
            pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(rows)} generated-sim rows to {args.output}")

    labels, results = evaluate_scores(rows, popu=args.popu, n_popu=args.n_popu)
    valid_counts = np.asarray([row.get("num_valid_responses", 0) for row in rows], dtype=np.int64)
    chosen_rejected_sims = score_array(rows, "chosen_rejected_sim")
    mean_sim_chosen = np.asarray([finite_mean(row.get("sim_chosen", [])) for row in rows], dtype=np.float64)
    mean_sim_rejected = np.asarray([finite_mean(row.get("sim_rejected", [])) for row in rows], dtype=np.float64)

    print(f"Loaded {len(rows)} rows")
    print(f"member: {int(labels.sum())} nonmember: {int((labels == 0).sum())}")
    print(
        f"num_responses={args.num_responses} topk={args.topk} "
        f"sim_filter={args.sim_filter} prompt_format={args.prompt_format}"
    )
    print(
        "valid responses: "
        f"mean={valid_counts.mean():.2f}, median={np.median(valid_counts):.2f}, "
        f"min={valid_counts.min()}, max={valid_counts.max()}"
    )
    print(
        "similarity diagnostics: "
        f"chosen_rejected={chosen_rejected_sims.mean():.4f}, "
        f"gen_chosen={mean_sim_chosen.mean():.4f}, "
        f"gen_rejected={mean_sim_rejected.mean():.4f}"
    )
    print("=== Generated Similarity Ranking MIA ===")
    for name, auc, tpr, n in results:
        print(f"{name:>24s}: AUROC={auc:.4f}, TPR@FPR=0.05={tpr:.4f}, n={n}")


if __name__ == "__main__":
    main()
