import argparse
import pickle

import numpy as np
import torch
from tqdm import tqdm

from cal_store import make_labeled_input, pad_batch
from common import MODEL_DICT, model_paths
from eval_tools import clean_array, mia_auc_tpr, population_wrapper
from output_store import load_lora_model, load_mia_datasets, load_tokenizer


EPS = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(description="MIA from generated-response likelihood position ranking")
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
    parser.add_argument("--logprob_batch_size", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--min_words", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--popu", type=int, default=1)
    parser.add_argument("--n_popu", type=int, default=1000)
    parser.add_argument("--eps", type=float, default=EPS)
    parser.add_argument("--input", type=str, default=None, help="optional cached pkl with generated responses")
    parser.add_argument("--output", type=str, default=None, help="optional path to cache scored rows")
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


def build_generation_prompt(prompt):
    prompt = str(prompt).strip()
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
        prompts = [build_generation_prompt(sample["prompt"]) for sample in batch_samples]
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


def load_or_generate_rows(args, target_model, tokenizer):
    if args.input:
        with open(args.input, "rb") as handle:
            rows = pickle.load(handle)
        if not rows:
            raise ValueError(f"empty input: {args.input}")
        return rows

    train_ds, eval_ds = load_mia_datasets(args)
    samples = dataset_to_samples(train_ds, label=1) + dataset_to_samples(eval_ds, label=0)
    return generate_responses(samples, target_model, tokenizer, args)


def build_logprob_records(rows):
    records = []
    for row_idx, row in enumerate(rows):
        records.append({
            "row_idx": row_idx,
            "kind": "chosen",
            "response_idx": -1,
            "prompt": row["prompt"],
            "response": row["chosen"],
        })
        records.append({
            "row_idx": row_idx,
            "kind": "rejected",
            "response_idx": -1,
            "prompt": row["prompt"],
            "response": row["rejected"],
        })
        for response_idx, response in enumerate(row.get("responses", [])):
            records.append({
                "row_idx": row_idx,
                "kind": "generated",
                "response_idx": response_idx,
                "prompt": row["prompt"],
                "response": response,
            })
    return records


def compute_logprobs(records, model, tokenizer, max_length, batch_size):
    encoded = [
        make_labeled_input(tokenizer, record["prompt"], record["response"], max_length)
        for record in records
    ]
    scored = []
    device = model.device

    for start in tqdm(range(0, len(records), batch_size), desc="target logprobs"):
        batch_records = records[start: start + batch_size]
        batch_encoded = encoded[start: start + batch_size]
        batch = pad_batch(tokenizer, batch_encoded)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits, dim=-1)

        for i, record in enumerate(batch_records):
            answer_start = batch["answer_starts"][i]
            answer_len = batch["answer_lens"][i]
            total = 0.0
            count = 0
            for pos in range(answer_start, answer_start + answer_len):
                if pos == 0:
                    continue
                token_id = input_ids[i, pos]
                total += float(log_probs[i, pos - 1, token_id].detach().cpu())
                count += 1
            out = dict(record)
            out["logp"] = float(total)
            out["avg_logp"] = float(total / max(count, 1))
            out["num_tokens"] = int(count)
            scored.append(out)

        del logits, log_probs

    return scored


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


def bottomk_mean(values, k):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0
    k = max(1, min(int(k), len(values)))
    return float(np.sort(values)[:k].mean())


def attach_rank_scores(rows, scored_records, args):
    grouped = [
        {
            "generated": [],
        }
        for _ in rows
    ]
    for record in scored_records:
        row = grouped[record["row_idx"]]
        if record["kind"] == "chosen":
            row["chosen"] = record
        elif record["kind"] == "rejected":
            row["rejected"] = record
        else:
            row["generated"].append(record)

    for row, group in zip(rows, grouped):
        chosen_logp = float(group["chosen"]["logp"])
        rejected_logp = float(group["rejected"]["logp"])
        gen_logps = np.asarray([record["logp"] for record in group["generated"]], dtype=np.float64)

        margin = chosen_logp - rejected_logp
        upper = chosen_logp - gen_logps
        lower = gen_logps - rejected_logp
        interval_min = np.minimum(upper, lower)
        pos = (gen_logps - rejected_logp) / (abs(margin) + args.eps)
        chosen_gap = -np.abs(pos - 1.0)

        row["target_chosen_logp"] = chosen_logp
        row["target_rejected_logp"] = rejected_logp
        row["target_margin"] = margin
        row["generated_logps"] = gen_logps.tolist()
        row["position"] = pos.tolist()
        row["interval_min"] = interval_min.tolist()

        row["pos_mean"] = finite_mean(pos)
        row["pos_median"] = float(np.median(pos)) if len(pos) else 0.0
        row["pos_topk"] = topk_mean(pos, args.topk)
        row["pos_std"] = finite_std(pos)
        row["pos_stability"] = row["pos_mean"] / (row["pos_std"] + EPS)
        row["between_rate"] = float(np.mean((upper >= 0) & (lower >= 0))) if len(gen_logps) else 0.0
        row["chosen_side_rate"] = float(np.mean(pos >= 0.5)) if len(pos) else 0.0
        row["above_chosen_rate"] = float(np.mean(gen_logps > chosen_logp)) if len(gen_logps) else 0.0
        row["yw_above_gen_rate"] = float(np.mean(chosen_logp >= gen_logps)) if len(gen_logps) else 0.0
        row["gen_above_yl_rate"] = float(np.mean(gen_logps >= rejected_logp)) if len(gen_logps) else 0.0
        row["chosen_gap_mean"] = finite_mean(chosen_gap)
        row["chosen_gap_topk"] = topk_mean(chosen_gap, args.topk)
        row["interval_min_mean"] = finite_mean(interval_min)
        row["interval_min_topk"] = topk_mean(interval_min, args.topk)
        row["interval_min_worst"] = bottomk_mean(interval_min, args.topk)
        row["num_valid_responses"] = int(len(gen_logps))

    return rows


def score_array(rows, key):
    return np.asarray([row.get(key, 0.0) for row in rows], dtype=np.float64)


def evaluate_scores(rows, popu=1, n_popu=1000):
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    score_names = [
        "pos_mean",
        "pos_median",
        "pos_topk",
        "pos_stability",
        "between_rate",
        "chosen_side_rate",
        "above_chosen_rate",
        "yw_above_gen_rate",
        "gen_above_yl_rate",
        "chosen_gap_mean",
        "chosen_gap_topk",
        "interval_min_mean",
        "interval_min_topk",
        "interval_min_worst",
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

    base_model = MODEL_DICT[args.model_name]
    paths = model_paths(args.model_name, args.sft_data, args.dpo_data)
    tokenizer = load_tokenizer(base_model, padding_side="left")
    target_model = load_lora_model(base_model, paths["dpo"])

    rows = load_or_generate_rows(args, target_model, tokenizer)
    records = build_logprob_records(rows)
    scored_records = compute_logprobs(
        records,
        target_model,
        tokenizer,
        max_length=args.max_prompt_length + args.max_new_tokens,
        batch_size=args.logprob_batch_size,
    )
    rows = attach_rank_scores(rows, scored_records, args)

    if args.output:
        with open(args.output, "wb") as handle:
            pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(rows)} generated-rank rows to {args.output}")

    labels, results = evaluate_scores(rows, popu=args.popu, n_popu=args.n_popu)
    valid_counts = np.asarray([row.get("num_valid_responses", 0) for row in rows], dtype=np.int64)

    print(f"Loaded {len(rows)} rows")
    print(f"member: {int(labels.sum())} nonmember: {int((labels == 0).sum())}")
    print(f"num_responses={args.num_responses} topk={args.topk}")
    print(
        "valid responses: "
        f"mean={valid_counts.mean():.2f}, median={np.median(valid_counts):.2f}, "
        f"min={valid_counts.min()}, max={valid_counts.max()}"
    )
    print("=== Generated Preference Position MIA ===")
    for name, auc, tpr, n in results:
        print(f"{name:>24s}: AUROC={auc:.4f}, TPR@FPR=0.05={tpr:.4f}, n={n}")


if __name__ == "__main__":
    main()
