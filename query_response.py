import argparse
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm

from common import MODEL_DICT, model_paths
from output_store import load_lora_model, load_mia_datasets, load_plain_model, load_tokenizer
from cal_store import dataset_to_samples, make_labeled_input, pad_batch


DEFAULT_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate target responses and compare them with chosen/rejected")
    parser.add_argument("--model_name", type=str, default="qwen-0.5b", choices=MODEL_DICT.keys())
    parser.add_argument("--sft_data", type=str, default="uf")
    parser.add_argument("--dpo_data", type=str, default="uf")
    parser.add_argument("--n_member", type=int, default=1000)
    parser.add_argument("--n_nonmember", type=int, default=1000)
    parser.add_argument("--num_responses", type=int, default=8)
    parser.add_argument("--cal_model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--generation_batch_size", type=int, default=4)
    parser.add_argument("--logprob_batch_size", type=int, default=8)
    parser.add_argument("--embed_model", type=str, default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--embed_batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def check_target_path(args):
    paths = model_paths(args.model_name, args.sft_data, args.dpo_data)
    dpo_path = os.path.abspath(paths["dpo"])
    if not os.path.exists(dpo_path):
        raise FileNotFoundError(f"DPO/target path does not exist: {dpo_path}")
    return paths


def encode_prompts(tokenizer, prompts, max_prompt_length):
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )


def clean_generation(text):
    return " ".join(str(text).split())


def build_generation_prompt(prompt):
    prompt = str(prompt).strip()
    return (
        "You are a helpful assistant. Provide a direct answer to the user's request.\n\n"
        f"User:\n{prompt}\n\n"
        "Assistant:"
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
            rows.append({
                "label": int(sample["label"]),
                "prompt": sample["prompt"],
                "chosen": sample["chosen"],
                "rejected": sample["rejected"],
                "responses": decoded[left:right],
            })

    return rows


def generate_cal_responses(rows, model, tokenizer, args):
    device = model.device

    for start in tqdm(range(0, len(rows), args.generation_batch_size), desc="generate cal responses"):
        batch_rows = rows[start: start + args.generation_batch_size]
        prompts = [build_generation_prompt(row["prompt"]) for row in batch_rows]
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

        for batch_idx, row in enumerate(batch_rows):
            left = batch_idx * args.num_responses
            right = left + args.num_responses
            row["cal_responses"] = decoded[left:right]

    return rows


def build_logprob_records(rows, response_key="responses"):
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
        for response_idx, response in enumerate(row.get(response_key, [])):
            records.append({
                "row_idx": row_idx,
                "kind": "generated",
                "response_idx": response_idx,
                "prompt": row["prompt"],
                "response": response,
            })
    return records


def compute_logprobs(records, model, tokenizer, max_length=1024, batch_size=8):
    encoded = [
        make_labeled_input(tokenizer, record["prompt"], record["response"], max_length)
        for record in records
    ]
    scored = []
    device = model.device

    for start in tqdm(range(0, len(records), batch_size), desc="response logprobs"):
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
            out["logp"] = total
            out["num_tokens"] = count
            scored.append(out)

        del logits, log_probs

    return scored


def attach_logprob_margins(
    rows,
    scored,
    chosen_key="chosen_logp",
    rejected_key="rejected_logp",
    response_key="response_logps",
    mean_chosen_margin_key="mean_chosen_margin",
    mean_rejected_margin_key="mean_rejected_margin",
):
    response_logps = [[] for _ in rows]

    for row in rows:
        row[chosen_key] = 0.0
        row[rejected_key] = 0.0

    for record in scored:
        row = rows[record["row_idx"]]
        if record["kind"] == "chosen":
            row[chosen_key] = float(record["logp"])
        elif record["kind"] == "rejected":
            row[rejected_key] = float(record["logp"])
        else:
            response_logps[record["row_idx"]].append(float(record["logp"]))

    for row_idx, row in enumerate(rows):
        logps = np.asarray(response_logps[row_idx], dtype=np.float64)
        chosen_logp = float(row[chosen_key])
        rejected_logp = float(row[rejected_key])

        chosen_margins = chosen_logp - logps
        rejected_margins = logps - rejected_logp

        row[response_key] = logps.tolist()
        row[mean_chosen_margin_key] = float(chosen_margins.mean()) if len(chosen_margins) else 0.0
        row[mean_rejected_margin_key] = float(rejected_margins.mean()) if len(rejected_margins) else 0.0

    return rows

def load_embedder(model_name):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise ImportError(
            "query_response.py requires sentence-transformers for semantic similarity. "
            "Install it or use an environment where it is available."
        ) from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def cosine_matrix(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T


def response_similarity_records(
    texts,
    chosen_emb,
    rejected_emb,
    response_embs,
    response_logps=None,
    chosen_logp=None,
    rejected_logp=None,
):
    if len(response_embs) == 0:
        chosen_sims = np.asarray([], dtype=np.float64)
        rejected_sims = np.asarray([], dtype=np.float64)
    else:
        chosen_sims = cosine_matrix(response_embs, chosen_emb).reshape(-1).astype(np.float64)
        rejected_sims = cosine_matrix(response_embs, rejected_emb).reshape(-1).astype(np.float64)
    sim_margins = chosen_sims - rejected_sims

    records = []
    for idx, (text, sim_chosen, sim_rejected, sim_margin) in enumerate(
        zip(texts, chosen_sims, rejected_sims, sim_margins)
    ):
        record = {
            "text": text,
            "sim_chosen": float(sim_chosen),
            "sim_rejected": float(sim_rejected),
            "sim_margin": float(sim_margin),
        }
        if response_logps is not None and chosen_logp is not None and rejected_logp is not None:
            response_logp = float(response_logps[idx]) if idx < len(response_logps) else 0.0
            record.update({
                "response_logp": response_logp,
                "chosen_margin": float(chosen_logp - response_logp),
                "rejected_margin": float(response_logp - rejected_logp),
            })
        records.append(record)

    summary = {
        "mean_sim_chosen": float(chosen_sims.mean()) if len(chosen_sims) else 0.0,
        "mean_sim_rejected": float(rejected_sims.mean()) if len(rejected_sims) else 0.0,
        "mean_sim_margin": float(sim_margins.mean()) if len(sim_margins) else 0.0,
        "max_sim_chosen": float(chosen_sims.max()) if len(chosen_sims) else 0.0,
        "max_sim_rejected": float(rejected_sims.max()) if len(rejected_sims) else 0.0,
        "max_sim_margin": float(sim_margins.max()) if len(sim_margins) else 0.0,
    }
    return records, summary


def attach_similarity(rows, embedder, batch_size=64):
    all_texts = []
    row_spans = []
    for row in rows:
        start = len(all_texts)
        target_responses = row.get("responses", [])
        cal_responses = row.get("cal_responses", [])
        all_texts.append(row["chosen"])
        all_texts.append(row["rejected"])
        all_texts.extend(target_responses)
        all_texts.extend(cal_responses)
        row_spans.append((start, len(target_responses), len(cal_responses)))

    embeddings = embedder.encode(
        all_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for row, (start, n_target, n_cal) in zip(rows, row_spans):
        chosen_emb = embeddings[start: start + 1]
        rejected_emb = embeddings[start + 1: start + 2]
        target_start = start + 2
        target_end = target_start + n_target
        cal_start = target_end
        cal_end = cal_start + n_cal

        target_texts = row.get("responses", [])
        target_embs = embeddings[target_start:target_end]
        target_records, target_summary = response_similarity_records(
            target_texts,
            chosen_emb,
            rejected_emb,
            target_embs,
            response_logps=row.get("response_logps", []),
            chosen_logp=row.get("chosen_logp", 0.0),
            rejected_logp=row.get("rejected_logp", 0.0),
        )
        row["responses"] = target_records
        row.update(target_summary)

        if n_cal > 0:
            cal_texts = row.get("cal_responses", [])
            cal_embs = embeddings[cal_start:cal_end]
            cal_records, cal_summary = response_similarity_records(
                cal_texts,
                chosen_emb,
                rejected_emb,
                cal_embs,
                response_logps=row.get("cal_response_logps", []),
                chosen_logp=row.get("cal_chosen_logp", 0.0),
                rejected_logp=row.get("cal_rejected_logp", 0.0),
            )
            row["cal_responses"] = cal_records
            for key, value in cal_summary.items():
                row[f"cal_{key}"] = value

    return rows



def slim_saved_rows(rows):
    slim_rows = []
    for row in rows:
        target_responses = row.get("responses", [])
        slim = {
            "label": int(row["label"]),
            "target": {
                "chosen_logp": float(row.get("chosen_logp", 0.0)),
                "rejected_logp": float(row.get("rejected_logp", 0.0)),
                "response_logps": [float(x) for x in row.get("response_logps", [])],
                "sim_chosen": [float(x.get("sim_chosen", 0.0)) for x in target_responses],
                "sim_rejected": [float(x.get("sim_rejected", 0.0)) for x in target_responses],
            },
        }

        if "cal_responses" in row:
            cal_responses = row.get("cal_responses", [])
            slim["cal"] = {
                "chosen_logp": float(row.get("cal_chosen_logp", 0.0)),
                "rejected_logp": float(row.get("cal_rejected_logp", 0.0)),
                "response_logps": [float(x) for x in row.get("cal_response_logps", [])],
                "sim_chosen": [float(x.get("sim_chosen", 0.0)) for x in cal_responses],
                "sim_rejected": [float(x.get("sim_rejected", 0.0)) for x in cal_responses],
            }

        slim_rows.append(slim)
    return slim_rows

def default_output_path(args):
    cal_tag = "" if not args.cal_model else "_cal" + args.cal_model.split("/")[-1]
    return (
        f"response_query_{args.sft_data}_{args.dpo_data}_{args.model_name}_"
        f"T{args.temperature}_n{args.num_responses}{cal_tag}.pkl"
    )


def main():
    args = parse_args()
    if args.num_responses <= 0:
        raise ValueError("--num_responses must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_ds, eval_ds = load_mia_datasets(args)
    samples = dataset_to_samples(train_ds, label=1) + dataset_to_samples(eval_ds, label=0)

    base_model = MODEL_DICT[args.model_name]
    paths = check_target_path(args)
    tokenizer = load_tokenizer(base_model, padding_side="left")
    target_model = load_lora_model(base_model, paths["dpo"])

    print(f"Loaded {len(samples)} samples: member={len(train_ds)} nonmember={len(eval_ds)}")
    print(f"Target base: {base_model}")
    print(f"Target DPO adapter: {paths['dpo']}")
    print(f"Generation: T={args.temperature}, top_p={args.top_p}, n={args.num_responses}")

    rows = generate_responses(samples, target_model, tokenizer, args)
    logprob_records = build_logprob_records(rows)
    scored_logprobs = compute_logprobs(
        logprob_records,
        target_model,
        tokenizer,
        max_length=args.max_prompt_length + args.max_new_tokens,
        batch_size=args.logprob_batch_size,
    )
    rows = attach_logprob_margins(rows, scored_logprobs)

    del target_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.cal_model:
        print(f"Calibration generation model: {args.cal_model}")
        cal_tokenizer = load_tokenizer(args.cal_model, padding_side="left")
        cal_model = load_plain_model(args.cal_model)
        rows = generate_cal_responses(rows, cal_model, cal_tokenizer, args)
        cal_logprob_records = build_logprob_records(rows, response_key="cal_responses")
        cal_scored_logprobs = compute_logprobs(
            cal_logprob_records,
            cal_model,
            cal_tokenizer,
            max_length=args.max_prompt_length + args.max_new_tokens,
            batch_size=args.logprob_batch_size,
        )
        rows = attach_logprob_margins(
            rows,
            cal_scored_logprobs,
            chosen_key="cal_chosen_logp",
            rejected_key="cal_rejected_logp",
            response_key="cal_response_logps",
            mean_chosen_margin_key="cal_mean_chosen_margin",
            mean_rejected_margin_key="cal_mean_rejected_margin",
        )
        del cal_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Embedding model: {args.embed_model}")
    embedder = load_embedder(args.embed_model)
    rows = attach_similarity(rows, embedder, batch_size=args.embed_batch_size)

    save_rows = slim_saved_rows(rows)
    save_path = args.output or default_output_path(args)
    with open(save_path, "wb") as handle:
        pickle.dump(save_rows, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(save_rows)} response-query samples to {save_path}")


if __name__ == "__main__":
    main()
