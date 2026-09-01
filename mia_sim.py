import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from _args import parse_args
from common import MODEL_DICT, default_split_sizes, model_paths
from explore_datasets import DATASET_LOADERS


def load_tokenizer(path, padding_side="right"):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_lora_model(base_model, adapter_path):
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def load_plain_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    model.eval()
    return model


def clean_generated_text(text):
    return str(text).replace("\x00", "").strip()


def is_valid_generated_text(text, min_words=2):
    text = clean_generated_text(text)
    if not text:
        return False
    if len(text) < 3:
        return False
    if len(set(text)) <= 2 and len(text) >= 10:
        return False
    most_common_char_ratio = max(text.count(char) for char in set(text)) / len(text)
    if most_common_char_ratio > 0.6 and len(text) >= 20:
        return False
    if not any(char.isalnum() for char in text):
        return False
    if len(text.split()) < min_words:
        return False
    return True


def filter_generated_responses(responses, min_words=2):
    filtered = []
    seen = set()
    for response in responses:
        response = clean_generated_text(response)
        key = " ".join(response.split())
        if not is_valid_generated_text(response, min_words=min_words):
            continue
        if key in seen:
            continue
        seen.add(key)
        filtered.append(response)
    return filtered


def generate_k_responses(model, tokenizer, prompt, k, max_new_tokens=256):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=k,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    responses = []
    for output in outputs:
        generated_ids = output[prompt_len:]
        response = clean_generated_text(tokenizer.decode(generated_ids, skip_special_tokens=True))
        responses.append(response)
    return filter_generated_responses(responses)


def generate_batch_k_responses(model, tokenizer, prompts, k, max_new_tokens=256):
    device = next(model.parameters()).device
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    tokenizer.padding_side = old_padding_side
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            num_return_sequences=k,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    responses_by_prompt = []
    for prompt_idx in range(len(prompts)):
        responses = []
        for response_idx in range(k):
            output_idx = prompt_idx * k + response_idx
            generated_ids = outputs[output_idx][prompt_len:]
            response = clean_generated_text(tokenizer.decode(generated_ids, skip_special_tokens=True))
            responses.append(response)
        responses_by_prompt.append(filter_generated_responses(responses))
    return responses_by_prompt


def load_embedder(model_name="sentence-transformers/all-mpnet-base-v2"):
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def cosine_matrix(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return a @ b.T


def response_sim_summary(chosen_emb, rejected_emb, response_embs):
    if len(response_embs) == 0:
        return {
            "sim_chosen": [],
            "sim_rejected": [],
            "sim_margin": [],
        }

    sim_chosen = cosine_matrix(response_embs, chosen_emb).reshape(-1).astype(np.float64)
    sim_rejected = cosine_matrix(response_embs, rejected_emb).reshape(-1).astype(np.float64)
    sim_margin = sim_chosen - sim_rejected
    return {
        "sim_chosen": sim_chosen.tolist(),
        "sim_rejected": sim_rejected.tolist(),
        "sim_margin": sim_margin.tolist(),
    }


def collect_response_rows(args, tokenizer, train_ds, eval_ds, model, ref_model=None, max_samples=2000):
    rows = []
    k = args.num_return_sequences
    max_new_tokens = args.max_new_tokens
    print(f"Generating {k} target/ref responses per sample...")

    def process(dataset, label):
        seen = 0
        for start in tqdm(range(0, len(dataset), args.sem_generation_batch_size), desc=f"label={label}"):
            if seen >= max_samples:
                break

            batch_samples = dataset[start: start + args.sem_generation_batch_size]
            prompts = list(batch_samples["prompt"])
            chosen_texts = list(batch_samples["chosen"])
            rejected_texts = list(batch_samples["rejected"])
            keep = min(len(prompts), max_samples - seen)
            prompts = prompts[:keep]
            chosen_texts = chosen_texts[:keep]
            rejected_texts = rejected_texts[:keep]

            responses_target = generate_batch_k_responses(
                model,
                tokenizer,
                prompts,
                k=k,
                max_new_tokens=max_new_tokens,
            )

            if ref_model is not None:
                responses_ref = generate_batch_k_responses(
                    ref_model,
                    tokenizer,
                    prompts,
                    k=k,
                    max_new_tokens=max_new_tokens,
                )
            else:
                responses_ref = [[] for _ in prompts]

            for i in range(keep):
                rows.append({
                    "label": label,
                    "chosen": chosen_texts[i],
                    "rejected": rejected_texts[i],
                    "target_response": responses_target[i],
                    "ref_response": responses_ref[i],
                })
            seen += keep

    process(train_ds, label=1)
    process(eval_ds, label=0)
    return rows


def attach_all_similarity(rows, embedder, batch_size=64):
    all_texts = []
    spans = []
    for row in rows:
        start = len(all_texts)
        target_responses = row["target_response"]
        ref_responses = row["ref_response"]
        all_texts.append(row["chosen"])
        all_texts.append(row["rejected"])
        all_texts.extend(target_responses)
        all_texts.extend(ref_responses)
        spans.append((start, len(target_responses), len(ref_responses)))

    embeddings = embedder.encode(
        all_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    for row, (start, n_target, n_ref) in zip(rows, spans):
        chosen_emb = embeddings[start: start + 1]
        rejected_emb = embeddings[start + 1: start + 2]
        target_start = start + 2
        target_end = target_start + n_target
        ref_start = target_end
        ref_end = ref_start + n_ref

        target_summary = response_sim_summary(
            chosen_emb,
            rejected_emb,
            embeddings[target_start:target_end],
        )
        ref_summary = response_sim_summary(
            chosen_emb,
            rejected_emb,
            embeddings[ref_start:ref_end],
        )
        row["target_sim_chosen"] = target_summary["sim_chosen"]
        row["target_sim_rejected"] = target_summary["sim_rejected"]
        row["ref_sim_chosen"] = ref_summary["sim_chosen"]
        row["ref_sim_rejected"] = ref_summary["sim_rejected"]

    return rows


def cal_sim_response(args, tokenizer, train_ds, eval_ds, model, ref_model=None, max_samples=2000):
    rows = collect_response_rows(
        args,
        tokenizer,
        train_ds,
        eval_ds,
        model,
        ref_model,
        max_samples=max_samples,
    )
    embedder = load_embedder("sentence-transformers/all-mpnet-base-v2")
    print("Calculating response similarities...")
    return attach_all_similarity(rows, embedder, batch_size=args.embed_batch_size)


def load_mia_datasets(args):
    n_train, n_eval = default_split_sizes(args.dpo_data)
    loader = DATASET_LOADERS[args.dpo_data]
    return loader(
        n_train,
        n_eval,
        for_dpo=True,
        n_members=args.n_member,
        n_nonmembers=args.n_nonmember,
    )


def main():
    args = parse_args()
    train_ds, eval_ds = load_mia_datasets(args)

    base_model = MODEL_DICT[args.model_name]
    paths = model_paths(args.model_name, args.sft_data, args.dpo_data)
    tokenizer = load_tokenizer(base_model)
    ref_model = load_lora_model(base_model, paths["sft"])
    target_model = load_lora_model(base_model, paths["dpo"])
    save_path = args.output or f"response/sim_{args.sft_data}_{args.dpo_data}_{args.model_name}.npy"
    rows = cal_sim_response(
        args,
        tokenizer,
        train_ds,
        eval_ds,
        target_model,
        ref_model,
        max_samples=max(args.n_member, args.n_nonmember),
    )
    np.save(save_path, rows)
    print(f"Saved {len(rows)} samples to {save_path}")


if __name__ == "__main__":
    main()
