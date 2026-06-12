import argparse
import os
import pickle

import torch
from tqdm import tqdm

from common import MODEL_DICT, model_paths
from output_store import load_lora_model, load_mia_datasets, load_tokenizer


DEFAULT_TEMPS = "0.3"


def parse_args():
    parser = argparse.ArgumentParser(description="Save token-level temperature logprobs for preference tuples")
    parser.add_argument("--model_name", type=str, default="gpt2", choices=MODEL_DICT.keys())
    parser.add_argument("--sft_data", type=str, default="imdb")
    parser.add_argument("--dpo_data", type=str, default="imdb")
    parser.add_argument("--n_member", type=int, default=1000)
    parser.add_argument("--n_nonmember", type=int, default=1000)
    parser.add_argument("--temps", type=str, default=DEFAULT_TEMPS)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def parse_temps(text):
    temps = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            value = float(item)
            if value <= 0:
                raise ValueError("Temperatures must be positive")
            temps.append(value)
    if not temps:
        raise ValueError("--temps must contain at least one temperature")
    return temps


def check_model_paths(paths):
    dpo_path = os.path.abspath(paths["dpo"])
    if not os.path.exists(dpo_path):
        raise FileNotFoundError(f"DPO/target path does not exist: {dpo_path}")


def dataset_to_samples(dataset, label):
    return [
        {
            "prompt": row["prompt"],
            "chosen": row["chosen"],
            "rejected": row["rejected"],
            "label": label,
        }
        for row in dataset
    ]


def make_labeled_input(tokenizer, prompt, response, max_length):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    if not prompt_ids:
        prompt_ids = [tokenizer.eos_token_id]
    if not response_ids:
        response_ids = [tokenizer.eos_token_id]

    truncated_prompt = False
    truncated_response = False
    if len(response_ids) >= max_length:
        response_ids = response_ids[: max_length - 1]
        prompt_ids = prompt_ids[-1:]
        truncated_response = True
        truncated_prompt = True
    else:
        max_prompt_len = max_length - len(response_ids)
        if len(prompt_ids) > max_prompt_len:
            truncated_prompt = True
        prompt_ids = prompt_ids[-max_prompt_len:]

    input_ids = prompt_ids + response_ids
    answer_start = len(prompt_ids)
    answer_len = len(response_ids)
    return input_ids, answer_start, answer_len, truncated_prompt, truncated_response


def pad_batch(tokenizer, encoded):
    max_len = max(len(x[0]) for x in encoded)
    pad_id = tokenizer.pad_token_id
    input_ids = []
    attention_mask = []
    answer_starts = []
    answer_lens = []

    for ids, answer_start, answer_len, _, _ in encoded:
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        answer_starts.append(answer_start)
        answer_lens.append(answer_len)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "answer_starts": answer_starts,
        "answer_lens": answer_lens,
    }


def build_pair_records(samples):
    records = []
    for sample_idx, sample in enumerate(samples):
        records.append({
            "sample_idx": sample_idx,
            "side": "chosen",
            "prompt": sample["prompt"],
            "response": sample["chosen"],
        })
        records.append({
            "sample_idx": sample_idx,
            "side": "rejected",
            "prompt": sample["prompt"],
            "response": sample["rejected"],
        })
    return records


def compute_token_logprobs(records, model, tokenizer, temps, max_length=1024, batch_size=4):
    encoded = [
        make_labeled_input(tokenizer, record["prompt"], record["response"], max_length)
        for record in records
    ]
    scored = []
    device = model.device

    for start in tqdm(range(0, len(records), batch_size), desc="token temperature logprobs"):
        batch_records = records[start: start + batch_size]
        batch_encoded = encoded[start: start + batch_size]
        batch = pad_batch(tokenizer, batch_encoded)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        batch_token_logps = [{float(temp): [] for temp in temps} for _ in batch_records]

        for temp in temps:
            log_probs = torch.log_softmax(logits / temp, dim=-1)
            for i, (answer_start, answer_len) in enumerate(zip(batch["answer_starts"], batch["answer_lens"])):
                token_scores = []
                for pos in range(answer_start, answer_start + answer_len):
                    if pos == 0:
                        token_scores.append(float("nan"))
                        continue
                    token_id = input_ids[i, pos]
                    token_scores.append(float(log_probs[i, pos - 1, token_id].detach().cpu()))
                batch_token_logps[i][float(temp)] = token_scores
            del log_probs

        for i, record in enumerate(batch_records):
            logps_t = batch_token_logps[i]
            scored.append({
                "sample_idx": int(record["sample_idx"]),
                "side": record["side"],
                "token_logps_T": logps_t,
            })

    return scored


def attach_rows(samples, scored_records):
    grouped = [{} for _ in samples]
    for record in scored_records:
        grouped[record["sample_idx"]][record["side"]] = record["token_logps_T"]

    rows = []
    for sample_idx, sample in enumerate(samples):
        rows.append({
            "label": int(sample["label"]),
            "chosen_token_logps": grouped[sample_idx]["chosen"],
            "rejected_token_logps": grouped[sample_idx]["rejected"],
        })
    return rows


def default_output_path(args):
    temp_tag = "-".join(str(x).replace(".", "p") for x in parse_temps(args.temps))
    return f"token_query_{args.sft_data}_{args.dpo_data}_{args.model_name}_T{temp_tag}.pkl"


def main():
    args = parse_args()
    temps = parse_temps(args.temps)

    train_ds, eval_ds = load_mia_datasets(args)
    samples = dataset_to_samples(train_ds, label=1) + dataset_to_samples(eval_ds, label=0)

    base_model = MODEL_DICT[args.model_name]
    paths = model_paths(args.model_name, args.sft_data, args.dpo_data)
    check_model_paths(paths)

    tokenizer = load_tokenizer(base_model)
    target_model = load_lora_model(base_model, paths["dpo"])

    records = build_pair_records(samples)
    scored_records = compute_token_logprobs(
        records,
        target_model,
        tokenizer,
        temps,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    rows = attach_rows(samples, scored_records)

    save_path = args.output or default_output_path(args)
    with open(save_path, "wb") as handle:
        pickle.dump(rows, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(rows)} token-level samples to {save_path}")
    print(f"Temperatures: {temps}")


if __name__ == "__main__":
    main()
