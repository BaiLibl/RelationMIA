import numpy as np
import torch
from datasets import Dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

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


def build_dpo_trainer(model, tokenizer, train_ds, eval_ds):
    cfg = DPOConfig(
        output_dir="saved/dummy_dpo_trainer",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        fp16=torch.cuda.is_available(),
        max_length=1024,
        max_prompt_length=256,
        truncation_mode="keep_end",
        report_to="none",
    )
    return DPOTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )


def collect_logprobs(trainer, model, ref_model=None, max_samples=2000):
    rows = []
    print(f"Collecting log probabilities for up to {max_samples} samples from train and eval datasets...")

    def process(dataloader, label):
        seen = 0
        for batch in tqdm(dataloader):
            if seen >= max_samples:
                break
            with torch.no_grad():
                out = trainer.concatenated_forward(model, batch)
                row = {
                    "m_chosen": out["chosen_logps"].detach().cpu().numpy(),
                    "m_rejected": out["rejected_logps"].detach().cpu().numpy(),
                }
                if ref_model is not None:
                    ref = trainer.concatenated_forward(ref_model, batch)
                    row["r_chosen"] = ref["chosen_logps"].detach().cpu().numpy()
                    row["r_rejected"] = ref["rejected_logps"].detach().cpu().numpy()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            batch_size = len(row["m_chosen"])
            for i in range(min(batch_size, max_samples - seen)):
                sample = {key: value[i] for key, value in row.items()}
                sample["label"] = label
                rows.append(sample)
            seen += batch_size

    process(trainer.get_train_dataloader(), label=1)
    process(trainer.get_eval_dataloader(), label=0)
    return rows


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


def compute_response_logps(model, tokenizer, prompt, responses):
    if not responses:
        return []

    dataset = Dataset.from_list([
        {
            "prompt": prompt,
            "chosen": response,
            "rejected": response,
        }
        for response in responses
    ])
    trainer = build_dpo_trainer(model, tokenizer, dataset, dataset)
    dataloader = trainer.get_eval_dataloader()

    logps = []
    for batch in dataloader:
        with torch.no_grad():
            out = trainer.concatenated_forward(model, batch)
        logps.extend(out["chosen_logps"].detach().cpu().numpy().tolist())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return [float(logp) for logp in logps]


def compute_batch_response_logps(model, tokenizer, prompts, responses_by_prompt):
    records = []
    for row_idx, (prompt, responses) in enumerate(zip(prompts, responses_by_prompt)):
        for response in responses:
            records.append({
                "row_idx": row_idx,
                "prompt": prompt,
                "chosen": response,
                "rejected": response,
            })

    logps_by_prompt = [[] for _ in prompts]
    if not records:
        return logps_by_prompt

    dataset = Dataset.from_list([
        {
            "prompt": record["prompt"],
            "chosen": record["chosen"],
            "rejected": record["rejected"],
        }
        for record in records
    ])
    score_trainer = build_dpo_trainer(model, tokenizer, dataset, dataset)
    dataloader = score_trainer.get_eval_dataloader()

    offset = 0
    for batch in dataloader:
        with torch.no_grad():
            out = score_trainer.concatenated_forward(model, batch)
        logps = out["chosen_logps"].detach().cpu().numpy().tolist()
        for idx, logp in enumerate(logps):
            row_idx = records[offset + idx]["row_idx"]
            logps_by_prompt[row_idx].append(float(logp))
        offset += len(logps)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return logps_by_prompt


def decode_batch_text(tokenizer, batch, key, index):
    ids = batch[key][index]
    if torch.is_tensor(ids):
        ids = ids.detach().cpu().tolist()
    if tokenizer.pad_token_id is not None:
        ids = [token_id for token_id in ids if token_id != tokenizer.pad_token_id]
    return tokenizer.decode(ids, skip_special_tokens=True)


def collect_logprobs_generated(args, tokenizer, trainer, model, ref_model=None, max_samples=2000):
    rows = []
    k = args.num_return_sequences
    max_new_tokens = args.max_new_tokens
    print(f"Generating {k} responses per sample and collecting target logps...")

    def process(dataloader, label):
        seen = 0
        for batch in tqdm(dataloader, desc=f"label={label}"):
            if seen >= max_samples:
                break

            with torch.no_grad():
                out = trainer.concatenated_forward(model, batch)
                row = {
                    "m_chosen": out["chosen_logps"].detach().cpu().numpy(),
                    "m_rejected": out["rejected_logps"].detach().cpu().numpy(),
                }
                if ref_model is not None:
                    ref = trainer.concatenated_forward(ref_model, batch)
                    row["r_chosen"] = ref["chosen_logps"].detach().cpu().numpy()
                    row["r_rejected"] = ref["rejected_logps"].detach().cpu().numpy()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            batch_size = len(row["m_chosen"])
            keep = min(batch_size, max_samples - seen)
            prompts = [
                decode_batch_text(tokenizer, batch, "prompt_input_ids", i)
                for i in range(keep)
            ]
            responses_by_prompt = generate_batch_k_responses(
                model,
                tokenizer,
                prompts,
                k=k,
                max_new_tokens=max_new_tokens,
            )
            m_response_logps = compute_batch_response_logps(
                model,
                tokenizer,
                prompts,
                responses_by_prompt,
            )
            r_response_logps = [[] for _ in prompts]
            if ref_model is not None:
                r_response_logps = compute_batch_response_logps(
                    ref_model,
                    tokenizer,
                    prompts,
                    responses_by_prompt,
                )

            for i in range(keep):
                prompt = decode_batch_text(tokenizer, batch, "prompt_input_ids", i)
                sample = {key: value[i] for key, value in row.items()}
                sample["label"] = label
                sample["response"] = responses_by_prompt[i]
                sample["m_response"] = m_response_logps[i]
                sample["r_response"] = r_response_logps[i]
                rows.append(sample)
            seen += batch_size

    process(trainer.get_train_dataloader(), label=1)
    process(trainer.get_eval_dataloader(), label=0)
    return rows


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
    save_path = args.output or f"premia_{args.sft_data}_{args.dpo_data}_{args.model_name}.npy"
    trainer = build_dpo_trainer(target_model, tokenizer, train_ds, eval_ds)
    # rows = collect_logprobs(trainer, target_model, ref_model, max_samples=max(args.n_member, args.n_nonmember))
    rows = collect_logprobs_generated(
        args,
        tokenizer,
        trainer,
        target_model,
        ref_model,
        max_samples=max(args.n_member, args.n_nonmember),
    )
    np.save(save_path, rows)
    print(f"Saved {len(rows)} samples to {save_path}")


if __name__ == "__main__":
    main()
