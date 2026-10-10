import numpy as np
import torch
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


def collect_logprobs_responses(trainer, model, ref_model, tokenizer,
                              max_samples=2000, max_new_tokens=256,
                              num_return_sequences=4, response_batch_size=8):
    """Keep original scores and add [2, K] response scores on both models.

    Row 0 contains target-generated responses; row 1 reference-generated ones.
    Columns retain sampling order, including duplicate samples. Score the exact generated
    tokens (including EOS, excluding padding) using TRL concatenated_forward.
    """
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if num_return_sequences <= 0 or response_batch_size <= 0:
        raise ValueError("num_return_sequences and response_batch_size must be positive")
    k = num_return_sequences
    rows = []
    pad_id = tokenizer.pad_token_id

    def generate_responses(generator, prompt_ids, prompt_mask):
        # Generation requires left padding; preserve the trainer's exact prompt.
        prompts = [ids[mask.bool()] for ids, mask in zip(prompt_ids, prompt_mask)]
        width = max(len(ids) for ids in prompts)
        budget = min(max_new_tokens, trainer.args.max_length - width)
        if budget <= 0:
            raise ValueError("Prompt leaves no room for responses within max_length")
        device = next(generator.parameters()).device
        inputs = torch.full((len(prompts), width), pad_id,
                            dtype=torch.long, device=device)
        mask = torch.zeros_like(inputs)
        for i, ids in enumerate(prompts):
            if not len(ids):
                raise ValueError("Cannot generate from an empty prompt")
            inputs[i, -len(ids):] = ids.to(device)
            mask[i, -len(ids):] = 1
        generated = generator.generate(
            input_ids=inputs, attention_mask=mask,
            do_sample=True, 
            temperature=1.0, 
            top_p=1.0, top_k=0,
            num_beams=1, num_return_sequences=k,
            max_new_tokens=budget, pad_token_id=pad_id,
            return_dict_in_generate=False,
        )[:, width:]
        eos_ids = generator.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = tokenizer.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        eos_ids = set(eos_ids or [])
        responses = []
        for ids in generated:
            tokens = ids.tolist()
            end = next((j + 1 for j, token in enumerate(tokens)
                        if token in eos_ids), len(tokens))
            responses.append(ids[:end].cpu())
        return responses

    def pad_responses(responses, device):
        width = max(len(ids) for ids in responses)
        ids = torch.full((len(responses), width), pad_id,
                         dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, response in enumerate(responses):
            ids[i, :len(response)] = response.to(device)
            mask[i, :len(response)] = 1
        return ids, mask

    def process(dataloader, label):
        seen = 0
        for batch in tqdm(dataloader, desc=f"collect responses label={label}"):
            if seen >= max_samples:
                break
            size = batch["prompt_input_ids"].shape[0]
            keep = min(size, max_samples - seen)
            batch = {key: value[:keep] if torch.is_tensor(value)
                     and value.ndim > 0 and value.shape[0] == size else value
                     for key, value in batch.items()}
            with torch.inference_mode():
                target = trainer.concatenated_forward(model, batch)
                reference = trainer.concatenated_forward(ref_model, batch)
                # HF generation order: sample0/k responses, sample1/k responses, ...
                generated_groups = []
                prompt_chunk_size = max(1, response_batch_size // k)
                for generator in (model, ref_model):
                    responses = []
                    for start in range(0, keep, prompt_chunk_size):
                        stop = min(start + prompt_chunk_size, keep)
                        responses.extend(generate_responses(
                            generator, batch["prompt_input_ids"][start:stop],
                            batch["prompt_attention_mask"][start:stop]))
                    generated_groups.append(responses)

                response_scores = [[], []]
                for start in range(0, keep * k, response_batch_size):
                    stop = min(start + response_batch_size, keep * k)
                    device = batch["prompt_input_ids"].device
                    sample_indices = torch.arange(start, stop, device=device) // k
                    response_batch = {
                        key: batch[key].index_select(0, sample_indices)
                        for key in ("prompt_input_ids", "prompt_attention_mask")
                    }
                    # Slots encode generation source, not preference labels.
                    for name, responses in zip(("chosen", "rejected"), generated_groups):
                        ids, mask = pad_responses(responses[start:stop], device)
                        response_batch[f"{name}_input_ids"] = ids
                        response_batch[f"{name}_attention_mask"] = mask
                    for scorer, chunks in zip((model, ref_model), response_scores):
                        result = trainer.concatenated_forward(scorer, response_batch)
                        chunks.append(torch.stack(
                            (result["chosen_logps"], result["rejected_logps"]),
                            dim=-1).detach().cpu())
                # Flat [B*K, source] -> [B, source, K].
                target_z, reference_z = [
                    torch.cat(chunks).reshape(keep, k, 2).transpose(1, 2).tolist()
                    for chunks in response_scores
                ]

            for i in range(keep):
                rows.append({
                    "label": label,
                    "m_chosen": target["chosen_logps"][i].item(),
                    "m_rejected": target["rejected_logps"][i].item(),
                    "r_chosen": reference["chosen_logps"][i].item(),
                    "r_rejected": reference["rejected_logps"][i].item(),
                    "m_response": target_z[i],
                    "r_response": reference_z[i],
                })
            seen += keep

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
    rows = collect_logprobs_responses(
        trainer, target_model, ref_model, tokenizer,
        max_samples=max(args.n_member, args.n_nonmember),
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
        response_batch_size=args.sem_generation_batch_size,
    )

    np.save(save_path, rows)
    print(f"Saved {len(rows)} samples to {save_path}")


if __name__ == "__main__":
    main()
