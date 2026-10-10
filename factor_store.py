import os

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


def load_embedder(model_name):
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def response_similarity(embedder, pairs, batch_size=64):
    texts = [text for pair in pairs for text in pair]
    embeddings = embedder.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    # Normalized embeddings: dot product equals cosine similarity.
    return np.clip(np.sum(embeddings[0::2] * embeddings[1::2], axis=1), -1, 1)


def collect_logprobs(trainer, model, ref_model=None, max_samples=2000,
                     tokenizer=None, embedder=None, embed_batch_size=64):
    if tokenizer is None or embedder is None:
        raise ValueError("Pass tokenizer and embedder for factor scoring")
    if embed_batch_size <= 0:
        raise ValueError("embed_batch_size must be positive")
    rows = []
    print(f"Collecting log probabilities for up to {max_samples} samples from train and eval datasets...")

    def process(dataloader, label):
        seen = 0
        for batch in tqdm(dataloader):
            if seen >= max_samples:
                break
            size = batch["prompt_input_ids"].shape[0]
            keep = min(size, max_samples - seen)
            batch = {key: value[:keep] if torch.is_tensor(value)
                     and value.ndim > 0 and value.shape[0] == size else value
                     for key, value in batch.items()}
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

            # Use masks, not token != pad: EOS may also be the padding token.
            pairs = []
            for name in ("chosen", "rejected"):
                row[f"{name}_len"] = batch[f"{name}_attention_mask"].sum(dim=1).cpu().numpy()
            for i in range(keep):
                texts = []
                for name in ("chosen", "rejected"):
                    ids = batch[f"{name}_input_ids"][i]
                    mask = batch[f"{name}_attention_mask"][i].bool()
                    texts.append(tokenizer.decode(ids[mask].tolist(), skip_special_tokens=True))
                pairs.append(texts)
            row["similarity"] = response_similarity(embedder, pairs, embed_batch_size)
            batch_size = len(row["m_chosen"])
            for i in range(min(batch_size, max_samples - seen)):
                sample = {key: value[i] for key, value in row.items()}
                sample["label"] = label
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
    embedder = load_embedder(args.embed_model)
    save_path = args.output or f"factor_{args.sft_data}_{args.dpo_data}_{args.model_name}.npy"
    trainer = build_dpo_trainer(target_model, tokenizer, train_ds, eval_ds)
    rows = collect_logprobs(
        trainer, target_model, ref_model,
        max_samples=max(args.n_member, args.n_nonmember),
        tokenizer=tokenizer, embedder=embedder,
        embed_batch_size=args.embed_batch_size,
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    np.save(save_path, rows)
    print(f"Saved {len(rows)} samples to {save_path}")


if __name__ == "__main__":
    main()
