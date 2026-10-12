import os
import random
import warnings

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


class SynonymPerturber:
    """English POS tagging + WordNet synonyms, with inflected replacements."""

    def __init__(self, model_name, seed):
        import spacy
        from nltk.corpus import wordnet
        import lemminflect  # Registers Token._.inflect.

        self.nlp = spacy.load(model_name)
        self.wordnet = wordnet
        try:
            wordnet.ensure_loaded()
        except LookupError as exc:
            raise RuntimeError("Install WordNet with: python -m nltk.downloader wordnet") from exc
        self.rng = random.Random(seed)

    def variants(self, doc, rates):
        pos_map = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}
        words = sum(token.is_alpha for token in doc)
        candidates = []
        for token in doc:
            if (token.pos_ not in pos_map or not token.is_alpha or token.is_stop
                    or token.ent_type_):
                continue
            synsets = self.wordnet.synsets(token.lemma_, pos=pos_map[token.pos_])
            alternatives = set()
            # First dictionary sense is a heuristic, not contextual disambiguation.
            for synset in synsets[:1]:
                for lemma in synset.lemmas():
                    word = lemma.name()
                    if not word.isalpha() or word.lower() == token.lemma_.lower():
                        continue
                    replacement_doc = self.nlp(word)
                    replacement = replacement_doc[0]._.inflect(token.tag_)
                    if replacement is None:
                        continue
                    if token.text.istitle():
                        replacement = replacement.capitalize()
                    elif token.text.isupper():
                        replacement = replacement.upper()
                    if replacement.lower() != token.text.lower():
                        alternatives.add(replacement)
            if alternatives:
                candidates.append((token.i, self.rng.choice(sorted(alternatives))))
        self.rng.shuffle(candidates)
        outputs = []
        for rate in rates:
            requested = int(words * rate / 100 + 0.5)
            selected = dict(candidates[:min(requested, len(candidates))])
            text = "".join(selected.get(token.i, token.text) + token.whitespace_ for token in doc)
            outputs.append((text, 100 * len(selected) / words if words else 0.0,
                            len(selected)))
        # Nested subsets make differences across rates less sensitive to randomness.
        return outputs


def collect_logprobs(trainer, model, ref_model=None, max_samples=2000,
                     tokenizer=None, perturber=None, rates=(5, 10, 20),
                     score_batch_size=8):
    if tokenizer is None or perturber is None or ref_model is None:
        raise ValueError("tokenizer, perturber, and ref_model are required")
    if not rates or any(not 0 <= rate <= 100 for rate in rates) or score_batch_size <= 0:
        raise ValueError("Rates must be in [0,100] and score_batch_size must be positive")
    rows = []
    eos = tokenizer.eos_token_id

    def completion_ids(text):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if eos is not None and (not ids or ids[-1] != eos):
            ids.append(eos)
        if not ids:
            raise ValueError("Empty perturbed completion without EOS")
        return ids

    def pad_completions(sequences, device):
        ids = torch.full((len(sequences), max(map(len, sequences))),
                         tokenizer.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, sequence in enumerate(sequences):
            ids[i, :len(sequence)] = torch.tensor(sequence, device=device)
            mask[i, :len(sequence)] = 1
        return ids, mask
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

            original_ids, texts = [], []
            for i in range(keep):
                pair = []
                for name in ("chosen", "rejected"):
                    ids = batch[f"{name}_input_ids"][i][batch[f"{name}_attention_mask"][i].bool()].tolist()
                    pair.append(ids)
                    texts.append(tokenizer.decode(ids, skip_special_tokens=True))
                original_ids.append(pair)
            docs = list(perturber.nlp.pipe(texts))
            variants = [perturber.variants(doc, rates) for doc in docs]
            chosen_ids, rejected_ids, actual_rates, changed_counts = [], [], [], []
            for i in range(keep):
                sample_rates, sample_counts = [], []
                for j, rate in enumerate(rates):
                    pair = [variants[2 * i][j], variants[2 * i + 1][j]]
                    # Zero replacements preserve exact original tokens/EOS.
                    ids = [original_ids[i][side] if item[2] == 0 else completion_ids(item[0])
                           for side, item in enumerate(pair)]
                    chosen_ids.append(ids[0])
                    rejected_ids.append(ids[1])
                    sample_rates.append([item[1] for item in pair])
                    sample_counts.append([item[2] for item in pair])
                actual_rates.append(sample_rates)
                changed_counts.append(sample_counts)
            chunks = []
            device = batch["prompt_input_ids"].device
            with torch.inference_mode():
                for start in range(0, len(chosen_ids), score_batch_size):
                    stop = min(start + score_batch_size, len(chosen_ids))
                    indices = torch.arange(start, stop, device=device) // len(rates)
                    pert_batch = {key: batch[key].index_select(0, indices)
                                  for key in ("prompt_input_ids", "prompt_attention_mask")}
                    for name, sequences in (("chosen", chosen_ids), ("rejected", rejected_ids)):
                        ids, mask = pad_completions(sequences[start:stop], device)
                        pert_batch[f"{name}_input_ids"] = ids
                        pert_batch[f"{name}_attention_mask"] = mask
                    target = trainer.concatenated_forward(model, pert_batch)
                    reference = trainer.concatenated_forward(ref_model, pert_batch)
                    chunks.append(torch.stack((target["chosen_logps"], target["rejected_logps"],
                                               reference["chosen_logps"], reference["rejected_logps"]),
                                              dim=-1).cpu())
            perturbation_scores = torch.cat(chunks).reshape(keep, len(rates), 4).tolist()
            batch_size = len(row["m_chosen"])
            for i in range(min(batch_size, max_samples - seen)):
                sample = {key: value[i] for key, value in row.items()}
                sample["label"] = label
                sample["pertu_score"] = perturbation_scores[i]
                sample["pert_rates"] = list(rates)
                sample["pert_actual_rates"] = actual_rates[i]
                sample["pert_changed_counts"] = changed_counts[i]
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
    perturber = SynonymPerturber(args.pert_nlp_model, args.pert_seed)
    warnings.warn("WordNet synonyms may change meaning; requested rates may exceed replaceable words. "
                  "TRL may truncate retokenized completions to its scoring length limit.")
    save_path = args.output or f"pert_{args.sft_data}_{args.dpo_data}_{args.model_name}.npy"
    trainer = build_dpo_trainer(target_model, tokenizer, train_ds, eval_ds)
    rows = collect_logprobs(
        trainer, target_model, ref_model, max_samples=max(args.n_member, args.n_nonmember),
        tokenizer=tokenizer, perturber=perturber, rates=args.pert_rates,
        score_batch_size=args.sem_generation_batch_size,
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    np.save(save_path, rows)
    print(f"Saved {len(rows)} samples to {save_path}")


if __name__ == "__main__":
    main()
