from dataclasses import dataclass
from difflib import SequenceMatcher

import torch


@dataclass
class SemanticQueryConfig:
    num_return_sequences: int = 4
    n_prompt_neighbors: int = 2
    n_chosen_neighbors: int = 2
    n_rejected_neighbors: int = 2
    semantic_threshold: float = 0.85
    edit_threshold: float = 0.10
    max_length: int = 1024
    max_probes_per_tuple: int = 64
    paraphrase_model: str = "facebook/bart-large-cnn"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    disable_generation: bool = False


def normalized_edit_distance(a, b):
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def unique_texts(texts):
    seen = set()
    out = []
    for text in texts:
        key = " ".join(str(text).strip().split())
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


class SemanticNeighborBuilder:
    def __init__(self, cfg):
        self.cfg = cfg
        self._generator = None
        self._generator_tokenizer = None
        self._embedder = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(self, text, limit):
        if limit <= 0:
            return []
        candidates = self.generate(text)
        candidates = self.filter(text, candidates)
        return candidates[:limit]

    def generate(self, text):
        if self.cfg.disable_generation or self.cfg.num_return_sequences <= 0:
            return []
        try:
            self._load_generator()
            inputs = self._generator_tokenizer(
                [text],
                return_tensors="pt",
                max_length=min(self.cfg.max_length, 1024),
                truncation=True,
            ).to(self._device)
            with torch.no_grad():
                outputs = self._generator.generate(
                    **inputs,
                    num_return_sequences=self.cfg.num_return_sequences,
                    num_beams=max(4, self.cfg.num_return_sequences),
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    max_length=min(256, max(32, len(text.split()) * 2)),
                    no_repeat_ngram_size=2,
                )
            return [
                self._generator_tokenizer.decode(x, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for x in outputs
            ]
        except Exception as exc:
            print(f"[sem-query] paraphrase generation failed; using original only. Error: {exc}")
            return []

    def filter(self, original, candidates):
        candidates = unique_texts(candidates)
        if not candidates:
            return []

        out = []
        orig_emb = None
        try:
            self._load_embedder()
            orig_emb = self._embedder.encode(original, convert_to_tensor=True)
        except Exception as exc:
            print(f"[sem-query] semantic filter unavailable; using edit filter only. Error: {exc}")

        for candidate in candidates:
            edit_dist = normalized_edit_distance(original, candidate)
            if edit_dist < self.cfg.edit_threshold:
                continue

            if orig_emb is not None:
                cand_emb = self._embedder.encode(candidate, convert_to_tensor=True)
                sim = torch.nn.functional.cosine_similarity(orig_emb, cand_emb, dim=0).item()
                if sim < self.cfg.semantic_threshold:
                    continue

            out.append(candidate)
        return out

    def _load_generator(self):
        if self._generator is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._generator_tokenizer = AutoTokenizer.from_pretrained(self.cfg.paraphrase_model)
        self._generator = AutoModelForSeq2SeqLM.from_pretrained(
            self.cfg.paraphrase_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).to(self._device)
        self._generator.eval()

    def _load_embedder(self):
        if self._embedder is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self.cfg.embed_model, device=str(self._device))


def build_semantic_relation_queries(sample, neighbor_builder, cfg):
    return {
        "prompt_neighbors": [sample["prompt"]]
        + neighbor_builder.build(sample["prompt"], cfg.n_prompt_neighbors),
        "chosen_neighbors": [sample["chosen"]]
        + neighbor_builder.build(sample["chosen"], cfg.n_chosen_neighbors),
        "rejected_neighbors": [sample["rejected"]]
        + neighbor_builder.build(sample["rejected"], cfg.n_rejected_neighbors),
    }


def build_semantic_queries(samples, cfg):
    neighbor_builder = SemanticNeighborBuilder(cfg)
    rows = []
    for sample in samples:
        row = build_semantic_relation_queries(sample, neighbor_builder, cfg)
        row["label"] = int(sample.get("label", -1))
        rows.append(row)
    return rows
