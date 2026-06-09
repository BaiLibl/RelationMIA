from dataclasses import dataclass
from difflib import SequenceMatcher
from textwrap import dedent

import torch


@dataclass
class SemanticQueryConfig:
    num_return_sequences: int = 4
    n_prompt_neighbors: int = 4
    n_chosen_neighbors: int = 4
    n_rejected_neighbors: int = 4
    semantic_threshold: float = 0.80
    edit_threshold: float = 0.20 # higher means more different
    max_length: int = 1024
    max_new_tokens: int = 256
    max_probes_per_tuple: int = 64
    sem_generation_batch_size: int = 8
    paraphrase_model: str = "Qwen/Qwen2.5-3B-Instruct"
    embed_model: str = "sentence-transformers/all-mpnet-base-v2"
    rewrite_profile: str = "balanced"
    temperature: float = 0.8
    min_length_ratio: float = 0.5
    max_length_ratio: float = 2.0
    top_p: float = 0.9
    disable_generation: bool = False
    debug: bool = False
    debug_max_chars: int = 500


def normalized_edit_distance(a, b):
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()

def length_ratio(original, candidate):
    orig_len = max(1, len(str(original).split()))
    cand_len = max(1, len(str(candidate).split()))
    return cand_len / orig_len


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

    def build(self, text, limit, field_name="text"):
        return self.build_many([(text, limit, field_name)])[0]

    def build_many(self, items):
        if not items:
            return []

        candidates_list = self.generate_many([(text, field_name) for text, _, field_name in items])
        outputs = []
        for (text, limit, field_name), candidates in zip(items, candidates_list):
            if limit <= 0:
                outputs.append([])
                continue
            if self.cfg.debug:
                self._print_candidates(field_name, text, candidates, title="RAW GENERATED")
            filtered = self.filter(text, candidates, field_name=field_name)
            kept = filtered[:limit]
            if self.cfg.debug:
                self._print_candidates(field_name, text, kept, title="FILTERED KEPT")
            outputs.append(kept)
        return outputs

    def generate(self, text, field_name="text"):
        return self.generate_many([(text, field_name)])[0]

    def generate_many(self, items):
        if self.cfg.disable_generation or self.cfg.num_return_sequences <= 0:
            return [[] for _ in items]
        if not items:
            return []
        try:
            self._load_generator()
            prompts = [self._build_rewrite_prompt(text, field_name) for text, field_name in items]
            inputs = self._encode_instruction_many(prompts)
            with torch.no_grad():
                outputs = self._generator.generate(
                    **inputs,
                    num_return_sequences=self.cfg.num_return_sequences,
                    do_sample=True,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    max_new_tokens=self.cfg.max_new_tokens,
                    pad_token_id=self._generator_tokenizer.pad_token_id,
                    eos_token_id=self._generator_tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[-1]
            decoded = []
            for output in outputs:
                new_tokens = output[input_len:]
                text_out = self._generator_tokenizer.decode(new_tokens, skip_special_tokens=True)
                decoded.append(self._clean_generation(text_out))

            grouped = []
            n = self.cfg.num_return_sequences
            for idx in range(len(items)):
                grouped.append(decoded[idx * n: (idx + 1) * n])
            return grouped
        except Exception as exc:
            print(f"[sem-query] rewrite generation failed; using original only. Error: {exc}")
            return [[] for _ in items]

    def filter(self, original, candidates, field_name="text"):
        candidates = unique_texts(candidates)
        if not candidates:
            return []

        out = []
        similarities = [None] * len(candidates)
        try:
            self._load_embedder()
            embeddings = self._embedder.encode([original] + candidates, convert_to_tensor=True)
            orig_emb = embeddings[0].unsqueeze(0)
            cand_embs = embeddings[1:]
            similarities = torch.nn.functional.cosine_similarity(orig_emb, cand_embs, dim=1).detach().cpu().tolist()
        except Exception as exc:
            print(f"[sem-query] semantic filter unavailable; using edit/length filters only. Error: {exc}")

        for idx, candidate in enumerate(candidates):
            edit_dist = normalized_edit_distance(original, candidate)
            ratio = length_ratio(original, candidate)
            sim = similarities[idx]
            keep = True
            reasons = []

            if edit_dist < self.cfg.edit_threshold:
                keep = False
                reasons.append(f"edit<{self.cfg.edit_threshold:.3f}")
            # if ratio < self.cfg.min_length_ratio:
            #     keep = False
            #     reasons.append(f"len_ratio<{self.cfg.min_length_ratio:.2f}")
            # if ratio > self.cfg.max_length_ratio:
            #     keep = False
            #     reasons.append(f"len_ratio>{self.cfg.max_length_ratio:.2f}")

            if sim is not None and sim < self.cfg.semantic_threshold:
                keep = False
                reasons.append(f"sim<{self.cfg.semantic_threshold:.3f}")

            if self.cfg.debug:
                status = "KEEP" if keep else "DROP"
                reason = ", ".join(reasons) if reasons else "pass"
                sim_text = "NA" if sim is None else f"{sim:.4f}"
                print(
                    f"[sem-query][{field_name}] candidate={idx} status={status} "
                    f"sim={sim_text} edit={edit_dist:.4f} len_ratio={ratio:.3f} reason={reason}"
                )
                self._print_text("candidate", candidate)

            if keep:
                out.append(candidate)
        return out

    def _build_rewrite_prompt(self, text, field_name):
        if field_name == "prompt":
            instruction = """
            Your task is to rewrite the user prompt below. The rewrite must be semantically equivalent to the original prompt.

            Meaning preservation requirements:
            - Preserve the exact task type, user intent, constraints, entities, numbers, examples, language requirements, and expected output format.
            - Preserve all answer choices, labels, code words, quoted text, variable names, and input data exactly unless only their surrounding wording is being changed.
            - Preserve the same scope: do not make the task broader, narrower, easier, harder, safer, more polite, or more detailed.
            - Preserve polarity and modality: do not change yes/no, true/false, allowed/not allowed, should/should not, can/cannot, or any negation.

            Prohibited changes:
            - Do not answer the prompt.
            - Do not continue, complete, solve, explain, summarize, or add examples.
            - Do not add or remove facts, requirements, constraints, options, entities, numbers, code, or conclusions.
            - Do not introduce new topics, assumptions, caveats, links, emails, comments, or metadata.
            - Do not focus only on the last question. Preserve all examples / QA pairs / options / labels.

            Before outputting, internally check that the rewrite has exactly the same meaning and constraints as the original.
            If an exact meaning-preserving rewrite is not possible, output the original prompt unchanged.
            Return only the rewritten prompt.

            User prompt:
            {text}
            """
        else:
            instruction = """
            Your task is to rewrite the answer below. The rewrite must be semantically equivalent to the original answer and must preserve its quality level.

            Meaning and quality preservation requirements:
            - Preserve the exact meaning, factual claims, conclusion, uncertainty, specificity, tone, and style.
            - Preserve all mistakes, omissions, ambiguity, awkwardness, incompleteness, confidence statements, and low-quality aspects.
            - Preserve polarity and final answer: do not change yes/no, true/false, correct/incorrect, expensive/not expensive, allowed/not allowed, can/cannot, or any negation.
            - Preserve numbers, units, names, options, code, equations, citations, quoted text, and formatting-sensitive content.
            - If the original answer is wrong, keep it wrong in the same way. If it is vague, keep it vague. If it is incomplete, keep it incomplete.

            Prohibited changes:
            - Do not answer the original question.
            - Do not continue, complete, summarize, expand, explain, improve, correct, clarify, or make the answer more helpful.
            - Do not add or remove facts, examples, code, numbers, citations, links, conclusions, or caveats.
            - Do not change the answer's usefulness, correctness, safety, politeness, or level of detail.
            - Do not change final answer or polarity.

            Before outputting, internally check that the rewrite has exactly the same meaning, final answer, and quality level as the original.
            If an exact meaning-preserving rewrite is not possible, output the original answer unchanged.
            Return only the rewritten answer.

            Answer:
            {text}
            """

        return dedent(instruction).strip().format(text=text)

    def _instruction_messages(self, prompt):
        return [
            {
                "role": "system",
                "content": (
                    "You are a strict meaning-preserving rewriting engine. Your only job is to change wording "
                    "without changing semantics, polarity, facts, constraints, final answers, mistakes, or quality level. "
                    "If preservation is uncertain, copy the original text unchanged."
                ),
            },
            {"role": "user", "content": prompt},
        ]

    def _encode_instruction(self, prompt):
        return self._encode_instruction_many([prompt])

    def _encode_instruction_many(self, prompts):
        if hasattr(self._generator_tokenizer, "apply_chat_template"):
            texts = [
                self._generator_tokenizer.apply_chat_template(
                    self._instruction_messages(prompt),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]
        else:
            texts = ["\n\n".join(m["content"] for m in self._instruction_messages(prompt)) for prompt in prompts]

        return self._generator_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
        ).to(self._device)

    def _clean_generation(self, text):
        text = str(text).strip()
        for prefix in ["Rewritten prompt:", "Rewritten answer:", "Rewrite:", "Paraphrase:"]:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ['"', "'"]:
            text = text[1:-1].strip()
        return text

    def _print_text(self, name, text):
        text = " ".join(str(text).split())
        if len(text) > self.cfg.debug_max_chars:
            text = text[: self.cfg.debug_max_chars] + " ..."
        print(f"  [{name}] {text}")

    def _print_candidates(self, field_name, original, candidates, title):
        print("=" * 100)
        print(f"[sem-query][{field_name}] {title}: n={len(candidates)}")
        self._print_text("original", original)
        for idx, candidate in enumerate(candidates):
            self._print_text(f"candidate {idx}", candidate)

    def _load_generator(self):
        if self._generator is not None:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._generator_tokenizer = AutoTokenizer.from_pretrained(self.cfg.paraphrase_model, use_fast=False)
        if self._generator_tokenizer.pad_token is None:
            self._generator_tokenizer.pad_token = self._generator_tokenizer.eos_token
        self._generator_tokenizer.padding_side = "left"
        self._generator = AutoModelForCausalLM.from_pretrained(
            self.cfg.paraphrase_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self._generator.eval()

    def _load_embedder(self):
        if self._embedder is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._embedder = SentenceTransformer(self.cfg.embed_model, device=str(self._device))


def build_semantic_relation_queries(sample, neighbor_builder, cfg):
    return {
        "prompt_neighbors": [sample["prompt"]]
        + neighbor_builder.build(sample["prompt"], cfg.n_prompt_neighbors, field_name="prompt"),
        "chosen_neighbors": [sample["chosen"]]
        + neighbor_builder.build(sample["chosen"], cfg.n_chosen_neighbors, field_name="chosen"),
        "rejected_neighbors": [sample["rejected"]]
        + neighbor_builder.build(sample["rejected"], cfg.n_rejected_neighbors, field_name="rejected"),
    }


def build_semantic_relation_queries_batch(samples, neighbor_builder, cfg):
    rows = []
    work_items = []
    for sample_idx, sample in enumerate(samples):
        rows.append({
            "prompt_neighbors": [sample["prompt"]],
            "chosen_neighbors": [sample["chosen"]],
            "rejected_neighbors": [sample["rejected"]],
        })
        fields = [
            ("prompt", sample["prompt"], cfg.n_prompt_neighbors),
            ("chosen", sample["chosen"], cfg.n_chosen_neighbors),
            ("rejected", sample["rejected"], cfg.n_rejected_neighbors),
        ]
        for field_name, text, limit in fields:
            if limit > 0:
                work_items.append((sample_idx, field_name, text, limit))

    batch_size = max(1, int(cfg.sem_generation_batch_size))
    for start in range(0, len(work_items), batch_size):
        batch = work_items[start: start + batch_size]
        built = neighbor_builder.build_many([(text, limit, field_name) for _, field_name, text, limit in batch])
        for (sample_idx, field_name, _, _), neighbors in zip(batch, built):
            rows[sample_idx][f"{field_name}_neighbors"].extend(neighbors)
    return rows


def build_semantic_queries(samples, cfg):
    neighbor_builder = SemanticNeighborBuilder(cfg)
    rows = build_semantic_relation_queries_batch(samples, neighbor_builder, cfg)
    for row, sample in zip(rows, samples):
        row["label"] = int(sample.get("label", -1))
    return rows
