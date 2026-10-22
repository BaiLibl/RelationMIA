from datasets import load_dataset

from common import MODEL_DICT


def extract_from_chat(entry):
    prompt_parts = []
    response_parts = []
    for turn in entry:
        if turn["role"] == "user":
            prompt_parts.append(turn["content"])
        elif turn["role"] == "assistant":
            response_parts.append(turn["content"])
    return "\n".join(prompt_parts), "\n".join(response_parts)


def _limit_splits(train_raw, eval_raw, n_members=0, n_nonmembers=0):
    if n_members > 0 and n_nonmembers > 0:
        train_raw = train_raw.select(range(min(n_members, len(train_raw))))
        eval_raw = eval_raw.select(range(min(n_nonmembers, len(eval_raw))))
    return train_raw, eval_raw


def load_imdb_dataset(n_train, n_eval, for_dpo=False, n_members=0, n_nonmembers=0):
    dataset = load_dataset("csv", data_files="imdb_rlhf_pairs_iterfive.csv")["train"]
    train_raw = dataset.select(range(min(n_train, len(dataset))))
    eval_raw = dataset.select(range(n_train, min(n_train + n_eval, len(dataset))))
    train_raw, eval_raw = _limit_splits(train_raw, eval_raw, n_members, n_nonmembers)

    def fmt(row):
        prompt = row["Prompt"].rstrip()
        chosen = row["positive_response"][len(prompt):].strip()
        if for_dpo:
            rejected = row["negative_response"][len(prompt):].strip()
            return {"prompt": prompt + "\n", "chosen": chosen, "rejected": rejected}
        return {"prompt": prompt + "\n", "completion": " " + chosen.lstrip()}

    return (
        train_raw.map(fmt, remove_columns=train_raw.column_names),
        eval_raw.map(fmt, remove_columns=eval_raw.column_names),
    )


def load_uf_dataset(n_train, n_eval, for_dpo=False, n_members=0, n_nonmembers=0):
    dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    n_train = min(n_train, len(dataset))
    n_eval = min(n_eval, len(dataset) - n_train)
    train_raw = dataset.select(range(n_train))
    eval_raw = dataset.select(range(n_train, n_train + n_eval))
    train_raw, eval_raw = _limit_splits(train_raw, eval_raw, n_members, n_nonmembers)

    def fmt(example):
        prompt, chosen = extract_from_chat(example["chosen"])
        _, rejected = extract_from_chat(example["rejected"])
        if for_dpo:
            return {
                "prompt": prompt.rstrip() + "\n",
                "chosen": chosen.strip(),
                "rejected": rejected.strip(),
                "score_chosen": example["score_chosen"],
                "score_rejected": example["score_rejected"],
                "len_chosen": len(chosen.strip()),
                "len_rejected": len(rejected.strip()),
            }
        return {"prompt": prompt.rstrip() + "\n", "completion": " " + chosen.lstrip()}

    return (
        train_raw.map(fmt, remove_columns=train_raw.column_names),
        eval_raw.map(fmt, remove_columns=eval_raw.column_names),
    )


def load_stackexchange(n_train, n_eval, for_dpo=False, n_members=0, n_nonmembers=0):
    train_raw = load_dataset(
        "lvwerra/stack-exchange-paired", split="train", data_dir="data/rl"
    ).shuffle(seed=42)
    eval_raw = load_dataset(
        "lvwerra/stack-exchange-paired", split="train", data_dir="data/evaluation"
    ).shuffle(seed=42)
    train_raw = train_raw.select(range(min(n_train, len(train_raw))))
    eval_raw = eval_raw.select(range(min(n_eval, len(eval_raw))))
    train_raw, eval_raw = _limit_splits(train_raw, eval_raw, n_members, n_nonmembers)

    def fmt(row):
        if for_dpo:
            return {
                "prompt": row["question"].rstrip() + "\n",
                "chosen": row["response_j"].strip(),
                "rejected": row["response_k"].strip(),
                "score_chosen": 0,
                "score_rejected": 0,
                "len_chosen": len(row["response_j"].strip()),
                "len_rejected": len(row["response_k"].strip()),
            }
        return {"prompt": row["question"].rstrip() + "\n", "completion": " " + row["response_j"].lstrip()}

    columns = train_raw.column_names
    return (
        train_raw.map(fmt, remove_columns=columns),
        eval_raw.map(fmt, remove_columns=columns),
    )


def load_prism(n_train, n_eval, for_dpo=False, n_members=0, n_nonmembers=0):
    train_raw = load_dataset("json", data_files="../data/PRISM/train.jsonl", split="train")
    eval_raw = load_dataset("json", data_files="../data/PRISM/test.jsonl", split="train")

    train_raw = train_raw.select(range(min(n_train, len(train_raw))))
    eval_raw = eval_raw.select(range(min(n_eval, len(eval_raw))))
    train_raw, eval_raw = _limit_splits(train_raw, eval_raw, n_members, n_nonmembers)

    def fmt(row):
        prompt = row["prompt"].rstrip() + "\n"
        chosen = row["chosen"].strip()
        rejected = row["rejected"].strip()

        if for_dpo:
            return {
                "user_id": row["user_id"],
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "score_chosen": row["chosen_score"],
                "score_rejected": row["rejected_score"],
                # "len_chosen": len(chosen),
                # "len_rejected": len(rejected),
                # "age": row["age"],
                # "gender": row["gender"],
                # "employment_status": row["employment_status"],
                # "education": row["education"],
                # "marital_status": row["marital_status"],
            }

        return {"prompt": prompt, "completion": " " + chosen.lstrip()}

    return (
        train_raw.map(fmt, remove_columns=train_raw.column_names),
        eval_raw.map(fmt, remove_columns=eval_raw.column_names),
    )

DATASET_LOADERS = {
    "imdb": load_imdb_dataset,
    "uf": load_uf_dataset,
    "se": load_stackexchange,
    "prism": load_prism
}
