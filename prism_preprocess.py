from datasets import load_dataset
from pathlib import Path
import pandas as pd

DATA_DIR = Path("./data/PRISM")
DATA_DIR.mkdir(parents=True, exist_ok=True)

REPO = "HannahRoseKirk/prism-alignment"

for config in ["survey", "conversations", "utterances"]:
    print(f"\nDownloading {config}...")

    ds = load_dataset(REPO, config, split="train")

    output_file = DATA_DIR / f"{config}.jsonl"
    ds.to_json(str(output_file), force_ascii=False)

    print(f"Saved: {output_file}")
    print(f"Rows: {len(ds)}")
    print(f"Columns: {ds.column_names}")


survey = pd.read_json(DATA_DIR / "survey.jsonl", lines=True)
conversations = pd.read_json(DATA_DIR / "conversations.jsonl", lines=True)
utterances = pd.read_json(DATA_DIR / "utterances.jsonl", lines=True)

datasets = {
    "survey": survey,
    "conversations": conversations,
    "utterances": utterances,
}

# PRISM demographic attributes
demographic_cols = [
    "user_id",
    "age",
    "gender",
    "employment_status",
    "education",
    "marital_status",
]

# 提取用户人口统计属性
user_demographics = survey[demographic_cols].copy()

# 关联 utterances
utterances = utterances.merge(
    user_demographics,
    on="user_id",
    how="left",
    validate="many_to_one",
)


def build_dpo_pairs(utterances):
    df = utterances.copy()

    # 确保关键字段存在
    required_cols = [
        "user_id",
        "conversation_id",
        "turn",
        "interaction_id",
        "user_prompt",
        "model_response",
        "score",
        "if_chosen",
    ]

    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 过滤无效数据
    df = df.dropna(
        subset=[
            "user_id",
            "user_prompt",
            "model_response",
            "if_chosen",
        ]
    )

    # 转换评分为数值
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # 确保 if_chosen 是布尔值
    df = df[df["if_chosen"].isin([True, False])]

    # 同一次交互的候选回答
    group_cols = [
        "user_id",
        "conversation_id",
        "turn",
        "interaction_id",
    ]

    records = []

    for _, group in df.groupby(group_cols, dropna=False):

        chosen_rows = group[group["if_chosen"] == True]
        rejected_rows = group[group["if_chosen"] == False]

        # 要求恰好一个 chosen，至少一个 rejected
        if len(chosen_rows) != 1 or len(rejected_rows) == 0:
            continue

        chosen_row = chosen_rows.iloc[0]

        for _, rejected_row in rejected_rows.iterrows():

            # 保证 chosen/rejected 使用同一个 prompt
            if (
                chosen_row["user_prompt"]
                != rejected_row["user_prompt"]
            ):
                continue

            # 排除完全相同的回答
            if (
                chosen_row["model_response"]
                == rejected_row["model_response"]
            ):
                continue

            records.append({
                "user_id": chosen_row["user_id"],
                "prompt": chosen_row["user_prompt"],
                "chosen": chosen_row["model_response"],
                "rejected": rejected_row["model_response"],
                "chosen_score": chosen_row["score"],
                "rejected_score": rejected_row["score"],
            })

    dpo_df = pd.DataFrame(
        records,
        columns=[
            "user_id",
            "prompt",
            "chosen",
            "rejected",
            "chosen_score",
            "rejected_score",
            "age",
            "gender",
            "employment_status",
            "education",
            "marital_status",
        ],
    )

    return dpo_df


# 构造 DPO 数据
dpo_df = build_dpo_pairs(utterances)

print("DPO pairs:123", len(dpo_df))
print("Users:", dpo_df["user_id"].nunique())
print(dpo_df.head())

# 保存
dpo_df.to_json(
    "./data/PRISM/dpo_pairs.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)
