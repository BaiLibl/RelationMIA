"""User-disjoint text-only demographic AIA on preprocessed PRISM pairs.

pip install numpy pandas scikit-learn sentence-transformers joblib
python text-aia.py --train_input ../data/PRISM/train.jsonl --test_input ../data/PRISM/test.jsonl
"""

import os
import pandas as pd
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder


# TEXTS = ["prompt", "chosen", "rejected"]

def build_user_features(path):
    print("build feature .....")
    df = pd.read_json(path, lines=True)
    encoder = SentenceTransformer("BAAI/bge-m3", device="cuda" if torch.cuda.is_available() else "cpu")
    encoder.max_seq_length = 512

    texts = df["prompt"].tolist() + df["chosen"].tolist() + df["rejected"].tolist()
    emb = encoder.encode(texts, batch_size=64,
                         normalize_embeddings=True, convert_to_numpy=True)
    print("feature done")
    n = len(df)
    p, c, r = emb[:n], emb[n:2*n], emb[2*n:]
    row_feat = np.concatenate([p, c, r, c - r], axis=1)  # (n_rows, 4*dim)

    feat_df = pd.DataFrame(row_feat)
    feat_df.columns = [f"f{i}" for i in range(row_feat.shape[1])]
    feat_df["user_id"] = df["user_id"].to_numpy()

    return df, feat_df

def aggregate_by_k(df, feat_df, k=None):
     # 每个 user 只保留前 k 行
    if k is not None:
        feat_df = feat_df.groupby("user_id", sort=False).head(k)

    # 2. 同一 user 的 feature 做 mean pooling
    feat_cols = [c for c in feat_df.columns if c != "user_id"]
    user_feat = feat_df.groupby("user_id", sort=False)[feat_cols].mean()

    profile_cols = ["age", "gender", "employment_status", "education", "marital_status"]
    user_profile = df.groupby("user_id", sort=False)[profile_cols].first()

    out = user_feat.join(user_profile).reset_index()
    out["feature"] = out[feat_cols].apply(lambda row: row.to_numpy(dtype=np.float32), axis=1)
    out = out[["user_id", "feature", *profile_cols]]
    return out 

def run_attribute(train_df, test_df, label_name="age", k=None):

    # text label -> label number
    le = LabelEncoder()
    train_Y = le.fit_transform(train_df[label_name])
    test_Y = le.transform(test_df[label_name])
    n_classes = len(le.classes_)

    # list[np.ndarray] -> 2D array
    train_X = np.stack(train_df["feature"].to_numpy()).astype(np.float32)
    test_X = np.stack(test_df["feature"].to_numpy()).astype(np.float32)

    print(train_X.shape, test_X.shape)

    # 标准化（对 LogisticRegression 很重要，尤其 feature 各块尺度不同）
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    # LogisticRegression 多分类
    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        multi_class="multinomial",  # 多分类用 softmax
    )
    clf.fit(train_X, train_Y)

    # 预测
    pred_Y = clf.predict(test_X)
    prob_Y = clf.predict_proba(test_X)

    # 指标：F1 + AUC
    precision_macro = precision_score(test_Y, pred_Y, average="macro", zero_division=0)
    recall_macro = recall_score(test_Y, pred_Y, average="macro", zero_division=0)
    f1_macro = f1_score(test_Y, pred_Y, average="macro", zero_division=0)


    if n_classes == 2:
        # 二分类：正类概率
        auc = roc_auc_score(test_Y, prob_Y[:, 1])
    else:
        # 多分类：one-vs-rest 宏平均
        try:
            auc = roc_auc_score(test_Y, prob_Y, multi_class="ovr", average="macro")
        except ValueError:
            # 测试集某些类别缺失时 AUC 无法计算
            auc = float("nan")

    print(f"[{label_name}] F1(macro)={f1_macro:.4f} AUC={auc:.4f}")

    return {
        "label": label_name,
        "classes": len(le.classes_),
        "k":k,
        "f1_macro": f1_macro,
        "precision": precision_macro,
        "recall": recall_macro,
        "auc": auc,
    }


if __name__ == "__main__":
    ATTRIBUTES = ["age", "gender", "employment_status", "education", "marital_status"]
    CSV_PATH = "results.csv"

    def append_res(row, csv_path=CSV_PATH):
        df = pd.DataFrame([row])                       # ← 关键：列表包一层
        header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode="a", header=header, index=False, encoding="utf-8")

    train_df, train_feature_df = build_user_features("../data/PRISM/train.jsonl")
    test_df, test_feature_df = build_user_features("../data/PRISM/test.jsonl")

    for att in ATTRIBUTES:
        for k in [None, 2, 4, 8]:
            train_data = aggregate_by_k(train_df, train_feature_df, k)
            test_data  = aggregate_by_k(test_df, test_feature_df, k)
            res = run_attribute(train_data, test_data, att, k)
            append_res(res)