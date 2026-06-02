import numpy as np
from sklearn import metrics


def clean_array(values, clip=1e4):
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)

    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=np.max(x[finite]),
        neginf=np.min(x[finite]),
    )
    return np.clip(x, -clip, clip)


def mia_auc_tpr(scores, labels, fpr_threshold=0.05):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    auc = metrics.auc(fpr, tpr)
    if auc < 0.5:
        fpr, tpr, _ = metrics.roc_curve(labels, -scores)
        auc = metrics.auc(fpr, tpr)

    idx = np.searchsorted(fpr, fpr_threshold, side="right") - 1
    idx = max(idx, 0)
    return auc, tpr[idx]


def to_population_level(scores, popu, n_popu=1000, seed=42):
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=np.float64)
    return np.array([scores[rng.choice(len(scores), size=popu, replace=True)].mean() for _ in range(n_popu)])


def population_wrapper(values, labels, popu, n_popu=1000):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    member = to_population_level(values[labels == 1], popu, n_popu=n_popu)
    nonmember = to_population_level(values[labels == 0], popu, n_popu=n_popu)
    return np.concatenate([member, nonmember]), np.concatenate([np.ones(len(member)), np.zeros(len(nonmember))])


def premia_scores(m_chosen, m_rejected, r_chosen, r_rejected):
    chosen = m_chosen - r_chosen
    rejected = r_rejected - m_rejected
    pair = (m_chosen - m_rejected) - (r_chosen - r_rejected)
    return {
        "chosen": chosen,
        "rejected": rejected,
        "pair": pair,
    }
