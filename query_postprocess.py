import argparse
import os
import re

import numpy as np

from eval_tools import clean_array, mia_auc_tpr, population_wrapper


VALID_NUMBER_RE = re.compile(r"^\s*[\[\(\{]?\s*(-1|0|1)\s*[\]\)\}]?\s*[\.,;:\(]?\s*$")

TASK_SPECIFIC_MAP = {
    "strengthener": 1,
    "weakener": -1,
    "smaller": -1,
}

EPS = 1e-8


def parse_response_position(text):
    lowered = str(text).strip().lower()
    if not lowered:
        return None
    if re.search(r"\bresponse[\s_:-]*1\b", lowered):
        return 1
    if re.search(r"\bresponse[\s_:-]*2\b", lowered):
        return 2
    if re.search(r"\brespond[\s_:-]*1\b", lowered):
        return 1
    if re.search(r"\brespond[\s_:-]*2\b", lowered):
        return 2
    return None


def parse_answer_number(text):
    lowered = str(text).strip().lower()
    match = re.search(r"\banswer\s*:?\s*(-1|0|1)\b", lowered)
    if match:
        return int(match.group(1))
    return None


def map_response_position(prompt_type, position):
    # prompt_file.py order:
    # 0 direct_satisfy:       Response 1 = chosen,   Response 2 = rejected
    # 1 direct_annotator:     Response 1 = chosen,   Response 2 = rejected
    # 2 direct_safe_helpful:  Response 1 = chosen,   Response 2 = rejected
    # 3 direct_rank:          Response 1 = chosen,   Response 2 = rejected
    # 4 direct_satisfy_swap:  Response 1 = rejected, Response 2 = chosen
    # 5 weakness:             1 if Response 2 has weaknesses, -1 if Response 1 has weaknesses
    # 6 weakness_swap:        1 if Response 1 has weaknesses, -1 if Response 2 has weaknesses
    if prompt_type in {0, 1, 2, 3}:
        return 1 if position == 1 else -1
    if prompt_type == 4:
        return -1 if position == 1 else 1
    if prompt_type == 5:
        return -1 if position == 1 else 1
    if prompt_type == 6:
        return 1 if position == 1 else -1
    return 0


def parse_output(text, prompt_type):
    text = str(text).strip()
    match = VALID_NUMBER_RE.match(text)
    if match:
        return int(match.group(1))

    answer = parse_answer_number(text)
    if answer is not None:
        return answer

    position = parse_response_position(text)
    if position is not None:
        return map_response_position(prompt_type, position)

    lowered = text.lower()
    for keyword, value in TASK_SPECIFIC_MAP.items():
        if keyword in lowered:
            return value

    return 0


def postprocess_outputs(outputs):
    n_prompt_types = 7
    n_queries = max(len(outputs) // n_prompt_types, 1)
    return [
        parse_output(output, (idx // n_queries) % n_prompt_types)
        for idx, output in enumerate(outputs)
    ]


def postprocess_rows(rows):
    processed = []
    for row in rows:
        processed.append({
            "label": int(row["label"]),
            "m_outputs": postprocess_outputs(row.get("m_outputs", [])),
            "r_outputs": postprocess_outputs(row.get("r_outputs", [])),
        })
    return processed


def output_distribution(outputs):
    outputs = np.asarray(outputs, dtype=np.int64)
    if len(outputs) == 0:
        return {
            "p1": 0.0,
            "p0": 1.0,
            "pn1": 0.0,
            "entropy": 0.0,
            "mean": 0.0,
        }

    p1 = float(np.mean(outputs == 1))
    p0 = float(np.mean(outputs == 0))
    pn1 = float(np.mean(outputs == -1))
    probs = np.asarray([p1, p0, pn1], dtype=np.float64)
    entropy = float(-np.sum(probs * np.log(probs + EPS)))
    return {
        "p1": p1,
        "p0": p0,
        "pn1": pn1,
        "entropy": entropy,
        "mean": float(np.mean(outputs)),
    }


def row_signals(row):
    m = np.asarray(row.get("m_outputs", []), dtype=np.int64)
    r = np.asarray(row.get("r_outputs", []), dtype=np.int64)
    m_dist = output_distribution(m)
    r_dist = output_distribution(r)

    n = min(len(m), len(r))
    if n > 0:
        m_pair = m[:n]
        r_pair = r[:n]
        agreement = float(np.mean(m_pair == r_pair))
        upgrade_rate = float(np.mean((r_pair != 1) & (m_pair == 1)))
        downgrade_rate = float(np.mean((r_pair == 1) & (m_pair <= 0)))
        transition_shift = upgrade_rate - downgrade_rate
    else:
        agreement = 0.0
        upgrade_rate = 0.0
        downgrade_rate = 0.0
        transition_shift = 0.0

    d_l1 = (
        abs(m_dist["p1"] - r_dist["p1"])
        + abs(m_dist["p0"] - r_dist["p0"])
        + abs(m_dist["pn1"] - r_dist["pn1"])
    )
    s_shift = (m_dist["p1"] - m_dist["pn1"]) - (r_dist["p1"] - r_dist["pn1"])
    s_odds = np.log((m_dist["p1"] + EPS) / (m_dist["pn1"] + EPS)) - np.log(
        (r_dist["p1"] + EPS) / (r_dist["pn1"] + EPS)
    )

    return {
        "p_m_1": m_dist["p1"],
        "p_m_0": m_dist["p0"],
        "p_m_n1": m_dist["pn1"],
        "entropy_m": m_dist["entropy"],
        "mean_m": m_dist["mean"],
        "delta_p_1": m_dist["p1"] - r_dist["p1"],
        "delta_p_0": m_dist["p0"] - r_dist["p0"],
        "delta_p_n1": m_dist["pn1"] - r_dist["pn1"],
        "delta_entropy": m_dist["entropy"] - r_dist["entropy"],
        "delta_mean": m_dist["mean"] - r_dist["mean"],
        "d_l1": d_l1,
        "s_shift": s_shift,
        "s_odds": float(s_odds),
        "agreement": agreement,
        "upgrade_rate": upgrade_rate,
        "s_transition": transition_shift,
    }


def collect_signal_arrays(rows):
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    signal_rows = [row_signals(row) for row in rows]
    names = list(signal_rows[0].keys()) if signal_rows else []
    signals = {
        name: clean_array([signal[name] for signal in signal_rows])
        for name in names
    }
    return labels, signals


def evaluate_signals(rows, popu=1, n_popu=1000):
    labels, signals = collect_signal_arrays(rows)
    results = []
    for name, scores in signals.items():
        eval_scores = scores
        eval_labels = labels
        if popu > 1:
            eval_scores, eval_labels = population_wrapper(scores, labels, popu, n_popu=n_popu)
        auc, tpr = mia_auc_tpr(eval_scores, eval_labels)
        results.append((name, auc, tpr, len(eval_scores)))
    return results


def default_output_path(input_path):
    root, ext = os.path.splitext(input_path)
    return f"{root}_post{ext or '.npy'}"


def main():
    parser = argparse.ArgumentParser(description="Postprocess query outputs into -1/0/1 labels")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--popu", type=int, default=1)
    parser.add_argument("--n_popu", type=int, default=1000)
    args = parser.parse_args()

    rows = np.load(args.input, allow_pickle=True)
    processed = postprocess_rows(rows)
    output_path = args.output or default_output_path(args.input)
    np.save(output_path, processed)
    print(f"Loaded {len(rows)} rows from {args.input}")
    print(f"Saved {len(processed)} postprocessed rows to {output_path}")
    print("=== Query Membership Signals ===")
    for name, auc, tpr, n in evaluate_signals(processed, popu=args.popu, n_popu=args.n_popu):
        print(f"{name:>16s}: AUROC={auc:.4f}, TPR@FPR=0.05={tpr:.4f}, n={n}")


if __name__ == "__main__":
    main()
