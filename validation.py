from typing import Any, Dict, Iterable, List, Set

import numpy as np
import pandas as pd


def _parse_skill_cell(value: Any) -> Set[str]:
    if value is None:
        return set()
    if isinstance(value, (set, list, tuple)):
        return {str(item).strip() for item in value if str(item).strip()}
    text = str(value).strip()
    if not text:
        return set()
    parts = [item.strip() for item in text.replace(";", ",").split(",")]
    return {item for item in parts if item}


def evaluate_skill_precision_recall(records: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    precisions = []
    recalls = []
    f1s = []

    for record in records:
        predicted = _parse_skill_cell(record.get("predicted_skills"))
        ground_truth = _parse_skill_cell(record.get("ground_truth_skills"))
        if not predicted and not ground_truth:
            continue

        tp = len(predicted.intersection(ground_truth))
        fp = len(predicted - ground_truth)
        fn = len(ground_truth - predicted)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    if not precisions:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "samples": 0}

    return {
        "precision": round(float(np.mean(precisions)), 4),
        "recall": round(float(np.mean(recalls)), 4),
        "f1": round(float(np.mean(f1s)), 4),
        "samples": len(precisions),
    }


def evaluate_score_callback_correlation(records: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    scores = []
    callbacks = []
    for record in records:
        score = record.get("overall_score")
        callback = record.get("callback_outcome")
        if score is None or callback is None:
            continue
        try:
            scores.append(float(score))
            callbacks.append(float(callback))
        except Exception:
            continue

    if len(scores) < 3:
        return {"correlation": 0.0, "samples": len(scores)}

    corr = float(np.corrcoef(scores, callbacks)[0, 1])
    if np.isnan(corr):
        corr = 0.0

    return {"correlation": round(corr, 4), "samples": len(scores)}


def evaluate_from_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    records = df.to_dict("records")
    skill_metrics = evaluate_skill_precision_recall(records)
    callback_metrics = evaluate_score_callback_correlation(records)
    return {
        "skill_matching": skill_metrics,
        "callback_correlation": callback_metrics,
    }
