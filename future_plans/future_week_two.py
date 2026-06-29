"""
Week 2 evaluation harness scaffold.

Goal:
- Add repeatable evaluation for fact-check subcomponents and end-task labels.
- Track metrics stability across seeds/splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass
class PredictionRow:
    y_true: str
    y_pred: str


@dataclass
class ClaimMetricRow:
    claim_id: str
    gold_verdict: str
    pred_verdict: str


def compute_label_metrics(rows: Iterable[PredictionRow]) -> Dict[str, float]:
    """Placeholder for label-level metrics (accuracy, macro-F1, per-class F1)."""
    rows_list = list(rows)
    if not rows_list:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    correct = sum(1 for r in rows_list if r.y_true == r.y_pred)
    return {
        "accuracy": correct / len(rows_list),
        "macro_f1": 0.0,  # TODO (week 2): compute true macro-F1 per class.
    }


def compute_claim_metrics(rows: Iterable[ClaimMetricRow]) -> Dict[str, float]:
    """Placeholder for claim-level verification metrics.

    TODO (week 2):
    - Add precision/recall/F1 for supported/contradicted/insufficient.
    - Add calibration/error breakdown.
    """
    rows_list = list(rows)
    if not rows_list:
        return {"claim_accuracy": 0.0}

    correct = sum(1 for r in rows_list if r.gold_verdict == r.pred_verdict)
    return {"claim_accuracy": correct / len(rows_list)}


def run_seeded_evaluations(seeds: List[int]) -> Dict[int, Dict[str, float]]:
    """Run repeated seeded evals.

    TODO (week 2): wire this into real training/inference pipelines.
    """
    results: Dict[int, Dict[str, float]] = {}
    for seed in seeds:
        _ = seed
        # Placeholder empty run until integrated with real data.
        results[seed] = {"accuracy": 0.0, "macro_f1": 0.0}
    return results


if __name__ == "__main__":
    demo = [PredictionRow("analysis", "op_response"), PredictionRow("response", "response")]
    print(compute_label_metrics(demo))
    print(run_seeded_evaluations([42, 7, 123]))
