"""
Week 3 production-readiness scaffold.

Goal:
- Add safety rails, monitoring hooks, and review/abstain behavior.
- Define interfaces for deployable fact-aware classification service.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional


Decision = Literal["label", "needs_human_review"]


@dataclass
class ServiceInput:
    comment_id: str
    text: str


@dataclass
class ServiceOutput:
    decision: Decision
    label: Optional[str]
    confidence: float
    reason: str


def should_abstain(label_confidence: float, evidence_confidence: float, threshold: float = 0.6) -> bool:
    """Decide whether to abstain and route to human review."""
    combined = 0.5 * label_confidence + 0.5 * evidence_confidence
    return combined < threshold


def classify_with_guardrails(text: str) -> Dict[str, float]:
    """Placeholder classification response.

    TODO (week 3):
    - Integrate real model score.
    - Include evidence-aware confidence.
    """
    _ = text
    return {"label": "op_response", "label_confidence": 0.55, "evidence_confidence": 0.40}


def serve(req: ServiceInput) -> ServiceOutput:
    """Main service handler with abstain pathway."""
    out = classify_with_guardrails(req.text)
    label = str(out["label"])
    lc = float(out["label_confidence"])
    ec = float(out["evidence_confidence"])

    if should_abstain(lc, ec):
        return ServiceOutput(
            decision="needs_human_review",
            label=None,
            confidence=(lc + ec) / 2.0,
            reason="Low combined confidence from classifier and verifier.",
        )

    return ServiceOutput(
        decision="label",
        label=label,
        confidence=(lc + ec) / 2.0,
        reason="Automated decision above confidence threshold.",
    )


if __name__ == "__main__":
    sample = ServiceInput(comment_id="demo-1", text="This is a sample claim-heavy comment.")
    print(serve(sample))
