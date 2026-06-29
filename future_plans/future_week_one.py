"""
Week 1 MVP scaffold: claim extraction + retrieval + simple verification hooks.

Goal:
- Build a minimal fact-aware pipeline around the existing classifier workflow.
- Keep this module standalone so it can be iterated without breaking current app/training code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional


Verdict = Literal["supported", "contradicted", "insufficient"]


@dataclass
class Claim:
    text: str


@dataclass
class Evidence:
    source: str
    title: str
    snippet: str
    url: str


@dataclass
class ClaimCheckResult:
    claim: Claim
    verdict: Verdict
    confidence: float
    evidence: List[Evidence]


def extract_claims(comment: str) -> List[Claim]:
    """Extract potentially checkable claims from a comment.

    TODO (week 1):
    - Replace this placeholder with real claim extraction (regex + NER or LLM-based parser).
    - Split multi-claim sentences into atomic claims.
    """
    text = comment.strip()
    if not text:
        return []
    return [Claim(text=text)]


def retrieve_evidence(claim: Claim, top_k: int = 3) -> List[Evidence]:
    """Retrieve evidence snippets from trusted sources.

    TODO (week 1):
    - Implement Wikipedia retrieval (REST API/search endpoint).
    - Optionally add one more source and rank by relevance.
    """
    _ = top_k
    return []


def verify_claim(claim: Claim, evidence: List[Evidence]) -> ClaimCheckResult:
    """Map claim + evidence to a verification verdict.

    TODO (week 1):
    - Start with a simple heuristic baseline.
    - Upgrade later to NLI/entailment model scoring.
    """
    verdict: Verdict = "insufficient" if not evidence else "supported"
    confidence = 0.0 if not evidence else 0.5
    return ClaimCheckResult(claim=claim, verdict=verdict, confidence=confidence, evidence=evidence)


def run_week_one_pipeline(comment: str) -> List[ClaimCheckResult]:
    """End-to-end MVP pipeline for one comment."""
    claims = extract_claims(comment)
    results: List[ClaimCheckResult] = []
    for claim in claims:
        evidence = retrieve_evidence(claim)
        results.append(verify_claim(claim, evidence))
    return results


if __name__ == "__main__":
    sample = "Jefferson Davis was president of the Confederacy during the Civil War."
    out = run_week_one_pipeline(sample)
    print(f"claims_checked={len(out)}")
    for row in out:
        print(f"- claim={row.claim.text!r} verdict={row.verdict} confidence={row.confidence}")
