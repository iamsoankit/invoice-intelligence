"""Evaluation metrics over the labeled dataset in evals/dataset.jsonl."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import PACKAGE_DIR
from .schemas import PipelineResult

DATASET_PATH = PACKAGE_DIR.parent / "evals" / "dataset.jsonl"


def load_dataset(path: Path = DATASET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _matches(expected, actual) -> bool:
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return abs(float(expected) - float(actual)) < 0.01
        except (TypeError, ValueError):
            return False
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    return expected == actual


@dataclass
class CaseScore:
    id: str
    expected_tier: str
    predicted_tier: str
    field_hits: int
    field_total: int
    policy_tp: int
    policy_fp: int
    policy_fn: int
    latency_ms: float
    tokens: int
    field_misses: list[str] = field(default_factory=list)

    @property
    def tier_correct(self) -> bool:
        return self.expected_tier == self.predicted_tier


def score_case(case: dict, result: PipelineResult) -> CaseScore:
    expected = case["expected"]
    extracted = result.extraction.model_dump()
    misses = [k for k, v in expected["fields"].items() if not _matches(v, extracted.get(k))]
    predicted_ids = {pid for f in result.audit.findings for pid in f.policy_ids}
    expected_ids = set(expected["policy_ids"])
    return CaseScore(
        id=case["id"],
        expected_tier=expected["risk_tier"],
        predicted_tier=result.audit.risk_tier.value,
        field_hits=len(expected["fields"]) - len(misses),
        field_total=len(expected["fields"]),
        policy_tp=len(predicted_ids & expected_ids),
        policy_fp=len(predicted_ids - expected_ids),
        policy_fn=len(expected_ids - predicted_ids),
        latency_ms=result.total_latency_ms,
        tokens=result.total_tokens,
        field_misses=misses,
    )


def summarize(scores: list[CaseScore]) -> dict:
    tp = sum(s.policy_tp for s in scores)
    fp = sum(s.policy_fp for s in scores)
    fn = sum(s.policy_fn for s in scores)
    n = max(len(scores), 1)
    return {
        "cases": len(scores),
        "tier_accuracy": sum(s.tier_correct for s in scores) / n,
        "field_accuracy": sum(s.field_hits for s in scores) / max(sum(s.field_total for s in scores), 1),
        "policy_precision": tp / (tp + fp) if tp + fp else 1.0,
        "policy_recall": tp / (tp + fn) if tp + fn else 1.0,
        "avg_latency_ms": sum(s.latency_ms for s in scores) / n,
        "total_tokens": sum(s.tokens for s in scores),
    }
