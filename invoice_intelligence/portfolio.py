"""Aggregate many audits into a portfolio view for the batch screen."""

from __future__ import annotations

from collections import Counter

from .schemas import PipelineResult, RiskTier


def portfolio_rows(named_results: list[tuple[str, PipelineResult]]) -> list[dict]:
    rows = []
    for name, r in named_results:
        top = r.audit.findings[0].title if r.audit.findings else "—"
        rows.append({
            "document": name,
            "vendor": r.extraction.vendor_name or "—",
            "amount": r.extraction.total_amount,
            "currency": r.extraction.currency or "",
            "tier": r.audit.risk_tier.value,
            "score": r.audit.risk_score,
            "findings": len(r.audit.findings),
            "policies": ", ".join(dict.fromkeys(p for f in r.audit.findings for p in f.policy_ids)),
            "top_issue": top,
            "action": "HOLD" if r.audit.risk_tier is RiskTier.CRITICAL
            else "REVIEW" if r.audit.risk_tier is RiskTier.MODERATE else "PAY",
        })
    return sorted(rows, key=lambda row: row["score"], reverse=True)


def portfolio_summary(rows: list[dict]) -> dict:
    tiers = Counter(row["tier"] for row in rows)
    held: Counter = Counter()
    for row in rows:
        if row["action"] == "HOLD" and row["amount"]:
            held[row["currency"] or "N/A"] += row["amount"]
    policy_hits = Counter(p.strip() for row in rows for p in row["policies"].split(",") if p.strip())
    return {
        "documents": len(rows),
        "tiers": {t.value: tiers.get(t.value, 0) for t in RiskTier},
        # Keyed by currency: amounts in different currencies are never summed together.
        "amount_on_hold": dict(held.most_common()),
        "top_policies": policy_hits.most_common(5),
    }
