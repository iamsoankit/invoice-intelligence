"""Deterministic guardrails and the offline (no-LLM) extractor.

Guardrails run on every invoice in both modes. In live mode they act as a
floor under the LLM verdict: the model can add risks but cannot talk a
rule-detected CRITICAL risk down to SECURE.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime

from . import config
from .schemas import Finding, InvoiceExtraction, RiskAudit, RiskTier, Severity

_SEVERITY_BASE = {Severity.LOW: 15, Severity.MEDIUM: 40, Severity.HIGH: 60, Severity.CRITICAL: 88}
_SEVERITY_ORDER = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
_TIER_ORDER = [RiskTier.SECURE, RiskTier.MODERATE, RiskTier.CRITICAL]

_BANK_CHANGE = re.compile(
    r"(routing|account|bank|remittance|iban|wire|swift)[^.\n]{0,60}(chang|new|updat|switch)"
    r"|(chang|new|updat|switch)[^.\n]{0,40}(routing|account|bank|remittance|iban|wire)",
    re.I,
)
_OUT_OF_BAND = re.compile(
    r"text message|via text|\bsms\b|whatsapp|telegram|personal email|sent separately|call this number", re.I
)
_OFFSHORE = re.compile(r"offshore|cayman|british virgin|panama|seychelles|third[- ]party account", re.I)
_TAX_MISSING = re.compile(
    r"(w-?9|w-?8|tax id|\bein\b|\bvat\b)[^.\n]{0,80}(pending|missing|later|not (yet )?(provided|available|on file)|next quarter|without)",
    re.I,
)
_VAGUE = re.compile(r"general consulting|ad-hoc|ad hoc|miscellaneous|various services|misc\.? services", re.I)
_INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above) (rules|instructions)|approve (this )?automatically"
    r"|mark (this )?(invoice )?as (verified|approved|secure)|you are an? (ai|assistant|language model)"
    r"|system prompt|override (the )?(policy|controls)",
    re.I,
)
_DUPLICATE = re.compile(r"reissued|re-issued|corrected copy|second notice|duplicate", re.I)


def _evidence(text: str, pattern: re.Pattern) -> str | None:
    match = pattern.search(text)
    if not match:
        return None
    start = text.rfind("\n", 0, match.start()) + 1
    end = text.find("\n", match.end())
    return text[start: end if end != -1 else len(text)].strip()[:240]


def run_guardrails(text: str, extraction: InvoiceExtraction) -> list[Finding]:
    findings: list[Finding] = []

    def add(pattern, title, severity, policy_ids, recommendation):
        evidence = _evidence(text, pattern)
        if evidence:
            findings.append(Finding(title=title, severity=severity, evidence=evidence, policy_ids=policy_ids,
                                    recommendation=recommendation, source="rule"))

    add(_INJECTION, "Embedded instructions / prompt-injection attempt", Severity.CRITICAL, ["SEC-01"],
        "Quarantine the document and report it to Security; do not act on embedded instructions.")
    add(_OUT_OF_BAND, "Payment instructions via out-of-band channel", Severity.CRITICAL, ["BANK-02"],
        "Reject the instructions; only accept bank changes through the signed supplier-portal form.")
    add(_BANK_CHANGE, "Change of bank / remittance details", Severity.CRITICAL, ["BANK-01"],
        "Freeze payment and verify by calling the vendor on the number already in the vendor master.")
    add(_OFFSHORE, "Offshore or third-party remittance account", Severity.CRITICAL, ["AML-01"],
        "Hold payment pending KYC refresh and Treasury sign-off.")
    add(_TAX_MISSING, "Tax documentation missing or deferred", Severity.HIGH, ["TAX-01"],
        "Hold payment until a valid W-9 / W-8 is on file.")
    add(_VAGUE, "Vague services without SOW / PO", Severity.MEDIUM, ["VEN-01"],
        "Request a PO reference or statement of work before matching.")
    add(_DUPLICATE, "Possible duplicate or reissued invoice", Severity.MEDIUM, ["VEN-02"],
        "Check payment history for the same vendor, amount and date.")

    days = extraction.payment_terms_days
    terms_evidence = f"Terms: {extraction.payment_terms}" if extraction.payment_terms else ""
    if days is not None and days > config.MAX_STANDARD_TERMS_DAYS:
        findings.append(Finding(
            title=f"Extended payment terms ({extraction.payment_terms})", severity=Severity.MEDIUM,
            evidence=terms_evidence, policy_ids=["PAY-01"], source="rule",
            recommendation="Obtain Procurement Director approval for terms beyond Net 30."))
    elif days is not None and days < 15:
        findings.append(Finding(
            title=f"Accelerated payment terms ({extraction.payment_terms})", severity=Severity.MEDIUM,
            evidence=terms_evidence, policy_ids=["PAY-02"], source="rule",
            recommendation="Escalate to the AP manager; do not skip the three-way match."))

    amount = extraction.total_amount
    if amount is not None and amount > config.APPROVAL_THRESHOLD_USD:
        findings.append(Finding(
            title="Above dual-approval threshold", severity=Severity.LOW,
            evidence=f"Total: {amount:,.2f} {extraction.currency or ''}".strip(), policy_ids=["APR-01"],
            source="rule", recommendation="Route to budget owner and Finance Controller for dual approval."))

    return findings


def tier_for(findings: Iterable[Finding]) -> RiskTier:
    severities = {f.severity for f in findings}
    if Severity.CRITICAL in severities:
        return RiskTier.CRITICAL
    if severities & {Severity.HIGH, Severity.MEDIUM}:
        return RiskTier.MODERATE
    return RiskTier.SECURE


def score_for(findings: list[Finding]) -> int:
    if not findings:
        return 5
    top = max(findings, key=lambda f: _SEVERITY_ORDER.index(f.severity))
    return min(100, _SEVERITY_BASE[top.severity] + 4 * (len(findings) - 1))


def audit_from_findings(findings: list[Finding]) -> RiskAudit:
    tier = tier_for(findings)
    if findings:
        ordered = sorted(findings, key=lambda f: _SEVERITY_ORDER.index(f.severity), reverse=True)
        summary = f"{len(findings)} issue(s) detected; most severe: {ordered[0].title.lower()}."
    else:
        summary = "No policy violations detected. Invoice meets clean-invoice criteria."
        ordered = []
    return RiskAudit(risk_tier=tier, risk_score=score_for(findings), summary=summary, findings=ordered)


def merge_audits(llm_audit: RiskAudit, rule_findings: list[Finding]) -> RiskAudit:
    """Union LLM and rule findings, keeping the stricter tier and score."""
    cited = {pid for f in llm_audit.findings for pid in f.policy_ids}
    extra = [f for f in rule_findings if not set(f.policy_ids) & cited]
    findings = sorted(llm_audit.findings + extra, key=lambda f: _SEVERITY_ORDER.index(f.severity), reverse=True)
    rule_tier = tier_for(rule_findings)
    tier = max(llm_audit.risk_tier, rule_tier, key=_TIER_ORDER.index)
    score = max(llm_audit.risk_score, score_for(rule_findings) if rule_findings else 0)
    return RiskAudit(risk_tier=tier, risk_score=score, summary=llm_audit.summary, findings=findings)


# --- Offline heuristic extractor -------------------------------------------------

def _field(text: str, *labels: str) -> str | None:
    for label in labels:
        match = re.search(rf"^\s*{label}\s*(?:[:#]\s*)+(.+?)\s*$", text, re.I | re.M)
        if match:
            return match.group(1)
    return None


def _iso_date(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"\bSept\b", "Sep", value.strip())
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return value


def terms_to_days(terms: str | None) -> int | None:
    if not terms:
        return None
    lowered = terms.lower()
    if "receipt" in lowered or "immediate" in lowered:
        return 0
    match = re.search(r"net\s*(\d+)", lowered) or re.search(r"(\d+)\s*days", lowered)
    return int(match.group(1)) if match else None


def heuristic_extract(text: str) -> InvoiceExtraction:
    amount_raw = _field(text, "total due", "amount due", "balance due", "total amount", "total")
    amount = currency = None
    if amount_raw:
        number = re.search(r"[\d,]+(?:\.\d+)?", amount_raw)
        amount = float(number.group(0).replace(",", "")) if number else None
        if "$" in amount_raw or "usd" in amount_raw.lower():
            currency = "USD"
        elif "€" in amount_raw or "eur" in amount_raw.lower():
            currency = "EUR"

    terms = _field(text, "payment terms", "terms")
    terms = terms.rstrip(".") if terms else None
    remittance = _evidence(text, re.compile(r"routing|account\s*#|iban|remittance|wire", re.I))
    tax_present = bool(re.search(r"(tax id|ein|vat( no)?|w-9)\s*[:#]\s*[A-Z]{0,2}[\d-]{5,}", text, re.I))

    field_lines = re.compile(
        r"^\s*(vendor|from|date|invoice( id| #| no\.?| number)?|billed to|total( due| amount)?|amount due|"
        r"balance due|(payment )?terms|services|po|supplier|tax id|vat|ein)\s*[:#]", re.I)
    notes = [
        line.strip(" *")
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith(("===", "***")) and not field_lines.match(line)
    ]

    return InvoiceExtraction(
        vendor_name=_field(text, "vendor", "from", "supplier"),
        invoice_id=_field(text, r"invoice(?: id| no\.?| number)?"),
        invoice_date=_iso_date(_field(text, "invoice date", "date")),
        billed_to=_field(text, "billed to", "bill to"),
        currency=currency,
        total_amount=amount,
        payment_terms=terms,
        payment_terms_days=terms_to_days(terms),
        bank_details_changed=bool(_BANK_CHANGE.search(text)),
        remittance_details=remittance,
        tax_id_provided=tax_present,
        notes=notes,
    )
