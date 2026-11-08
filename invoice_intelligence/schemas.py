"""Typed contracts shared by the extractor, auditor, UI and evals.

The same Pydantic models are passed to Gemini as `response_schema`, so the
model is constrained to emit exactly these shapes.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskTier(str, Enum):
    SECURE = "SECURE"
    MODERATE = "MODERATE"
    CRITICAL = "CRITICAL"


class InvoiceExtraction(BaseModel):
    vendor_name: Optional[str] = Field(None, description="Legal name of the vendor issuing the invoice.")
    invoice_id: Optional[str] = Field(None, description="Invoice number or ID, verbatim.")
    invoice_date: Optional[str] = Field(None, description="Invoice date normalised to ISO-8601 (YYYY-MM-DD) when possible.")
    billed_to: Optional[str] = Field(None, description="Customer the invoice is addressed to.")
    currency: Optional[str] = Field(None, description="ISO-4217 currency code, e.g. USD.")
    total_amount: Optional[float] = Field(None, description="Total amount due as a number, no symbols.")
    payment_terms: Optional[str] = Field(None, description="Payment terms verbatim, e.g. 'Net 30' or 'Due on Receipt'.")
    payment_terms_days: Optional[int] = Field(None, description="Terms converted to days. 'Due on Receipt' = 0.")
    bank_details_changed: bool = Field(False, description="True if the document announces new or changed bank/remittance details.")
    remittance_details: Optional[str] = Field(None, description="Any account, routing or remittance details mentioned.")
    tax_id_provided: bool = Field(False, description="True only if a tax ID / W-9 / VAT number is actually present.")
    notes: list[str] = Field(default_factory=list, description="Other notable statements, warnings or requests, verbatim.")


class Finding(BaseModel):
    title: str = Field(description="Short headline for the risk.")
    severity: Severity
    evidence: str = Field(description="Verbatim quote from the invoice that supports the finding.")
    policy_ids: list[str] = Field(default_factory=list, description="IDs of the policy chunks (e.g. 'PAY-02') that this finding relies on.")
    recommendation: str = Field(description="Concrete next action for the AP team.")
    source: str = Field("llm", description="'llm' for model findings, 'rule' for deterministic guardrails.")


class RiskAudit(BaseModel):
    risk_tier: RiskTier
    risk_score: int = Field(ge=0, le=100, description="0 = no risk, 100 = certain fraud.")
    summary: str = Field(description="One or two sentence executive summary.")
    findings: list[Finding] = Field(default_factory=list)


class PolicyChunk(BaseModel):
    id: str
    title: str
    text: str
    source: str


class RetrievedChunk(BaseModel):
    chunk: PolicyChunk
    score: float
    retriever: str


class TraceStep(BaseModel):
    name: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    detail: str = ""


class PipelineResult(BaseModel):
    mode: str
    extraction: InvoiceExtraction
    audit: RiskAudit
    retrieved: list[RetrievedChunk]
    trace: list[TraceStep]

    @property
    def total_latency_ms(self) -> float:
        return sum(step.latency_ms for step in self.trace)

    @property
    def total_tokens(self) -> int:
        return sum(step.input_tokens + step.output_tokens for step in self.trace)
