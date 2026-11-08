"""End-to-end pipeline: extract -> guardrails -> retrieve -> audit -> merge.

`mode="offline"` uses the heuristic extractor, BM25 and rule-based audit, so
it is deterministic, free, and works without an API key. `mode="live"` uses
Gemini for extraction (text, PDF or image), hybrid retrieval, and a
policy-grounded auditor whose verdict is floored by the guardrails.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from .retrieval import BM25Retriever, EmbeddingRetriever, HybridRetriever, load_policy_chunks
from .rules import audit_from_findings, heuristic_extract, merge_audits, run_guardrails
from .schemas import InvoiceExtraction, PipelineResult, RetrievedChunk, TraceStep


def build_retrieval_query(extraction: InvoiceExtraction, document: str) -> str:
    parts = [
        f"payment terms {extraction.payment_terms or ''}",
        f"amount {extraction.total_amount or ''}",
        "bank details changed remittance" if extraction.bank_details_changed else "",
        "" if extraction.tax_id_provided else "missing tax id W-9 documentation",
        extraction.remittance_details or "",
        " ".join(extraction.notes),
        document[:1500],
    ]
    return " ".join(p for p in parts if p)


class InvoicePipeline:
    def __init__(self, client=None, top_k: int = 4):
        self.client = client
        self.top_k = top_k
        self.chunks = load_policy_chunks()
        self.bm25 = BM25Retriever(self.chunks)
        self._dense: EmbeddingRetriever | None = None
        self._agents = None

    @property
    def agents(self):
        if self._agents is None:
            from .llm import GeminiAgents

            self._agents = GeminiAgents(self.client)
        return self._agents

    def run(self, document: str = "", file_bytes: bytes | None = None, mime_type: str | None = None,
            mode: str = "offline") -> PipelineResult:
        if mode == "live" and self.client is None:
            raise ValueError("Live mode needs a Gemini client (set GOOGLE_API_KEY).")
        if mode == "offline" and file_bytes is not None:
            raise ValueError("File uploads need live mode (Gemini reads PDFs/images natively).")

        trace: list[TraceStep] = []

        @contextmanager
        def step(name: str):
            record = TraceStep(name=name, latency_ms=0)
            start = time.perf_counter()
            yield record
            record.latency_ms = round((time.perf_counter() - start) * 1000, 1)
            trace.append(record)

        with step("1 · Extractor") as s:
            if mode == "live":
                extraction, usage = self.agents.extract(document or None, file_bytes, mime_type)
                s.input_tokens, s.output_tokens = usage.input_tokens, usage.output_tokens
                s.detail = "Gemini · response_schema=InvoiceExtraction"
            else:
                extraction = heuristic_extract(document)
                s.detail = "Heuristic regex extractor"

        # Uploaded files have no raw text, so guardrails inspect the extracted fields instead.
        rules_text = document or "\n".join(
            [f"Terms: {extraction.payment_terms or ''}", extraction.remittance_details or "", *extraction.notes]
        )

        with step("2 · Guardrails") as s:
            rule_findings = run_guardrails(rules_text, extraction)
            s.detail = f"{len(rule_findings)} deterministic rule hit(s)"

        with step("3 · Retriever") as s:
            query = build_retrieval_query(extraction, rules_text)
            retriever = self.bm25
            if mode == "live":
                try:
                    if self._dense is None:
                        self._dense = EmbeddingRetriever(self.chunks, self.client)
                    retriever = HybridRetriever([self.bm25, self._dense])
                except Exception as exc:  # embeddings are an enhancement; never fail the audit on them
                    s.detail = f"Embedding index unavailable ({type(exc).__name__}); "
            retrieved = retriever.search(query, k=self.top_k)
            # Policies cited by guardrails are always in context, even if ranked lower.
            have = {r.chunk.id for r in retrieved}
            by_id = {c.id: c for c in self.chunks}
            for pid in dict.fromkeys(p for f in rule_findings for p in f.policy_ids):
                if pid not in have and pid in by_id:
                    retrieved.append(RetrievedChunk(chunk=by_id[pid], score=0.0, retriever="guardrail"))
            s.detail += f"{retriever.name} · top-{self.top_k} of {len(self.chunks)} policy chunks"

        with step("4 · Auditor") as s:
            if mode == "live":
                llm_audit, usage = self.agents.audit(rules_text, extraction, retrieved)
                s.input_tokens, s.output_tokens = usage.input_tokens, usage.output_tokens
                audit = merge_audits(llm_audit, rule_findings)
                s.detail = "Gemini · response_schema=RiskAudit · merged with guardrail floor"
            else:
                audit = audit_from_findings(rule_findings)
                s.detail = "Rule-based scoring"

        return PipelineResult(mode=mode, extraction=extraction, audit=audit, retrieved=retrieved, trace=trace)
