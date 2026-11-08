import json
from types import SimpleNamespace

import pytest

from invoice_intelligence.analysis import extract_risk_tier, json_to_csv
from invoice_intelligence.evaluation import load_dataset, score_case, summarize
from invoice_intelligence.pipeline import InvoicePipeline
from invoice_intelligence.retrieval import (
    BM25Retriever, EmbeddingRetriever, HybridRetriever, load_policy_chunks,
)
from invoice_intelligence.rules import heuristic_extract, merge_audits, run_guardrails, terms_to_days
from invoice_intelligence.schemas import (
    Finding, InvoiceExtraction, RetrievedChunk, RiskAudit, RiskTier, Severity,
)

CHUNKS = load_policy_chunks()
CASES = {c["id"]: c for c in load_dataset()}


# --- Fakes ------------------------------------------------------------------------

class FakeModels:
    """Stands in for client.models; records calls and returns canned JSON."""

    def __init__(self, extraction: dict, audit: dict):
        self.extraction, self.audit = extraction, audit
        self.generate_calls, self.embed_calls = [], []

    def generate_content(self, model, contents, config):
        self.generate_calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        payload = self.extraction if config.response_schema is InvoiceExtraction else self.audit
        usage = SimpleNamespace(prompt_token_count=100, candidates_token_count=20)
        return SimpleNamespace(text=json.dumps(payload), usage_metadata=usage)

    def embed_content(self, model, contents, config):
        self.embed_calls.append(contents)
        # Toy embedding: bag of a few keywords, enough to rank meaningfully.
        vocab = ["bank", "routing", "text", "tax", "offshore", "terms", "approval", "instructions"]
        low = contents.lower()
        values = [float(low.count(w)) + 0.01 for w in vocab]
        return SimpleNamespace(embeddings=[SimpleNamespace(values=values)])


def fake_client(extraction=None, audit=None):
    return SimpleNamespace(models=FakeModels(extraction or {}, audit or {}))


# --- Retrieval ------------------------------------------------------------------

def test_knowledge_base_chunks_have_unique_ids():
    ids = [c.id for c in CHUNKS]
    assert len(ids) == len(set(ids)) >= 10


def test_bm25_ranks_out_of_band_policy_first():
    hits = BM25Retriever(CHUNKS).search("new wire instructions will be sent via text message SMS", k=3)
    assert hits[0].chunk.id == "BANK-02"


def test_hybrid_rrf_rewards_agreement():
    a = SimpleNamespace(name="a", search=lambda q, k: [RetrievedChunk(chunk=CHUNKS[0], score=1, retriever="a"),
                                                      RetrievedChunk(chunk=CHUNKS[1], score=.5, retriever="a")])
    b = SimpleNamespace(name="b", search=lambda q, k: [RetrievedChunk(chunk=CHUNKS[1], score=1, retriever="b")])
    hits = HybridRetriever([a, b]).search("x", k=2)
    assert [h.chunk.id for h in hits] == [CHUNKS[1].id, CHUNKS[0].id]


def test_embedding_index_is_cached(tmp_path):
    client = fake_client()
    EmbeddingRetriever(CHUNKS, client, cache_dir=tmp_path)
    assert len(client.models.embed_calls) == len(CHUNKS)
    retriever = EmbeddingRetriever(CHUNKS, client, cache_dir=tmp_path)
    assert len(client.models.embed_calls) == len(CHUNKS)  # no re-embedding
    retriever.search("routing number changed")
    assert client.models.embed_calls[-1].startswith("task: search result | query:")


# --- Guardrails & extraction -----------------------------------------------------

@pytest.mark.parametrize("terms,days", [("Net 90", 90), ("Due on Receipt", 0), ("30 days", 30), ("TBD", None)])
def test_terms_to_days(terms, days):
    assert terms_to_days(terms) == days


def test_prompt_injection_is_flagged():
    doc = CASES["prompt-injection"]["document"]
    findings = run_guardrails(doc, heuristic_extract(doc))
    assert any(f.policy_ids == ["SEC-01"] and f.severity is Severity.CRITICAL for f in findings)


def test_clean_invoice_has_no_findings():
    doc = CASES["clean-net30"]["document"]
    assert run_guardrails(doc, heuristic_extract(doc)) == []


def test_guardrails_floor_overrides_permissive_llm():
    llm = RiskAudit(risk_tier=RiskTier.SECURE, risk_score=5, summary="Looks fine.", findings=[])
    rule = Finding(title="Bank change", severity=Severity.CRITICAL, evidence="x", policy_ids=["BANK-01"],
                   recommendation="call back", source="rule")
    merged = merge_audits(llm, [rule])
    assert merged.risk_tier is RiskTier.CRITICAL and merged.risk_score >= 80 and merged.findings == [rule]


# --- Pipeline -------------------------------------------------------------------

@pytest.mark.parametrize("case_id", list(CASES))
def test_offline_pipeline_matches_labels(case_id):
    case = CASES[case_id]
    result = InvoicePipeline().run(case["document"], mode="offline")
    score = score_case(case, result)
    assert score.tier_correct and not score.field_misses


def test_live_pipeline_with_mocked_gemini(tmp_path, monkeypatch):
    monkeypatch.setattr("invoice_intelligence.config.CACHE_DIR", tmp_path)
    case = CASES["bec-routing-sms"]
    client = fake_client(
        extraction={"vendor_name": "Global Logistics Inc.", "total_amount": 145000, "payment_terms": "Net 15",
                    "payment_terms_days": 15, "bank_details_changed": True, "tax_id_provided": False, "notes": []},
        audit={"risk_tier": "MODERATE", "risk_score": 50, "summary": "Bank change requested.",
               "findings": [{"title": "Bank change", "severity": "HIGH", "evidence": "routing number has changed",
                             "policy_ids": ["BANK-01", "MADE-UP-99"], "recommendation": "Call back."}]},
    )
    pipeline = InvoicePipeline(client=client)
    pipeline._dense = EmbeddingRetriever(pipeline.chunks, client, cache_dir=tmp_path)
    result = pipeline.run(case["document"], mode="live")

    extract_call, audit_call = client.models.generate_calls
    assert extract_call.config.response_mime_type == "application/json"
    assert "untrusted" in extract_call.config.system_instruction
    assert "<invoice_document>" in extract_call.contents[0]
    assert "<policies>" in audit_call.contents

    llm_finding = next(f for f in result.audit.findings if f.source == "llm")
    assert llm_finding.policy_ids == ["BANK-01"]  # hallucinated citation removed
    assert result.audit.risk_tier is RiskTier.CRITICAL  # BANK-02 guardrail floor
    assert any(f.source == "rule" and "BANK-02" in f.policy_ids for f in result.audit.findings)
    assert result.total_tokens == 240
    assert [s.name for s in result.trace][0] == "1 · Extractor"


def test_offline_rejects_file_upload():
    with pytest.raises(ValueError):
        InvoicePipeline().run(file_bytes=b"%PDF", mime_type="application/pdf", mode="offline")


def test_eval_summary_offline_baseline():
    pipeline = InvoicePipeline()
    summary = summarize([score_case(c, pipeline.run(c["document"])) for c in CASES.values()])
    assert summary["tier_accuracy"] == 1.0 and summary["policy_recall"] == 1.0


# --- Legacy helpers ----------------------------------------------------------------

def test_legacy_helpers():
    assert extract_risk_tier("[CRITICAL]\n- x") == "CRITICAL"
    assert extract_risk_tier("nothing") == "MODERATE"
    assert json_to_csv('{"a": 1, "b": "x"}').splitlines() == ["a,b", "1,x"]
