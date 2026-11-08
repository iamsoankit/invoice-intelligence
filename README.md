# Invoice Intelligence ⚡

An AI accounts-payable auditor built on **Google Gemini**. It reads invoices (text, PDF or image), extracts them into a typed schema, retrieves the relevant company policies with **hybrid RAG**, and produces a **policy-cited** risk audit. Deterministic guardrails sit under the LLM, and a **labeled eval harness** measures the whole pipeline.

## Architecture

```
            ┌────────────┐   ┌─────────────┐   ┌──────────────────┐   ┌──────────────┐
 invoice ─▶ │ 1 EXTRACTOR│─▶ │ 2 GUARDRAILS│─▶ │ 3 RETRIEVER (RAG)│─▶ │ 4 AUDITOR    │─▶ audit
 txt/pdf/img│ Gemini +   │   │ regex rules │   │ BM25 + Gemini    │   │ Gemini +     │   + citations
            │ Pydantic   │   │ fraud / BEC │   │ embeddings, RRF  │   │ RiskAudit    │   + trace
            │ schema     │   │ / injection │   │ fusion, cached   │   │ schema       │
            └────────────┘   └─────────────┘   └──────────────────┘   └──────────────┘
                                    └──────── risk floor: LLM can't downgrade a rule hit ───────┘
```

| Concept | Where |
|---|---|
| Structured output (`response_schema` = Pydantic models) | `invoice_intelligence/schemas.py`, `llm.py` |
| Multimodal extraction (PDF / image via `Part.from_bytes`) | `llm.py` |
| RAG: chunking, BM25, dense embeddings (`gemini-embedding-2`), Reciprocal Rank Fusion, on-disk index cache | `retrieval.py`, `knowledge_base/` |
| Grounding: auditor must cite retrieved policy IDs; hallucinated citations are stripped | `llm.py` |
| Prompt-injection defence: untrusted-data delimiters + detection rule `SEC-01` | `llm.py`, `rules.py` |
| Guardrails as a risk floor under the LLM | `rules.py` (`merge_audits`) |
| Observability: per-step latency and token usage | `pipeline.py` → UI "Pipeline trace" tab |
| Evaluation: tier accuracy, field accuracy, citation precision/recall | `evaluation.py`, `evals/` |

## App tabs

* **Audit**: one invoice (paste, load a test case, or upload PDF/image). You get a pay / review / hold verdict, findings with evidence quotes and policy citations, the extracted JSON, retrieved policies, and a per-step trace.
* **Batch**: audit many invoices at once into a payment queue, with risk-distribution and most-violated-policy charts and a CSV export.
* **Policy Q&A**: RAG over the policy knowledge base. Every answer cites policy IDs and says when the policy doesn't cover the question.
* **Evals**: run the labeled eval set in the current mode.
* **Architecture**: pipeline diagram, a map from each concept to its file, and a knowledge-base browser.

## Modes

* **Offline**: heuristic extractor, BM25 and a rule engine. Deterministic, free, no API key. It is also the eval baseline.
* **Live**: Gemini structured extraction (text, PDF, image), hybrid retrieval and an LLM auditor merged with the guardrails.

## Setup

```bash
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your_gemini_api_key_here" > .env   # only needed for live mode
streamlit run app.py
```

Optional overrides: `GEMINI_MODEL` (default `gemini-3.6-flash`), `GEMINI_EMBEDDING_MODEL` (default `gemini-embedding-2`), `GEMINI_EMBEDDING_DIM` (default `768`).

## CLI

```bash
python main.py --sample bec-routing-sms          # offline
python main.py invoice.pdf --live                 # multimodal, needs API key
```

## Evaluation

```bash
python -m evals.run_eval                 # offline baseline
python -m evals.run_eval --mode live     # Gemini pipeline (spends tokens)
```

`evals/dataset.jsonl` holds labeled invoices covering BEC bank-change fraud, offshore remittance, missing W-9, prompt injection, duplicates, and clean invoices. Each case has an expected tier, the policy IDs it should cite, and ground-truth fields. The offline rules were written alongside this set, so their 100% score is a regression baseline rather than a generalisation claim. Add harder and real-world cases to measure the live pipeline honestly. The eval also runs from the app's **Evals** tab. Add `--output report.json` to save a per-case report.

## CI workflows

* `ci.yml` (every PR and push to `main`): `ruff` lint, `pytest`, and the offline eval as a regression gate. The eval report is uploaded as an artifact.
* `live-eval.yml` (manual or weekly): runs the eval against the real Gemini API and posts a scorecard to the job summary. It needs a `GOOGLE_API_KEY` repository secret and skips cleanly without one.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tests cover retrieval, guardrails, the risk floor, and the live pipeline against a mocked Gemini client. The mock checks schema-constrained requests, citation filtering and token accounting, and needs no API key. CI runs lint, the tests, and an offline eval regression gate.

## Extending the knowledge base

Add `## [ID] Title` sections to any markdown file in `invoice_intelligence/knowledge_base/`. Each section becomes one retrievable chunk, and the embedding index rebuilds automatically for changed chunks.
