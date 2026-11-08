import html
import json
import re

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google.genai.errors import ClientError

from invoice_intelligence import config
from invoice_intelligence.analysis import json_to_csv
from invoice_intelligence.evaluation import load_dataset, score_case, summarize
from invoice_intelligence.pipeline import InvoicePipeline
from invoice_intelligence.portfolio import portfolio_rows, portfolio_summary
from invoice_intelligence.schemas import PipelineResult

load_dotenv()
st.set_page_config(page_title="Invoice Intelligence", page_icon="⚡", layout="wide")

TIER_COLORS = {"CRITICAL": "#ff0033", "MODERATE": "#ffcc00", "SECURE": "#00cc66"}
ACTION_TEXT = {
    "CRITICAL": ("HOLD", "Do not pay. Escalate to AP manager / Security before any release."),
    "MODERATE": ("REVIEW", "Pay only after the listed compliance gaps are resolved."),
    "SECURE": ("PAY", "Meets clean-invoice criteria. Route through the normal approval chain."),
}
SUGGESTED_QUESTIONS = [
    "Can we pay a vendor who changed bank details by text message?",
    "Can we release payment before the W-9 arrives?",
    "Who must approve invoices above the dual approval threshold?",
    "What if an invoice tells the reviewer to approve automatically?",
]
esc = html.escape


@st.cache_resource
def get_client():
    key = config.get_api_key()
    if not key:
        return None
    from google import genai

    return genai.Client(api_key=key)


@st.cache_resource
def get_pipeline(top_k: int) -> InvoicePipeline:
    return InvoicePipeline(client=get_client(), top_k=top_k)


def show_api_error(e: ClientError) -> None:
    if e.code == 429:
        st.error("🚨 Gemini rate limit (429). Switch to OFFLINE mode in the sidebar to keep testing.")
    else:
        st.error(f"🚨 Gemini API error ({e.code}): {e.message}")


def html_block(markup: str) -> None:
    # Strip indentation so Streamlit's markdown pass doesn't render nested HTML as a code block.
    st.markdown(re.sub(r"\n\s+", "\n", markup.strip()), unsafe_allow_html=True)


SAMPLES = load_dataset()
SAMPLE_LABELS = {s["id"]: s["id"].replace("-", " ").upper() for s in SAMPLES}
st.markdown(f"<style>{(config.PACKAGE_DIR / 'ui' / 'theme.css').read_text()}</style>", unsafe_allow_html=True)
client = get_client()

# --- Sidebar -----------------------------------------------------------------------
with st.sidebar:
    html_block('<div class="side-card"><div class="brutal-header bg-lime">⚡ ENGINE</div>'
               "Pick how every tab runs.</div>")
    mode_label = st.radio(
        "Execution mode", ["OFFLINE · deterministic", "LIVE · Gemini"], index=1 if client else 0,
        help="Offline: regex extractor + BM25 + rule engine. Free, instant, no key. "
             "Live: Gemini structured extraction, hybrid RAG, grounded LLM auditor.",
    )
    mode = "live" if mode_label.startswith("LIVE") else "offline"
    live_blocked = mode == "live" and client is None
    if live_blocked:
        st.error("Set GOOGLE_API_KEY in .env to enable live mode.")
    top_k = st.slider("Policies retrieved (top-k)", 2, 8, 4)
    pipeline = get_pipeline(top_k)

    html_block(f"""
    <div class="side-card"><div class="brutal-header bg-cyan">🧠 GENAI STACK</div>
    <b>LLM:</b> {esc(config.GENERATION_MODEL)}<br>
    <b>Embeddings:</b> {esc(config.EMBEDDING_MODEL)} ({config.EMBEDDING_DIM}d)<br>
    <b>Retrieval:</b> BM25 + dense · RRF<br>
    <b>Output:</b> Pydantic response_schema<br>
    <b>Knowledge base:</b> {len(pipeline.chunks)} policy chunks<br>
    <b>Evals:</b> {len(SAMPLES)} labeled invoices
    </div>
    <div class="side-card"><div class="brutal-header bg-orange">⚙️ PROJECT</div>
    <b>ARCHITECT:</b> Ankit Mohapatra<br>
    <b>REPO:</b> iamsoankit/invoice-intelligence
    </div>""")

# --- Header ------------------------------------------------------------------------
mode_pill = f"LIVE · {config.GENERATION_MODEL.upper()}" if mode == "live" else "OFFLINE ENGINE"
html_block(f"""
<div class="hero"><div class="title-block"><span>PROCUREMENT AUDIT ENGINE</span></div></div>
<div class="subtitle">AI accounts-payable auditor: extracts invoices into a typed schema, retrieves the relevant
company policy with RAG, and flags fraud &amp; compliance risk with cited evidence.</div>
<div style="margin-bottom:14px">
<span class="pill bg-lime">● {esc(mode_pill)}</span>
<span class="pill bg-cyan">{"HYBRID" if mode == "live" else "BM25"} RAG · {len(pipeline.chunks)} POLICIES</span>
<span class="pill bg-yellow">STRUCTURED OUTPUT</span>
<span class="pill bg-pink">PROMPT-INJECTION GUARD</span>
<span class="pill bg-white">EVAL-GATED CI</span>
</div>""")

tab_audit, tab_batch, tab_qa, tab_eval, tab_arch = st.tabs(
    ["🔍 AUDIT", "🗂️ BATCH", "💬 POLICY Q&A", "📊 EVALS", "🏗️ ARCHITECTURE"])


# === AUDIT =========================================================================
def load_sample():
    sample = next(s for s in SAMPLES if s["id"] == st.session_state.sample_id)
    st.session_state.invoice_text = sample["document"]


def render_result(result: PipelineResult) -> None:
    audit = result.audit
    tier = audit.risk_tier.value
    action, action_text = ACTION_TEXT[tier]

    html_block(f"""<div class="verdict tier-{tier}"><div class="action">{action}</div>
    <div class="text">{esc(audit.summary)}<br>{action_text}</div></div>""")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        html_block(f"""<div class="kpi tier-{tier}"><div class="label">RISK TIER</div>
        <div class="value">{tier}</div><div class="scorebar"><div style="width:{audit.risk_score}%"></div></div>
        <div class="label" style="margin-top:4px">SCORE {audit.risk_score}/100</div></div>""")
    with k2:
        html_block(f"""<div class="kpi bg-yellow"><div class="label">FINDINGS</div>
        <div class="value">{len(audit.findings)}</div>
        <div class="label">{sum(f.source == "rule" for f in audit.findings)} RULE ·
        {sum(f.source == "llm" for f in audit.findings)} LLM</div></div>""")
    with k3:
        html_block(f"""<div class="kpi bg-cyan"><div class="label">LATENCY</div>
        <div class="value">{result.total_latency_ms:,.1f} ms</div>
        <div class="label">{len(result.trace)} PIPELINE STEPS</div></div>""")
    with k4:
        html_block(f"""<div class="kpi bg-lime"><div class="label">TOKENS</div>
        <div class="value">{result.total_tokens:,}</div>
        <div class="label">{result.mode.upper()} MODE</div></div>""")

    st.write("")
    t_find, t_json, t_rag, t_trace = st.tabs(
        ["🚩 FINDINGS", "📦 EXTRACTED JSON", "📚 RETRIEVED POLICIES", "🧭 PIPELINE TRACE"])

    with t_find:
        if not audit.findings:
            html_block('<div class="finding sev-LOW"><div class="head">✅ No policy violations found</div></div>')
        for f in audit.findings:
            chips = "".join(f'<span class="chip">{esc(p)}</span>' for p in f.policy_ids)
            html_block(f"""<div class="finding sev-{f.severity.value}">
            <div class="head"><span>{esc(f.title)}</span>
            <span><span class="chip sev">{f.severity.value}</span>
            <span class="chip src-{esc(f.source)}">{esc(f.source.upper())}</span></span></div>
            <div class="quote">“{esc(f.evidence)}”</div>
            <div>{chips} <b>▸ ACTION:</b> {esc(f.recommendation)}</div></div>""")

    with t_json:
        payload = result.extraction.model_dump()
        st.json(payload)
        flat = {k: "; ".join(v) if isinstance(v, list) else v for k, v in payload.items()}
        report = {"extraction": payload, "audit": audit.model_dump(mode="json"),
                  "retrieved_policies": [r.chunk.id for r in result.retrieved]}
        d1, d2, d3 = st.columns(3)
        d1.download_button("📥 JSON", json.dumps(payload, indent=2), "invoice_payload.json", "application/json")
        d2.download_button("📥 CSV", json_to_csv(json.dumps(flat)), "invoice_payload.csv", "text/csv")
        d3.download_button("📥 FULL AUDIT", json.dumps(report, indent=2), "invoice_audit.json", "application/json")

    with t_rag:
        st.caption("Policy chunks placed in the auditor's context. "
                   "`guardrail` = force-included because a rule cited it.")
        for r in result.retrieved:
            html_block(f"""<div class="finding sev-LOW"><div class="head">
            <span>[{esc(r.chunk.id)}] {esc(r.chunk.title)}</span>
            <span><span class="chip sev">{esc(r.retriever)}</span><span class="chip">score {r.score:.4f}</span></span></div>
            <div style="font-size:.87rem;margin-top:6px">{esc(r.chunk.text)}</div></div>""")

    with t_trace:
        st.dataframe(pd.DataFrame([s.model_dump() for s in result.trace]), hide_index=True, width="stretch")


with tab_audit:
    st.session_state.setdefault("invoice_text", "")
    col_in, col_side = st.columns([1.55, 1], gap="large")
    with col_in:
        src_text, src_file = st.tabs(["📝 PASTE TEXT", "📎 UPLOAD PDF / IMAGE"])
        with src_text:
            st.text_area("RAW DOCUMENT PAYLOAD", key="invoice_text", height=250,
                         placeholder="PASTE AN INVOICE OR CONTRACT HERE…")
        with src_file:
            upload = st.file_uploader("Invoice file", type=["pdf", "png", "jpg", "jpeg", "webp", "txt"],
                                      help="PDFs and images are read natively by Gemini (multimodal) in live mode.")
            if upload is not None and upload.type != "text/plain" and mode == "offline":
                st.warning("PDF / image analysis needs LIVE mode. Text files work in both modes.")
        run_clicked = st.button("RUN COMPLIANCE AUDIT", type="primary", disabled=live_blocked)

    with col_side:
        with st.container(border=True):
            html_block('<div class="brutal-header bg-yellow" style="margin:-16px -16px 12px -16px">'
                       "💡 TEST INVOICES</div>")
            st.selectbox("Scenario", list(SAMPLE_LABELS), format_func=SAMPLE_LABELS.get, key="sample_id")
            st.button("⤵ LOAD INTO EDITOR", on_click=load_sample)
            preview = next(s for s in SAMPLES if s["id"] == st.session_state.sample_id)
            # Streamlit's markdown pass collapses newlines inside raw HTML, so use explicit <br>.
            html_block(f'<div class="doc">{esc(preview["document"]).replace(chr(10), "<br>")}</div>')

    if run_clicked:
        text, file_bytes, mime = st.session_state.invoice_text.strip(), None, None
        if upload is not None:
            if upload.type == "text/plain":
                text = upload.getvalue().decode("utf-8", errors="replace")
            else:
                file_bytes, mime = upload.getvalue(), upload.type
        if not text and file_bytes is None:
            st.warning("Paste an invoice, load a test invoice, or upload a file first.")
        else:
            try:
                with st.spinner(f"RUNNING {mode.upper()} PIPELINE…"):
                    st.session_state.result = pipeline.run(text, file_bytes, mime, mode=mode)
            except ClientError as e:
                st.session_state.pop("result", None)
                show_api_error(e)
            except ValueError as e:
                st.session_state.pop("result", None)
                st.error(f"🚨 {e}")

    if "result" in st.session_state:
        render_result(st.session_state.result)


# === BATCH =========================================================================
with tab_batch:
    st.markdown("Audit a whole stack of invoices and get a pay / review / hold queue. "
                "Text files work offline; PDFs and images need LIVE mode.")
    b1, b2 = st.columns([1.6, 1], gap="large")
    with b1:
        batch_files = st.file_uploader("Invoices", accept_multiple_files=True,
                                       type=["txt", "pdf", "png", "jpg", "jpeg", "webp"], key="batch_files")
    with b2:
        st.write("")
        run_batch = st.button("▶ AUDIT UPLOADED FILES", type="primary", disabled=live_blocked or not batch_files)
        run_samples = st.button(f"▶ AUDIT ALL {len(SAMPLES)} TEST INVOICES", disabled=live_blocked)

    if run_batch or run_samples:
        jobs = [(s["id"], s["document"], None, None) for s in SAMPLES] if run_samples else [
            (f.name, f.getvalue().decode("utf-8", errors="replace"), None, None) if f.type == "text/plain"
            else (f.name, "", f.getvalue(), f.type) for f in batch_files]
        done, errors = [], []
        progress = st.progress(0.0, text="Auditing…")
        for i, (name, text, data, mime) in enumerate(jobs, start=1):
            try:
                done.append((name, pipeline.run(text, data, mime, mode=mode)))
            except ClientError as e:
                show_api_error(e)
                break
            except ValueError as e:
                errors.append(f"{name}: {e}")
            progress.progress(i / len(jobs), text=f"Audited {i}/{len(jobs)}")
        progress.empty()
        for err in errors:
            st.warning(err)
        st.session_state.batch_rows = portfolio_rows(done)

    rows = st.session_state.get("batch_rows")
    if rows:
        summary = portfolio_summary(rows)
        tiers = summary["tiers"]
        c = st.columns(4)
        cards = [
            ("DOCUMENTS", summary["documents"], "bg-white", "AUDITED"),
            ("ON HOLD", tiers["CRITICAL"], "tier-CRITICAL", f"{summary['amount_on_hold']:,.0f} BLOCKED"),
            ("NEEDS REVIEW", tiers["MODERATE"], "tier-MODERATE", "COMPLIANCE GAPS"),
            ("CLEAR TO PAY", tiers["SECURE"], "tier-SECURE", "CLEAN"),
        ]
        for col, (label, value, cls, foot) in zip(c, cards, strict=True):
            with col:
                html_block(f'<div class="kpi {cls}"><div class="label">{label}</div>'
                           f'<div class="value">{value}</div><div class="label">{foot}</div></div>')
        st.write("")

        g1, g2 = st.columns(2, gap="large")
        tier_df = pd.DataFrame([{"tier": t, "count": n} for t, n in tiers.items()])
        bars = alt.Chart(tier_df).mark_bar(stroke="#000", strokeWidth=3).encode(
            x=alt.X("tier:N", sort=list(TIER_COLORS), title=None),
            y=alt.Y("count:Q", title="invoices", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color("tier:N", scale=alt.Scale(domain=list(TIER_COLORS), range=list(TIER_COLORS.values())),
                            legend=None),
        ).properties(height=230, title="RISK DISTRIBUTION")
        g1.altair_chart(bars, width="stretch")
        if summary["top_policies"]:
            pol_df = pd.DataFrame(summary["top_policies"], columns=["policy", "hits"])
            pol = alt.Chart(pol_df).mark_bar(stroke="#000", strokeWidth=3, color="#00d9ff").encode(
                y=alt.Y("policy:N", sort="-x", title=None),
                x=alt.X("hits:Q", title="findings", axis=alt.Axis(tickMinStep=1)),
            ).properties(height=230, title="MOST-VIOLATED POLICIES")
            g2.altair_chart(pol, width="stretch")

        df = pd.DataFrame(rows)
        st.dataframe(
            df, hide_index=True, width="stretch",
            column_config={
                "score": st.column_config.ProgressColumn("score", min_value=0, max_value=100, format="%d"),
                "amount": st.column_config.NumberColumn("amount", format="%.2f"),
            },
        )
        st.download_button("📥 DOWNLOAD PAYMENT QUEUE (CSV)", df.to_csv(index=False), "payment_queue.csv", "text/csv")


# === POLICY Q&A ====================================================================
def set_question(q: str) -> None:
    st.session_state.qa_question = q


with tab_qa:
    st.markdown("Ask the AP policy anything. Answers come **only** from retrieved policy chunks and cite them. "
                "Offline mode returns the best-matching policy sentences; LIVE mode has Gemini compose the answer.")
    q_cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, q in zip(q_cols, SUGGESTED_QUESTIONS, strict=True):
        col.button(q, on_click=set_question, args=(q,), key=f"sq-{q}")
    st.session_state.setdefault("qa_question", "")
    st.text_input("YOUR QUESTION", key="qa_question", placeholder="e.g. What happens if a vendor asks for Net 90?")
    ask = st.button("ASK THE POLICY", type="primary", disabled=live_blocked)

    if ask and st.session_state.qa_question.strip():
        try:
            with st.spinner("Retrieving and answering…"):
                answer, retrieved, step = pipeline.ask(st.session_state.qa_question.strip(), mode=mode)
            st.session_state.setdefault("qa_history", []).insert(
                0, (st.session_state.qa_question.strip(), answer, retrieved, step))
        except ClientError as e:
            show_api_error(e)

    for question, answer, retrieved, step in st.session_state.get("qa_history", [])[:5]:
        cited = re.sub(r"\[([A-Z]+-\d+)\]", r'<span class="cite">\1</span>', esc(answer.answer))
        badge = "" if answer.grounded else '<span class="chip" style="background:#ff9000">NOT IN POLICY</span>'
        html_block(f'<div class="question">Q ▸ {esc(question)}</div>'
                   f'<div class="answer">{badge}{cited}</div>')
        with st.expander(f"Sources · {step.detail} · {step.latency_ms:.1f} ms · "
                         f"{step.input_tokens + step.output_tokens} tokens", expanded=False):
            for r in retrieved:
                mark = "✅" if r.chunk.id in answer.cited_ids else "▫️"
                st.markdown(f"{mark} **[{r.chunk.id}] {r.chunk.title}** · score `{r.score:.4f}`  \n{r.chunk.text}")


# === EVALS =========================================================================
with tab_eval:
    st.markdown(f"Score the pipeline on **{len(SAMPLES)} labeled invoices**: risk-tier accuracy, field-level "
                "extraction accuracy, and policy-citation precision / recall. The same harness gates CI "
                "(`python -m evals.run_eval`). LIVE mode spends Gemini tokens.")
    if st.button(f"▶ RUN EVAL ({mode.upper()})", type="primary", disabled=live_blocked):
        try:
            with st.spinner("Evaluating…"):
                scores = [score_case(c, pipeline.run(c["document"], mode=mode)) for c in SAMPLES]
            st.session_state.eval = (mode, scores, summarize(scores))
        except ClientError as e:
            show_api_error(e)

    if "eval" in st.session_state:
        eval_mode, scores, summary = st.session_state.eval
        st.caption(f"Last run: {eval_mode.upper()} mode")
        m = st.columns(5)
        m[0].metric("Tier accuracy", f"{summary['tier_accuracy']:.0%}")
        m[1].metric("Field accuracy", f"{summary['field_accuracy']:.0%}")
        m[2].metric("Citation precision", f"{summary['policy_precision']:.0%}")
        m[3].metric("Citation recall", f"{summary['policy_recall']:.0%}")
        m[4].metric("Avg latency", f"{summary['avg_latency_ms']:.1f} ms")
        st.dataframe(pd.DataFrame([{
            "case": s.id, "expected": s.expected_tier, "predicted": s.predicted_tier,
            "tier ✓": s.tier_correct, "fields": f"{s.field_hits}/{s.field_total}",
            "missed fields": ", ".join(s.field_misses), "citations TP/FP/FN": f"{s.policy_tp}/{s.policy_fp}/{s.policy_fn}",
            "latency ms": round(s.latency_ms, 1), "tokens": s.tokens,
        } for s in scores]), hide_index=True, width="stretch")
    with st.expander("View the labeled dataset"):
        for case in SAMPLES:
            exp = case["expected"]
            st.markdown(f"**{case['id']}** → `{exp['risk_tier']}` · cites {', '.join(exp['policy_ids']) or '—'}")


# === ARCHITECTURE ==================================================================
with tab_arch:
    html_block("""
    <div class="arch">
    <div class="node bg-yellow"><div class="n">1 · EXTRACTOR</div><div class="d">Gemini with Pydantic
    <code>response_schema</code>. Reads text, PDF or image natively. Invoice wrapped as untrusted data.</div></div>
    <div class="arrow">→</div>
    <div class="node bg-pink"><div class="n">2 · GUARDRAILS</div><div class="d">Deterministic rules for BEC,
    bank changes, offshore accounts, missing W-9, duplicates and prompt injection.</div></div>
    <div class="arrow">→</div>
    <div class="node bg-cyan"><div class="n">3 · RETRIEVER</div><div class="d">BM25 + gemini-embedding-2 dense
    search, fused with Reciprocal Rank Fusion. Rule-cited policies always included.</div></div>
    <div class="arrow">→</div>
    <div class="node bg-lime"><div class="n">4 · AUDITOR</div><div class="d">Gemini must cite retrieved policy
    IDs; invented citations are stripped. Rule findings act as a risk floor.</div></div>
    </div>""")
    a1, a2 = st.columns([1.2, 1], gap="large")
    with a1:
        st.markdown("""
| GenAI concept | Where it lives |
|---|---|
| Structured output (`response_schema`) | `invoice_intelligence/schemas.py`, `llm.py` |
| Multimodal extraction (`Part.from_bytes`) | `llm.py` |
| Chunking, BM25, dense embeddings, RRF, index cache | `retrieval.py` |
| Grounded answers with citation filtering | `llm.py`, `qa.py` |
| Prompt-injection defence | `llm.py` system prompts + rule `SEC-01` |
| Guardrails as a risk floor | `rules.py` → `merge_audits` |
| Tracing (latency + tokens per step) | `pipeline.py` |
| Offline evals + CI regression gate | `evaluation.py`, `evals/`, `.github/workflows/ci.yml` |
| Live evals against Gemini | `.github/workflows/live-eval.yml` |
""")
    with a2:
        st.markdown(f"**Knowledge base — {len(pipeline.chunks)} policy chunks**")
        for chunk in pipeline.chunks:
            with st.expander(f"[{chunk.id}] {chunk.title}"):
                st.write(chunk.text)
