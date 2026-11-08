import html
import json

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google.genai.errors import ClientError

from invoice_intelligence import config
from invoice_intelligence.analysis import json_to_csv
from invoice_intelligence.evaluation import load_dataset, score_case, summarize
from invoice_intelligence.pipeline import InvoicePipeline
from invoice_intelligence.schemas import PipelineResult

load_dotenv()

st.set_page_config(page_title="Invoice Intelligence", page_icon="⚡", layout="wide")


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


SAMPLES = load_dataset()
SAMPLE_LABELS = {s["id"]: s["id"].replace("-", " ").upper() for s in SAMPLES}
esc = html.escape

# --- Neo-Brutalist CSS -------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #f4f1ea !important; }
[data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 4px solid #000 !important; }
h1, h2, h3, p, label, li, b, div, code, pre { font-family: 'Courier New', Courier, monospace !important; }
h1, h2, h3, p, label, li { color: #000 !important; font-weight: 700 !important; }

div[data-baseweb="textarea"] > textarea {
    background: #fff !important; color: #000 !important; border: 4px solid #000 !important;
    border-radius: 0 !important; box-shadow: 8px 8px 0 #000 !important; padding: 15px !important;
    font-size: 15px !important; font-weight: bold !important;
}
div.stButton > button, div.stDownloadButton > button {
    border: 3px solid #000 !important; border-radius: 0 !important; font-weight: 900 !important;
    text-transform: uppercase !important; box-shadow: 4px 4px 0 #000 !important;
    transition: all .1s ease !important; width: 100%; color: #000 !important; background: #fff !important;
}
div.stButton > button[kind="primary"] {
    background: #ccff00 !important; border-width: 4px !important; padding: .8rem 2.5rem !important;
    font-size: 1.15rem !important; box-shadow: 8px 8px 0 #000 !important;
}
div.stButton > button:hover, div.stDownloadButton > button:hover { transform: translate(2px,2px); box-shadow: 2px 2px 0 #000 !important; }
div.stButton > button:active { transform: translate(4px,4px); box-shadow: 0 0 0 #000 !important; }

[data-testid="stVerticalBlockBorderWrapper"] { border: 3px solid #000 !important; border-radius: 0 !important;
    box-shadow: 5px 5px 0 #000; background: #fff; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border: 3px solid #000; background: #fff; padding: 6px 14px; font-weight: 900; }
.stTabs [aria-selected="true"] { background: #ccff00 !important; box-shadow: 3px 3px 0 #000; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

.brutal-card { background: #fff; padding: 18px; border: 4px solid #000; box-shadow: 8px 8px 0 #000;
    margin-bottom: 22px; color: #000 !important; }
.brutal-header { padding: 8px 14px; border-bottom: 4px solid #000; margin: -18px -18px 16px -18px;
    font-weight: 900 !important; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px; }
.side-card { border: 3px solid #000; padding: 12px; margin-bottom: 18px; background: #fff;
    box-shadow: 5px 5px 0 #000; font-size: .88rem; color: #000 !important; }
.side-card .brutal-header { margin: -12px -12px 12px -12px; font-size: .95rem; }
.bg-pink { background: #ff006e !important; color: #fff !important; }
.bg-cyan { background: #00d9ff !important; } .bg-orange { background: #ff9000 !important; }
.bg-yellow { background: #ffea00 !important; } .bg-lime { background: #ccff00 !important; }
.bg-black { background: #000 !important; color: #fff !important; }

.title-block { display: inline-block; font-size: 2.1rem; font-weight: 900; text-transform: uppercase;
    background: #000; padding: 10px 18px; box-shadow: 6px 6px 0 #ff006e; }
.title-block span { color: #fff !important; }
.pill { display: inline-block; border: 3px solid #000; padding: 4px 10px; margin: 0 8px 8px 0;
    font-weight: 900; font-size: .8rem; box-shadow: 3px 3px 0 #000; color: #000 !important; }

.kpi { background: #fff; border: 4px solid #000; box-shadow: 6px 6px 0 #000; padding: 12px 14px; height: 100%; }
.kpi .label { font-size: .75rem; font-weight: 900; letter-spacing: 1px; }
.kpi .value { font-size: 1.8rem; font-weight: 900; line-height: 1.2; }
.tier-CRITICAL { background: #ff0033 !important; color: #fff !important; }
.tier-MODERATE { background: #ffcc00 !important; }
.tier-SECURE { background: #00cc66 !important; }
.tier-CRITICAL .label, .tier-CRITICAL .value { color: #fff !important; }
.scorebar { height: 14px; border: 3px solid #000; background: #fff; margin-top: 6px; }
.scorebar > div { height: 100%; background: #000; }

.finding { background: #fff; border: 3px solid #000; border-left-width: 14px; padding: 12px 16px;
    margin-bottom: 14px; box-shadow: 5px 5px 0 #000; }
.sev-CRITICAL { border-left-color: #ff0033; } .sev-HIGH { border-left-color: #ff9000; }
.sev-MEDIUM { border-left-color: #ffcc00; } .sev-LOW { border-left-color: #00d9ff; }
.finding .head { display: flex; justify-content: space-between; gap: 10px; font-weight: 900; font-size: 1.02rem; }
.finding .quote { background: #f4f1ea; border: 2px dashed #000; padding: 6px 10px; margin: 8px 0; font-size: .88rem; }
.chip { display: inline-block; border: 2px solid #000; padding: 1px 7px; margin-right: 6px; font-size: .75rem;
    font-weight: 900; background: #00d9ff; }
.chip.src-rule { background: #000; color: #fff !important; } .chip.src-llm { background: #ccff00; }
.chip.sev { background: #fff; }
.doc { background: #f4f1ea; padding: 10px; border: 2px solid #000; color: #000 !important;
    white-space: pre-wrap; font-size: .8rem; font-weight: bold; margin: 0; }
</style>
""", unsafe_allow_html=True)

client = get_client()

# --- Sidebar -----------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="side-card"><div class="brutal-header bg-lime">⚡ ENGINE</div>'
                "Choose how the pipeline runs.</div>", unsafe_allow_html=True)
    mode_label = st.radio(
        "Execution mode",
        ["OFFLINE · deterministic", "LIVE · Gemini"],
        index=1 if client else 0,
        help="Offline: regex extractor + BM25 + rule engine. Free, instant, no key. "
             "Live: Gemini structured extraction, hybrid RAG, grounded LLM auditor.",
    )
    mode = "live" if mode_label.startswith("LIVE") else "offline"
    if mode == "live" and client is None:
        st.error("Set GOOGLE_API_KEY in .env to enable live mode.")
    top_k = st.slider("Policies retrieved (top-k)", 2, 8, 4)

    kb_size = len(get_pipeline(top_k).chunks)
    st.markdown(f"""
<div class="side-card"><div class="brutal-header bg-cyan">🧠 GENAI STACK</div>
<b>LLM:</b> {esc(config.GENERATION_MODEL)}<br>
<b>Embeddings:</b> {esc(config.EMBEDDING_MODEL)} ({config.EMBEDDING_DIM}d)<br>
<b>Retrieval:</b> BM25 + dense · RRF fusion<br>
<b>Output:</b> Pydantic response_schema<br>
<b>Knowledge base:</b> {kb_size} policy chunks<br>
<b>Evals:</b> {len(SAMPLES)} labeled invoices
</div>
<div class="side-card"><div class="brutal-header bg-pink">🧭 PIPELINE</div>
<b>1. EXTRACTOR</b> → text / PDF / image into a typed schema<br>
<b>2. GUARDRAILS</b> → deterministic fraud &amp; injection rules<br>
<b>3. RETRIEVER</b> → hybrid search over AP policy<br>
<b>4. AUDITOR</b> → cites policy IDs; rules act as a risk floor
</div>
<div class="side-card"><div class="brutal-header bg-orange">⚙️ PROJECT</div>
<b>ARCHITECT:</b> Ankit Mohapatra
</div>
""", unsafe_allow_html=True)

# --- Header ------------------------------------------------------------------------
retriever_name = "HYBRID RAG" if mode == "live" else "BM25 RAG"
st.markdown(f"""
<div style="margin-bottom:14px"><div class="title-block"><span>PROCUREMENT AUDIT ENGINE</span></div></div>
<div style="margin-bottom:18px">
  <span class="pill bg-lime">● {"LIVE · " + esc(config.GENERATION_MODEL.upper()) if mode == "live" else "OFFLINE MODE"}</span>
  <span class="pill bg-cyan">{retriever_name} · {kb_size} POLICIES</span>
  <span class="pill bg-yellow">STRUCTURED OUTPUT</span>
  <span class="pill bg-pink" style="color:#fff !important">PROMPT-INJECTION GUARD</span>
</div>
""", unsafe_allow_html=True)

if "invoice_text" not in st.session_state:
    st.session_state.invoice_text = ""


def load_sample():
    sample = next(s for s in SAMPLES if s["id"] == st.session_state.sample_id)
    st.session_state.invoice_text = sample["document"]


col_in, col_side = st.columns([1.55, 1], gap="large")

with col_in:
    tab_text, tab_file = st.tabs(["📝 PASTE TEXT", "📎 UPLOAD PDF / IMAGE"])
    with tab_text:
        st.text_area("RAW DOCUMENT PAYLOAD", key="invoice_text", height=260,
                     placeholder="PASTE AN INVOICE OR CONTRACT HERE…")
    with tab_file:
        upload = st.file_uploader("Invoice file", type=["pdf", "png", "jpg", "jpeg", "webp", "txt"],
                                  help="PDFs and images are read natively by Gemini (multimodal) in live mode.")
        if upload is not None and upload.type != "text/plain" and mode == "offline":
            st.warning("PDF / image analysis needs LIVE mode. Text files work in both modes.")
    run_clicked = st.button("RUN COMPLIANCE AUDIT", type="primary")

with col_side:
    with st.container(border=True):
        st.markdown('<div class="brutal-header bg-yellow" style="margin:-16px -16px 12px -16px">'
                    "💡 TEST INVOICES</div>", unsafe_allow_html=True)
        st.selectbox("Scenario", list(SAMPLE_LABELS), format_func=SAMPLE_LABELS.get, key="sample_id")
        st.button("⤵ LOAD INTO EDITOR", on_click=load_sample)
        preview = next(s for s in SAMPLES if s["id"] == st.session_state.sample_id)
        # Streamlit's markdown pass collapses newlines inside raw HTML, so use explicit <br>.
        preview_html = esc(preview["document"]).replace("\n", "<br>")
        st.markdown(f'<div class="doc">{preview_html}</div>', unsafe_allow_html=True)

# --- Execution ---------------------------------------------------------------------
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
                st.session_state.result = get_pipeline(top_k).run(text, file_bytes, mime, mode=mode)
        except ClientError as e:
            st.session_state.pop("result", None)
            if e.code == 429:
                st.error("🚨 Gemini rate limit (429). Switch to OFFLINE mode in the sidebar to keep testing.")
            else:
                st.error(f"🚨 Gemini API error ({e.code}): {e.message}")
        except ValueError as e:
            st.session_state.pop("result", None)
            st.error(f"🚨 {e}")


def render_result(result: PipelineResult) -> None:
    audit = result.audit
    st.markdown("<hr style='border-top:4px solid #000;opacity:1'>", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    tier = audit.risk_tier.value
    k1.markdown(f"""<div class="kpi tier-{tier}"><div class="label">RISK TIER</div>
        <div class="value">{tier}</div><div class="scorebar"><div style="width:{audit.risk_score}%"></div></div>
        <div class="label" style="margin-top:4px">SCORE {audit.risk_score}/100</div></div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="kpi bg-yellow"><div class="label">FINDINGS</div>
        <div class="value">{len(audit.findings)}</div>
        <div class="label">{sum(f.source == "rule" for f in audit.findings)} RULE ·
        {sum(f.source == "llm" for f in audit.findings)} LLM</div></div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="kpi bg-cyan"><div class="label">LATENCY</div>
        <div class="value">{result.total_latency_ms:,.1f} ms</div>
        <div class="label">{len(result.trace)} PIPELINE STEPS</div></div>""", unsafe_allow_html=True)
    k4.markdown(f"""<div class="kpi bg-lime"><div class="label">TOKENS</div>
        <div class="value">{result.total_tokens:,}</div>
        <div class="label">{result.mode.upper()} MODE</div></div>""", unsafe_allow_html=True)

    st.markdown(f'<div class="brutal-card" style="margin-top:22px"><b>EXECUTIVE SUMMARY ▸</b> '
                f'{esc(audit.summary)}</div>', unsafe_allow_html=True)

    t_find, t_json, t_rag, t_trace = st.tabs(
        ["🚩 FINDINGS", "📦 EXTRACTED JSON", "📚 RETRIEVED POLICIES", "🧭 PIPELINE TRACE"])

    with t_find:
        if not audit.findings:
            st.markdown('<div class="finding sev-LOW"><div class="head">✅ No policy violations found</div></div>',
                        unsafe_allow_html=True)
        for f in audit.findings:
            chips = "".join(f'<span class="chip">{esc(p)}</span>' for p in f.policy_ids)
            st.markdown(f"""<div class="finding sev-{f.severity.value}">
                <div class="head"><span>{esc(f.title)}</span>
                <span><span class="chip sev">{f.severity.value}</span>
                <span class="chip src-{esc(f.source)}">{esc(f.source.upper())}</span></span></div>
                <div class="quote">“{esc(f.evidence)}”</div>
                <div>{chips} <b>▸ ACTION:</b> {esc(f.recommendation)}</div></div>""", unsafe_allow_html=True)

    with t_json:
        payload = result.extraction.model_dump()
        st.json(payload)
        flat = {k: "; ".join(v) if isinstance(v, list) else v for k, v in payload.items()}
        report = {"extraction": payload, "audit": audit.model_dump(mode="json")}
        d1, d2, d3 = st.columns(3)
        d1.download_button("📥 JSON", json.dumps(payload, indent=2), "invoice_payload.json", "application/json")
        d2.download_button("📥 CSV", json_to_csv(json.dumps(flat)), "invoice_payload.csv", "text/csv")
        d3.download_button("📥 FULL AUDIT", json.dumps(report, indent=2), "invoice_audit.json", "application/json")

    with t_rag:
        st.caption("Policy chunks placed in the auditor's context. `guardrail` = force-included because a rule cited it.")
        for r in result.retrieved:
            st.markdown(f"""<div class="finding sev-LOW"><div class="head">
                <span>[{esc(r.chunk.id)}] {esc(r.chunk.title)}</span>
                <span><span class="chip sev">{esc(r.retriever)}</span><span class="chip">score {r.score:.4f}</span></span></div>
                <div style="font-size:.88rem;margin-top:6px">{esc(r.chunk.text)}</div></div>""", unsafe_allow_html=True)

    with t_trace:
        st.dataframe(pd.DataFrame([s.model_dump() for s in result.trace]), hide_index=True, width="stretch")


if "result" in st.session_state:
    render_result(st.session_state.result)

# --- Evaluation harness ------------------------------------------------------------
st.markdown("<hr style='border-top:4px solid #000;opacity:1'>", unsafe_allow_html=True)
with st.expander("📊 EVALUATION HARNESS — score the pipeline on the labeled dataset", expanded=False):
    st.caption(f"{len(SAMPLES)} labeled invoices · metrics: risk-tier accuracy, field-level extraction accuracy, "
               "policy-citation precision / recall. Live mode spends Gemini tokens.")
    if st.button(f"▶ RUN EVAL ({mode.upper()})", disabled=mode == "live" and client is None):
        pipeline = get_pipeline(top_k)
        with st.spinner("Evaluating…"):
            scores = [score_case(c, pipeline.run(c["document"], mode=mode)) for c in SAMPLES]
        summary = summarize(scores)
        m = st.columns(4)
        m[0].metric("Tier accuracy", f"{summary['tier_accuracy']:.0%}")
        m[1].metric("Field accuracy", f"{summary['field_accuracy']:.0%}")
        m[2].metric("Citation precision", f"{summary['policy_precision']:.0%}")
        m[3].metric("Citation recall", f"{summary['policy_recall']:.0%}")
        st.dataframe(pd.DataFrame([{
            "case": s.id, "expected": s.expected_tier, "predicted": s.predicted_tier,
            "tier ✓": s.tier_correct, "fields": f"{s.field_hits}/{s.field_total}",
            "missed fields": ", ".join(s.field_misses), "latency ms": round(s.latency_ms, 1), "tokens": s.tokens,
        } for s in scores]), hide_index=True, width="stretch")
