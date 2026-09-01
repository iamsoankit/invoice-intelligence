import os
import time
import json
import streamlit as st
from google import genai
from google.genai.errors import ClientError
from dotenv import load_dotenv
from invoice_intelligence.analysis import extract_risk_tier, json_to_csv

load_dotenv()

# Safely initialize client only if key exists
api_key = os.environ.get("GOOGLE_API_KEY", "")
client = genai.Client(api_key=api_key) if api_key else None

st.set_page_config(page_title="Procurement AI", page_icon="⚡", layout="wide")

# --- Manage State for Test Data Cycling ---
if 'test_idx' not in st.session_state:
    st.session_state.test_idx = 0

test_invoices = [
    """=== INVOICE DOCUMENT ===
Vendor: Global Logistics Inc.
Date: August 15, 2026
TOTAL DUE: $145,000.00
Terms: Net 15

*** IMPORTANT ***
Our routing number has changed. Do not send to the previous account. New wire instructions will be sent via text message.""",

    """=== INVOICE DOCUMENT ===
From: TechSphere Solutions
Invoice ID: TS-881
Billed To: ACME Corp
Amount Due: $125,500.00

Payment Terms: Due on Receipt.
WARNING: Remittance address changed to offshore account #4492.""",

    """=== INVOICE DOCUMENT ===
Vendor: Apex Marketing LLC
Date: Aug 10, 2026
Services: General Consulting & Ad-hoc PR
TOTAL: $45,000.00
Terms: Net 90

Note: Corporate Tax ID (W-9) is currently pending and will be submitted next quarter. Please process payment without it for now."""
]

# --- Neo-Brutalist Custom CSS ---
st.markdown("""
    <style>
    /* Absolute White & Background Setup */
    [data-testid="stAppViewContainer"] { background-color: #f4f1ea !important; }
    [data-testid="stSidebar"] { background-color: #ffffff !important; border-right: 4px solid #000000 !important; }
    
    /* Apply brutalist font globally */
    h1, h2, h3, p, label, li, b, div { font-family: 'Courier New', Courier, monospace !important; }
    
    /* Force default text to black for headings and paragraphs */
    h1, h2, h3, p, label, li { color: #000000 !important; font-weight: 700 !important; }
    
    /* Brutalist Input Area */
    div[data-baseweb="textarea"] > textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 4px solid #000000 !important;
        border-radius: 0px !important;
        box-shadow: 8px 8px 0px #000000 !important;
        padding: 15px !important;
        font-size: 16px !important;
        font-weight: bold !important;
    }
    
    /* PRIMARY BUTTON (Analyze) */
    div.stButton > button[kind="primary"] {
        background-color: #ccff00 !important;
        color: #000000 !important;
        border: 4px solid #000000 !important;
        border-radius: 0px !important;
        padding: 0.8rem 2.5rem !important;
        font-weight: 900 !important;
        font-size: 1.2rem !important;
        text-transform: uppercase !important;
        box-shadow: 8px 8px 0px #000000 !important;
        transition: all 0.1s ease !important;
        width: 100%;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translate(2px, 2px) !important;
        box-shadow: 6px 6px 0px #000000 !important;
    }
    div.stButton > button[kind="primary"]:active {
        transform: translate(8px, 8px) !important;
        box-shadow: 0px 0px 0px #000000 !important;
    }
    
    /* SECONDARY BUTTON (Cycle Test Data / Download) */
    div.stButton > button[kind="secondary"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 3px solid #000000 !important;
        border-radius: 0px !important;
        padding: 0.4rem 1rem !important;
        font-weight: 900 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase !important;
        box-shadow: 4px 4px 0px #000000 !important;
        transition: all 0.1s ease !important;
        width: 100%;
        margin-bottom: 5px;
    }
    div.stButton > button[kind="secondary"]:active {
        transform: translate(4px, 4px) !important;
        box-shadow: 0px 0px 0px #000000 !important;
    }
    
    /* Brutalist Output Cards */
    .brutal-card {
        background-color: #ffffff;
        padding: 20px;
        border: 4px solid #000000;
        box-shadow: 8px 8px 0px #000000;
        margin-bottom: 25px;
        height: 100%;
        color: #000000 !important;
    }
    .brutal-header { 
        padding: 10px 15px; 
        border-bottom: 4px solid #000000; 
        margin: -20px -20px 20px -20px; 
        font-weight: 900 !important;
        font-size: 1.2rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Color Badges */
    .bg-pink { background-color: #ff006e !important; color: #ffffff !important; }
    .bg-cyan { background-color: #00d9ff !important; color: #000000 !important; }
    .bg-orange { background-color: #ff9000 !important; color: #000000 !important; }
    .bg-yellow { background-color: #ffea00 !important; color: #000000 !important; }
    .badge-critical { background-color: #ff0033; color: #ffffff; padding: 4px 10px; border: 2px solid #000; font-weight: 900; display: inline-block; margin-bottom: 12px; box-shadow: 3px 3px 0px #000; }
    .badge-moderate { background-color: #ffcc00; color: #000000; padding: 4px 10px; border: 2px solid #000; font-weight: 900; display: inline-block; margin-bottom: 12px; box-shadow: 3px 3px 0px #000; }
    .badge-secure { background-color: #00cc66; color: #000000; padding: 4px 10px; border: 2px solid #000; font-weight: 900; display: inline-block; margin-bottom: 12px; box-shadow: 3px 3px 0px #000; }
    
    /* Sidebar styling */
    .sidebar-section {
        border: 3px solid #000000;
        padding: 15px;
        margin-bottom: 20px;
        background-color: #ffffff;
        box-shadow: 5px 5px 0px #000000;
        color: #000000 !important;
        font-size: 0.95rem;
    }
    .sidebar-section b, .sidebar-section p, .sidebar-section div {
        color: #000000 !important;
    }
    hr {
        border-top: 4px solid #000000 !important;
        opacity: 1 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar: Documentation & Demo Mode Toggle ---
with st.sidebar:
    st.write("")
    
    # Demo Mode Toggle Section
    st.markdown("""
    <div class="sidebar-section" style="background-color: #ccff00;">
        <div class="brutal-header bg-black" style="color: white !important;">⚡ CONFIG</div>
    """, unsafe_allow_html=True)
    demo_mode = st.toggle("ENABLE DEMO MODE", value=True, help="Bypasses API rate limits by using instant local simulation.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="brutal-header bg-orange">⚙️ SYSTEM INFO</div>
        <b>PROJECT:</b> Invoice Intelligence<br>
        <b>ARCHITECT:</b> Ankit Mohapatra<br>
        <b>STATUS:</b> Online & Ready<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="brutal-header bg-cyan">🛠️ TECH STACK</div>
        • <b>Frontend:</b> Streamlit & CSS<br>
        • <b>Backend:</b> Python<br>
        • <b>AI Core:</b> Google GenAI SDK<br>
        • <b>Model:</b> Gemini 3.6 Flash<br>
        • <b>Paradigm:</b> Multi-Agent System<br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section">
        <div class="brutal-header bg-pink">📖 HOW IT WORKS</div>
        <b>1. EXTRACTOR NODE</b><br>
        Parses raw text into strict JSON schema.<br><br>
        <b>2. AUDITOR NODE</b><br>
        Scores risk tiers (Critical/Moderate/Secure) & flags compliance gaps.
    </div>
    """, unsafe_allow_html=True)

# --- Main Dashboard Header ---
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 15px;">
    <div style="font-size: 2.2rem; font-weight: 900; text-transform: uppercase; background-color: #000000; color: #ffffff !important; padding: 10px 18px; border: 3px solid #000000; box-shadow: 5px 5px 0px #ff006e;">
        PROCUREMENT AUDIT ENGINE
    </div>
    <div style="background-color: #ccff00; color: #000000; font-weight: 900; padding: 8px 14px; border: 3px solid #000000; box-shadow: 4px 4px 0px #000000; font-size: 0.95rem;">
        ● MULTI-AGENT ACTIVE
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<p style='font-size: 1.1rem; max-width: 900px; margin-bottom: 25px; background-color: #ffffff; padding: 12px 15px; border: 3px solid #000; box-shadow: 4px 4px 0px #000; color: #000000 !important;'>Drop an unstructured vendor contract or invoice below. The multi-agent pipeline extracts data, scores risk tiers, and flags compliance anomalies.</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])

with col1:
    invoice_text = st.text_area("RAW DOCUMENT PAYLOAD", height=250, placeholder="PASTE CONTRACT OR INVOICE HERE...")
    analyze_clicked = st.button("INITIALIZE COMPLIANCE AUDIT", type="primary")

with col2:
    if st.button("🔄 CYCLE TEST DATA PAYLOADS", type="secondary"):
        st.session_state.test_idx = (st.session_state.test_idx + 1) % len(test_invoices)
        
    current_invoice = test_invoices[st.session_state.test_idx]
    current_num = st.session_state.test_idx + 1

    st.markdown(f"""
    <div class="brutal-card">
        <div class="brutal-header bg-yellow">💡 TEST DATA 0{current_num}</div>
        <p style='font-size: 0.85rem; color: #000000 !important; font-weight: bold;'>COPY AND PASTE THIS TO TEST THE ENGINE:</p>
        <pre style="background:#f4f1ea; padding:8px; border:2px solid #000; color:#000000 !important; font-weight: bold; white-space: pre-wrap; font-size: 0.8rem;">{current_invoice}</pre>
    </div>
    """, unsafe_allow_html=True)

# --- Execution Logic ---
if analyze_clicked and invoice_text.strip():
    st.markdown("<hr>", unsafe_allow_html=True)
    
    if demo_mode:
        # --- DEMO / MOCK MODE (Zero API Usage) ---
        with st.spinner("RUNNING MOCK MULTI-AGENT PIPELINE (DEMO MODE)..."):
            time.sleep(0.6)
            
            # Generate smart mock responses based on input text
            if "Global Logistics" in invoice_text or "routing number" in invoice_text.lower():
                risk_tier = "CRITICAL"
                audit_text = "[CRITICAL]\n- HIGH FRAUD RISK: Unverified request to change wire routing instructions via text message.\n- Action Required: Validate bank account change via phone callback with official vendor on file.\n- Compliance Violation: Segregation of duties breach."
                extracted_json = json.dumps({
                    "vendor_name": "Global Logistics Inc.",
                    "invoice_date": "August 15, 2026",
                    "total_due_usd": 145000.00,
                    "payment_terms": "Net 15",
                    "flagged_anomalies": ["Routing number change via SMS", "High-value disbursement"]
                }, indent=2)
            elif "TechSphere" in invoice_text or "offshore" in invoice_text.lower():
                risk_tier = "CRITICAL"
                audit_text = "[CRITICAL]\n- AML / COMPLIANCE WARNING: Remittance address points to an unverified offshore account (#4492).\n- Action Required: Freeze payment pending KYC verification and tax compliance clearance."
                extracted_json = json.dumps({
                    "vendor_name": "TechSphere Solutions",
                    "invoice_id": "TS-881",
                    "billed_to": "ACME Corp",
                    "amount_due_usd": 125500.00,
                    "payment_terms": "Due on Receipt",
                    "remittance_account": "Offshore #4492"
                }, indent=2)
            else:
                risk_tier = "MODERATE"
                audit_text = "[MODERATE]\n- TAX COMPLIANCE GAP: Corporate Tax ID (W-9) is currently missing.\n- Net 90 payment terms exceed standard corporate policy limits (Net 30 max)."
                extracted_json = json.dumps({
                    "vendor_name": "Apex Marketing LLC",
                    "invoice_date": "August 10, 2026",
                    "total_due_usd": 45000.00,
                    "payment_terms": "Net 90",
                    "tax_id_status": "Pending W-9 submission"
                }, indent=2)
                
        formatted_audit_html = audit_text.replace('\n', '<br>')
        
    else:
        # --- LIVE API MODE ---
        try:
            with st.spinner("EXECUTING LIVE MULTI-AGENT PIPELINE..."):
                extractor_prompt = f"Extract Vendor Name, Total Amount, Date, Terms, and Important Notes. Output ONLY a valid JSON object. Text: {invoice_text}"
                extracted_response = client.models.generate_content(model='gemini-3.6-flash', contents=extractor_prompt)
                extracted_json = extracted_response.text.strip()
                time.sleep(0.5) 
                
                analyst_prompt = f"""
                Review this JSON. First line MUST be exactly one of these tags: [CRITICAL], [MODERATE], or [SECURE]. 
                Followed by short bullet points detailing financial risks or compliance status.
                JSON Data: {extracted_json}
                """
                raw_analysis_response = client.models.generate_content(model='gemini-3.6-flash', contents=analyst_prompt)
                raw_analysis = raw_analysis_response.text.strip()
                
                risk_tier = extract_risk_tier(raw_analysis)
                audit_text = raw_analysis

                for tier in ("CRITICAL", "SECURE", "MODERATE"):
                    audit_text = audit_text.replace(f"[{tier}]", "").strip()

            formatted_audit_html = audit_text.replace('\n', '<br>')
            
        except ClientError as e:
            if e.code == 429:
                st.error("🚨 **API Rate Limit Exceeded (429)**: Free tier limit reached. Switch **ENABLE DEMO MODE** on in the sidebar to continue testing instantly without limits!")
            else:
                st.error(f"🚨 **API Error ({e.code})**: {e.message}")
            st.stop()
        except Exception as ex:
            st.error(f"🚨 **An unexpected error occurred:** {str(ex)}")
            st.stop()

    # --- Render Results ---
    res1, res2 = st.columns(2)
    
    with res1:
        badge_class = "badge-moderate"
        if risk_tier == "CRITICAL": badge_class = "badge-critical"
        elif risk_tier == "SECURE": badge_class = "badge-secure"
        
        st.markdown(f"""
        <div class="brutal-card">
            <div class="brutal-header bg-pink">⚠️ RISK AUDIT & SCORING</div>
            <div class="{badge_class}">RISK TIER: {risk_tier}</div>
            <div style="font-size: 1.05rem; line-height: 1.6; color: #000000 !important; font-weight: bold;">
                {formatted_audit_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with res2:
        st.markdown(f"""
        <div class="brutal-card">
            <div class="brutal-header bg-cyan">📦 JSON PAYLOAD</div>
            <pre style="background:#f4f1ea; color:#000000 !important; font-weight:bold; border: 3px solid #000; padding: 15px; margin-bottom: 15px;"><code>{extracted_json}</code></pre>
        </div>
        """, unsafe_allow_html=True)
        
        # Add Export Download Buttons below JSON card
        try:
            parsed_json = json.loads(extracted_json)
            json_string = json.dumps(parsed_json, indent=2)
            
            csv_string = json_to_csv(extracted_json)
            
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.download_button(
                    label="📥 Download JSON",
                    data=json_string,
                    file_name="invoice_payload.json",
                    mime="application/json",
                    type="secondary"
                )
            with d_col2:
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_string,
                    file_name="invoice_payload.csv",
                    mime="text/csv",
                    type="secondary"
                )
        except Exception:
            st.download_button(
                label="📥 Download Raw Payload",
                data=extracted_json,
                file_name="invoice_payload.json",
                mime="application/json",
                type="secondary"
            )
