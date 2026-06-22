import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import time

# Add code folder to path
sys.path.append(os.path.dirname(__file__))
import importlib
import agent
importlib.reload(agent)
from agent import ClaimsAgent

# Set page config
st.set_page_config(
    page_title="Multi-Model Claim Evidence Review",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme State
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

IS_DARK = st.session_state.theme == "dark"

# Theme colors mapping
bg = "#09090b" if IS_DARK else "#ffffff"
bg_subtle = "#0c0c0f" if IS_DARK else "#f9fafb"
card = "#0c0c0f" if IS_DARK else "#ffffff"
card_hover = "#131316" if IS_DARK else "#f4f4f5"
border = "#1e1e24" if IS_DARK else "#e4e4e7"
border_subtle = "#16161a" if IS_DARK else "#f0f0f2"
text = "#fafafa" if IS_DARK else "#09090b"
text_muted = "#71717a"
text_dim = "#52525b" if IS_DARK else "#a1a1aa"
green = "#22c55e" if IS_DARK else "#16a34a"
green_muted = "rgba(34,197,94,0.12)" if IS_DARK else "rgba(22,163,74,0.08)"
red = "#ef4444" if IS_DARK else "#dc2626"
red_muted = "rgba(239,68,68,0.12)" if IS_DARK else "rgba(220,38,38,0.08)"
amber = "#f59e0b" if IS_DARK else "#d97706"
amber_muted = "rgba(245,158,11,0.12)" if IS_DARK else "rgba(217,119,6,0.08)"
shadow = "none" if IS_DARK else "0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.03)"

# Inject Custom CSS Design System
css = f"""
<style>
:root {{
    --bg: {bg};
    --bg-subtle: {bg_subtle};
    --card: {card};
    --card-hover: {card_hover};
    --border: {border};
    --border-subtle: {border_subtle};
    --text: {text};
    --text-muted: {text_muted};
    --text-dim: {text_dim};
    --accent: #2563eb;
    --accent-muted: #1d4ed8;
    --green: {green};
    --green-muted: {green_muted};
    --red: {red};
    --red-muted: {red_muted};
    --amber: {amber};
    --amber-muted: {amber_muted};
    --shadow: {shadow};
    --radius: 10px;
}}

/* Hide default streamlit headers/footers */
header[data-testid="stHeader"], footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], .stDeployButton {{
    display: none !important;
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], .main, .block-container, section[data-testid="stMain"] {{
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}}

.block-container {{
    padding: 1.5rem 2.5rem 2rem !important;
    max-width: 1360px !important;
}}

/* Card components */
.panel-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.4rem;
    box-shadow: var(--shadow);
    margin-bottom: 1.25rem;
}}

.metric-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.4rem;
    box-shadow: var(--shadow);
    margin-bottom: 1rem;
}}
.metric-label {{
    font-size: 0.78rem;
    color: var(--text-muted);
    font-weight: 500;
}}
.metric-value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.03em;
    margin-top: 0.2rem;
}}

/* Table Styling */
.data-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.8rem;
    margin-top: 0.5rem;
}}
.data-table th {{
    text-align: left;
    padding: 0.6rem 0.8rem;
    color: var(--text-muted);
    font-weight: 500;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    border-bottom: 1px solid var(--border);
}}
.data-table td {{
    padding: 0.65rem 0.8rem;
    color: var(--text);
    border-bottom: 1px solid var(--border-subtle);
}}
.data-table tr:last-child td {{
    border-bottom: none;
}}

/* Badges */
.badge {{
    display: inline-block;
    padding: 2px 9px;
    border-radius: 6px;
    font-size: 0.72rem;
    font-weight: 500;
}}
.badge-green {{ color: var(--green); background: var(--green-muted); }}
.badge-red {{ color: var(--red); background: var(--red-muted); }}
.badge-amber {{ color: var(--amber); background: var(--amber-muted); }}
.badge-blue {{ color: var(--accent); background: rgba(37,99,235,0.1); }}
.badge-gray {{ color: var(--text-muted); background: var(--border); }}

/* Chat Transcript bubbles */
.chat-container {{
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    background: var(--bg-subtle);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    margin-bottom: 1rem;
}}
.chat-bubble {{
    padding: 0.6rem 0.85rem;
    border-radius: 8px;
    max-width: 85%;
    font-size: 0.82rem;
    line-height: 1.4;
}}
.chat-user {{
    background: var(--card);
    border: 1px solid var(--border);
    color: var(--text);
    align-self: flex-end;
    border-bottom-right-radius: 2px;
}}
.chat-agent {{
    background: rgba(37, 99, 235, 0.08);
    border: 1px solid rgba(37, 99, 235, 0.2);
    color: var(--text);
    align-self: flex-start;
    border-bottom-left-radius: 2px;
}}
.chat-speaker {{
    font-weight: 600;
    font-size: 0.72rem;
    margin-bottom: 0.15rem;
    color: var(--text-muted);
}}

/* Brand */
.brand {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 1.5rem;
}}
.brand-name {{
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: var(--text);
}}
.brand-tag {{
    font-size: 0.7rem;
    background: var(--accent);
    color: white;
    padding: 2px 6px;
    border-radius: 4px;
    font-weight: 600;
}}

/* Sidebar styling overrides */
section[data-testid="stSidebar"] {{
    background-color: var(--bg-subtle) !important;
    border-right: 1px solid var(--border) !important;
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Plotly default layout theme setup
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#71717a" if not IS_DARK else "#a1a1aa", size=11),
    margin=dict(l=20, r=20, t=30, b=20),
    xaxis=dict(
        gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        zerolinecolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        tickfont=dict(size=10, color="#71717a"),
    ),
    yaxis=dict(
        gridcolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        zerolinecolor="rgba(0,0,0,0.04)" if not IS_DARK else "rgba(255,255,255,0.04)",
        tickfont=dict(size=10, color="#71717a"),
    ),
)

# Header
head_left, head_right = st.columns([8, 2])
with head_left:
    st.markdown("""
    <div class="brand">
        <span class="brand-name">🛡️ ClaimsGuard AI</span>
        <span class="brand-tag">Multi-Model Agent</span>
    </div>
    """, unsafe_allow_html=True)
with head_right:
    theme_label = "☀️ Light Theme" if IS_DARK else "🌙 Dark Theme"
    st.button(theme_label, on_click=toggle_theme, use_container_width=True)

# Locate directories
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

with st.sidebar.expander("⚙️ LLM & VLM Settings", expanded=False):
    gemini_key = st.text_input("Gemini API Key", value="", type="password")
    groq_key = st.text_input("Groq API Key", value="", type="password")

        
    reasoner_provider = st.selectbox("Reasoner Provider", ["Gemini", "Groq"], index=1)
    
    if reasoner_provider == "Gemini":
        gemini_model = st.selectbox("Gemini Reasoner Model", ["gemma-4-31b-it", "gemma-4-26b-a4b-it", "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"])
        reasoner_model = gemini_model
    else:
        groq_reasoner_model = st.selectbox("Groq Reasoner Model", [
            "meta-llama/llama-4-scout-17b-16e-instruct", 
            "llama-3.3-70b-versatile", 
            "llama-3.1-8b-instant", 
            "qwen/qwen3-32b", 
            "qwen/qwen3.6-27b", 
            "allam-2-7b", 
            "groq/compound", 
            "groq/compound-mini", 
            "openai/gpt-oss-120b"
        ], index=2)
        reasoner_model = groq_reasoner_model
        gemini_model = "gemma-4-31b-it"

    groq_model = st.selectbox("Groq Vision Model", [
        "meta-llama/llama-4-scout-17b-16e-instruct", 
        "llama-3.2-11b-vision-preview", 
        "llama-3.2-90b-vision-preview"
    ])
    
    # Sidebar Reindex button
    if st.button("🔄 Reindex ChromaDB Rules", use_container_width=True):
        with st.spinner("Reindexing rules..."):
            agent.init_chroma()
            st.success("ChromaDB indexing completed!")

# Initialize agent with caching to avoid reloading SentenceTransformer/Chroma on every interaction
@st.cache_resource
def get_agent(gemini_key, groq_key, gemini_model, groq_model, repo_root, reasoner_provider, reasoner_model):
    agent_inst = ClaimsAgent(
        gemini_key=gemini_key,
        groq_key=groq_key,
        gemini_model=gemini_model,
        groq_model=groq_model,
        repo_root=repo_root,
        reasoner_provider=reasoner_provider,
        reasoner_model=reasoner_model
    )
    # Eagerly initialize Chroma and SentenceTransformer once
    agent_inst.init_chroma()
    return agent_inst

agent = get_agent(gemini_key, groq_key, gemini_model, groq_model, REPO_ROOT, reasoner_provider.lower(), reasoner_model)

if not hasattr(agent, "chat_with_data"):
    st.cache_resource.clear()
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# Load Datasets
@st.cache_data
def load_claims_data():
    sample_path = os.path.join(REPO_ROOT, "dataset", "sample_claims.csv")
    test_path = os.path.join(REPO_ROOT, "dataset", "claims.csv")
    sample_df = pd.read_csv(sample_path) if os.path.exists(sample_path) else pd.DataFrame()
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()
    return sample_df, test_df

sample_df, test_df = load_claims_data()

# Column Identifier Helper
def identify_columns(df):
    cols = [str(c) for c in df.columns]
    mapping = {}
    
    def find_match(candidates, default):
        for col in cols:
            col_lower = col.lower()
            if col_lower in candidates:
                return col
        for col in cols:
            col_lower = col.lower()
            for cand in candidates:
                if cand in col_lower or col_lower in cand:
                    return col
        return cols[0] if cols else default
        
    mapping["user_id"] = find_match(["user_id", "userid", "user", "client", "customer", "client_id"], "user_id")
    mapping["image_paths"] = find_match(["image_paths", "image_path", "images", "image", "paths", "photos", "pics", "photo", "evidence"], "image_paths")
    mapping["user_claim"] = find_match(["user_claim", "claim", "transcript", "conversation", "text", "chat", "message", "description", "details"], "user_claim")
    mapping["claim_object"] = find_match(["claim_object", "object", "type", "item", "category", "claim_type", "object_type"], "claim_object")
    return mapping

# Sidebar Dataset Upload
st.sidebar.markdown("### 📁 Dataset Management")
uploaded_file = st.sidebar.file_uploader("Upload custom claims dataset (CSV)", type=["csv"])

has_custom = False
if uploaded_file is not None:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        
        # Identify columns
        mapping = identify_columns(uploaded_df)
        
        # Show column mapping customization in an expander
        with st.sidebar.expander("🛠️ Column Auto-Mapping", expanded=True):
            st.info("Successfully identified columns! Adjust if needed:")
            col_options = list(uploaded_df.columns)
            user_id_col = st.selectbox("User ID Column", col_options, index=col_options.index(mapping["user_id"]) if mapping["user_id"] in col_options else 0)
            image_paths_col = st.selectbox("Image Paths Column", col_options, index=col_options.index(mapping["image_paths"]) if mapping["image_paths"] in col_options else 0)
            user_claim_col = st.selectbox("Transcript Column", col_options, index=col_options.index(mapping["user_claim"]) if mapping["user_claim"] in col_options else 0)
            claim_object_col = st.selectbox("Object Type Column", col_options, index=col_options.index(mapping["claim_object"]) if mapping["claim_object"] in col_options else 0)
            
        # Re-map the dataframe to standard names
        mapped_df = pd.DataFrame()
        mapped_df["user_id"] = uploaded_df[user_id_col]
        mapped_df["image_paths"] = uploaded_df[image_paths_col]
        mapped_df["user_claim"] = uploaded_df[user_claim_col]
        mapped_df["claim_object"] = uploaded_df[claim_object_col]
        
        for col in ["evidence_standard_met", "evidence_standard_met_reason", "risk_flags", "issue_type", "object_part", "claim_status", "claim_status_justification", "supporting_image_ids", "valid_image", "severity"]:
            if col in uploaded_df.columns:
                mapped_df[col] = uploaded_df[col]
                
        st.session_state.custom_df = mapped_df
        st.session_state.custom_df_raw = uploaded_df
        has_custom = True
        st.sidebar.success("Custom dataset loaded!")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

# Navigation Tabs
tab_single, tab_batch, tab_chat = st.tabs([
    "🔍 Single Claim Verification", 
    "📊 Batch Runner & Analytics", 
    "💬 Chat with Claims Dataset"
])

# Custom Helper to render Chat Transcript bubbles
def render_transcript(transcript):
    bubbles_html = ""
    lines = [line.strip() for line in transcript.split("|") if line.strip()]
    for line in lines:
        speaker = "System"
        msg = line
        if ":" in line:
            parts = line.split(":", 1)
            speaker = parts[0].strip()
            msg = parts[1].strip()
            
        is_user = "customer" in speaker.lower() or "cliente" in speaker.lower()
        bubble_class = "chat-user" if is_user else "chat-agent"
        bubbles_html += f'<div class="chat-bubble {bubble_class}"><div class="chat-speaker">{speaker}</div><div>{msg}</div></div>'
    st.markdown(f'<div class="chat-container">{bubbles_html}</div>', unsafe_allow_html=True)

# -----------------
# Tab 1: Single Claim Inspector
# -----------------
with tab_single:
    st.markdown("### Inspect and Verify Claim")
    
    # Dataset Selector
    sources = ["Sample Dataset (sample_claims.csv)", "Test Dataset (claims.csv)"]
    if has_custom:
        sources.append("Uploaded Custom Dataset")
    source_choice = st.radio("Claim Source", sources, horizontal=True)
    
    if source_choice == "Uploaded Custom Dataset" and "custom_df" in st.session_state:
        active_df = st.session_state.custom_df
    elif "Sample" in source_choice:
        active_df = sample_df
    else:
        active_df = test_df
    
    if active_df.empty:
        st.warning("No rows loaded in the selected dataset.")
    else:
        # Row Selector
        row_idx = st.selectbox("Select Row ID / Case", active_df.index, format_func=lambda idx: f"Row {idx+1}: User {active_df.iloc[idx]['user_id']} ({active_df.iloc[idx]['claim_object']})")
        claim = active_df.iloc[row_idx]
        
        # Display Grid
        c_left, c_right = st.columns([6, 4])
        
        with c_left:
            # 1. Conversation transcript
            st.markdown("##### Chat Transcript")
            render_transcript(claim['user_claim'])
            
            # 2. Submitted images
            st.markdown("##### Submitted Images")
            img_paths = [p.strip() for p in claim['image_paths'].split(";") if p.strip()]
            img_cols = st.columns(max(len(img_paths), 1))
            for i, path in enumerate(img_paths):
                abs_img_path = os.path.join(REPO_ROOT, "dataset", path.replace("\\", "/"))
                with img_cols[i]:
                    st.markdown(f"**Image ID: {os.path.basename(path).split('.')[0]}**")
                    if os.path.exists(abs_img_path):
                        img = Image.open(abs_img_path)
                        st.image(img, use_container_width=True)
                    else:
                        st.error(f"Image not found at {path}")
                        
        with c_right:
            # 3. User history
            st.markdown("##### User Claim History Profile")
            user_hist = agent.get_user_history(claim['user_id'])
            
            # Render History Info in a zinc card
            st.markdown(f"""
            <div class="panel-card">
                <div><strong>User ID:</strong> <span class="badge badge-gray">{user_hist['user_id']}</span></div>
                <div style="margin-top:0.4rem;"><strong>Prior Claims:</strong> {user_hist['past_claim_count']} (Accepted: {user_hist['accept_claim']} | Rejected: {user_hist['rejected_claim']} | Review: {user_hist['manual_review_claim']})</div>
                <div style="margin-top:0.4rem;"><strong>History Flags:</strong> 
                    {' '.join([f'<span class="badge badge-red">{f}</span>' if f != 'none' else '<span class="badge badge-green">none</span>' for f in user_hist['history_flags'].split(';') if f])}
                </div>
                <div style="margin-top:0.4rem; font-style: italic; font-size: 0.8rem; color: var(--text-muted);">"{user_hist['history_summary']}"</div>
            </div>
            """, unsafe_allow_html=True)
            
            # 4. Retrieved evidence requirements
            st.markdown("##### Retrieved Minimum Evidence Rules")
            with st.spinner("Retrieving evidence requirements from ChromaDB..."):
                requirements = agent.retrieve_requirements(claim['user_claim'], claim['claim_object'])
                reqs_html = ""
                for req in requirements:
                    reqs_html += f"<div style='margin-bottom:0.5rem; font-size:0.8rem;'>📄 {req}</div>"
                st.markdown(f"<div class='panel-card'>{reqs_html}</div>", unsafe_allow_html=True)
            
            # Run Verification Button
            if st.button("🚀 Verify Claim Evidence (Multi-Model Agent)", use_container_width=True):
                with st.spinner("Analyzing image visual facts with Groq & verifying decision with Gemini..."):
                    agent.reset_token_usage()
                    start_time = time.time()
                    
                    # Run Agent Pipeline
                    res = agent.verify_claim(
                        user_id=claim['user_id'],
                        image_paths=claim['image_paths'],
                        user_claim=claim['user_claim'],
                        claim_object=claim['claim_object']
                    )
                    latency = time.time() - start_time
                    
                    st.markdown("---")
                    st.markdown("#### Agent Verification Output")
                    
                    # KPI Outputs
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        met = "true" if res.get("evidence_standard_met") else "false"
                        met_badge = "badge-green" if met == "true" else "badge-red"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Evidence Standard Met</div>
                            <div class="metric-value"><span class="badge {met_badge}" style="font-size:1.1rem; padding: 4px 12px;">{met.upper()}</span></div>
                            <div style="font-size:0.7rem; color:var(--text-muted); margin-top:0.4rem;">{res.get('evidence_standard_met_reason')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c2:
                        status = res.get("claim_status", "unknown").upper()
                        status_badge = "badge-green" if status == "SUPPORTED" else ("badge-red" if status == "CONTRADICTED" else "badge-amber")
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Claim Status Decision</div>
                            <div class="metric-value"><span class="badge {status_badge}" style="font-size:1.1rem; padding: 4px 12px;">{status}</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                    with c3:
                        severity = res.get("severity", "unknown").upper()
                        sev_badge = "badge-red" if severity in ["HIGH", "MEDIUM"] else ("badge-green" if severity == "LOW" else "badge-gray")
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Severity Level</div>
                            <div class="metric-value"><span class="badge {sev_badge}" style="font-size:1.1rem; padding: 4px 12px;">{severity}</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Details
                    st.markdown(f"""
                    <div class="panel-card">
                        <div><strong>Object Part Visited:</strong> <span class="badge badge-gray">{res.get('object_part', 'unknown').upper()}</span></div>
                        <div style="margin-top:0.4rem;"><strong>Detected Issue Type:</strong> <span class="badge badge-blue">{res.get('issue_type', 'unknown').upper()}</span></div>
                        <div style="margin-top:0.4rem;"><strong>Supporting Image IDs:</strong> {res.get('supporting_image_ids', 'none')}</div>
                        <div style="margin-top:0.4rem;"><strong>Risk Flags Raised:</strong> 
                            {' '.join([f'<span class="badge badge-red">{f}</span>' if f != 'none' else '<span class="badge badge-green">none</span>' for f in res.get('risk_flags', 'none').split(';')])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("##### Grounded Justification")
                    st.info(res.get("claim_status_justification", "No justification provided."))
                    
                    # Tech Metrics inside an expander for a cleaner UI
                    with st.expander("🛠️ Show Technical Details (Latency, Tokens, and Cost)"):
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("Total Latency", f"{latency:.2f}s")
                        with c2:
                            st.metric("Gemini Tokens (In / Out)", f"{agent.token_usage['gemini_input']} / {agent.token_usage['gemini_output']}")
                        with c3:
                            st.metric("Groq Tokens (In / Out)", f"{agent.token_usage['groq_input']} / {agent.token_usage['groq_output']}")
                        with c4:
                            # calculate cost
                            gemini_cost = (agent.token_usage["gemini_input"] * 0.075 + agent.token_usage["gemini_output"] * 0.30) / 1_000_000
                            groq_cost = (agent.token_usage["groq_input"] * 0.20 + agent.token_usage["groq_output"] * 0.20) / 1_000_000
                            st.metric("Estimated Cost", f"${gemini_cost + groq_cost:.5f}")

# -----------------
# Tab 2: Batch Runner & Analytics
# -----------------
with tab_batch:
    st.markdown("### Batch Processing & Metrics Analytics")
    
    bc_left, bc_right = st.columns([4, 6])
    
    with bc_left:
        st.markdown("##### Execute Bulk Verification")
        batch_options = ["dataset/sample_claims.csv (20 Claims)", "dataset/claims.csv (45 Claims)"]
        if has_custom:
            batch_options.append("Uploaded Custom Dataset")
        batch_source = st.selectbox("Choose Batch Target", batch_options)
        
        run_bulk = st.button("▶️ Run Bulk Prediction Pipeline")
        
        if run_bulk:
            if batch_source == "Uploaded Custom Dataset" and "custom_df" in st.session_state:
                df_target = st.session_state.custom_df
            else:
                target_file = os.path.join(REPO_ROOT, batch_source.split(" ")[0])
                df_target = pd.read_csv(target_file)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            agent.reset_token_usage()
            start_time = time.time()
            
            bulk_results = []
            total = len(df_target)
            
            for idx, row in df_target.iterrows():
                status_text.text(f"Processing row {idx+1}/{total} (User {row['user_id']})...")
                res = agent.verify_claim(
                    user_id=row['user_id'],
                    image_paths=row['image_paths'],
                    user_claim=row['user_claim'],
                    claim_object=row['claim_object']
                )
                
                res_row = {
                    "user_id": row['user_id'],
                    "image_paths": row['image_paths'],
                    "user_claim": row['user_claim'],
                    "claim_object": row['claim_object'],
                    "evidence_standard_met": str(res.get("evidence_standard_met", False)).lower(),
                    "evidence_standard_met_reason": res.get("evidence_standard_met_reason", "none"),
                    "risk_flags": res.get("risk_flags", "none"),
                    "issue_type": res.get("issue_type", "unknown"),
                    "object_part": res.get("object_part", "unknown"),
                    "claim_status": res.get("claim_status", "not_enough_information"),
                    "claim_status_justification": res.get("claim_status_justification", "none"),
                    "supporting_image_ids": res.get("supporting_image_ids", "none"),
                    "valid_image": str(res.get("valid_image", True)).lower(),
                    "severity": res.get("severity", "unknown")
                }
                bulk_results.append(res_row)
                progress_bar.progress((idx + 1) / total)
                
            elapsed = time.time() - start_time
            progress_bar.empty()
            status_text.success(f"Successfully processed {total} claims in {elapsed:.1f}s!")
            
            # Save results to session state and disk
            res_df = pd.DataFrame(bulk_results)
            st.session_state.bulk_results_df = res_df
            st.session_state.bulk_metrics = {
                "elapsed": elapsed,
                "token_usage": agent.token_usage.copy()
            }
            
            # Save to disk
            try:
                output_csv_dataset = os.path.join(REPO_ROOT, "dataset", "output.csv")
                output_csv_root = os.path.join(REPO_ROOT, "output.csv")
                res_df.to_csv(output_csv_dataset, index=False)
                res_df.to_csv(output_csv_root, index=False)
                st.success(f"💾 Saved predictions to: \n- `{output_csv_root}`\n- `{output_csv_dataset}`")
            except Exception as disk_err:
                st.error(f"Error saving output.csv to disk: {disk_err}")
                
        # Show download button if bulk results are in session state
        if "bulk_results_df" in st.session_state:
            csv_data = st.session_state.bulk_results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions (output.csv)",
                data=csv_data,
                file_name="output.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    with bc_right:
        st.markdown("##### Evaluation Report & Metrics")
        report_path = os.path.join(REPO_ROOT, "code", "evaluation", "evaluation_report.md")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Evaluation report not yet generated. Run evaluation from terminal `python code/evaluation/main.py` or run bulk prediction above.")

    # Show charts if bulk results exist in session state
    if "bulk_results_df" in st.session_state:
        st.markdown("---")
        st.markdown("#### Batch Run Visualization Dashboard")
        df_res = st.session_state.bulk_results_df
        
        c1, c2, c3 = st.columns(3)
        
        with c1:
            # Claim Status Chart
            status_counts = df_res['claim_status'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Count']
            fig_status = px.bar(status_counts, x='Status', y='Count', color='Status',
                                color_discrete_map={"supported": "#22c55e", "contradicted": "#ef4444", "not_enough_information": "#f59e0b"},
                                title="Claims Decision Breakdown")
            fig_status.update_layout(PLOT_LAYOUT)
            st.plotly_chart(fig_status, use_container_width=True, config={"displayModeBar": False})
            
        with c2:
            # Severity Chart
            sev_counts = df_res['severity'].value_counts().reset_index()
            sev_counts.columns = ['Severity', 'Count']
            fig_sev = px.pie(sev_counts, names='Severity', values='Count', hole=0.4,
                             title="Claim Severity Distribution")
            fig_sev.update_layout(PLOT_LAYOUT)
            st.plotly_chart(fig_sev, use_container_width=True, config={"displayModeBar": False})
            
        with c3:
            # Object type counts
            obj_counts = df_res['claim_object'].value_counts().reset_index()
            obj_counts.columns = ['Object', 'Count']
            fig_obj = px.bar(obj_counts, x='Object', y='Count', title="Claims by Object Family")
            fig_obj.update_layout(PLOT_LAYOUT)
            st.plotly_chart(fig_obj, use_container_width=True, config={"displayModeBar": False})

# -----------------
# Tab 3: Chat with Claims Dataset
# -----------------
with tab_chat:
    st.markdown("### Chat with Claims Dataset")
    st.markdown("Ask natural language questions about your claims dataset. The assistant will answer your questions and run secure database analyses when needed.")
    
    # Select target dataset for Chat
    chat_sources = ["Sample Dataset (sample_claims.csv)", "Test Dataset (claims.csv)"]
    if has_custom:
        chat_sources.append("Uploaded Custom Dataset")
    active_chat_source = st.selectbox("Target Dataset for Chat Query", chat_sources, key="chat_target_selectbox")
    
    if active_chat_source == "Uploaded Custom Dataset" and "custom_df" in st.session_state:
        df_chat = st.session_state.custom_df.copy()
    elif "Sample" in active_chat_source:
        df_chat = sample_df.copy()
    else:
        df_chat = test_df.copy()

    # Setup chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            
    # Chat Input
    query = st.chat_input("Ask a question (e.g. 'What is the most common issue type for cars?', 'List all claims with high severity')")
    
    if query:
        # Render user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    chat_res = agent.chat_with_data(
                        query=query,
                        history=st.session_state.messages[:-1],
                        df=df_chat
                    )
                    
                    response_text = chat_res["conversational_response"]
                    code = chat_res["code"]
                    ans_data = chat_res["answer_data"]
                    
                    # Display the conversational response
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                    # Display the execution results if code was executed
                    if code:
                        if isinstance(ans_data, (pd.DataFrame, pd.Series)):
                            st.dataframe(ans_data)
                        
                        # Collapsible expander for clean UI
                        with st.expander("💻 View Executed Analysis Code"):
                            st.code(code, language="python")
                            
                except Exception as e:
                    error_msg = f"Failed to query dataset: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": f"Sorry, I couldn't compute the answer. Error: {str(e)}"})

