"""
app.py  –  HeartVigil AI  |  Main Streamlit Application
=========================================================
OTP-based email authentication  •  7 pages  •  4 AI agents
"""

from __future__ import annotations

import os
import io
import random
import string
import logging
import datetime
import requests
from typing import Any, Dict, List, Optional
# ========== COMPATIBILITY WRAPPERS ==========
# For data_agent compatibility
try:
    from data_agent import save_health_data as run_data_agent
except ImportError:
    from data_agent import DataAgent
    def run_data_agent(supabase, user_id, health_data, risk_score=None, risk_label=None, source="manual"):
        from data_agent import compute_delta, _fetch_latest_record
        # Implementation
        pass

# For risk_agent compatibility
try:
    from risk_agent import run_risk_agent
except ImportError:
    from risk_agent import doctor_ai_agent
    def run_risk_agent(health_data, use_groq=True, previous_record=None, delta=None):
        result = doctor_ai_agent(health_data, use_groq=use_groq)
        # Add direction info if previous_record exists
        if previous_record:
            result["direction_info"] = {
                "direction": "first" if not previous_record else "stable",
                "direction_icon": "🆕",
                "direction_color": "#6B46C1",
                "score_change": 0,
                "prev_label": None,
                "changed_fields": []
            }
        return result

# For monitor_agent compatibility
try:
    from monitor_agent import run_monitor_agent
except ImportError:
    import monitor_agent as ma
    def run_monitor_agent(supabase, user_id, use_groq=True, limit=100):
        records = ma.get_user_history(user_id)
        df = pd.DataFrame(records) if records else pd.DataFrame()
        trends = {}
        alerts = []
        for field in ["trestbps", "chol", "thalach", "oldpeak"]:
            latest, pct, sym = ma.compute_trends(records, field)
            if latest is not None:
                trends[field] = {"direction": sym, "pct_change": pct}
        alerts = ma.detect_trends(records)
        return {
            "has_history": len(records) > 0,
            "history": df,
            "trends": trends,
            "alerts": [{"severity": "MEDIUM", "message": a, "pct": ""} for a in alerts],
            "ai_summary": ma.generate_ai_summary(records) if len(records) >= 2 else "",
            "field_progress": []
        }

# For reco_agent compatibility
try:
    from reco_agent import run_reco_agent
except ImportError:
    from reco_agent import generate_recommendations
    def run_reco_agent(health_data, risk_output=None, progress_summary="", use_groq=True):
        tips = generate_recommendations(health_data)
        return {
            "tips": tips,
            "ai_narrative": risk_output.get("ai_explanation", "") if risk_output else "",
            "source": "rule-based",
            "disclaimer": "These recommendations are AI-generated and should be discussed with your healthcare provider."
        }

# For pdf_extractor compatibility
try:
    from pdf_extractor import extract_features_from_pdf
except ImportError:
    from pdf_extractor import parse_pdf_health_data as extract_features_from_pdf

# Helper function for _fetch_latest_record
def _fetch_latest_record(supabase, user_id):
    try:
        res = supabase.table("health_records").select("*").eq("user_id", user_id).order("created_at", desc=True).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception:
        return None

# Load .env file so SENDER_EMAIL, SENDER_PASSWORD, etc. are available
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)
except ImportError:
    pass  # dotenv not installed; fall back to st.secrets or system env

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Local modules ──────────────────────────────────────────────────────────────
from supabase_client import get_supabase
from data_agent      import run_data_agent, validate_fields, SAFE_RANGES
from risk_agent      import run_risk_agent, RISK_COLORS, FEATURE_COLS
from monitor_agent   import run_monitor_agent, METRIC_LABELS
from reco_agent      import run_reco_agent
from pdf_extractor   import extract_features_from_pdf

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Logo loader ───────────────────────────────────────────────────────────────
_LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")

def _logo_b64() -> str:
    """Return base64-encoded logo for HTML <img> tags. Empty string if not found."""
    if os.path.exists(_LOGO_PATH):
        import base64
        with open(_LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# ── Page config ───────────────────────────────────────────────────────────────
_page_icon = _LOGO_PATH if os.path.exists(_LOGO_PATH) else "❤️"
st.set_page_config(
    page_title="HeartVigil AI",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
def _inject_css():
    st.markdown(
        """
        <style>
        /* ════════════════════════════════════════════════════════
           FONTS
        ════════════════════════════════════════════════════════ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

        html, body, [class*="css"], [class*="st-"] {
            font-family: 'Inter', 'Plus Jakarta Sans', sans-serif !important;
        }

        /* ════════════════════════════════════════════════════════
           LIGHT MODE TOKENS
        ════════════════════════════════════════════════════════ */
        :root {
            --primary:       #7C3AED;
            --primary-d:     #5B21B6;
            --primary-l:     #A78BFA;
            --primary-glow:  rgba(124,58,237,.18);
            --accent:        #EC4899;
            --accent-l:      #F472B6;

            --low:           #059669;
            --low-bg:        #D1FAE5;
            --low-text:      #064E3B;
            --medium:        #D97706;
            --medium-bg:     #FEF3C7;
            --medium-text:   #78350F;
            --high:          #DC2626;
            --high-bg:       #FEE2E2;
            --high-text:     #7F1D1D;

            --bg:            #F4F3FF;
            --bg2:           #FFFFFF;
            --surface:       #FFFFFF;
            --surface2:      #F8F7FF;
            --surface3:      #EDE9FE;
            --text:          #1E1B2E;
            --text-2:        #3D3552;
            --text-muted:    #6B7280;
            --border:        #DDD6FE;
            --border-2:      #E5E7EB;
            --shadow:        0 4px 24px rgba(124,58,237,.10);
            --shadow-lg:     0 12px 48px rgba(124,58,237,.18);
            --radius:        16px;
            --radius-sm:     10px;
        }

        /* ════════════════════════════════════════════════════════
           DARK MODE TOKENS  (Streamlit sets data-theme="dark")
        ════════════════════════════════════════════════════════ */
        [data-theme="dark"] {
            --primary:       #A78BFA;
            --primary-d:     #7C3AED;
            --primary-l:     #C4B5FD;
            --primary-glow:  rgba(167,139,250,.20);
            --accent:        #F472B6;
            --accent-l:      #FBCFE8;

            --low:           #34D399;
            --low-bg:        #064E3B;
            --low-text:      #A7F3D0;
            --medium:        #FBBF24;
            --medium-bg:     #451A03;
            --medium-text:   #FDE68A;
            --high:          #F87171;
            --high-bg:       #450A0A;
            --high-text:     #FCA5A5;

            --bg:            #0F0B1A;
            --bg2:           #16102B;
            --surface:       #1C1533;
            --surface2:      #231B40;
            --surface3:      #2D2250;
            --text:          #F3F0FF;
            --text-2:        #DDD6FE;
            --text-muted:    #9CA3AF;
            --border:        #3D3166;
            --border-2:      #2D2550;
            --shadow:        0 4px 24px rgba(0,0,0,.40);
            --shadow-lg:     0 12px 48px rgba(0,0,0,.60);
        }

        /* ════════════════════════════════════════════════════════
           BASE / LAYOUT
        ════════════════════════════════════════════════════════ */
        .main, .stApp {
            background: var(--bg) !important;
        }
        .block-container {
            padding: 1.8rem 2rem 4rem !important;
            max-width: 1240px !important;
        }
        /* ── Main content text: scoped to block-container only ── */
        [data-testid="stAppViewContainer"] > .main .block-container p,
        [data-testid="stAppViewContainer"] > .main .block-container li,
        [data-testid="stAppViewContainer"] > .main [data-testid="stMarkdownContainer"] p,
        [data-testid="stAppViewContainer"] > .main [data-testid="stMarkdownContainer"] li {
            color: var(--text) !important;
        }
        [data-testid="stAppViewContainer"] > .main h1,
        [data-testid="stAppViewContainer"] > .main h2,
        [data-testid="stAppViewContainer"] > .main h3,
        [data-testid="stAppViewContainer"] > .main h4 {
            color: var(--text) !important;
        }
        [data-testid="stAppViewContainer"] > .main .stCaption > div,
        [data-testid="stAppViewContainer"] > .main [data-testid="stCaptionContainer"] p {
            color: var(--text-muted) !important;
        }
        /* Widget labels */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] label {
            color: var(--text-2) !important;
        }
        /* Input text values */
        .stTextInput input, .stNumberInput input {
            color: var(--text) !important;
        }


        /* ════════════════════════════════════════════════════════
           SIDEBAR
        ════════════════════════════════════════════════════════ */
        [data-testid="stSidebar"] {
            background: linear-gradient(165deg, #0F0B1A 0%, #1a0d38 45%, #2D1B69 100%) !important;
            border-right: 1px solid rgba(124,58,237,.3) !important;
        }
        [data-testid="stSidebar"] * { color: #EDE9FE !important; }
        [data-testid="stSidebar"] .stButton > button {
            background: rgba(255,255,255,.06) !important;
            border: 1px solid rgba(167,139,250,.25) !important;
            color: #DDD6FE !important;
            border-radius: 12px !important;
            text-align: left !important;
            width: 100% !important;
            padding: .65rem 1rem !important;
            font-size: .91rem !important;
            font-weight: 500 !important;
            letter-spacing: .01em !important;
            transition: all .2s ease !important;
            margin-bottom: 2px !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background: rgba(124,58,237,.35) !important;
            border-color: rgba(167,139,250,.5) !important;
            transform: translateX(5px) !important;
            color: #fff !important;
        }
        [data-testid="stSidebar"] .stButton > button:focus {
            background: linear-gradient(135deg,#7C3AED,#A855F7) !important;
            box-shadow: 0 0 0 2px #C4B5FD !important;
            color: #fff !important;
        }
        /* Sidebar toggle */
        [data-testid="stSidebar"] .stToggle label { color: #DDD6FE !important; }

        /* ════════════════════════════════════════════════════════
           CARDS
        ════════════════════════════════════════════════════════ */
        .hv-card {
            background: var(--surface);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 1.5rem;
            border: 1px solid var(--border);
            transition: transform .22s ease, box-shadow .22s ease;
            color: var(--text);
        }
        .hv-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-lg);
        }
        .hv-card * { color: var(--text) !important; }

        /* Risk cards — forced bright backgrounds with dark text for readability */
        .risk-card-low {
            background: linear-gradient(135deg, var(--low-bg), #A7F3D0) !important;
            border-left: 5px solid var(--low) !important;
        }
        .risk-card-low * { color: var(--low-text) !important; }
        .risk-card-medium {
            background: linear-gradient(135deg, var(--medium-bg), #FDE68A) !important;
            border-left: 5px solid var(--medium) !important;
        }
        .risk-card-medium * { color: var(--medium-text) !important; }
        .risk-card-high {
            background: linear-gradient(135deg, var(--high-bg), #FCA5A5) !important;
            border-left: 5px solid var(--high) !important;
        }
        .risk-card-high * { color: var(--high-text) !important; }

        /* ════════════════════════════════════════════════════════
           METRIC PILL
        ════════════════════════════════════════════════════════ */
        .metric-pill {
            display: inline-flex;
            align-items: center;
            gap: .5rem;
            background: var(--surface2);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: .45rem 1.1rem;
            font-size: .88rem;
            font-weight: 600;
            color: var(--text) !important;
            box-shadow: 0 2px 8px rgba(0,0,0,.07);
            margin: .2rem .3rem .2rem 0;
        }
        .metric-pill .arrow-up   { color: var(--high)   !important; }
        .metric-pill .arrow-down { color: var(--low)    !important; }
        .metric-pill .arrow-flat { color: var(--medium) !important; }

        /* ════════════════════════════════════════════════════════
           HERO BANNER
        ════════════════════════════════════════════════════════ */
        .hero {
            background: linear-gradient(135deg, #1a0533 0%, #3D1A78 50%, #7C3AED 100%);
            border-radius: 22px;
            padding: 3.2rem 2.8rem;
            color: #fff !important;
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(167,139,250,.3);
        }
        .hero::before {
            content: '❤️';
            position: absolute;
            right: 2.5rem; top: 50%;
            transform: translateY(-50%);
            font-size: 9rem;
            opacity: .07;
            pointer-events: none;
        }
        .hero::after {
            content: '';
            position: absolute;
            top: -60px; right: -60px;
            width: 250px; height: 250px;
            background: radial-gradient(circle, rgba(236,72,153,.35) 0%, transparent 70%);
            pointer-events: none;
        }
        .hero * { color: #fff !important; }
        .hero h1 { font-size: clamp(1.7rem,4vw,2.6rem); font-weight: 900; margin: 0 0 .6rem; line-height: 1.2; }
        .hero p  { font-size: clamp(.93rem,2vw,1.1rem); opacity: .88; max-width: 600px; line-height: 1.65; }
        .hero .badge {
            display: inline-block;
            background: rgba(255,255,255,.12);
            border: 1px solid rgba(255,255,255,.28);
            backdrop-filter: blur(8px);
            border-radius: 999px;
            padding: .3rem 1rem;
            font-size: .79rem;
            font-weight: 700;
            margin-bottom: 1rem;
            letter-spacing: .07em;
            text-transform: uppercase;
        }

        /* ════════════════════════════════════════════════════════
           FEATURE GRID
        ════════════════════════════════════════════════════════ */
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.2rem;
            margin-bottom: 2rem;
        }
        @media(max-width:768px) { .feature-grid { grid-template-columns: 1fr; } }
        .feature-card {
            background: var(--surface);
            border-radius: var(--radius);
            padding: 1.6rem 1.4rem;
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
            text-align: center;
            transition: transform .22s, box-shadow .22s;
        }
        .feature-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-lg); }
        .feature-card .icon { font-size: 2.4rem; margin-bottom: .6rem; display: block; }
        .feature-card h3 { font-size: 1rem; font-weight: 700; color: var(--text) !important; margin: 0 0 .4rem; }
        .feature-card p  { font-size: .86rem; color: var(--text-muted) !important; margin: 0; line-height: 1.55; }

        /* ════════════════════════════════════════════════════════
           STATS GRID
        ════════════════════════════════════════════════════════ */
        .stats-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 1.1rem; margin-bottom: 2rem; }
        @media(max-width:600px) { .stats-grid { grid-template-columns: 1fr; } }
        .stat-card {
            background: linear-gradient(135deg, #6D28D9, #7C3AED, #A855F7);
            border-radius: var(--radius);
            padding: 1.4rem 1.5rem;
            color: #fff !important;
            text-align: center;
            box-shadow: 0 8px 24px rgba(109,40,217,.35);
            border: 1px solid rgba(196,181,253,.2);
            transition: transform .2s;
        }
        .stat-card:hover { transform: translateY(-3px); }
        .stat-card * { color: #fff !important; }
        .stat-card .num { font-size: 2.1rem; font-weight: 900; line-height: 1; }
        .stat-card .lbl { font-size: .81rem; opacity: .85; margin-top: .3rem; }

        /* ════════════════════════════════════════════════════════
           ALERTS
        ════════════════════════════════════════════════════════ */
        .alert-high {
            background: var(--high-bg);
            border-left: 4px solid var(--high);
            border-radius: 10px;
            padding: .75rem 1rem;
            margin: .45rem 0;
            color: var(--high-text) !important;
            font-size: .9rem;
            font-weight: 500;
        }
        .alert-medium {
            background: var(--medium-bg);
            border-left: 4px solid var(--medium);
            border-radius: 10px;
            padding: .75rem 1rem;
            margin: .45rem 0;
            color: var(--medium-text) !important;
            font-size: .9rem;
            font-weight: 500;
        }
        .alert-low {
            background: var(--low-bg);
            border-left: 4px solid var(--low);
            border-radius: 10px;
            padding: .75rem 1rem;
            margin: .45rem 0;
            color: var(--low-text) !important;
            font-size: .9rem;
            font-weight: 500;
        }
        .alert-high *, .alert-medium *, .alert-low * { color: inherit !important; }

        /* ════════════════════════════════════════════════════════
           SECTION HEADING
        ════════════════════════════════════════════════════════ */
        .section-title {
            font-size: 1.2rem;
            font-weight: 800;
            color: var(--text) !important;
            margin: 1.4rem 0 .9rem;
            display: flex;
            align-items: center;
            gap: .6rem;
            letter-spacing: -.01em;
        }
        .section-title::after {
            content: '';
            flex: 1;
            height: 2px;
            background: linear-gradient(90deg, var(--primary), transparent);
            border-radius: 1px;
            opacity: .7;
        }

        /* ════════════════════════════════════════════════════════
           TIP CARD
        ════════════════════════════════════════════════════════ */
        .tip-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 1rem 1.25rem;
            margin-bottom: .75rem;
            display: flex;
            align-items: flex-start;
            gap: .9rem;
            box-shadow: var(--shadow);
            transition: transform .18s, box-shadow .18s;
        }
        .tip-card:hover { transform: translateX(5px); box-shadow: var(--shadow-lg); }
        .tip-card * { color: var(--text) !important; }
        .tip-num {
            background: linear-gradient(135deg, var(--primary-d), var(--primary));
            color: #fff !important;
            border-radius: 50%;
            width: 30px; height: 30px;
            display: flex; align-items: center; justify-content: center;
            font-size: .8rem; font-weight: 800;
            flex-shrink: 0; margin-top: .1rem;
        }

        /* ════════════════════════════════════════════════════════
           FORM INPUTS
        ════════════════════════════════════════════════════════ */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            background: var(--surface2) !important;
            border: 1.5px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            color: var(--text) !important;
            font-size: .95rem !important;
            transition: border-color .18s, box-shadow .18s !important;
        }
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px var(--primary-glow) !important;
            outline: none !important;
        }
        .stTextInput > div > div > input::placeholder,
        .stNumberInput > div > div > input::placeholder {
            color: var(--text-muted) !important;
        }
        .stSelectbox > div > div {
            background: var(--surface2) !important;
            border: 1.5px solid var(--border) !important;
            border-radius: var(--radius-sm) !important;
            color: var(--text) !important;
        }
        /* Labels */
        .stTextInput label, .stNumberInput label,
        .stSelectbox label, .stDateInput label,
        [data-testid="stWidgetLabel"] p {
            color: var(--text-2) !important;
            font-weight: 600 !important;
            font-size: .88rem !important;
        }

        /* ════════════════════════════════════════════════════════
           BUTTONS
        ════════════════════════════════════════════════════════ */
        .stButton > button {
            border-radius: var(--radius-sm) !important;
            font-weight: 700 !important;
            font-size: .93rem !important;
            transition: all .2s ease !important;
            border: 1.5px solid var(--border) !important;
            color: var(--text) !important;
            background: var(--surface2) !important;
        }
        .stButton > button:hover {
            border-color: var(--primary) !important;
            color: var(--primary) !important;
            box-shadow: 0 4px 16px var(--primary-glow) !important;
            transform: translateY(-2px) !important;
        }
        /* Primary buttons */
        [data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, var(--primary-d), var(--primary)) !important;
            color: #fff !important;
            border: none !important;
            box-shadow: 0 4px 16px var(--primary-glow) !important;
        }
        [data-testid="baseButton-primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 24px var(--primary-glow) !important;
            color: #fff !important;
        }

        /* ════════════════════════════════════════════════════════
           TABS
        ════════════════════════════════════════════════════════ */
        [data-baseweb="tab-list"] {
            gap: .4rem;
            background: var(--surface2) !important;
            border-radius: 12px !important;
            padding: .3rem !important;
            border: 1px solid var(--border) !important;
        }
        [data-baseweb="tab"] {
            background: transparent !important;
            border-radius: 9px !important;
            border: none !important;
            padding: .48rem 1.1rem !important;
            font-weight: 600 !important;
            font-size: .87rem !important;
            color: var(--text-muted) !important;
            transition: all .18s !important;
        }
        [data-baseweb="tab"]:hover {
            background: var(--surface3) !important;
            color: var(--text) !important;
        }
        [aria-selected="true"] {
            background: linear-gradient(135deg, var(--primary-d), var(--primary)) !important;
            color: #fff !important;
            box-shadow: 0 3px 10px var(--primary-glow) !important;
        }

        /* ════════════════════════════════════════════════════════
           EXPANDER
        ════════════════════════════════════════════════════════ */
        [data-testid="stExpander"] {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: var(--radius) !important;
        }
        [data-testid="stExpander"] summary p {
            color: var(--text) !important;
            font-weight: 600 !important;
        }

        /* ════════════════════════════════════════════════════════
           DATAFRAME / TABLE
        ════════════════════════════════════════════════════════ */
        [data-testid="stDataFrame"] {
            border-radius: var(--radius) !important;
            overflow: hidden !important;
            border: 1px solid var(--border) !important;
        }

        /* ════════════════════════════════════════════════════════
           CHART CONTAINER
        ════════════════════════════════════════════════════════ */
        .js-plotly-plot { border-radius: var(--radius); overflow: hidden; }

        /* ════════════════════════════════════════════════════════
           MISC
        ════════════════════════════════════════════════════════ */
        .stSpinner > div { border-top-color: var(--primary) !important; }
        .stSuccess, .stInfo, .stWarning, .stError {
            border-radius: var(--radius-sm) !important;
        }
        [data-testid="stToggle"] {
            accent-color: var(--primary) !important;
        }
        hr { border-color: var(--border) !important; opacity: .6; }
        ::selection { background: var(--primary); color: #fff; }

        /* ════════════════════════════════════════════════════════
           RESPONSIVE
        ════════════════════════════════════════════════════════ */
        @media (max-width: 768px) {
            .block-container { padding: .8rem .9rem 3rem !important; }
            .hero { padding: 2rem 1.3rem; }
            .hero::before, .hero::after { display: none; }
            .stats-grid { grid-template-columns: 1fr 1fr; }
            .feature-grid { grid-template-columns: 1fr; }
        }
        @media (max-width: 480px) {
            .stats-grid { grid-template-columns: 1fr; }
        }

        /* ════════════════════════════════════════════════════════
           ANIMATIONS
        ════════════════════════════════════════════════════════ */
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(18px); }
            to   { opacity: 1; transform: translateY(0);    }
        }
        @keyframes pulse-ring {
            0%   { transform: scale(.95); box-shadow: 0 0 0 0 rgba(124,58,237,.5); }
            70%  { transform: scale(1);   box-shadow: 0 0 0 14px rgba(124,58,237,0); }
            100% { transform: scale(.95); box-shadow: 0 0 0 0 rgba(124,58,237,0); }
        }
        @keyframes shimmer {
            0%   { background-position: -400px 0; }
            100% { background-position:  400px 0; }
        }
        .hv-card, .feature-card, .stat-card, .tip-card {
            animation: fadeUp .35s ease both;
        }

        /* ════════════════════════════════════════════════════════
           AUTH PAGE OVERRIDE  (separate block injected in _auth_page)
        ════════════════════════════════════════════════════════ */
        </style>
        """,
        unsafe_allow_html=True,
    )




# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS  ─  EMAIL / OTP
# ══════════════════════════════════════════════════════════════════════════════
def _generate_otp(length: int = 6) -> str:
    return "".join(random.choices(string.digits, k=length))


def send_otp_email(to_email: str, otp: str) -> bool:
    """Send OTP via Gmail SMTP (app-password). Returns True on success."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    sender  = os.environ.get("SENDER_EMAIL", "")
    pw      = os.environ.get("SENDER_PASSWORD", "")
    server  = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    port    = int(os.environ.get("SMTP_PORT", 587))

    if not sender:
        try:
            sender = st.secrets.get("SENDER_EMAIL", "")
            pw     = st.secrets.get("SENDER_PASSWORD", "")
        except Exception:
            pass

    if not sender:
        logger.warning("SMTP not configured – OTP is %s (dev mode)", otp)
        return True  # dev mode: skip sending

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "HeartVigil AI – Your Login OTP"
        msg["From"]    = f"HeartVigil AI <{sender}>"
        msg["To"]      = to_email

        html = f"""
        <div style="font-family:Inter,sans-serif;max-width:480px;margin:auto;
                    background:#fff;border-radius:16px;overflow:hidden;
                    box-shadow:0 4px 24px rgba(0,0,0,.1);">
          <div style="background:linear-gradient(135deg,#1a0533,#6B46C1);
                      padding:2rem;text-align:center;color:#fff;">
            <div style="font-size:3rem;">❤️</div>
            <h1 style="margin:.5rem 0;font-size:1.5rem;font-weight:800;">HeartVigil AI</h1>
          </div>
          <div style="padding:2.5rem;text-align:center;">
            <p style="color:#6c757d;margin-bottom:1.5rem;">Your one-time login code</p>
            <div style="background:#f0ebff;border-radius:12px;padding:1.5rem;
                        display:inline-block;margin-bottom:1.5rem;">
              <span style="font-size:2.5rem;font-weight:900;letter-spacing:.5rem;
                           color:#6B46C1;">{otp}</span>
            </div>
            <p style="color:#6c757d;font-size:.85rem;">
              This code expires in <strong>10 minutes</strong>.<br>
              Never share this code with anyone.
            </p>
          </div>
          <div style="background:#f8f9fa;padding:1rem;text-align:center;
                      color:#adb5bd;font-size:.78rem;">
            HeartVigil AI – For educational purposes only.
          </div>
        </div>
        """
        msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(server, port) as srv:
            srv.starttls()
            srv.login(sender, pw)
            srv.sendmail(sender, to_email, msg.as_string())
        logger.info("OTP email sent to %s", to_email)
        return True
    except Exception as exc:
        logger.error("Email send error: %s", exc)
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS  ─  SUPABASE USERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _supabase():
    return get_supabase()


def _get_or_create_user(email: str) -> Optional[Dict]:
    try:
        sb = _supabase()
        res = sb.table("users").select("*").eq("email", email).execute()
        if res.data:
            return res.data[0]
        ins = sb.table("users").insert({"email": email}).execute()
        return ins.data[0] if ins.data else None
    except Exception as exc:
        logger.error("User upsert error: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS  ─  PDF REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════
def _generate_pdf_report(
    user_email:   str,
    health_data:  Dict,
    risk_output:  Dict,
    reco_output:  Dict,
) -> bytes:
    """Generate a professional PDF assessment report using ReportLab."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib           import colors
        from reportlab.lib.units     import cm
        from reportlab.platypus      import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
        )
        from reportlab.lib.styles    import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums     import TA_CENTER, TA_LEFT

        buf    = io.BytesIO()
        doc    = SimpleDocTemplate(buf, pagesize=A4,
                                   leftMargin=2*cm, rightMargin=2*cm,
                                   topMargin=2*cm, bottomMargin=2*cm)
        styles = getSampleStyleSheet()

        PURPLE  = colors.HexColor("#6B46C1")
        DARK    = colors.HexColor("#1a1f2e")
        MUTED   = colors.HexColor("#6c757d")
        RISK_C  = {
            "LOW":    colors.HexColor("#10B981"),
            "MEDIUM": colors.HexColor("#F59E0B"),
            "HIGH":   colors.HexColor("#EF4444"),
        }

        def style(name, **kw):
            s = ParagraphStyle(name, parent=styles["Normal"], **kw)
            return s

        title_s  = style("title",  fontSize=22, textColor=PURPLE, fontName="Helvetica-Bold", spaceAfter=4, alignment=TA_CENTER)
        sub_s    = style("sub",    fontSize=10, textColor=MUTED,  spaceAfter=2, alignment=TA_CENTER)
        h2_s     = style("h2",     fontSize=13, textColor=PURPLE, fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
        body_s   = style("body",   fontSize=10, textColor=DARK,   leading=14)
        small_s  = style("small",  fontSize=8,  textColor=MUTED)

        rl = risk_output.get("risk_label", "UNKNOWN")
        rc = RISK_C.get(rl, PURPLE)

        story = [
            Spacer(1, .3*cm),
            Paragraph("❤ HeartVigil AI", title_s),
            Paragraph("Heart Disease Risk Assessment Report", sub_s),
            Paragraph(
                f"Generated: {datetime.datetime.now().strftime('%B %d, %Y  %I:%M %p')}  |  "
                f"Patient: {user_email}",
                small_s,
            ),
            HRFlowable(width="100%", thickness=1, color=PURPLE, spaceAfter=12, spaceBefore=8),
            Paragraph("Risk Summary", h2_s),
        ]

        risk_rows = [
            ["Risk Level", rl],
            ["Probability", f"{risk_output.get('risk_pct', 0):.1f}%"],
        ]
        risk_tbl = Table(risk_rows, colWidths=[5*cm, 10*cm])
        risk_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#f0ebff")),
            ("BACKGROUND", (1,0), (1,0), rc),
            ("TEXTCOLOR",  (1,0), (1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
            ("FONTNAME",   (0,0), (0,-1),  "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 10),
            ("ROWBACKGROUNDS", (1,1), (-1,-1), [colors.white, colors.HexColor("#fafafa")]),
            ("GRID",       (0,0), (-1,-1), .5, colors.HexColor("#E5E7EB")),
            ("ROUNDEDCORNERS", [6]),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ]))
        story += [risk_tbl, Spacer(1, .5*cm)]

        story.append(Paragraph("Key Clinical Risk Factors", h2_s))
        for r in risk_output.get("reasons", []):
            story.append(Paragraph(f"• {r}", body_s))
        story.append(Spacer(1, .4*cm))

        ai_exp = risk_output.get("ai_explanation", "")
        if ai_exp:
            story.append(Paragraph("AI Explanation", h2_s))
            story.append(Paragraph(ai_exp, body_s))
            story.append(Spacer(1, .4*cm))

        story.append(Paragraph("Health Metrics", h2_s))
        labels = {
            "age":"Age (years)","sex":"Sex (1=M,0=F)","cp":"Chest Pain Type",
            "trestbps":"Resting BP (mmHg)","chol":"Cholesterol (mg/dL)",
            "fbs":"Fasting BS>120","restecg":"Resting ECG","thalach":"Max HR (bpm)",
            "exang":"Exercise Angina","oldpeak":"ST Depression","slope":"ST Slope",
            "ca":"Vessels Colored","thal":"Thal",
        }
        metric_rows = [["Metric", "Value"]] + [
            [labels.get(k, k), str(v)]
            for k, v in health_data.items()
            if k in labels
        ]
        m_tbl = Table(metric_rows, colWidths=[8*cm, 7*cm])
        m_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), PURPLE),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 9.5),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f3ff")]),
            ("GRID",       (0,0), (-1,-1), .4, colors.HexColor("#E5E7EB")),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING", (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ]))
        story += [m_tbl, Spacer(1, .5*cm)]

        story.append(Paragraph("Personalised Recommendations", h2_s))
        for i, tip in enumerate(reco_output.get("tips", []), 1):
            story.append(Paragraph(f"{i}. {tip}", body_s))
        story.append(Spacer(1, .5*cm))

        story.append(HRFlowable(width="100%", thickness=.5, color=MUTED, spaceAfter=6))
        story.append(Paragraph(
            "⚠ Disclaimer: This report is AI-generated for educational purposes only. "
            "It does not constitute medical advice. Always consult a qualified healthcare "
            "professional before making changes to your health routine.",
            style("disc", fontSize=8, textColor=MUTED, leading=11),
        ))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        return b""
    except Exception as exc:
        logger.error("PDF generation error: %s", exc)
        return b""


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
def _init_session():
    defaults = {
        "authenticated":  False,
        "user":           None,
        "page":           "Dashboard",
        "otp":            None,
        "otp_email":      None,
        "otp_expiry":     None,
        "otp_resends":    0,
        "email_input":    "",
        "risk_output":    None,
        "health_data":    None,
        "monitor_output": None,
        "reco_output":    None,
        "use_groq":       True,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH PAGE
# ══════════════════════════════════════════════════════════════════════════════
def _auth_page():
    st.markdown("""
    <style>
    #MainMenu, footer, header { visibility: hidden !important; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    .auth-bg {
        min-height: 100vh; display: flex; align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #0F0B1A 0%, #1a0d38 40%, #2D1B69 70%, #5B21B6 100%);
        padding: 2rem 1rem; position: relative; overflow: hidden;
    }
    .auth-bg::before {
        content: ""; position: fixed; top: -120px; right: -120px;
        width: 450px; height: 450px;
        background: radial-gradient(circle, rgba(236,72,153,.28) 0%, transparent 65%);
        pointer-events: none; animation: blob1 9s ease-in-out infinite;
    }
    .auth-bg::after {
        content: ""; position: fixed; bottom: -100px; left: -100px;
        width: 380px; height: 380px;
        background: radial-gradient(circle, rgba(124,58,237,.32) 0%, transparent 65%);
        pointer-events: none; animation: blob1 11s ease-in-out infinite reverse;
    }
    @keyframes blob1 {
        0%,100% { transform: translateY(0) scale(1); }
        50%      { transform: translateY(-28px) scale(1.06); }
    }
    .auth-card {
        background: rgba(255,255,255,.055);
        backdrop-filter: blur(28px); -webkit-backdrop-filter: blur(28px);
        border: 1px solid rgba(196,181,253,.18);
        border-radius: 28px; padding: 2.8rem 2.5rem 2.4rem;
        width: 100%; max-width: 460px;
        box-shadow: 0 32px 80px rgba(0,0,0,.55), inset 0 1px 0 rgba(255,255,255,.08);
        position: relative; z-index: 1;
        animation: cardIn .5s cubic-bezier(.22,1,.36,1) both;
    }
    @keyframes cardIn {
        from { opacity:0; transform: translateY(28px) scale(.97); }
        to   { opacity:1; transform: translateY(0)    scale(1);   }
    }
    .auth-heart-ring {
        display: inline-flex; align-items: center; justify-content: center;
        width: 76px; height: 76px; border-radius: 50%;
        background: linear-gradient(135deg, #6D28D9, #EC4899);
        font-size: 2.2rem; margin-bottom: .9rem;
        animation: heartPulse 2.6s ease infinite;
        box-shadow: 0 0 0 0 rgba(124,58,237,.5);
    }
    @keyframes heartPulse {
        0%,100% { box-shadow: 0 0 0 0    rgba(124,58,237,.45); transform: scale(1);    }
        50%      { box-shadow: 0 0 0 14px rgba(124,58,237,0);   transform: scale(1.07); }
    }
    .auth-title {
        font-size: 1.85rem; font-weight: 900; letter-spacing:-.025em;
        color: #fff !important;
        text-shadow: 0 2px 16px rgba(124,58,237,.6);
        margin: 0 0 .15rem;
    }
    .auth-subtitle { font-size:.86rem; color:rgba(196,181,253,.7) !important; margin:0 0 2rem; }
    .auth-step-pill {
        display:inline-block; background:rgba(124,58,237,.3);
        border:1px solid rgba(196,181,253,.3); border-radius:999px;
        padding:.22rem .8rem; font-size:.72rem; font-weight:700;
        color:#C4B5FD !important; letter-spacing:.07em;
        text-transform:uppercase; margin-bottom:.75rem;
    }
    .auth-heading { font-size:1.2rem; font-weight:800; color:#fff !important; margin-bottom:.3rem; }
    .auth-caption { font-size:.85rem; color:rgba(196,181,253,.68) !important; margin-bottom:1.3rem; line-height:1.6; }
    .auth-card .stTextInput>div>div>input {
        background:rgba(255,255,255,.07) !important;
        border:1.5px solid rgba(196,181,253,.28) !important;
        border-radius:13px !important; color:#fff !important;
        font-size:.97rem !important; padding:.68rem 1rem !important;
        transition: border-color .18s, box-shadow .18s !important;
    }
    .auth-card .stTextInput>div>div>input::placeholder { color:rgba(196,181,253,.38) !important; }
    .auth-card .stTextInput>div>div>input:focus {
        border-color:#A78BFA !important;
        box-shadow:0 0 0 3px rgba(124,58,237,.28) !important; outline:none !important;
    }
    .auth-card .stTextInput label { color:rgba(196,181,253,.75) !important; font-weight:600 !important; font-size:.83rem !important; }
    .auth-card [data-testid="baseButton-primary"] {
        background: linear-gradient(135deg,#5B21B6,#7C3AED,#A855F7) !important;
        border:none !important; border-radius:13px !important;
        font-size:.97rem !important; font-weight:800 !important; color:#fff !important;
        box-shadow:0 6px 22px rgba(91,33,182,.55) !important;
        letter-spacing:.015em; padding:.72rem !important;
        transition: all .2s !important;
    }
    .auth-card [data-testid="baseButton-primary"]:hover {
        transform:translateY(-2px) !important;
        box-shadow:0 10px 30px rgba(91,33,182,.7) !important;
    }
    .auth-card .stButton>button:not([data-testid="baseButton-primary"]) {
        background:rgba(255,255,255,.06) !important;
        border:1px solid rgba(196,181,253,.22) !important;
        border-radius:13px !important; color:#C4B5FD !important; font-weight:600 !important;
    }
    .auth-card .stButton>button:not([data-testid="baseButton-primary"]):hover {
        background:rgba(255,255,255,.12) !important;
        border-color:rgba(196,181,253,.45) !important;
    }
    .auth-email-badge {
        background:rgba(124,58,237,.22); border:1px solid rgba(196,181,253,.28);
        border-radius:11px; padding:.5rem .9rem; margin-bottom:1.1rem;
        font-size:.86rem; color:#C4B5FD !important; font-weight:600;
        display:flex; align-items:center; gap:.5rem;
    }
    .auth-chips { display:flex; gap:.45rem; flex-wrap:wrap; justify-content:center; margin-top:1.6rem; }
    .auth-chip {
        background:rgba(255,255,255,.05); border:1px solid rgba(196,181,253,.18);
        border-radius:999px; padding:.22rem .75rem;
        font-size:.71rem; color:rgba(196,181,253,.6) !important; font-weight:500;
    }
    .auth-divider {
        border:none; border-top:1px solid rgba(196,181,253,.12); margin:1.5rem 0 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        _b64 = _logo_b64()
        logo_inner = (
            f'<img src="data:image/png;base64,{_b64}" '
            f'style="width:48px;height:48px;object-fit:contain;border-radius:50%;">'
            if _b64 else "🧡"
        )
        st.markdown(f"""
        <div style="text-align:center;padding-top:2rem;">
          <div class="auth-heart-ring">{logo_inner}</div>
          <div class="auth-title">HeartVigil AI</div>
          <div class="auth-subtitle">AI-Powered Heart Disease Risk Platform</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.otp is None:
            # ── Step 1: email entry ────────────────────────────────────────
            st.markdown('<div class="auth-step-pill">Step 1 of 2</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-heading">Sign In or Register</div>', unsafe_allow_html=True)
            st.markdown(
                "<div class=\"auth-caption\">Enter your email and we'll send a secure "
                "one-time code. No password required.</div>",
                unsafe_allow_html=True,
            )

            email = st.text_input(
                "Email address",
                placeholder="you@example.com",
                key="auth_email_input",
            )
            send_btn = st.button(
                "📨  Send Secure Code",
                use_container_width=True,
                type="primary",
            )
            if send_btn:
                email = email.strip().lower()
                if "@" not in email or "." not in email:
                    st.error("Please enter a valid email address.")
                else:
                    otp    = _generate_otp()
                    expiry = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)
                    with st.spinner("Sending secure code…"):
                        ok = send_otp_email(email, otp)
                    if ok:
                        st.session_state.otp         = otp
                        st.session_state.otp_email   = email
                        st.session_state.otp_expiry  = expiry
                        st.session_state.otp_resends = 0
                        st.success(f"Code sent to **{email}**. Check your inbox!")
                        st.rerun()
                    else:
                        st.error("Failed to send code. Check SMTP settings.")
        else:
            # ── Step 2: OTP verify ─────────────────────────────────────────
            st.markdown('<div class="auth-step-pill">Step 2 of 2</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-heading">Enter Your Code</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="auth-email-badge">'
                f'📧  Code sent to <strong>{st.session_state.otp_email}</strong>'
                f'</div>',
                unsafe_allow_html=True,
            )
            entered = st.text_input(
                "6-digit code",
                max_chars=6,
                placeholder="Enter 6-digit code",
                key="auth_otp_input",
            )
            col1, col2 = st.columns([3, 2])
            with col1:
                verify_btn = st.button(
                    "✓  Verify & Sign In",
                    use_container_width=True,
                    type="primary",
                )
            with col2:
                back_btn = st.button("← Back", use_container_width=True)

            if back_btn:
                st.session_state.otp       = None
                st.session_state.otp_email = None
                st.rerun()

            now = datetime.datetime.utcnow()
            if st.session_state.otp_resends < 2:
                if st.button("🔄 Resend Code", use_container_width=True):
                    new_otp = _generate_otp()
                    with st.spinner("Sending new code…"):
                        send_otp_email(st.session_state.otp_email, new_otp)
                    st.session_state.otp         = new_otp
                    st.session_state.otp_expiry  = now + datetime.timedelta(minutes=10)
                    st.session_state.otp_resends += 1
                    st.success("New code sent!")
                    st.rerun()

            if verify_btn:
                if now > st.session_state.otp_expiry:
                    st.error("Code expired. Please request a new one.")
                    st.session_state.otp = None
                    st.rerun()
                elif entered.strip() == st.session_state.otp:
                    with st.spinner("Signing you in…"):
                        user = _get_or_create_user(st.session_state.otp_email)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user           = user
                        st.session_state.otp            = None
                        st.success("✨ Welcome! Loading dashboard…")
                        st.rerun()
                    else:
                        st.error("Could not sign in. Check Supabase settings.")
                else:
                    st.error("Incorrect code — please try again.")

        st.markdown('<hr class="auth-divider">', unsafe_allow_html=True)
        st.markdown("""
        <div class="auth-chips">
          <span class="auth-chip">🔒 Passwordless Auth</span>
          <span class="auth-chip">🤖 4 AI Agents</span>
          <span class="auth-chip">📈 Risk Tracking</span>
          <span class="auth-chip">📋 PDF Reports</span>
        </div>
        <p style="text-align:center;font-size:.72rem;color:rgba(196,181,253,.4);margin-top:1.1rem;">
          For educational and research purposes only
        </p>
        """, unsafe_allow_html=True)



def _sidebar():
    PAGES = [
        ("\U0001f3e0", "Dashboard"),
        ("\U0001f4dd", "Assessment"),
        ("\U0001f6e1\ufe0f", "Risk Analysis"),
        ("\U0001f4ca", "Data Agent"),
        ("\U0001f4c8", "Monitoring"),
        ("\U0001f4a1", "Recommendations"),
        ("\U0001f464", "Profile"),
    ]
    with st.sidebar:
        _b64 = _logo_b64()
        if _b64:
            st.markdown(f"""
            <div style="text-align:center;padding:1rem 0 1.2rem;">
              <img src="data:image/png;base64,{_b64}"
                   style="width:80px;height:80px;object-fit:contain;
                          filter:drop-shadow(0 0 12px rgba(167,139,250,.6));">
              <div style="font-size:1.1rem;font-weight:800;color:#e9d8fd;margin-top:.4rem;">HeartVigil AI</div>
              <div style="font-size:.72rem;color:#c4b5fd;margin-top:.15rem;">Heart Health Monitor</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:1rem 0 1.5rem;">
              <div style="font-size:3rem;line-height:1;">❤️</div>
              <div style="font-size:1.1rem;font-weight:800;color:#e9d8fd;margin-top:.5rem;">HeartVigil AI</div>
              <div style="font-size:.72rem;color:#c4b5fd;margin-top:.15rem;">Heart Health Monitor</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color:rgba(255,255,255,.1);margin:.5rem 0 1rem;'>", unsafe_allow_html=True)

        for icon, page in PAGES:
            if st.button(f"{icon}  {page}", key=f"nav_{page}", use_container_width=True):
                st.session_state.page = page
                st.rerun()

        st.markdown("<hr style='border-color:rgba(255,255,255,.1);margin:1rem 0;'>", unsafe_allow_html=True)

        # Groq toggle
        st.session_state.use_groq = st.toggle(
            "🤖 AI Explanations",
            value=st.session_state.use_groq,
            help="Enable Groq LLM for AI-powered explanations and recommendations.",
        )

        st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)

        # User info
        email = st.session_state.user.get("email", "") if st.session_state.user else ""
        st.markdown(
            f"<div style='font-size:.75rem;color:#c4b5fd;margin-bottom:.3rem;'>Signed in as</div>"
            f"<div style='font-size:.82rem;color:#e9d8fd;word-break:break-all;margin-bottom:.8rem;'>{email}</div>",
            unsafe_allow_html=True,
        )

        if st.button("🚪 Sign Out", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _gauge_chart(pct: float, label: str, color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 28, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ccc"},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#d1fae5"},
                {"range": [40, 70], "color": "#fef3c7"},
                {"range": [70, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": pct,
            },
        },
        title={"text": f"Risk Score<br><span style='font-size:.9em;color:{color}'>{label}</span>",
               "font": {"size": 14, "family": "Inter"}},
    ))
    fig.update_layout(
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"t": 60, "b": 20, "l": 30, "r": 30},
        font={"family": "Inter"},
    )
    return fig


def _comparison_chart(chart_data: List[Dict]) -> go.Figure:
    if not chart_data:
        return go.Figure()

    metrics   = [d["metric"] for d in chart_data]
    values    = [d["value"]  for d in chart_data]
    safe_lo   = [d["safe_lo"] for d in chart_data]
    safe_hi   = [d["safe_hi"] for d in chart_data]

    colors_bar = []
    for d in chart_data:
        v, lo, hi = d["value"], d["safe_lo"], d["safe_hi"]
        if lo <= v <= hi:
            colors_bar.append("#10B981")
        elif v < lo:
            colors_bar.append("#F59E0B")
        else:
            colors_bar.append("#EF4444")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metrics, y=values,
        name="Your Value",
        marker_color=colors_bar,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        x=metrics, y=safe_hi,
        mode="markers+lines",
        name="Safe Range Upper",
        line={"color": "#6B46C1", "dash": "dot", "width": 1.5},
        marker={"size": 6},
    ))
    fig.add_trace(go.Scatter(
        x=metrics, y=safe_lo,
        mode="markers+lines",
        name="Safe Range Lower",
        line={"color": "#A78BFA", "dash": "dot", "width": 1.5},
        marker={"size": 6},
    ))
    fig.update_layout(
        title={"text": "Your Values vs Safe Ranges", "font": {"size": 14, "family": "Inter"}},
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFAFA",
        legend={"orientation": "h", "y": -0.2},
        margin={"t": 50, "b": 60, "l": 30, "r": 10},
        font={"family": "Inter", "size": 11},
    )
    return fig


def _trend_chart(df: pd.DataFrame, metric: str, label: str) -> go.Figure:
    if df.empty or metric not in df.columns:
        return go.Figure()
    dates = df["created_at"].dt.strftime("%b %d").tolist()
    vals  = df[metric].tolist()
    if not vals:
        return go.Figure()
    n      = int(len(vals))
    window = max(1, min(3, n))
    try:
        roll = pd.Series(vals).rolling(window=window, min_periods=1).mean().tolist()
    except Exception:
        roll = vals

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=vals,
        name=label,
        mode="lines+markers",
        line={"color": "#6B46C1", "width": 2.5},
        marker={"size": 7},
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=roll,
        name="Rolling Avg",
        mode="lines",
        line={"color": "#EC4899", "width": 1.5, "dash": "dot"},
    ))
    fig.update_layout(
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FAFAFA",
        margin={"t": 30, "b": 40, "l": 30, "r": 10},
        font={"family": "Inter", "size": 11},
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def _page_dashboard():
    user  = st.session_state.user
    email = user.get("email", "")
    joined = user.get("created_at", "")[:10]

    st.markdown(f"""
    <div class="hero">
      <div class="badge">❤️ AI-Powered Cardiac Risk Assessment</div>
      <h1>Welcome back!</h1>
      <p>HeartVigil AI analyses your health metrics using a trained Random Forest + XGBoost
         ensemble with Groq AI explanations to help you understand your heart disease risk.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-grid">
      <div class="feature-card">
        <div class="icon">🔬</div>
        <h3>Smart Assessment</h3>
        <p>13-field health form or PDF upload. ML-powered risk in seconds.</p>
      </div>
      <div class="feature-card">
        <div class="icon">📈</div>
        <h3>Trend Monitoring</h3>
        <p>Track your metrics over time with date-range filtering and intelligent trend detection.</p>
      </div>
      <div class="feature-card">
        <div class="icon">🤖</div>
        <h3>AI Insights</h3>
        <p>Llama 3.3 70B generates personalised explanations citing your actual values.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        sb  = _supabase()
        res = sb.table("health_records").select("risk_label,risk_score").eq("user_id", user["id"]).execute()
        records = res.data or []
    except Exception:
        records = []

    n_total    = len(records)
    latest_lbl = records[-1]["risk_label"] if records else "N/A"
    _rs = records[-1].get("risk_score") if records else None
    latest_sc  = f"{_rs * 100:.1f}%" if _rs is not None else "N/A"

    st.markdown(f"""
    <div class="stats-grid">
      <div class="stat-card">
        <div class="num">{n_total}</div>
        <div class="lbl">Total Assessments</div>
      </div>
      <div class="stat-card">
        <div class="num">{latest_lbl}</div>
        <div class="lbl">Latest Risk Level</div>
      </div>
      <div class="stat-card">
        <div class="num">{latest_sc}</div>
        <div class="lbl">Latest Risk Score</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🏠 Welcome", "📊 Latest Assessment"])

    with tab1:
        st.markdown('<p class="section-title">🚀 Get Started</p>', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(
                "Take your first health assessment to get a personalised heart disease risk score, "
                "AI explanation, and health recommendations.",
                icon="ℹ️",
            )
            if st.button("🔬 Start New Assessment", type="primary", use_container_width=True):
                st.session_state.page = "Assessment"
                st.rerun()
        with col2:
            st.markdown(f"""
            <div class="hv-card">
              <div style="font-size:.82rem;color:#6c757d;">Account</div>
              <div style="font-weight:700;font-size:.95rem;margin:.3rem 0;word-break:break-all;">{email}</div>
              <div style="font-size:.8rem;color:#6c757d;">Joined: {joined}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        if st.session_state.risk_output is None:
            st.info("No assessment completed yet. Take your first assessment!")
            if st.button("📋 Go to Assessment", type="primary"):
                st.session_state.page = "Assessment"
                st.rerun()
        else:
            _show_latest_assessment_tab()


def _show_latest_assessment_tab():
    ro = st.session_state.risk_output
    hd = st.session_state.health_data
    if not ro or not hd:
        return

    label = ro["risk_label"]
    pct   = ro["risk_pct"]
    color = ro["risk_color"]
    cls   = f"risk-card-{label.lower()}"

    st.markdown(f"""
    <div class="hv-card {cls}" style="margin-bottom:1.5rem;">
      <div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
        <div>
          <div style="font-size:.82rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;opacity:.7;">Risk Level</div>
          <div style="font-size:2.5rem;font-weight:900;color:{color};">{label}</div>
        </div>
        <div style="text-align:right;flex:1;">
          <div style="font-size:.82rem;opacity:.7;">Probability</div>
          <div style="font-size:2rem;font-weight:800;">{pct:.1f}%</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<p class="section-title">🔑 Key Risk Factors</p>', unsafe_allow_html=True)
        for r in ro.get("reasons", []):
            st.markdown(f"<div class='alert-medium'>• {r}</div>", unsafe_allow_html=True)

    with col2:
        st.plotly_chart(_gauge_chart(pct, label, color), use_container_width=True)

    ai_exp = ro.get("ai_explanation", "")
    if ai_exp:
        with st.expander("🤖 AI Insights", expanded=True):
            st.markdown(f"<div style='line-height:1.7;font-size:.95rem;'>{ai_exp}</div>", unsafe_allow_html=True)

    from monitor_agent import build_comparison_chart_data
    cdata = build_comparison_chart_data(pd.Series(hd))
    if cdata:
        st.markdown('<p class="section-title">📊 Values vs Safe Ranges</p>', unsafe_allow_html=True)
        st.plotly_chart(_comparison_chart(cdata), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        reco = st.session_state.reco_output or {"tips": []}
        pdf  = _generate_pdf_report(
            st.session_state.user.get("email", ""),
            hd, ro, reco,
        )
        if pdf:
            st.download_button(
                "📥 Download PDF Report",
                data=pdf,
                file_name=f"heartvigil_report_{datetime.date.today()}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
    with col2:
        if st.button("🔬 Take New Assessment", use_container_width=True):
            st.session_state.page = "Assessment"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
def _page_assessment():
    st.markdown('<h2 style="color:var(--text,#1a1f2e);font-weight:800;">📋 Health Assessment</h2>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:var(--text-muted,#6c757d);margin-bottom:1.5rem;'>"
        "Fill in your health metrics or upload a PDF medical report to auto-fill the form.</p>",
        unsafe_allow_html=True,
    )

    with st.expander("Upload PDF Medical Report (Optional)", expanded=False):
        pdf_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            help="We'll attempt to extract your health data from the report.",
            key="pdf_uploader",
        )
        if pdf_file is not None and "_pdf_extracted" not in st.session_state:
            with st.spinner("Extracting data from PDF\u2026"):
                extracted = extract_features_from_pdf(pdf_file.read())
            n = sum(1 for v in extracted.values() if v is not None)
            if n > 0:
                st.session_state["_pdf_data"]      = extracted
                st.session_state["_pdf_extracted"] = True  # prevent re-run loop

                _CP  = [(0,"Typical Angina"),(1,"Atypical Angina"),(2,"Non-Anginal Pain"),(3,"Asymptomatic")]
                _FBS = [(0,"No  (\u2264120 mg/dL)"),(1,"Yes (>120 mg/dL)")]
                _ECG = [(0,"Normal"),(1,"ST-T Wave Abnormality"),(2,"LV Hypertrophy")]
                _EXA = [(0,"No"),(1,"Yes")]
                _SLP = [(0,"Upsloping"),(1,"Flat"),(2,"Downsloping")]
                _THL = [(1,"Normal"),(2,"Fixed Defect"),(3,"Reversible Defect")]

                def _si(val, lo, hi, default=0):
                    try: return max(lo, min(hi, int(float(val))))
                    except Exception: return default

                if extracted.get("age")      is not None: st.session_state["f_age"]   = _si(extracted["age"], 1, 120)
                if extracted.get("sex")      is not None: st.session_state["f_sex"]   = ("Male",1) if _si(extracted["sex"],0,1)==1 else ("Female",0)
                if extracted.get("cp")       is not None: st.session_state["f_cp"]    = _CP[_si(extracted["cp"],0,3)]
                if extracted.get("trestbps") is not None: st.session_state["f_bp"]    = _si(extracted["trestbps"],60,250)
                if extracted.get("chol")     is not None: st.session_state["f_chol"]  = _si(extracted["chol"],1,700)
                if extracted.get("fbs")      is not None: st.session_state["f_fbs"]   = _FBS[_si(extracted["fbs"],0,1)]
                if extracted.get("restecg")  is not None: st.session_state["f_ecg"]   = _ECG[_si(extracted["restecg"],0,2)]
                if extracted.get("thalach")  is not None: st.session_state["f_hr"]    = _si(extracted["thalach"],1,300)
                if extracted.get("exang")    is not None: st.session_state["f_exang"] = _EXA[_si(extracted["exang"],0,1)]
                if extracted.get("oldpeak")  is not None:
                    try: st.session_state["f_op"] = round(float(extracted["oldpeak"]),1)
                    except Exception: pass
                if extracted.get("slope")    is not None: st.session_state["f_slope"] = _SLP[_si(extracted["slope"],0,2)]
                if extracted.get("ca")       is not None: st.session_state["f_ca"]    = _si(extracted["ca"],0,4)
                if extracted.get("thal")     is not None:
                    tv = _si(extracted["thal"],1,3)
                    st.session_state["f_thal"] = _THL[{1:0,2:1,3:2}.get(tv,0)]

                st.success(f"Extracted {n} / 13 fields from PDF. Form filled below \u2193")
                st.rerun()
            else:
                st.warning("Could not extract data. Please fill the form manually.")
        elif pdf_file is not None and "_pdf_extracted" in st.session_state:
            n = sum(1 for v in st.session_state.get("_pdf_data", {}).values() if v is not None)
            st.success(f"Extracted {n} / 13 fields from PDF. Form filled below \u2193")



    _, col_btn = st.columns([3, 1])
    with col_btn:
        if st.button("🔄 New Assessment", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k.startswith("f_") or k in ("_pdf_data", "_pdf_extracted"):
                    del st.session_state[k]
            st.rerun()

    pdf_data = st.session_state.get("_pdf_data", {})

    def _d(key, fallback=None):
        v = pdf_data.get(key)
        return v if v is not None else fallback

    with st.form("assessment_form", clear_on_submit=False):

        st.markdown('<p class="section-title">👤 Personal Info</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        _age_v = _d("age")
        age = c1.number_input(
            "Age (years)", min_value=1, max_value=120,
            value=int(_age_v) if _age_v is not None else None,
            step=1, placeholder="e.g. 52",
            help="Patient age in whole years (1–120).",
            key="f_age",
        )
        sex = c2.selectbox(
            "Sex", options=[("Male", 1), ("Female", 0)],
            format_func=lambda x: x[0],
            help="Biological sex of the patient.",
            key="f_sex",
        )

        st.markdown('<p class="section-title">❤️ Cardiac Metrics</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        cp = c1.selectbox(
            "Chest Pain Type",
            options=[(0,"Typical Angina"),(1,"Atypical Angina"),(2,"Non-Anginal Pain"),(3,"Asymptomatic")],
            format_func=lambda x: f"{x[0]}: {x[1]}",
            help="0: Classic exertional chest pain\n1: Atypical features\n2: Non-cardiac origin\n3: No chest pain",
            key="f_cp",
        )
        _bp_v = _d("trestbps")
        trestbps = c2.number_input(
            "Resting BP (mmHg)", min_value=1, max_value=300,
            value=int(_bp_v) if _bp_v is not None else None,
            step=1, placeholder="e.g. 120",
            help="Normal: 90–120 mmHg. Stage 1 HTN >130. Stage 2 HTN >140. Crisis >180.",
            key="f_bp",
        )
        _chol_v = _d("chol")
        chol = c3.number_input(
            "Cholesterol (mg/dL)", min_value=1, max_value=700,
            value=int(_chol_v) if _chol_v is not None else None,
            step=1, placeholder="e.g. 240",
            help="Desirable: <200. Borderline high: 200–239. High: ≥240 mg/dL.",
            key="f_chol",
        )

        c1, c2, c3 = st.columns(3)
        fbs = c1.selectbox(
            "Fasting Blood Sugar > 120 mg/dL",
            options=[(0,"No  (≤120 mg/dL)"),(1,"Yes (>120 mg/dL)")],
            format_func=lambda x: x[1],
            help="Fasting glucose >120 mg/dL suggests diabetes or pre-diabetes.",
            key="f_fbs",
        )
        restecg = c2.selectbox(
            "Resting ECG",
            options=[(0,"Normal"),(1,"ST-T Wave Abnormality"),(2,"LV Hypertrophy")],
            format_func=lambda x: f"{x[0]}: {x[1]}",
            help="0: Normal\n1: T-wave inversions / ST changes\n2: Left ventricular hypertrophy (Estes criteria)",
            key="f_ecg",
        )
        _hr_v = _d("thalach")
        thalach = c3.number_input(
            "Max Heart Rate Achieved (bpm)", min_value=1, max_value=300,
            value=int(_hr_v) if _hr_v is not None else None,
            step=1, placeholder="e.g. 150",
            help="Peak HR during exercise stress test. Estimated max = 220 − Age.",
            key="f_hr",
        )

        st.markdown('<p class="section-title">🏥 Clinical Findings</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        exang = c1.selectbox(
            "Exercise Induced Angina",
            options=[(0,"No"),(1,"Yes")],
            format_func=lambda x: x[1],
            help="Did the patient develop chest pain during exercise testing?",
            key="f_exang",
        )
        _op_v = _d("oldpeak")
        oldpeak = c2.number_input(
            "ST Depression (oldpeak)", min_value=0.0, max_value=10.0,
            value=float(_op_v) if _op_v is not None else None,
            step=0.1, format="%.1f", placeholder="e.g. 1.5",
            help="ST depression induced by exercise vs rest. 0.0 = none. >2.0 = significant ischaemia.",
            key="f_op",
        )
        slope = c3.selectbox(
            "ST Slope (peak exercise)",
            options=[(0,"Upsloping"),(1,"Flat"),(2,"Downsloping")],
            format_func=lambda x: f"{x[0]}: {x[1]}",
            help="Flat or downsloping ST slope is associated with higher cardiac risk.",
            key="f_slope",
        )

        c1, c2 = st.columns(2)
        ca = c1.selectbox(
            "Major Vessels Coloured by Fluoroscopy (CA)",
            options=[0, 1, 2, 3, 4],
            help="Number of major coronary vessels visualised (0–4). More = greater burden of disease.",
            key="f_ca",
        )
        thal_opts = [(1,"Normal"),(2,"Fixed Defect"),(3,"Reversible Defect")]
        thal_idx  = {1:0, 2:1, 3:2}.get(int(_d("thal", 1)), 0)
        thal = c2.selectbox(
            "Thalassaemia (Thal)",
            options=thal_opts,
            format_func=lambda x: f"{x[0]}: {x[1]}",
            help="1: Normal perfusion\n2: Fixed Defect – no blood flow in part of heart\n3: Reversible – abnormal flow corrected at rest",
            key="f_thal",
        )

        submitted = st.form_submit_button("🔬 Analyse Risk", type="primary", use_container_width=True)

    if submitted:
        req_errors = []
        if age is None:
            req_errors.append("⚠️ **Age** is required. Enter the patient's age in years (1–120).")
        if trestbps is None:
            req_errors.append("⚠️ **Resting Blood Pressure** is required. Normal resting BP is 90–120 mmHg.")
        if chol is None:
            req_errors.append("⚠️ **Cholesterol** is required. Desirable serum cholesterol is below 200 mg/dL.")
        if thalach is None:
            req_errors.append("⚠️ **Max Heart Rate** is required. Typical range is 60–220 bpm.")
        if oldpeak is None:
            req_errors.append("⚠️ **ST Depression** is required. Enter 0.0 if none.")

        if req_errors:
            for e in req_errors:
                st.warning(e)
            st.stop()

        health_data = {
            "age":      int(age),
            "sex":      int(sex[1]),
            "cp":       int(cp[0]),
            "trestbps": int(trestbps),
            "chol":     int(chol),
            "fbs":      int(fbs[0]),
            "restecg":  int(restecg[0]),
            "thalach":  int(thalach),
            "exang":    int(exang[0]),
            "oldpeak":  float(oldpeak),
            "slope":    int(slope[0]),
            "ca":       int(ca),
            "thal":     int(thal[0]),
        }

        # Validate
        errors = validate_fields(health_data)
        if errors:
            st.error("Please fix the following errors:\n\n" + "\n".join(f"• {e}" for e in errors))
            st.stop()

        # ── Phase 2 Agent Pipeline ─────────────────────────────────────────────────
        # Step 1: Fetch previous record for delta (do NOT save yet)
        from data_agent import _fetch_latest_record
        previous_record = None
        delta           = {}
        try:
            previous_record = _fetch_latest_record(_supabase(), st.session_state.user["id"])
            if previous_record:
                from data_agent import compute_delta
                delta = compute_delta(health_data, previous_record)
        except Exception as exc:
            logger.warning("Could not fetch previous record: %s", exc)

        # Step 2: Risk Agent – uses real previous record for direction
        with st.spinner("🔬 Running risk analysis…"):
            risk_output = run_risk_agent(
                health_data,
                use_groq=st.session_state.use_groq,
                previous_record=previous_record,
                delta=delta,
            )

        # Step 3: Save to Supabase with REAL risk score (one-shot, no placeholder)
        _save_ok = False
        try:
            with st.spinner("Saving your assessment\u2026"):
                data_result = run_data_agent(
                    _supabase(),
                    st.session_state.user["id"],
                    health_data,
                    risk_score=risk_output["risk_score"],
                    risk_label=risk_output["risk_label"],
                    source="manual",
                )
            if data_result["success"]:
                _save_ok = True
            else:
                errs = data_result.get("errors") or []
                msg  = data_result.get("message", "Unknown error")
                if errs:
                    st.error(
                        "\u26a0\ufe0f **Could not save assessment.** Validation errors:\n\n" +
                        "\n".join(f"\u2022 {e}" for e in errs)
                    )
                else:
                    st.error(f"\u26a0\ufe0f **Database save failed:** {msg}")
        except Exception as exc:
            import traceback
            st.error(
                f"\u26a0\ufe0f **Database save exception:** {exc}\n\n"
                f"```\n{traceback.format_exc()}\n```"
            )
            logger.error("Data agent save exception: %s", exc)

        st.session_state.risk_output = risk_output
        st.session_state.health_data = health_data

        # Step 4: Monitor Agent – build progress summary for reco agent
        progress_summary = ""
        try:
            from monitor_agent import fetch_history, compute_trends, detect_alerts, build_progress_summary
            hist_df = fetch_history(_supabase(), st.session_state.user["id"], limit=10)
            if not hist_df.empty:
                hist_trends = compute_trends(hist_df)
                hist_alerts = detect_alerts(hist_trends)
                progress_summary = build_progress_summary(hist_df, hist_trends, hist_alerts)
        except Exception as exc:
            logger.warning("Could not build progress summary: %s", exc)

        # Step 5: Recommendation Agent – personalised with actual values + progress
        with st.spinner("💡 Generating personalised recommendations…"):
            try:
                reco_output = run_reco_agent(
                    health_data,
                    risk_output,
                    progress_summary=progress_summary,
                    use_groq=st.session_state.use_groq,
                )
                st.session_state.reco_output = reco_output
            except Exception as exc:
                logger.error("Reco agent error: %s", exc)

        st.success("✅ Analysis complete!")
        st.session_state.page = "Risk Analysis"
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def _page_risk_analysis():
    st.markdown('<h2 style="color:var(--text,#1a1f2e);font-weight:800;">📊 Risk Analysis</h2>', unsafe_allow_html=True)

    ro = st.session_state.risk_output
    hd = st.session_state.health_data

    if ro is None or hd is None:
        st.info("No assessment data available. Please complete an assessment first.")
        if st.button("📋 Go to Assessment", type="primary"):
            st.session_state.page = "Assessment"
            st.rerun()
        return

    label = ro["risk_label"] or "N/A"
    pct   = ro["risk_pct"]
    color = ro["risk_color"]
    cls   = f"risk-card-{label.lower()}" if label in ("HIGH", "MEDIUM", "LOW") else "hv-card"

    # ── Direction banner (Phase 2) ────────────────────────────────────────────
    dir_info = ro.get("direction_info", {})
    if dir_info and dir_info.get("direction") != "first" and dir_info.get("direction"):
        d       = dir_info["direction"]
        d_icon  = dir_info["direction_icon"]
        d_color = dir_info["direction_color"]
        d_chg   = dir_info["score_change"]
        d_prev  = dir_info.get("prev_label", "")
        d_sign  = f"+{d_chg:.1f}" if d_chg > 0 else f"{d_chg:.1f}"
        st.markdown(f"""
        <div style="background:{d_color}18;border:1.5px solid {d_color}44;border-radius:12px;
                    padding:.9rem 1.2rem;margin-bottom:1.2rem;
                    display:flex;align-items:center;gap:1rem;flex-wrap:wrap;">
          <div style="font-size:2rem;">{d_icon}</div>
          <div>
            <div style="font-size:.75rem;color:#6c757d;text-transform:uppercase;
                        letter-spacing:.07em;font-weight:600;">vs Previous Assessment</div>
            <div style="font-size:1.05rem;font-weight:700;color:{d_color};">
              Risk has <strong>{d.upper()}</strong>
              &nbsp;({d_sign} percentage points)
            </div>
            {f'<div style="font-size:.82rem;color:#6c757d;">Previous level: <strong>{d_prev}</strong></div>' if d_prev else ''}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Changed fields
        changed = dir_info.get("changed_fields", [])
        if changed:
            st.markdown('<p class="section-title">🔄 What Changed Since Last Assessment</p>', unsafe_allow_html=True)
            cf_cols = st.columns(min(len(changed), 4))
            for i, cf in enumerate(changed[:4]):
                cc = "#EF4444" if cf["concern"] else "#10B981"
                cf_cols[i].markdown(f"""
                <div style="background:{cc}11;border:1px solid {cc}33;border-radius:10px;
                            padding:.8rem 1rem;text-align:center;">
                  <div style="font-size:.72rem;color:#6c757d;">{cf['label']}</div>
                  <div style="font-size:1.1rem;font-weight:800;color:{cc};">
                    {cf['direction'].split()[0]}
                  </div>
                  <div style="font-size:.85rem;font-weight:600;color:{cc};">
                    {cf['pct']:+.1f}%
                  </div>
                </div>
                """, unsafe_allow_html=True)
    elif dir_info.get("direction") == "first":
        st.info("🆕 This is your first assessment. Future assessments will show your progress.", icon="🆕")

    # Risk banner
    st.markdown(f"""
    <div class="hv-card {cls}" style="margin-bottom:1.5rem;">
      <div style="display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;">
        <div>
          <div style="font-size:.82rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;opacity:.7;">Heart Disease Risk</div>
          <div style="font-size:3rem;font-weight:900;color:{color};line-height:1;">{label}</div>
        </div>
        <div style="flex:1;text-align:right;">
          <div style="font-size:.85rem;opacity:.7;">Probability Score</div>
          <div style="font-size:2.5rem;font-weight:800;">{pct:.1f}%</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(_gauge_chart(pct, label, color), use_container_width=True)
    with col2:
        st.markdown('<p class="section-title">🔑 Clinical Risk Factors</p>', unsafe_allow_html=True)
        for r in ro.get("reasons", []):
            severity = "high" if any(w in r.lower() for w in ["high", "elevated", "severe", "critical"]) else "medium"
            st.markdown(f"<div class='alert-{severity}'>• {r}</div>", unsafe_allow_html=True)

    # AI explanation
    ai_exp = ro.get("ai_explanation", "")
    if ai_exp:
        st.markdown('<p class="section-title">🤖 AI Clinical Explanation</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="hv-card">
          <div style="line-height:1.8;font-size:.97rem;color:#1a1f2e;">{ai_exp}</div>
        </div>
        """, unsafe_allow_html=True)

    # Feature importance
    importances = ro.get("feature_importances", {})
    if importances:
        st.markdown('<p class="section-title">📊 Feature Importance</p>', unsafe_allow_html=True)
        feat_labels = {
            "age":"Age","sex":"Sex","cp":"Chest Pain","trestbps":"Resting BP",
            "chol":"Cholesterol","fbs":"Fasting BS","restecg":"ECG","thalach":"Max HR",
            "exang":"Exercise Angina","oldpeak":"ST Depression","slope":"ST Slope",
            "ca":"Vessels","thal":"Thal",
        }
        items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
        labels_fi = [feat_labels.get(k, k) for k, _ in items]
        vals_fi   = [v for _, v in items]
        fig_fi = go.Figure(go.Bar(
            x=vals_fi, y=labels_fi,
            orientation="h",
            marker_color=["#6B46C1" if v > 0.1 else "#A78BFA" for v in vals_fi],
            text=[f"{v:.3f}" for v in vals_fi],
            textposition="outside",
        ))
        fig_fi.update_layout(
            height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FAFAFA",
            margin={"t": 20, "b": 20, "l": 10, "r": 60},
            font={"family": "Inter", "size": 11},
            xaxis_title="Importance Score",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # Comparison chart
    from monitor_agent import build_comparison_chart_data
    cdata = build_comparison_chart_data(pd.Series(hd))
    if cdata:
        st.markdown('<p class="section-title">📈 Your Values vs Safe Ranges</p>', unsafe_allow_html=True)
        st.plotly_chart(_comparison_chart(cdata), use_container_width=True)

    # Download PDF
    reco = st.session_state.reco_output or {"tips": []}
    pdf  = _generate_pdf_report(
        st.session_state.user.get("email", ""),
        hd, ro, reco,
    )
    if pdf:
        st.download_button(
            "📥 Download Full PDF Report",
            data=pdf,
            file_name=f"heartvigil_report_{datetime.date.today()}.pdf",
            mime="application/pdf",
            use_container_width=True,
            type="primary",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DATA AGENT  – Recent 5 Assessments + Comparison Graph
# ══════════════════════════════════════════════════════════════════════════════
def _page_data_agent():
    st.markdown('<h2 style="color:var(--text,#1a1f2e);font-weight:800;">🗃️ Data Agent — Assessment History</h2>', unsafe_allow_html=True)
    st.markdown(
        "<p style='color:var(--text-muted,#6c757d);margin-bottom:1.5rem;'>"
        "View your 5 most recent assessments, compare key metrics side-by-side, "
        "and track how your numbers have changed over time."
        "</p>",
        unsafe_allow_html=True,
    )

    user = st.session_state.user
    try:
        sb  = _supabase()
        res = (
            sb.table("health_records")
            .select("*")
            .eq("user_id", user["id"])
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )
        records = res.data or []
    except Exception as exc:
        st.error(f"Failed to load assessment history: {exc}")
        return

    if not records:
        st.info("No assessments found. Complete your first assessment to see your history here.", icon="ℹ️")
        if st.button("📋 Take Assessment", type="primary"):
            st.session_state.page = "Assessment"
            st.rerun()
        return

    df = pd.DataFrame(records)
    df["created_at"] = pd.to_datetime(df["created_at"])
    # Safely convert risk_score (may be None for old/unscored records)
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce").fillna(0.0) * 100
    df = df.sort_values("created_at").reset_index(drop=True)
    df["label"] = df["created_at"].dt.strftime("#%d %b %H:%M")

    # ── Summary cards (newest first) ──────────────────────────────────────────
    st.markdown('<p class="section-title">📋 Recent 5 Assessments</p>', unsafe_allow_html=True)
    for _, row in df.iloc[::-1].iterrows():
        rl    = row.get("risk_label") or None
        rl_display = rl if rl in ("HIGH", "MEDIUM", "LOW") else "Pending"
        rs    = row.get("risk_score")  # already *100 in df
        try:
            rs_num = float(rs) if rs is not None else None
        except (TypeError, ValueError):
            rs_num = None
        rs_display = f"{rs_num:.1f}%" if rs_num is not None and rs_num > 0 else "Pending"
        ts    = row["created_at"].strftime("%d %b %Y  %H:%M")
        color_map = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"}
        color = color_map.get(rl, "#6B46C1")
        cls   = f"risk-card-{rl.lower()}" if rl in color_map else "hv-card"
        bp    = row.get("trestbps") if row.get("trestbps") is not None else "–"
        chol  = row.get("chol")     if row.get("chol")     is not None else "–"
        hr    = row.get("thalach")  if row.get("thalach")  is not None else "–"
        op    = row.get("oldpeak")  if row.get("oldpeak")  is not None else "–"
        st.markdown(f"""
        <div class="hv-card {cls}" style="margin-bottom:1rem;">
          <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.5rem;">
            <div>
              <div style="font-size:.78rem;opacity:.7;">{ts}</div>
              <div style="font-size:1.6rem;font-weight:900;color:{color};">{rl_display}</div>
              <div style="font-size:.9rem;font-weight:600;">Risk Score: {rs_display}</div>
            </div>
            <div style="display:flex;gap:1.4rem;flex-wrap:wrap;">
              <div style="text-align:center;">
                <div style="font-size:.72rem;opacity:.7;">BP</div>
                <div style="font-weight:700;font-size:1rem;">{bp} <span style='font-size:.7rem;'>mmHg</span></div>
              </div>
              <div style="text-align:center;">
                <div style="font-size:.72rem;opacity:.7;">Cholesterol</div>
                <div style="font-weight:700;font-size:1rem;">{chol} <span style='font-size:.7rem;'>mg/dL</span></div>
              </div>
              <div style="text-align:center;">
                <div style="font-size:.72rem;opacity:.7;">Max HR</div>
                <div style="font-weight:700;font-size:1rem;">{hr} <span style='font-size:.7rem;'>bpm</span></div>
              </div>
              <div style="text-align:center;">
                <div style="font-size:.72rem;opacity:.7;">ST Dep.</div>
                <div style="font-weight:700;font-size:1rem;">{op}</div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── PDF download for latest record ────────────────────────────────────────
    latest = df.iloc[-1].to_dict()   # most recent (df sorted ascending)
    try:
        rl = latest.get("risk_label")
        if rl not in ("HIGH", "MEDIUM", "LOW"):
            rl = "UNKNOWN"
        rs_raw = latest.get("risk_score", 0) or 0   # already *100 in df
        dummy_risk = {
            "risk_label":          rl,
            "probability_percent": float(rs_raw),
            "clinical_reasons":    ["Historical record — no live risk run"],
            "ai_explanation":      "This report is generated from a saved assessment record.",
        }
        pdf_bytes = _generate_pdf_report(
            st.session_state.user.get("email", ""),
            latest,
            dummy_risk,
            {},
            [],
        )
        if pdf_bytes:
            st.download_button(
                label="📥 Download Latest Report as PDF",
                data=pdf_bytes,
                file_name=f"heartvigil_report_{datetime.date.today()}.pdf",
                mime="application/pdf",
                use_container_width=False,
            )
    except Exception as _pdf_exc:
        logger.warning("PDF generation skipped in data agent: %s", _pdf_exc)

    # ── Comparison chart ───────────────────────────────────────────────────────
    st.markdown('<p class="section-title">📊 Metric Comparison Across Assessments</p>', unsafe_allow_html=True)

    compare_metrics = [
        ("risk_score",  "Risk Score (%)",      "#6B46C1"),
        ("trestbps",    "Resting BP (mmHg)",   "#EF4444"),
        ("chol",        "Cholesterol (mg/dL)", "#F59E0B"),
        ("thalach",     "Max HR (bpm)",        "#10B981"),
        ("oldpeak",     "ST Depression (mm)",  "#3B82F6"),
    ]

    tab_labels = [lbl for _, lbl, _ in compare_metrics]
    tabs = st.tabs(tab_labels)

    safe_ranges_chart = {
        "risk_score": (0,   40),
        "trestbps":   (90,  120),
        "chol":       (0,   200),
        "thalach":    (60,  100),
        "oldpeak":    (0,   1.0),
    }

    for tab, (metric, label, color) in zip(tabs, compare_metrics):
        with tab:
            if metric not in df.columns:
                st.info(f"No data for {label}.")
                continue
            # Drop NaN rows for this metric before plotting
            valid_mask = df[metric].notna()
            xs   = [x for x, v in zip(df["label"].tolist(), valid_mask) if v]
            ys_raw = df.loc[valid_mask, metric].tolist()
            ys   = [float(v) for v in ys_raw]
            lo, hi = safe_ranges_chart.get(metric, (None, None))

            if not ys:
                st.info(f"No data for {label} in any assessment.")
                continue

            fig = go.Figure()
            bar_colors = []
            for y in ys:
                if lo is not None and hi is not None:
                    bar_colors.append(color if lo <= y <= hi else "#EF4444")
                else:
                    bar_colors.append(color)

            fig.add_trace(go.Bar(
                x=xs, y=ys,
                name=label,
                marker_color=bar_colors,
                text=[f"{v:.1f}" for v in ys],
                textposition="outside",
            ))
            if lo is not None and hi is not None and xs:
                fig.add_hrect(
                    y0=lo, y1=hi,
                    fillcolor="rgba(16,185,129,0.10)",
                    line_width=0,
                    annotation_text=f"Safe: {lo}–{hi}",
                    annotation_position="top right",
                    annotation_font_color="#10B981",
                )
                fig.add_hline(y=hi, line_dash="dot", line_color="#10B981", line_width=1.5)
            if len(ys) >= 2:
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    mode="lines+markers",
                    name="Trend",
                    line={"color": "rgba(107,70,193,0.6)", "width": 2, "dash": "dot"},
                    marker={"size": 8, "color": color},
                ))
            fig.update_layout(
                height=340,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#FAFAFA",
                margin={"t": 30, "b": 40, "l": 30, "r": 20},
                font={"family": "Inter", "size": 12},
                legend={"orientation": "h", "y": -0.25},
                yaxis_title=label,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Raw table ──────────────────────────────────────────────────────────────
    with st.expander("📋 Raw Data Table", expanded=False):
        disp_cols = ["created_at", "risk_label", "risk_score", "trestbps", "chol", "thalach", "oldpeak", "ca", "exang"]
        avail = [c for c in disp_cols if c in df.columns]
        df_disp = df[avail].copy()
        df_disp["created_at"] = df_disp["created_at"].dt.strftime("%Y-%m-%d %H:%M")
        df_disp["risk_label"]  = df_disp["risk_label"].apply(
            lambda x: x if x in ("HIGH", "MEDIUM", "LOW") else "Pending"
        )
        df_disp["risk_score"]  = df_disp["risk_score"].apply(
            lambda x: f"{float(x):.1f}%" if (x is not None and str(x) != "nan" and float(x) > 0) else "Pending"
        )
        df_disp.columns = [c.replace("_", " ").title() for c in df_disp.columns]
        st.dataframe(df_disp, use_container_width=True)



# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MONITORING  – Trend Charts + Date Range Filter
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MONITORING  ─  Premium Health Dashboard
# ══════════════════════════════════════════════════════════════════════════════
def _page_monitoring():
    st.markdown('<h2 style="color:var(--text,#1a1f2e);font-weight:800;">\U0001f4c8 Health Monitoring</h2>', unsafe_allow_html=True)

    user = st.session_state.user

    # ─── Load data ─────────────────────────────────────────────────────────────
    try:
        with st.spinner("Loading monitoring data\u2026"):
            monitor_out = run_monitor_agent(
                _supabase(), user["id"],
                use_groq=st.session_state.use_groq,
                limit=100,
            )
        st.session_state.monitor_output = monitor_out
    except Exception as exc:
        st.error(f"Monitoring agent error: {exc}")
        import traceback
        st.code(traceback.format_exc(), language="bash")
        return

    if not monitor_out.get("has_history"):
        st.info("No historical records found. Complete your first assessment to start tracking.", icon="\u2139\ufe0f")
        if st.button("\U0001f4cb Take Assessment", type="primary"):
            st.session_state.page = "Assessment"
            st.rerun()
        return

    df_full = monitor_out["history"]
    trends  = monitor_out["trends"]
    alerts  = monitor_out["alerts"]
    ai_sum  = monitor_out.get("ai_summary", "")

    # ─── Date range filter ─────────────────────────────────────────────────────
    st.markdown('<p class="section-title">\U0001f4c5 Date Range</p>', unsafe_allow_html=True)
    min_date = df_full["created_at"].min().date()
    max_date = df_full["created_at"].max().date()

    cd1, cd2, cd3 = st.columns([2, 2, 1])
    with cd1:
        start_date = st.date_input("From", value=min_date, min_value=min_date,
                                   max_value=max_date, key="monitor_start_date")
    with cd2:
        end_date   = st.date_input("To",   value=max_date, min_value=min_date,
                                   max_value=max_date, key="monitor_end_date")
    with cd3:
        st.markdown("<div style='height:28px'>", unsafe_allow_html=True)
        if st.button("\u21ba Reset", use_container_width=True):
            for k in ("monitor_start_date", "monitor_end_date"):
                if k in st.session_state: del st.session_state[k]
            st.rerun()

    if start_date > end_date:
        st.warning("Start date must be before end date.")
        return

    mask = (
        (df_full["created_at"].dt.date >= start_date) &
        (df_full["created_at"].dt.date <= end_date)
    )
    df = df_full[mask].copy().reset_index(drop=True)
    n_total, n_filt = len(df_full), len(df)
    st.caption(f"Showing **{n_filt}** of **{n_total}** assessments \u2014 {start_date} \u2192 {end_date}")

    if df.empty:
        st.info("No assessments in this date range. Adjust the filters above.", icon="\U0001f4c5")
        return

    latest = df.iloc[-1]

    # ─── Heart Health Score gauge ──────────────────────────────────────────────
    st.markdown('<p class="section-title">\u2764\ufe0f Heart Health Score</p>', unsafe_allow_html=True)

    risk_pct  = float(latest.get("risk_score") or 0)   # already *100 in df
    health_score = max(0, round(100 - risk_pct))
    hs_color = "#10B981" if health_score >= 70 else ("#F59E0B" if health_score >= 45 else "#EF4444")
    hs_grade = "Excellent" if health_score >= 80 else ("Good" if health_score >= 65 else ("Fair" if health_score >= 45 else "At Risk"))

    # Heart age estimate (simple Framingham-inspired heuristic)
    try:
        age_c = float(latest.get("age") or 45)
        bp_c  = float(latest.get("trestbps") or 120)
        ch_c  = float(latest.get("chol") or 200)
        smk   = float(latest.get("fbs") or 0)
        sex_c = int(latest.get("sex") or 1)
        heart_age = int(age_c
                        + (bp_c - 120) * 0.12
                        + (ch_c - 200) * 0.02
                        + smk * 3
                        + (2 if risk_pct > 50 else 0)
                        + (-3 if sex_c == 0 else 2))
        heart_age = max(int(age_c) - 15, min(int(age_c) + 25, heart_age))
        ha_delta  = heart_age - int(age_c)
        ha_color  = "#10B981" if ha_delta <= 0 else ("#F59E0B" if ha_delta <= 5 else "#EF4444")
        ha_sign   = f"+{ha_delta}" if ha_delta > 0 else str(ha_delta)
        ha_text   = f"{heart_age} yrs ({ha_sign} vs chronological)"
    except Exception:
        ha_text = "N/A"; ha_color = "#6c757d"

    hg1, hg2, hg3 = st.columns([1, 1, 1])

    # Gauge chart
    with hg1:
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=health_score,
            delta={"reference": 70, "valueformat": ".0f",
                   "increasing": {"color": "#10B981"}, "decreasing": {"color": "#EF4444"}},
            number={"suffix": "/100", "font": {"size": 28, "family": "Inter", "color": "#1a1f2e"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ccc",
                         "tickvals": [0, 25, 50, 75, 100]},
                "bar":  {"color": hs_color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  45],  "color": "rgba(239,68,68,.10)"},
                    {"range": [45, 70],  "color": "rgba(245,158,11,.10)"},
                    {"range": [70, 100], "color": "rgba(16,185,129,.10)"},
                ],
                "threshold": {"line": {"color": hs_color, "width": 3},
                              "thickness": 0.85, "value": health_score},
            },
            title={"text": f"<b>{hs_grade}</b>", "font": {"size": 15, "color": hs_color}},
        ))
        gauge_fig.update_layout(
            height=220, margin={"t": 30, "b": 0, "l": 20, "r": 20},
            paper_bgcolor="rgba(0,0,0,0)", font={"family": "Inter"},
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    with hg2:
        risk_label = latest.get("risk_label") or "N/A"
        rl_color   = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"}.get(risk_label, "#6B46C1")
        st.markdown(f"""
        <div style="background:{rl_color}12;border:1.5px solid {rl_color}44;
                    border-radius:16px;padding:1.4rem 1.2rem;text-align:center;height:190px;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;">
          <div style="font-size:.78rem;color:#6c757d;font-weight:600;text-transform:uppercase;letter-spacing:.07em;">Current Risk</div>
          <div style="font-size:2.6rem;font-weight:900;color:{rl_color};line-height:1.1;margin:.3rem 0;">{risk_label}</div>
          <div style="font-size:1.1rem;font-weight:700;color:{rl_color};">{risk_pct:.1f}%</div>
          <div style="font-size:.76rem;color:#6c757d;margin-top:.4rem;">{n_filt} assessments tracked</div>
        </div>
        """, unsafe_allow_html=True)

    with hg3:
        st.markdown(f"""
        <div style="background:#6B46C112;border:1.5px solid #6B46C144;
                    border-radius:16px;padding:1.4rem 1.2rem;text-align:center;height:190px;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;">
          <div style="font-size:.78rem;color:#6c757d;font-weight:600;text-transform:uppercase;letter-spacing:.07em;">Estimated Heart Age</div>
          <div style="font-size:2.0rem;font-weight:900;color:{ha_color};line-height:1.1;margin:.3rem 0;">{ha_text}</div>
          <div style="font-size:.76rem;color:#6c757d;margin-top:.4rem;">Framingham-inspired heuristic</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── KPI metric cards ──────────────────────────────────────────────────────
    st.markdown('<p class="section-title">\U0001f4ca Key Metrics (Latest)</p>', unsafe_allow_html=True)
    kpis = [
        ("trestbps",  "Resting BP",      "mmHg",  90, 120, "\U0001fa7a"),
        ("chol",      "Cholesterol",     "mg/dL",  0, 200, "\U0001f9ea"),
        ("thalach",   "Max Heart Rate",  "bpm",   60, 100, "\U0001f493"),
        ("oldpeak",   "ST Depression",   "mm",     0, 1.0, "\U0001f4c9"),
        ("ca",        "Vessels Colored", "",       0,   0, "\U0001f9ec"),
        ("risk_score","Risk Score",      "%",      0,  35, "\u2695\ufe0f"),
    ]
    k_cols = st.columns(len(kpis))
    for i, (metric, label, unit, lo, hi, icon) in enumerate(kpis):
        val  = latest.get(metric)
        info = trends.get(metric, {})
        dirn = info.get("direction", "\u2192")
        pct  = info.get("pct_change")
        if val is None:
            k_cols[i].markdown(f"""
            <div style="background:#f8f9fa;border:1px solid #e5e7eb;border-radius:14px;
                        padding:1rem .9rem;text-align:center;min-height:120px;">
              <div style="font-size:1.4rem;">{icon}</div>
              <div style="font-size:.72rem;color:#6c757d;font-weight:600;text-transform:uppercase;">{label}</div>
              <div style="font-size:1.1rem;font-weight:700;color:#6c757d;">N/A</div>
            </div>""", unsafe_allow_html=True)
            continue
        v = float(val)
        if hi == 0:
            status = "Abnormal" if v > 0 else "Normal"
            s_color = "#EF4444" if v > 0 else "#10B981"
        elif v < lo:
            status = "Low"; s_color = "#F59E0B"
        elif v > hi:
            status = "Elevated"; s_color = "#EF4444"
        else:
            status = "Normal"; s_color = "#10B981"
        pct_str = f"{pct:+.1f}%" if pct is not None else ""
        k_cols[i].markdown(f"""
        <div style="background:{s_color}0e;border:1.5px solid {s_color}44;border-radius:14px;
                    padding:1rem .9rem;text-align:center;min-height:120px;">
          <div style="font-size:1.4rem;">{icon}</div>
          <div style="font-size:.72rem;color:#6c757d;font-weight:600;text-transform:uppercase;letter-spacing:.05em;">{label}</div>
          <div style="font-size:1.35rem;font-weight:900;color:{s_color};margin:.2rem 0;">{v:.1f}<span style="font-size:.7rem;font-weight:500;"> {unit}</span></div>
          <div style="font-size:.72rem;font-weight:600;color:{s_color};">{status}</div>
          <div style="font-size:.68rem;color:#6c757d;">{dirn} {pct_str}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Alerts ────────────────────────────────────────────────────────────────
    if alerts:
        st.markdown('<p class="section-title">\u26a0\ufe0f Health Alerts</p>', unsafe_allow_html=True)
        for a in alerts:
            sev = a["severity"]
            color = "#EF4444" if sev == "HIGH" else "#F59E0B"
            st.markdown(
                f'<div style="background:{color}12;border-left:4px solid {color};'
                f'border-radius:0 10px 10px 0;padding:.7rem 1rem;margin-bottom:.5rem;">' +
                f'\u26a0 <strong>{a["message"]}</strong> \u2014 <em>{a["pct"]}</em></div>',
                unsafe_allow_html=True,
            )

    # ─── AI Summary ────────────────────────────────────────────────────────────
    if ai_sum:
        with st.expander("AI Health Trend Summary", expanded=True):
            st.markdown(f'<div style="line-height:1.75;font-size:.95rem;">{ai_sum}</div>', unsafe_allow_html=True)

    # ─── Risk Score Timeline (large, prominent) ────────────────────────────────
    st.markdown('<p class="section-title">\U0001f4c8 Risk Score Over Time</p>', unsafe_allow_html=True)
    if "risk_score" in df.columns and not df["risk_score"].dropna().empty:
        df_rs = df[df["risk_score"].notna()].copy()
        dates_rs = df_rs["created_at"].dt.strftime("%b %d %H:%M").tolist()
        vals_rs  = df_rs["risk_score"].tolist()
        labels_rs = df_rs["risk_label"].fillna("Pending").tolist()

        rs_colors = [{"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"}.get(l, "#6B46C1")
                     for l in labels_rs]

        rs_fig = go.Figure()
        rs_fig.add_hrect(y0=0,  y1=35, fillcolor="rgba(16,185,129,.07)",  line_width=0,
                         annotation_text="Low Risk Zone",     annotation_position="top left",
                         annotation_font={"color": "#10B981", "size": 10})
        rs_fig.add_hrect(y0=35, y1=65, fillcolor="rgba(245,158,11,.07)",  line_width=0,
                         annotation_text="Medium Risk Zone",  annotation_position="top left",
                         annotation_font={"color": "#F59E0B", "size": 10})
        rs_fig.add_hrect(y0=65, y1=100, fillcolor="rgba(239,68,68,.07)", line_width=0,
                         annotation_text="High Risk Zone",    annotation_position="top left",
                         annotation_font={"color": "#EF4444", "size": 10})
        rs_fig.add_trace(go.Scatter(
            x=dates_rs, y=vals_rs,
            mode="lines+markers+text",
            line={"color": "#6B46C1", "width": 3},
            marker={"color": rs_colors, "size": 12, "line": {"color": "white", "width": 2}},
            text=[f"{v:.1f}%" for v in vals_rs],
            textposition="top center",
            textfont={"size": 10, "family": "Inter"},
            name="Risk Score (%)",
            fill="tozeroy",
            fillcolor="rgba(107,70,193,.07)",
            hovertemplate="<b>%{x}</b><br>Risk: %{y:.1f}%<extra></extra>",
        ))
        if len(vals_rs) >= 2:
            roll = pd.Series(vals_rs).rolling(window=max(1,min(3,len(vals_rs))), min_periods=1).mean().tolist()
            rs_fig.add_trace(go.Scatter(
                x=dates_rs, y=roll, mode="lines",
                line={"color": "#EC4899", "width": 1.5, "dash": "dot"},
                name="Rolling Avg",
            ))
        rs_fig.update_layout(
            height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,251,1)",
            margin={"t": 20, "b": 50, "l": 40, "r": 20},
            font={"family": "Inter", "size": 11, "color": "#374151"},
            yaxis={"range": [0, 105], "title": "Risk Score (%)", "gridcolor": "#F0F0F0"},
            xaxis={"showgrid": False},
            legend={"orientation": "h", "y": -0.25},
        )
        st.plotly_chart(rs_fig, use_container_width=True)

    # ─── Radar (Spider) Chart ─────────────────────────────────────────────────
    st.markdown('<p class="section-title">\U0001f578\ufe0f Metrics vs Safe Ranges</p>', unsafe_allow_html=True)
    radar_metrics = [
        ("trestbps","BP",       120, 200),
        ("chol",    "Chol",     200, 400),
        ("thalach", "Max HR",   100,  60),   # inverted: lower = worse
        ("oldpeak", "ST Dep",   0,     6),
        ("ca",      "Vessels",  0,     4),
        ("risk_score","Risk",   35,  100),
    ]
    radar_vals = []; radar_lbls = []; radar_ref = []
    for metric, lbl, safe_hi, max_val in radar_metrics:
        v = latest.get(metric)
        if v is None: continue
        radar_vals.append(min(100, round(float(v) / max_val * 100, 1)))
        radar_ref.append(round(safe_hi / max_val * 100, 1))
        radar_lbls.append(lbl)

    if len(radar_vals) >= 3:
        radar_vals_closed = radar_vals + [radar_vals[0]]
        radar_ref_closed  = radar_ref  + [radar_ref[0]]
        radar_lbls_closed = radar_lbls + [radar_lbls[0]]

        r_fig = go.Figure()
        r_fig.add_trace(go.Scatterpolar(
            r=radar_ref_closed, theta=radar_lbls_closed,
            fill="toself", name="Safe Zone",
            line={"color": "#10B981", "dash": "dot", "width": 1.5},
            fillcolor="rgba(16,185,129,.08)",
        ))
        r_fig.add_trace(go.Scatterpolar(
            r=radar_vals_closed, theta=radar_lbls_closed,
            fill="toself", name="Your Metrics",
            line={"color": "#7C3AED", "width": 2.5},
            fillcolor="rgba(124,58,237,.15)",
            marker={"size": 8, "color": "#7C3AED"},
        ))
        r_fig.update_layout(
            polar={"radialaxis": {"visible": True, "range": [0, 110],
                                  "tickfont": {"size": 9, "family": "Inter"}},
                   "angularaxis": {"tickfont": {"size": 11, "family": "Inter", "color": "#374151"}}},
            showlegend=True, height=340,
            paper_bgcolor="rgba(0,0,0,0)", margin={"t": 30, "b": 30, "l": 30, "r": 30},
            legend={"font": {"family": "Inter", "size": 10}},
        )
        rc1, rc2 = st.columns([1, 1])
        with rc1:
            st.plotly_chart(r_fig, use_container_width=True)
        with rc2:
            # Sparkline mini-charts for BP and Cholesterol side by side
            for metric, label, color in [("trestbps","Resting BP","#EF4444"),("chol","Cholesterol","#F59E0B")]:
                if metric in df.columns and not df[metric].dropna().empty:
                    s_vals  = df[metric].dropna().tolist()
                    s_dates = df["created_at"].dropna().dt.strftime("%b %d").tolist()[-len(s_vals):]
                    sp_fig  = go.Figure(go.Scatter(
                        x=s_dates, y=s_vals, mode="lines+markers",
                        line={"color": color, "width": 2},
                        marker={"size": 6, "color": color},
                        fill="tozeroy", fillcolor="rgba(" + ",".join(str(int(color.lstrip("#")[i*2:i*2+2],16)) for i in range(3)) + ",0.09)",
                        hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>",
                    ))
                    sp_fig.update_layout(
                        title={"text": label, "font": {"size": 12, "family": "Inter", "color": "#374151"}},
                        height=150, margin={"t": 30, "b": 20, "l": 30, "r": 10},
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,251,1)",
                        xaxis={"showgrid": False, "tickfont": {"size": 9}},
                        yaxis={"gridcolor": "#f0f0f0", "tickfont": {"size": 9}},
                        showlegend=False,
                    )
                    st.plotly_chart(sp_fig, use_container_width=True)

    # ─── Per-metric progress cards ─────────────────────────────────────────────
    field_progress = monitor_out.get("field_progress", [])
    if field_progress:
        st.markdown('<p class="section-title">\U0001f4c8 Progress vs First Assessment</p>', unsafe_allow_html=True)
        pc = st.columns(min(len(field_progress), 4))
        for i, fp in enumerate(field_progress):
            prog    = fp["progress"]
            p_color = prog["color"]
            p_icon  = prog["icon"]
            p_desc  = prog["description"]
            val_str = f"{fp['latest']:.1f} {fp['unit']}".strip()
            pc[i % 4].markdown(f"""
            <div style="background:{p_color}12;border:1.5px solid {p_color}44;border-radius:14px;
                        padding:.9rem 1rem;margin-bottom:.6rem;text-align:center;">
              <div style="font-size:.68rem;color:#6c757d;font-weight:700;text-transform:uppercase;letter-spacing:.06em;">{fp["label"]}</div>
              <div style="font-size:1.7rem;line-height:1.1;margin:.15rem 0;">{p_icon}</div>
              <div style="font-size:.88rem;font-weight:800;color:{p_color};">{prog["status"].capitalize()}</div>
              <div style="font-size:.76rem;color:#374151;font-weight:600;">{val_str}</div>
              <div style="font-size:.68rem;color:{p_color};">{p_desc}</div>
            </div>""", unsafe_allow_html=True)
            if (i + 1) % 4 == 0 and i + 1 < len(field_progress):
                pc = st.columns(min(len(field_progress) - i - 1, 4))

    st.markdown("<br>", unsafe_allow_html=True)

    # ─── Trend Charts grid ─────────────────────────────────────────────────────
    st.markdown('<p class="section-title">\U0001f9ea Detailed Trend Analysis</p>', unsafe_allow_html=True)
    chart_metrics = [
        ("trestbps", "Resting BP (mmHg)",    "#EF4444", 90,  120),
        ("chol",     "Cholesterol (mg/dL)",   "#F59E0B", 0,   200),
        ("thalach",  "Max Heart Rate (bpm)",  "#10B981", 60,  100),
        ("oldpeak",  "ST Depression (mm)",    "#3B82F6", 0,   1.0),
    ]
    try:
        for i in range(0, len(chart_metrics), 2):
            cols = st.columns(2)
            for j, (metric, label, color, safe_lo, safe_hi) in enumerate(chart_metrics[i:i+2]):
                with cols[j]:
                    if metric not in df.columns or df[metric].dropna().empty:
                        st.caption(f"No data for {label}")
                        continue
                    d_vals  = df[metric].ffill().tolist()
                    d_dates = df["created_at"].dt.strftime("%b %d").tolist()
                    n       = len(d_vals)
                    roll    = pd.Series(d_vals).rolling(max(1,min(3,n)), min_periods=1).mean().tolist()

                    fig = go.Figure()
                    fig.add_hrect(y0=safe_lo, y1=safe_hi,
                                  fillcolor="rgba(16,185,129,.08)", line_width=0,
                                  annotation_text=f"Safe {safe_lo}\u2013{safe_hi}",
                                  annotation_font={"color": "#10B981", "size": 9},
                                  annotation_position="top right")
                    fig.add_trace(go.Scatter(
                        x=d_dates, y=d_vals, mode="lines+markers", name=label,
                        line={"color": color, "width": 2.5},
                        marker={"size": 7, "color": color, "line": {"color": "white", "width": 1.5}},
                        fill="tozeroy", fillcolor="rgba(" + ",".join(str(int(color.lstrip("#")[i*2:i*2+2],16)) for i in range(3)) + ",0.07)",
                        hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}<extra></extra>",
                    ))
                    if n >= 2:
                        fig.add_trace(go.Scatter(
                            x=d_dates, y=roll, mode="lines", name="Trend",
                            line={"color": "#EC4899", "width": 1.5, "dash": "dot"},
                        ))
                    fig.update_layout(
                        title={"text": label, "font": {"size": 12, "color": "#374151", "family": "Inter"}},
                        height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(248,249,251,1)",
                        margin={"t": 40, "b": 40, "l": 35, "r": 15},
                        font={"family": "Inter", "size": 10},
                        xaxis={"showgrid": False},
                        yaxis={"gridcolor": "#F0F0F0"},
                        legend={"orientation": "h", "y": -0.35, "font": {"size": 9}},
                    )
                    st.plotly_chart(fig, use_container_width=True)
    except Exception as ce:
        st.warning(f"Some trend charts could not render: {ce}")

    # ─── Stats summary table ───────────────────────────────────────────────────
    st.markdown('<p class="section-title">\U0001f4cb Assessment Records</p>', unsafe_allow_html=True)
    cols_show = ["created_at", "risk_label", "risk_score", "trestbps", "chol", "thalach", "oldpeak", "ca"]
    avail     = [c for c in cols_show if c in df.columns]
    df_show   = df[avail].copy().sort_values("created_at", ascending=False)
    df_show["created_at"]  = df_show["created_at"].dt.strftime("%Y-%m-%d %H:%M")
    df_show["risk_label"]  = df_show["risk_label"].apply(lambda x: x if x in ("HIGH","MEDIUM","LOW") else "Pending")
    df_show["risk_score"]  = df_show["risk_score"].apply(
        lambda x: f"{float(x):.1f}%" if (x is not None and str(x) != "nan" and float(x) > 0) else "Pending"
    )
    df_show.columns = [c.replace("_"," ").title() for c in df_show.columns]
    st.dataframe(df_show.reset_index(drop=True), use_container_width=True)

    # ─── CSV download ──────────────────────────────────────────────────────────
    csv_buf = df.drop(columns=["id","user_id"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button(
        "\U0001f4e5 Export Records as CSV", data=csv_buf,
        file_name=f"heartvigil_monitoring_{datetime.date.today()}.csv",
        mime="text/csv", use_container_width=False,
    )



def _page_reco():
    st.markdown('<h2 style="color:var(--text,#1a1f2e);font-weight:800;">💡 Personalised Recommendations</h2>', unsafe_allow_html=True)

    reco = st.session_state.reco_output
    hd   = st.session_state.health_data
    ro   = st.session_state.risk_output

    if reco is None:
        if hd is None:
            st.info("Complete an assessment first to receive personalised recommendations.")
            if st.button("📋 Take Assessment", type="primary"):
                st.session_state.page = "Assessment"
                st.rerun()
            return
        with st.spinner("💡 Generating your personalised recommendations…"):
            try:
                reco = run_reco_agent(
                    hd, ro or {},
                    progress_summary="",
                    use_groq=st.session_state.use_groq,
                )
                st.session_state.reco_output = reco
            except Exception as exc:
                st.error(f"Could not generate recommendations: {exc}")
                return

    tips         = reco.get("tips", [])
    ai_narrative = reco.get("ai_narrative", "")
    source       = reco.get("source", "rule-based")

    # ── Risk summary banner ───────────────────────────────────────────────────
    if ro:
        rl    = ro.get("risk_label", "N/A")
        rp    = ro.get("risk_pct", 0)
        color_map = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"}
        rc    = color_map.get(rl, "#6B46C1")
        cls   = f"risk-card-{rl.lower()}"
        src_badge = ("🤖 AI-powered" if source == "groq" else "📋 Rule-based")
        st.markdown(f"""
        <div class="hv-card {cls}" style="margin-bottom:1.5rem;">
          <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
            <div>
              <div style="font-size:.78rem;opacity:.7;text-transform:uppercase;letter-spacing:.08em;">Your Risk Level</div>
              <div style="font-size:2.4rem;font-weight:900;color:{rc};line-height:1;">{rl}</div>
              <div style="font-size:.88rem;font-weight:600;margin-top:.2rem;">Probability: {rp:.1f}%</div>
            </div>
            <div style="text-align:right;">
              <div style="font-size:.78rem;opacity:.7;">Recommendation Source</div>
              <div style="font-size:.9rem;font-weight:600;">{src_badge}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── AI narrative ──────────────────────────────────────────────────────────
    if ai_narrative:
        st.markdown('<p class="section-title">🤖 AI Health Summary</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="hv-card" style="margin-bottom:1.5rem;border-left:4px solid #6B46C1;">
          <div style="line-height:1.85;font-size:.97rem;color:#1a1f2e;">{ai_narrative}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Metric snapshot ───────────────────────────────────────────────────────
    if hd:
        _SAFE = {
            "trestbps": (90,  120,  "mmHg",  "🩺 Resting BP"),
            "chol":     (0,   200,  "mg/dL", "🔬 Cholesterol"),
            "thalach":  (60,  100,  "bpm",   "💓 Max HR (exercise)"),
            "oldpeak":  (0.0, 1.0,  "mm",    "📉 ST Depression"),
        }
        st.markdown('<p class="section-title">📊 Your Current Values</p>', unsafe_allow_html=True)
        metric_cols = st.columns(4)
        for i, (field, (lo, hi, unit, lbl)) in enumerate(_SAFE.items()):
            val = hd.get(field)
            if val is None:
                continue
            v = float(val)
            if v < lo:
                status = "⚠️ Low"; chip_color = "#F59E0B"; bg = "#fef3c7"
            elif v > hi:
                status = "⚠️ High"; chip_color = "#EF4444"; bg = "#fee2e2"
            else:
                status = "✅ OK"; chip_color = "#10B981"; bg = "#d1fae5"
            metric_cols[i].markdown(f"""
            <div style="background:{bg};border-radius:12px;padding:.9rem 1rem;
                        border:1px solid {chip_color}33;margin-bottom:.5rem;">
              <div style="font-size:.75rem;color:#6c757d;">{lbl}</div>
              <div style="font-size:1.5rem;font-weight:900;color:{chip_color};">{v:.1f}
                <span style="font-size:.7rem;font-weight:400;">{unit}</span>
              </div>
              <div style="font-size:.72rem;color:{chip_color};font-weight:600;">{status}</div>
              <div style="font-size:.68rem;color:#6c757d;">Safe: {lo}–{hi} {unit}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Personalised tips ─────────────────────────────────────────────────────
    if tips:
        st.markdown('<p class="section-title">✅ Personalised Action Plan</p>', unsafe_allow_html=True)
        priority_colors = ["#EF4444", "#F59E0B", "#6B46C1", "#3B82F6", "#10B981"]
        priority_labels = ["Priority 1", "Priority 2", "Priority 3", "Priority 4", "Priority 5"]
        for i, tip in enumerate(tips):
            pc = priority_colors[min(i, len(priority_colors)-1)]
            pl = priority_labels[min(i, len(priority_labels)-1)]
            st.markdown(f"""
            <div class="tip-card" style="border-left:4px solid {pc};">
              <div class="tip-num" style="background:linear-gradient(135deg,{pc},{pc}cc);">
                {i+1}
              </div>
              <div style="flex:1;">
                <div style="font-size:.72rem;color:{pc};font-weight:700;
                            text-transform:uppercase;letter-spacing:.06em;margin-bottom:.2rem;">
                  {pl}
                </div>
                <div style="font-size:.95rem;line-height:1.65;color:#1a1f2e;">{tip}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    disc = reco.get("disclaimer", "")
    if disc:
        st.markdown(f"""
        <div style="background:#f0ebff;border-radius:10px;padding:.85rem 1rem;
                    margin-top:1.5rem;font-size:.82rem;color:#5b21b6;">
          {disc}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Regenerate Recommendations", use_container_width=True):
            if hd:
                with st.spinner("Regenerating…"):
                    try:
                        reco = run_reco_agent(
                            hd, ro or {},
                            progress_summary="",
                            use_groq=st.session_state.use_groq,
                        )
                        st.session_state.reco_output = reco
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Error: {exc}")
            else:
                st.warning("Please complete an assessment first.")
    with col2:
        if st.button("📋 Take New Assessment", use_container_width=True):
            st.session_state.page = "Assessment"
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: PROFILE
# ══════════════════════════════════════════════════════════════════════════════
def _page_profile():
    st.markdown('<h2 style="color:var(--text,#1a1f2e);font-weight:800;">👤 Profile</h2>', unsafe_allow_html=True)

    user = st.session_state.user
    if not user:
        st.warning("Not authenticated.")
        return

    email   = user.get("email", "N/A")
    user_id = user.get("id", "N/A")
    joined  = user.get("created_at", "")[:10]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div class="hv-card" style="text-align:center;">
          <div style="font-size:4rem;margin-bottom:.8rem;">👤</div>
          <div style="font-weight:700;font-size:1.1rem;color:#1a1f2e;word-break:break-all;">{email}</div>
          <div style="font-size:.82rem;color:#6c757d;margin-top:.4rem;">Member since {joined}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-title">📋 Account Details</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="hv-card">
          <table style="width:100%;font-size:.95rem;border-collapse:collapse;">
            <tr><td style="padding:.5rem 0;color:#6c757d;width:40%;">Email</td>
                <td style="font-weight:600;">{email}</td></tr>
            <tr><td style="padding:.5rem 0;color:#6c757d;">User ID</td>
                <td style="font-family:monospace;font-size:.82rem;">{user_id}</td></tr>
            <tr><td style="padding:.5rem 0;color:#6c757d;">Joined</td>
                <td>{joined}</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)

    # Assessment history stats
    st.markdown('<p class="section-title">📊 Your Statistics</p>', unsafe_allow_html=True)
    try:
        sb  = _supabase()
        res = sb.table("health_records").select("*").eq("user_id", user_id).execute()
        records = res.data or []
    except Exception:
        records = []

    n_total  = len(records)
    n_high   = sum(1 for r in records if r.get("risk_label") == "HIGH")
    n_medium = sum(1 for r in records if r.get("risk_label") == "MEDIUM")
    n_low    = sum(1 for r in records if r.get("risk_label") == "LOW")

    c1, c2, c3, c4 = st.columns(4)
    for col, num, lbl, color in [
        (c1, n_total,  "Total Assessments", "#6B46C1"),
        (c2, n_low,    "Low Risk",          "#10B981"),
        (c3, n_medium, "Medium Risk",       "#F59E0B"),
        (c4, n_high,   "High Risk",         "#EF4444"),
    ]:
        col.markdown(f"""
        <div class="hv-card" style="text-align:center;border-top:4px solid {color};">
          <div style="font-size:2rem;font-weight:800;color:{color};">{num}</div>
          <div style="font-size:.82rem;color:#6c757d;">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    # Delete all data
    st.markdown("---")
    st.markdown("#### ⚠️ Danger Zone")
    with st.expander("Delete all my health data"):
        st.warning("This will permanently delete all your health records. This cannot be undone.")
        confirm = st.text_input("Type **DELETE** to confirm", key="delete_confirm")
        if st.button("🗑️ Delete All Data", type="primary"):
            if confirm == "DELETE":
                try:
                    _supabase().table("health_records").delete().eq("user_id", user_id).execute()
                    st.session_state.risk_output    = None
                    st.session_state.health_data    = None
                    st.session_state.monitor_output = None
                    st.session_state.reco_output    = None
                    st.success("All health data deleted.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")
            else:
                st.error("Please type DELETE to confirm.")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN ROUTER
# ══════════════════════════════════════════════════════════════════════════════
def main():
    _init_session()
    _inject_css()

    if not st.session_state.authenticated:
        _auth_page()
        return

    _sidebar()

    page = st.session_state.page
    if page == "Dashboard":
        _page_dashboard()
    elif page == "Assessment":
        _page_assessment()
    elif page == "Risk Analysis":
        _page_risk_analysis()
    elif page == "Data Agent":
        _page_data_agent()
    elif page == "Monitoring":
        _page_monitoring()
    elif page == "Recommendations":
        _page_reco()
    elif page == "Profile":
        _page_profile()
    else:
        _page_dashboard()


if __name__ == "__main__":
    main()
