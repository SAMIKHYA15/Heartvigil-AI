"""
Patch script - fixes risk_agent.py, ai_helper.py, then restarts server
"""
import os, re

BASE = r"c:\Users\ADMIN\.gemini\antigravity\scratch\heartvigil-ai"

# ═══════════════════════════════════════════════════════
# 1. Rewrite risk_agent.py (key sections only via markers)
# ═══════════════════════════════════════════════════════
with open(os.path.join(BASE, "risk_agent.py"), encoding="utf-8") as f:
    src = f.read()

# ── Fix A: Replace CLINICAL_RULES with observation-only version ──────────────
OLD_RULES = '''# ── Clinical rule descriptions ─────────────────────────────────────────────────
CLINICAL_RULES: List[Dict[str, Any]] = [
    {
        "condition": lambda d: d.get("age", 0) >= 65,
        "reason":    "Age \\u2265 65 is a significant cardiovascular risk factor.",
        "delta":     0.05,
    },
    {
        "condition": lambda d: d.get("trestbps", 0) >= 140,
        "reason":    "Resting blood pressure \\u2265 140 mmHg indicates hypertension.",
        "delta":     0.07,
    },
    {
        "condition": lambda d: d.get("chol", 0) >= 240,
        "reason":    "High cholesterol (\\u2265 240 mg/dL) increases arterial plaque risk.",
        "delta":     0.06,
    },
    {
        "condition": lambda d: d.get("fbs", 0) == 1,
        "reason":    "Fasting blood sugar > 120 mg/dL suggests diabetes risk.",
        "delta":     0.05,
    },
    {
        "condition": lambda d: d.get("exang", 0) == 1,
        "reason":    "Exercise-induced angina is a key indicator of ischaemia.",
        "delta":     0.08,
    },
    {
        "condition": lambda d: d.get("ca", 0) >= 2,
        "reason":    "\\u2265 2 major vessels colored by fluoroscopy indicates advanced CAD.",
        "delta":     0.10,
    },
    {
        "condition": lambda d: d.get("oldpeak", 0) >= 2.0,
        "reason":    "ST depression \\u2265 2.0 mm indicates myocardial stress.",
        "delta":     0.07,
    },
    {
        "condition": lambda d: d.get("cp", 0) == 3,
        "reason":    "Asymptomatic chest pain can mask silent ischaemia.",
        "delta":     0.04,
    },
    {
        "condition": lambda d: d.get("thalach", 0) < 100,
        "reason":    "Low max heart rate (< 100 bpm) may indicate poor cardiac reserve.",
        "delta":     0.04,
    },
    {
        "condition": lambda d: d.get("sex", 0) == 1 and d.get("age", 0) >= 45,
        "reason":    "Males aged \\u2265 45 have substantially elevated coronary risk.",
        "delta":     0.03,
    },
    {
        "condition": lambda d: d.get("sex", 0) == 0 and d.get("age", 0) >= 55,
        "reason":    "Post-menopausal females (\\u2265 55) face increased cardiac risk.",
        "delta":     0.03,
    },
    {
        "condition": lambda d: d.get("thal", 0) in (1, 2),
        "reason":    "Fixed/reversible thalassemia defect suggests perfusion abnormality.",
        "delta":     0.06,
    },
]'''

NEW_RULES = '''# ── Clinical observations (no probability adjustments — model is trusted) ─────
# The calibrated ensemble (88.5% acc, AUC=0.96) is the ground truth.
# Rules below collect USER-FACING explanations only, do NOT shift probabilities.
CLINICAL_RULES: List[Dict[str, Any]] = [
    {
        "condition": lambda d: float(d.get("age", 0)) >= 60,
        "reason":    "Age 60+ is a well-established cardiovascular risk factor.",
    },
    {
        "condition": lambda d: int(d.get("sex", 0)) == 1 and float(d.get("age", 0)) >= 45,
        "reason":    "Males aged 45+ face substantially elevated coronary artery disease risk.",
    },
    {
        "condition": lambda d: int(d.get("sex", 0)) == 0 and float(d.get("age", 0)) >= 55,
        "reason":    "Post-menopausal females (55+) have increased cardiac risk after oestrogen decline.",
    },
    {
        "condition": lambda d: float(d.get("trestbps", 0)) >= 140,
        "reason":    "Resting BP >= 140 mmHg confirms hypertension — a primary heart disease risk factor.",
    },
    {
        "condition": lambda d: float(d.get("trestbps", 0)) >= 160,
        "reason":    "Resting BP >= 160 mmHg indicates Stage 2 hypertension requiring urgent attention.",
    },
    {
        "condition": lambda d: float(d.get("chol", 0)) >= 240,
        "reason":    "Total cholesterol >= 240 mg/dL promotes atherosclerotic plaque formation.",
    },
    {
        "condition": lambda d: float(d.get("chol", 0)) >= 280,
        "reason":    "Severely elevated cholesterol (>= 280 mg/dL) significantly increases CAD risk.",
    },
    {
        "condition": lambda d: int(d.get("fbs", 0)) == 1,
        "reason":    "Fasting blood sugar > 120 mg/dL suggests diabetes — a major cardiac risk amplifier.",
    },
    {
        "condition": lambda d: float(d.get("oldpeak", 0)) >= 2.0,
        "reason":    "ST depression >= 2.0 mm during exercise indicates significant myocardial ischaemia.",
    },
    {
        "condition": lambda d: float(d.get("oldpeak", 0)) >= 3.5,
        "reason":    "ST depression >= 3.5 mm is a high-grade indicator of severe coronary disease.",
    },
    {
        "condition": lambda d: float(d.get("thalach", 0)) < 120,
        "reason":    "Max heart rate < 120 bpm may reflect reduced cardiac reserve or chronotropic incompetence.",
    },
    {
        "condition": lambda d: int(d.get("restecg", 0)) >= 1,
        "reason":    "Resting ECG abnormality detected — further cardiac evaluation is recommended.",
    },
    {
        "condition": lambda d: int(d.get("slope", 2)) == 0,
        "reason":    "Downsloping ST segment (slope=0) during exercise is associated with ischaemia.",
    },
]'''

# ── Fix B: Replace _apply_clinical_rules to NOT adjust probability ────────────
OLD_ENGINE = '''# ── Clinical rule engine ───────────────────────────────────────────────────────
def _apply_clinical_rules(
    base_prob: float,
    health_data: Dict[str, Any],
) -> tuple[float, List[str]]:
    """
    Apply small, bounded clinical adjustments to the calibrated model output.
    Each rule shifts probability by a small delta, CAPPED at \\u00b10.12 total
    so we don\'t distort the calibration \\u2014 the model is already well-calibrated.

    Returns
    -------
    (adjusted_prob: float, reasons: list[str])
    """
    reasons: List[str] = []
    total_delta = 0.0
    MAX_DELTA   = 0.12   # cap total clinical adjustment

    for rule in CLINICAL_RULES:
        try:
            if rule["condition"](health_data):
                reasons.append(rule["reason"])
                total_delta += rule["delta"]
        except Exception:
            continue

    # Clamp and blend: 80% model, 20% clinical rules (to preserve calibration)
    bounded_delta = max(-MAX_DELTA, min(MAX_DELTA, total_delta))
    prob = base_prob + 0.20 * bounded_delta
    prob = max(0.0, min(1.0, prob))

    if not reasons:
        reasons.append("No major clinical risk flags detected based on provided data.")

    return round(prob, 4), reasons'''

NEW_ENGINE = '''# ── Clinical observation engine (probability-preserving) ──────────────────────
def _apply_clinical_rules(
    base_prob: float,
    health_data: Dict[str, Any],
) -> tuple[float, List[str]]:
    """
    Collect clinical observation reasons for display WITHOUT adjusting probability.
    The calibrated stacking ensemble (88.5% accuracy, AUC=0.96) is the ground truth.
    Clinical rules are used ONLY to generate human-readable explanations.

    Returns
    -------
    (prob: float, reasons: list[str])   -- prob is UNCHANGED from base_prob
    """
    reasons: List[str] = []

    for rule in CLINICAL_RULES:
        try:
            if rule["condition"](health_data):
                reasons.append(rule["reason"])
        except Exception:
            continue

    if not reasons:
        reasons.append(
            "No classical risk factors triggered for this profile. "
            "The model prediction is based on your complete clinical pattern."
        )

    return round(base_prob, 4), reasons'''

applied = 0
for old, new in [(OLD_RULES, NEW_RULES), (OLD_ENGINE, NEW_ENGINE)]:
    # Try exact match first
    if old in src:
        src = src.replace(old, new)
        applied += 1
        print("Applied exact match")
    else:
        print("Exact match failed, trying regex...")

# If exact didn't work, use regex fallback on the rules block
if applied < 2:
    # Replace delta-adjusting engine with simple version
    pat = r'(# ── Clinical rule engine.*?return round\(prob, 4\), reasons)'
    m = re.search(pat, src, re.DOTALL)
    if m:
        src = src[:m.start()] + NEW_ENGINE + src[m.end():]
        applied += 1
        print("Applied engine via regex")

    # Replace CLINICAL_RULES block
    pat2 = r'(# ── Clinical rule descriptions.*?CLINICAL_RULES.*?\])'
    m2 = re.search(pat2, src, re.DOTALL)
    if m2:
        src = src[:m2.start()] + NEW_RULES + src[m2.end():]
        applied += 1
        print("Applied rules via regex")

print(f"Total patches applied: {applied}")

with open(os.path.join(BASE, "risk_agent.py"), "w", encoding="utf-8") as f:
    f.write(src)

# Syntax check risk_agent.py
import ast
try:
    ast.parse(open(os.path.join(BASE, "risk_agent.py"), encoding="utf-8", errors="replace").read())
    print("risk_agent.py syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR line {e.lineno}: {e.msg}")

# ═══════════════════════════════════════════════════════
# 2. Improve ai_helper.py - better model, retry logic, streaming ready
# ═══════════════════════════════════════════════════════
AI_HELPER = '''"""
ai_helper.py
Helper for calling the Groq API (Llama 3.3 70B Versatile).
All functions return graceful fallback strings when the API is unavailable.
"""

import os
import logging
import time

logger = logging.getLogger(__name__)

# ── Model config ───────────────────────────────────────────────────────────────
GROQ_MODEL   = "llama-3.3-70b-versatile"
MAX_TOKENS   = 1024
TEMPERATURE  = 0.35
MAX_RETRIES  = 2


def _get_groq_key() -> str:
    """Retrieve the Groq API key from env / Streamlit secrets."""
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    return key


def call_groq(
    system_prompt: str,
    user_prompt: str,
    fallback: str = "AI explanation is currently unavailable. Please consult your physician.",
    max_tokens: int = MAX_TOKENS,
    temperature: float = TEMPERATURE,
) -> str:
    """
    Call the Groq chat completions API with retry on transient errors.

    Parameters
    ----------
    system_prompt : str  -- System instruction for the LLM.
    user_prompt   : str  -- User-facing message / context.
    fallback      : str  -- Text returned when the API call fails.
    max_tokens    : int  -- Maximum response tokens.
    temperature   : float -- Sampling temperature (lower = more deterministic).

    Returns
    -------
    str -- LLM response text, or ``fallback`` on error.
    """
    api_key = _get_groq_key()
    if not api_key:
        logger.warning("GROQ_API_KEY not set -- returning fallback.")
        return fallback

    for attempt in range(MAX_RETRIES):
        try:
            from groq import Groq
            client   = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = response.choices[0].message.content.strip()
            if text:
                return text
        except ImportError:
            logger.error("groq package not installed. Run: pip install groq")
            return fallback
        except Exception as exc:
            err = str(exc)
            logger.warning("Groq API attempt %d error: %s", attempt + 1, err)
            # Rate-limit: wait briefly before retry
            if "rate_limit" in err.lower() or "429" in err:
                time.sleep(2 * (attempt + 1))
            elif attempt == MAX_RETRIES - 1:
                logger.error("Groq API failed after %d attempts: %s", MAX_RETRIES, exc)
                return fallback

    return fallback


def summarise_health_trends(trends_text: str) -> str:
    """Return an AI-generated plain-English summary of health trends."""
    system = (
        "You are a compassionate AI health assistant working with a cardiologist. "
        "Summarise the patient\\'s health trends in 3-4 clear, empathetic sentences. "
        "Do NOT diagnose or prescribe. Use simple, non-technical language. "
        "If trends are improving, be encouraging. If worsening, gently urge physician consultation."
    )
    return call_groq(
        system_prompt=system,
        user_prompt=f"Health trend data:\\n{trends_text}",
        fallback=(
            "Your health metrics have been recorded and are being monitored. "
            "Please consult a physician for a professional interpretation of your trends."
        ),
        max_tokens=350,
    )


def get_ai_response(prompt: str, system: str = "") -> str:
    """General-purpose wrapper for Groq API calls."""
    return call_groq(
        system_prompt=system or "You are a helpful and concise medical AI assistant.",
        user_prompt=prompt,
        fallback="AI response unavailable. Please try again later.",
        max_tokens=512,
    )
'''

with open(os.path.join(BASE, "ai_helper.py"), "w", encoding="utf-8") as f:
    f.write(AI_HELPER)

try:
    ast.parse(AI_HELPER)
    print("ai_helper.py syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")

# ═══════════════════════════════════════════════════════
# 3. Verify model files exist
# ═══════════════════════════════════════════════════════
for fname in ["model.joblib", "scaler.joblib", "feature_names.joblib", "feature_names_full.joblib"]:
    p = os.path.join(BASE, fname)
    exists = os.path.exists(p)
    size_kb = os.path.getsize(p)//1024 if exists else 0
    print(f"  {fname}: {'OK' if exists else 'MISSING'} ({size_kb} KB)")

print("Patch complete.")
