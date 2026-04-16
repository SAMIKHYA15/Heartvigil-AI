"""
risk_agent.py  –  Agent 2  (Phase 2)
Loads the trained ensemble model and generates a risk prediction with
clinical reasoning, AI explanation, AND risk direction vs previous assessment.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import joblib
import pandas as pd

# Fix: Import the correct function name
from ai_helper import call_groq

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR              = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH            = os.path.join(BASE_DIR, "model.joblib")
SCALER_PATH           = os.path.join(BASE_DIR, "scaler.joblib")
FEAT_NAMES_PATH       = os.path.join(BASE_DIR, "feature_names.joblib")

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal",
]

# ── Risk thresholds ─────────────────────────────────────────────────────────────
LOW_THRESH    = 0.35
MEDIUM_THRESH = 0.65

RISK_COLORS = {
    "LOW":    "#10B981",
    "MEDIUM": "#F59E0B",
    "HIGH":   "#EF4444",
}

# ── Clinical observations (no probability adjustments — model is trusted) ─────
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
]


# ── Model loading (cached) ────────────────────────────────────────────────────
_model  = None
_scaler = None


def _load_artifacts():
    """Load model and scaler lazily (once per process)."""
    global _model, _scaler
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "model.joblib not found. Run train.py first."
            )
        _model  = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
    return _model, _scaler


# ── Clinical observation engine (probability-preserving) ──────────────────────
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

    return round(base_prob, 4), reasons


# ── Risk label ─────────────────────────────────────────────────────────────────
def _label(prob: float) -> str:
    if prob < LOW_THRESH:
        return "LOW"
    elif prob < MEDIUM_THRESH:
        return "MEDIUM"
    return "HIGH"


# ── Risk direction vs previous assessment ─────────────────────────────────────
def _compute_direction(
    current_prob: float,
    previous: Optional[Dict],
    delta: Optional[Dict],
) -> Dict[str, Any]:
    """
    Compare current risk with the previous assessment and compute direction.

    Returns
    -------
    dict:
        direction       : "improved" | "worsened" | "stable" | "first"
        direction_icon  : emoji
        direction_color : hex
        score_change    : float (percentage points change)
        prev_label      : str
        changed_fields  : list[dict]  – fields that changed significantly
    """
    if previous is None:
        return {
            "direction":       "first",
            "direction_icon":  "\u2605",   # star
            "direction_color": "#6B46C1",
            "score_change":    0.0,
            "prev_label":      None,
            "changed_fields":  [],
        }

    raw_prev = float(previous.get("risk_score") or 0)
    # Detect scale: new records stored 0-1, old stored 0-100
    prev_score = raw_prev if raw_prev > 1.5 else raw_prev * 100

    curr_score = current_prob * 100
    change     = curr_score - prev_score
    prev_label = previous.get("risk_label", "N/A") or "N/A"

    if change <= -3:
        direction = "improved"
        icon  = "\u2713"   # check mark
        color = "#10B981"
    elif change >= 3:
        direction = "worsened"
        icon  = "\u26a0"   # warning triangle (no variation selector)
        color = "#EF4444"
    else:
        direction = "stable"
        icon  = "\u2192"   # right arrow
        color = "#F59E0B"

    # Which fields changed significantly?
    changed_fields = []
    if delta:
        field_labels = {
            "trestbps": "Blood Pressure",
            "chol":     "Cholesterol",
            "thalach":  "Max Heart Rate",
            "oldpeak":  "ST Depression",
            "ca":       "Vessels Coloured",
            "fbs":      "Fasting Blood Sugar",
            "exang":    "Exercise Angina",
        }
        for field, pct_change in delta.items():
            if pct_change is None or field not in field_labels:
                continue
            abs_chg = abs(pct_change)
            if abs_chg >= 5:
                direction_f = "\u25b2 increased" if pct_change > 0 else "\u25bc decreased"
                changed_fields.append({
                    "field":     field,
                    "label":     field_labels[field],
                    "pct":       pct_change,
                    "direction": direction_f,
                    "concern":   (pct_change > 0 and field in ("trestbps", "chol", "oldpeak", "ca"))
                                 or (pct_change < 0 and field == "thalach"),
                })
        # Sort by magnitude
        changed_fields.sort(key=lambda x: abs(x["pct"]), reverse=True)

    return {
        "direction":       direction,
        "direction_icon":  icon,
        "direction_color": color,
        "score_change":    round(change, 1),
        "prev_label":      prev_label,
        "changed_fields":  changed_fields[:5],  # top 5
    }


# ── AI explanation ─────────────────────────────────────────────────────────────
def _groq_explanation(
    health_data: Dict[str, Any],
    risk_label: str,
    risk_pct: float,
    reasons: List[str],
    direction_info: Optional[Dict] = None,
) -> str:
    """Generate AI explanation using Groq (or fallback)."""
    
    direction_ctx = ""
    if direction_info and direction_info.get("direction") != "first":
        direction_ctx = (
            f"\nCompared to previous assessment: risk has {direction_info['direction']} "
            f"(change: {direction_info['score_change']:+.1f} percentage points). "
            f"Previous risk level: {direction_info['prev_label']}."
        )
        if direction_info.get("changed_fields"):
            changes = ", ".join(
                f"{f['label']} {f['direction']} by {abs(f['pct']):.1f}%"
                for f in direction_info["changed_fields"][:3]
            )
            direction_ctx += f" Key metric changes: {changes}."

    system = (
        "You are a senior cardiologist AI assistant. "
        "Provide a concise, empathetic explanation of a patient's heart disease risk assessment. "
        "Use plain language (no jargon). Never diagnose or prescribe. "
        "Encourage the patient to consult a physician. Keep the response under 200 words."
    )
    user = (
        f"Patient risk level: {risk_label} ({risk_pct:.1f}%){direction_ctx}\n"
        f"Key clinical findings:\n"
        + "\n".join(f"- {r}" for r in reasons)
        + f"\n\nHealth data summary: age={health_data.get('age')}, "
        f"BP={health_data.get('trestbps')}, chol={health_data.get('chol')}, "
        f"maxHR={health_data.get('thalach')}."
    )
    
    return call_groq(
        system_prompt=system,
        user_prompt=user,
        fallback=(
            f"Your heart disease risk is assessed as **{risk_label}** "
            f"({risk_pct:.1f}%). "
            "Please discuss these results with your healthcare provider for "
            "a comprehensive evaluation and personalised advice."
        ),
        max_tokens=250,
    )


# ── Feature importance ─────────────────────────────────────────────────────────
def _get_feature_importances(model) -> Dict[str, float]:
    """Return feature importances if the model supports it."""
    try:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "estimators_"):
            # VotingClassifier – average importances from tree-based estimators
            imps = []
            for est in model.estimators_:
                if hasattr(est, "feature_importances_"):
                    imps.append(est.feature_importances_)
            if not imps:
                return {}
            imp = np.mean(imps, axis=0)
        else:
            return {}
        return {col: float(v) for col, v in zip(FEATURE_COLS, imp)}
    except Exception:
        return {}


# ── Public API (backward compatible with app.py) ───────────────────────────────
def doctor_ai_agent(
    health_data: Dict[str, Any],
    use_groq: bool = True,
    previous_record: Optional[Dict] = None,
    delta: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Main entry point for Agent 2 (called from app.py).

    Parameters
    ----------
    health_data     : dict   – Keys = 13 UCI feature names.
    use_groq        : bool   – Whether to call Groq for an AI explanation.
    previous_record : dict   – Previous health_record row from Supabase (or None).
    delta           : dict   – Percent-change per field from data_agent.compute_delta().

    Returns
    -------
    dict with keys expected by app.py:
        probability     : float  (0–100 percentage)
        risk_label      : str    ("LOW" | "MEDIUM" | "HIGH")
        risk_color      : str    (hex colour)
        reasons         : list[str]
        ai_explanation  : str
        risk_direction  : str    (for dashboard banner)
        probability_change : float
    """
    try:
        model, scaler = _load_artifacts()
    except FileNotFoundError as e:
        logger.warning("Model not found, using rule-based fallback: %s", e)
        base_prob = 0.5
        adjusted_prob, reasons = _apply_clinical_rules(base_prob, health_data)
        label = _label(adjusted_prob)
        pct   = round(adjusted_prob * 100, 1)
        dir_info = _compute_direction(adjusted_prob, previous_record, delta)
        explanation = _groq_explanation(health_data, label, pct, reasons, dir_info) if use_groq else ""
        return {
            "risk_prob":           adjusted_prob,
            "risk_pct":            pct,
            "risk_label":          label,
            "risk_score":          adjusted_prob,
            "risk_color":          RISK_COLORS[label],
            "reasons":             reasons,
            "ai_explanation":      explanation,
            "raw_prob":            base_prob,
            "direction_info":      dir_info,
            "feature_importances": {},
        }

    # Build 13-feature vector using DataFrame with proper column names
    feat_vec = {col: float(health_data.get(col, 0)) for col in FEATURE_COLS}
    feat_df     = pd.DataFrame([feat_vec], columns=FEATURE_COLS)
    feat_scaled = pd.DataFrame(
        scaler.transform(feat_df), columns=FEATURE_COLS
    )

    raw_prob = float(model.predict_proba(feat_scaled)[0][1])
    adj_prob, reasons = _apply_clinical_rules(raw_prob, health_data)
    label         = _label(adj_prob)
    pct           = round(adj_prob * 100, 1)

    dir_info = _compute_direction(adj_prob, previous_record, delta)

    explanation = ""
    if use_groq:
        explanation = _groq_explanation(health_data, label, pct, reasons, dir_info)

    return {
        "risk_prob":           adj_prob,
        "risk_pct":            pct,
        "risk_label":          label,
        "risk_score":          adj_prob,
        "risk_color":          RISK_COLORS[label],
        "reasons":             reasons,
        "ai_explanation":      explanation,
        "raw_prob":            raw_prob,
        "direction_info":      dir_info,
        "feature_importances": _get_feature_importances(model),
    }


# For backward compatibility with older app.py calls
def run_risk_agent(
    health_data: Dict[str, Any],
    use_groq: bool = True,
    previous_record: Optional[Dict] = None,
    delta: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Alias for doctor_ai_agent (backward compatibility)."""
    return doctor_ai_agent(health_data, use_groq, previous_record, delta)