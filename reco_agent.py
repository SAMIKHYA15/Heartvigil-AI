"""
reco_agent.py  –  Agent 4
Generates personalised, context-aware health recommendations using
Groq (Llama 3.3 70B) with a rule-based fallback.
Recommendations reference the patient's ACTUAL clinical values.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ai_helper import call_groq

logger = logging.getLogger(__name__)


# ── Safe reference ranges (for contextual commentary) ────────────────────────
_SAFE_RANGES = {
    "trestbps": (90,  120,  "mmHg",   "Resting Blood Pressure"),
    "chol":     (0,   200,  "mg/dL",  "Cholesterol"),
    "thalach":  (60,  100,  "bpm",    "Max Heart Rate (exercise)"),
    "oldpeak":  (0.0, 1.0,  "mm",     "ST Depression"),
    "fbs":      (0,   0,    "",       "Fasting Blood Sugar >120"),
    "exang":    (0,   0,    "",       "Exercise-Induced Angina"),
    "ca":       (0,   0,    "vessels","Major Vessels Coloured"),
}

_FIELD_LABELS = {
    "age":      "Age",
    "sex":      "Sex",
    "cp":       "Chest Pain Type",
    "trestbps": "Resting BP",
    "chol":     "Cholesterol",
    "fbs":      "Fasting BS >120",
    "restecg":  "Resting ECG",
    "thalach":  "Max Heart Rate",
    "exang":    "Exercise Angina",
    "oldpeak":  "ST Depression",
    "slope":    "ST Slope",
    "ca":       "Vessels Coloured",
    "thal":     "Thal",
}


# ── Rule-based fallback tip bank ───────────────────────────────────────────────
_RULE_TIPS: Dict[str, List[str]] = {
    "HIGH": [
        "🏥 **Urgent cardiology referral recommended.** Your risk score warrants immediate professional evaluation.",
        "💊 **Medication review:** Work with your doctor to optimise your current cardiovascular medications and treatment plan.",
        "🚶 **Supervised exercise:** Begin low-intensity activity such as daily 20-minute gentle walks only after medical clearance.",
        "🥗 **Heart-healthy diet:** Eliminate saturated fats and sodium. Adopt a Mediterranean diet rich in olive oil, fish, nuts, and vegetables.",
        "🧘 **Stress management:** Practice daily deep breathing, meditation, or yoga for at least 10–15 minutes to lower cortisol levels.",
    ],
    "MEDIUM": [
        "🩺 **Schedule a GP check-up** within the next 4–6 weeks to review your cardiovascular risk factors in detail.",
        "🏃 **Exercise target:** Aim for 150 minutes of moderate aerobic activity per week — brisk walking, cycling, or swimming.",
        "🥦 **Nutrition upgrade:** Increase fibre (30 g/day) and omega-3 intake via leafy greens, salmon, walnuts, and flaxseed.",
        "⚖ **Weight management:** Even a 5–10% body weight reduction can significantly lower blood pressure and cholesterol.",
        "🚭 **Smoking cessation:** If you smoke, seek cessation support — quitting is the single most impactful lifestyle change available.",
    ],
    "LOW": [
        "✅ **Excellent risk profile!** Your current metrics are well-controlled — maintain your healthy lifestyle to preserve this.",
        "🏋 **Stay active:** Continue 150+ min/week of moderate activity or 75 min/week of vigorous aerobic exercise.",
        "🥗 **Diet maintenance:** Keep eating a balanced diet rich in whole grains, fruits, vegetables, lean proteins, and healthy fats.",
        "😴 **Sleep hygiene:** Prioritise 7–9 hours of quality sleep each night; chronic sleep deprivation raises cardiovascular risk.",
        "📅 **Annual screening:** Maintain yearly health check-ups to monitor cholesterol, BP, and blood glucose proactively.",
    ],
}

_DEFAULT_TIPS = _RULE_TIPS["MEDIUM"]


# ── Build a human-readable metric snapshot ────────────────────────────────────
def _build_metric_context(health_data: Dict[str, Any]) -> str:
    """Return a concise text block listing key metrics with safe-range context."""
    lines = []
    for field, (lo, hi, unit, label) in _SAFE_RANGES.items():
        val = health_data.get(field)
        if val is None:
            continue
        v = float(val)
        if hi == 0:
            status = "⚠ ABNORMAL" if v > 0 else "✅ Normal"
        elif v < lo:
            status = "⚠ Below normal range"
        elif v > hi:
            status = f"⚠ Elevated (safe: {lo}–{hi} {unit})"
        else:
            status = f"✅ Within safe range ({lo}–{hi} {unit})"
        unit_str = f" {unit}" if unit else ""
        lines.append(f"  • {label}: {v:.1f}{unit_str} — {status}")
    # add age and sex for context
    if "age" in health_data:
        lines.insert(0, f"  • Age: {health_data['age']} years")
    if "sex" in health_data:
        lines.insert(1, f"  • Sex: {'Male' if health_data['sex'] == 1 else 'Female'}")
    return "\n".join(lines)


# ── Groq-powered personalised tips ────────────────────────────────────────────
def _ai_recommendations(
    health_data: Dict[str, Any],
    risk_output: Dict[str, Any],
    progress_summary: str,
) -> List[str]:
    """
    Request 5 personalised tips from Groq.
    Each tip must cite the patient's actual values.
    """
    label   = risk_output.get("risk_label", "MEDIUM")
    pct     = risk_output.get("risk_pct", 50)
    reasons = risk_output.get("reasons", [])

    metric_ctx = _build_metric_context(health_data)

    system = (
        "You are a senior preventive cardiologist and health coach AI. "
        "Generate exactly 5 numbered, personalised health recommendations. "
        "IMPORTANT: Each recommendation MUST reference the patient's actual measured value "
        "and explain WHY that value is concerning or positive. "
        "Crucially, you must provide HIGHLY SPECIFIC, actionable lifestyle advice: tell the user exactly what specific foods to eat, "
        "what beverages to drink, what exact ingredients to avoid, and what specific types of exercises they should do to improve that exact metric. "
        "Be highly specific, actionable, and empathetic. Use emoji prefixes. "
        "Do not diagnose, prescribe medications, or give specific drug names. "
        "Format each tip on its own line as: '1. [emoji] Recommendation text'\n"
        "Example of good format: '1. 🩺 Your resting blood pressure of 145 mmHg is above the safe range "
        "of 90–120 mmHg. To lower this naturally, limit sodium to under 2,000 mg/day, drink hibiscus tea or beet juice, snack on potassium-rich bananas, and aim for 30 minutes of brisk walking daily.'"
    )

    user = (
        f"Patient Risk Assessment:\n"
        f"  • Overall Risk: {label} ({pct:.1f}% probability)\n"
        f"  • Key clinical risk flags:\n"
        + "\n".join(f"    – {r}" for r in reasons)
        + f"\n\nPatient's Current Metrics:\n{metric_ctx}"
        + (f"\n\nHealth Trend Context:\n{progress_summary}" if progress_summary else "")
        + "\n\nGenerate 5 specific, value-referencing recommendations:"
    )

    raw = call_groq(
        system_prompt=system,
        user_prompt=user,
        fallback="",
        max_tokens=700,
    )

    if not raw:
        return []

    import re
    tips = re.findall(r"\d+\.\s+(.+)", raw)
    return tips[:5] if tips else []


# ── Groq-powered narrative ─────────────────────────────────────────────────────
def _ai_narrative(
    health_data: Dict[str, Any],
    risk_output: Dict[str, Any],
) -> str:
    """Generate a short empathetic narrative paragraph summarising the patient's situation."""
    label  = risk_output.get("risk_label", "MEDIUM")
    pct    = risk_output.get("risk_pct", 50)
    metric_ctx = _build_metric_context(health_data)

    system = (
        "You are a compassionate preventive cardiology AI assistant. "
        "Write a single paragraph (3–5 sentences) that warmly summarises the patient's "
        "heart health status, references their key measured values by name, "
        "and motivates them to take action. Keep the tone supportive, never alarmist. "
        "Do not diagnose or prescribe. Keep under 100 words."
    )
    user = (
        f"Risk: {label} ({pct:.1f}%)\n"
        f"Patient metrics:\n{metric_ctx}"
    )
    return call_groq(
        system_prompt=system,
        user_prompt=user,
        fallback=(
            f"Your heart health assessment shows a **{label}** risk level "
            f"with a probability score of **{pct:.1f}%**. "
            "Review the personalised recommendations below and discuss them with your healthcare provider."
        ),
        max_tokens=200,
    )


# ── Rule-based fallback ────────────────────────────────────────────────────────
def _rule_based_tips(
    risk_label: str,
    health_data: Dict[str, Any],
) -> List[str]:
    """Return rule-based tips enhanced with actual patient values where possible."""
    base = list(_RULE_TIPS.get(risk_label.upper(), _DEFAULT_TIPS))

    # Append value-specific tips at the end (replace last generic tip)
    specific = []
    bp = health_data.get("trestbps")
    if bp and float(bp) > 120:
        specific.append(
            f"🩺 **Blood pressure:** Your resting BP is **{bp} mmHg** (safe: 90–120 mmHg). "
            "To lower this naturally, strictly limit sodium to < 2,000 mg/day, drink hibiscus tea, eat potassium-rich foods like bananas and spinach, and walk briskly for 30 minutes daily."
        )
    chol = health_data.get("chol")
    if chol and float(chol) > 200:
        specific.append(
            f"🔬 **Cholesterol:** Your cholesterol is **{chol} mg/dL** (desirable: < 200 mg/dL). "
            "Lower it by eating a bowl of oatmeal for breakfast, snacking on raw almonds, drinking green tea, and swapping red meat for baked salmon twice a week."
        )
    hr = health_data.get("thalach")
    if hr and float(hr) < 100:
        specific.append(
            f"💓 **Max heart rate:** Achieved only **{hr} bpm** during exercise. "
            "Stay hydrated with water and electrolytes, and slowly build your cardiac endurance with light jogging, swimming, or cycling 3 times a week."
        )
    elif hr and float(hr) > 160:
        specific.append(
            f"💓 **Max heart rate:** Your heart rate reached **{hr} bpm**, showing high cardiac effort. "
            "Ensure you are drinking enough water before and during exercise, and incorporate cooling-down stretches and deep-breathing exercises to manage cardiovascular stress."
        )
    fbs = health_data.get("fbs")
    if fbs and int(fbs) == 1:
        specific.append(
            f"🩸 **Fasting Blood Sugar:** Your blood sugar is elevated (>120 mg/dL). "
            "Avoid sugary drinks and refined carbs completely. Drink water with freshly squeezed lemon, eat high-fibre foods like lentils, and take a 15-minute walk immediately after meals to improve your insulin sensitivity."
        )

    # Merge: keep first (5 - len(specific)) base tips + specific tips
    keep = max(0, 5 - len(specific))
    return (base[:keep] + specific)[:5]


# ── Agent entry-point ─────────────────────────────────────────────────────────
def run_reco_agent(
    health_data: Dict[str, Any],
    risk_output: Dict[str, Any],
    progress_summary: str = "",
    use_groq: bool = True,
) -> Dict[str, Any]:
    """
    Agent 4 full pipeline.

    Parameters
    ----------
    health_data      : dict  – 13 clinical features (age, trestbps, chol …)
    risk_output      : dict  – output from run_risk_agent()
    progress_summary : str   – output from monitor_agent.build_progress_summary()
    use_groq         : bool  – whether to attempt Groq API call

    Returns
    -------
    dict with keys:
        tips         : list[str]  – exactly 5 recommendation strings
        ai_narrative : str        – short empathetic summary paragraph
        metric_ctx   : str        – metric snapshot text
        source       : str        – "groq" | "rule-based"
        disclaimer   : str        – legal disclaimer
    """
    risk_label = risk_output.get("risk_label", "MEDIUM")
    tips: List[str] = []
    narrative = ""
    source = "rule-based"

    metric_ctx = _build_metric_context(health_data)

    if use_groq:
        try:
            tips = _ai_recommendations(health_data, risk_output, progress_summary)
            if tips:
                source = "groq"
        except Exception as exc:
            logger.error("AI recommendations failed: %s", exc)

        try:
            narrative = _ai_narrative(health_data, risk_output)
        except Exception as exc:
            logger.error("AI narrative failed: %s", exc)

    # Fall back to rule-based if Groq returned nothing
    if not tips:
        tips = _rule_based_tips(risk_label, health_data)
        source = "rule-based"

    if not narrative:
        pct = risk_output.get("risk_pct", 0)
        narrative = (
            f"Your heart health assessment shows a **{risk_label}** risk level "
            f"with a probability score of **{pct:.1f}%**. "
            "Review the personalised recommendations below and discuss them with your healthcare provider."
        )

    # Ensure exactly 5 tips
    while len(tips) < 5:
        tips.append("🔄 Continue monitoring your health metrics regularly with follow-up assessments.")
    tips = tips[:5]

    disclaimer = (
        "⚠ **Disclaimer:** These recommendations are AI-generated for "
        "educational purposes only and do not constitute medical advice. "
        "Always consult a qualified healthcare professional before making "
        "changes to your health routine."
    )

    return {
        "tips":         tips,
        "ai_narrative": narrative,
        "metric_ctx":   metric_ctx,
        "source":       source,
        "disclaimer":   disclaimer,
    }
