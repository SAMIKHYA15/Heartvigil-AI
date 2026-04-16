"""
monitor_agent.py  –  Agent 3  (Phase 2)
Fetches historical health records, computes per-field trends with
improved/worsened/stable status, detects clinically concerning patterns,
and builds a rich progress summary for Agent 4.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from ai_helper import call_groq

logger = logging.getLogger(__name__)

# ── Metric configuration ───────────────────────────────────────────────────────
NUMERIC_METRICS = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
    "ca", "fbs", "exang",
]

SAFE_RANGES: Dict[str, Tuple[float, float, str]] = {
    "trestbps": (90,  120,  "mmHg"),
    "chol":     (0,   200,  "mg/dL"),
    "thalach":  (60,  100,  "bpm"),
    "oldpeak":  (0,   1.0,  "mm"),
    "ca":       (0,   0,    "vessels"),
    "fbs":      (0,   0,    "(0=normal)"),
    "exang":    (0,   0,    "(0=normal)"),
}

METRIC_LABELS: Dict[str, str] = {
    "age":        "Age",
    "trestbps":   "Resting BP",
    "chol":       "Cholesterol",
    "thalach":    "Max Heart Rate",
    "oldpeak":    "ST Depression",
    "ca":         "Vessels Colored",
    "fbs":        "Fasting BS > 120",
    "exang":      "Exercise Angina",
    "risk_score": "Risk Score (%)",
}

# ── For per-field progress: higher = worse for these fields ───────────────────
# True  = rising value is BAD (e.g. BP, cholesterol)
# False = rising value is GOOD (e.g. max HR)
FIELD_HIGHER_IS_WORSE = {
    "trestbps": True,
    "chol":     True,
    "thalach":  False,    # higher max HR = better cardiac reserve
    "oldpeak":  True,
    "ca":       True,
    "fbs":      True,
    "exang":    True,
    "risk_score": True,
}

# Concerning trend rules: (field, direction, threshold_pct, message)
TREND_RULES = [
    ("trestbps",  "up",   5.0, "Rising blood pressure trend detected."),
    ("chol",      "up",   5.0, "Rising cholesterol trend detected."),
    ("thalach",   "down", 5.0, "Declining maximum heart rate – possible cardiac reserve reduction."),
    ("oldpeak",   "up",   10.0,"Increasing ST depression – possible worsening ischaemia."),
    ("ca",        "up",   0.0, "Increase in fluoroscopy-colored vessels – advancing CAD."),
    ("risk_score","up",   5.0, "Overall risk score is trending upward."),
]


# ── Fetch records ──────────────────────────────────────────────────────────────
def fetch_history(supabase, user_id: str, limit: int = 50) -> pd.DataFrame:
    """
    Return the user's last *limit* health records as a DataFrame.
    Columns include all 13 features plus risk_score, risk_label, created_at.
    """
    try:
        res = (
            supabase.table("health_records")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        if not res.data:
            return pd.DataFrame()
        df = pd.DataFrame(res.data)
        df["created_at"] = pd.to_datetime(df["created_at"])
        # Safely convert risk_score; old or placeholder records may be None/0
        df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")

        # Remove only TRULY empty placeholder rows (all 4 core clinical values are 0/null)
        # Do NOT filter on risk_label/risk_score alone — valid records may have null scores
        core_fields = ["age", "trestbps", "chol", "thalach"]
        for f in core_fields:
            if f in df.columns:
                df[f] = pd.to_numeric(df[f], errors="coerce")

        if all(f in df.columns for f in core_fields):
            empty_mask = (
                df["age"].fillna(0).eq(0) &
                df["trestbps"].fillna(0).eq(0) &
                df["chol"].fillna(0).eq(0) &
                df["thalach"].fillna(0).eq(0)
            )
            df = df[~empty_mask].copy()

        if df.empty:
            return pd.DataFrame()

        # Convert risk_score to 0-100 display scale
        # Some records may be stored as 0-1 (adj_prob), others as 0-100 (pct)
        # Detect by value magnitude: if max > 1.5, assume already 0-100
        rs_col = df["risk_score"].dropna()
        if len(rs_col) > 0 and rs_col.max() <= 1.5:
            df["risk_score"] = df["risk_score"].fillna(0.0) * 100  # convert 0-1 → 0-100
        else:
            df["risk_score"] = df["risk_score"].fillna(0.0)         # already 0-100
        return df.sort_values("created_at")


    except Exception as exc:
        logger.error("Failed to fetch records: %s", exc)
        return pd.DataFrame()


# ── Trend computation ──────────────────────────────────────────────────────────
def _pct_change(series: pd.Series) -> Optional[float]:
    """Percent change between first and last value in a series."""
    s = series.dropna()
    if len(s) < 2:
        return None
    first, last = s.iloc[0], s.iloc[-1]
    if first == 0:
        return None
    return round(((last - first) / abs(first)) * 100, 2)


def _field_progress(metric: str, pct_change: Optional[float]) -> Dict[str, str]:
    """
    Determine per-field progress status: improved / worsened / stable / insufficient.

    Returns dict with keys: status, icon, color, description
    """
    if pct_change is None:
        return {
            "status":      "insufficient",
            "icon":        "📊",
            "color":       "#6c757d",
            "description": "Insufficient data to determine trend",
        }

    higher_is_worse = FIELD_HIGHER_IS_WORSE.get(metric, True)

    # Threshold: < 3% change = stable
    if abs(pct_change) < 3:
        return {
            "status":      "stable",
            "icon":        "➡",
            "color":       "#F59E0B",
            "description": f"Stable ({pct_change:+.1f}%)",
        }

    is_worsening = (higher_is_worse and pct_change > 0) or (not higher_is_worse and pct_change < 0)

    if is_worsening:
        return {
            "status":      "worsened",
            "icon":        "▲" if pct_change > 0 else "▼",
            "color":       "#EF4444",
            "description": f"Worsened ({pct_change:+.1f}%)",
        }
    else:
        return {
            "status":      "improved",
            "icon":        "▼" if pct_change > 0 else "▲",
            "color":       "#10B981",
            "description": f"Improved ({pct_change:+.1f}%)",
        }


def compute_trends(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute per-metric percent changes, rolling averages, and progress status.

    Returns
    -------
    dict  {metric: {
        "pct_change"  : float,
        "direction"   : str,
        "values"      : list,
        "latest"      : float,
        "rolling_avg" : list,
        "progress"    : dict  (status/icon/color/description),
    }}
    """
    trends = {}
    if df.empty:
        return trends

    for metric in NUMERIC_METRICS + ["risk_score"]:
        if metric not in df.columns:
            continue
        s = df[metric].dropna()
        n = int(len(s))
        if n == 0:
            continue

        pct = _pct_change(s)
        if pct is None:
            direction = "→"
        elif pct > 2:
            direction = "↑"
        elif pct < -2:
            direction = "↓"
        else:
            direction = "→"

        window = max(1, min(3, n))
        try:
            rolling_avg = s.rolling(window=window, min_periods=1).mean().tolist()
        except Exception:
            rolling_avg = [float(s.mean())] * n

        progress = _field_progress(metric, pct)

        trends[metric] = {
            "pct_change":  pct,
            "direction":   direction,
            "values":      s.tolist(),
            "latest":      float(s.iloc[-1]),
            "rolling_avg": rolling_avg,
            "progress":    progress,
        }
    return trends


# ── Alert detection ────────────────────────────────────────────────────────────
def detect_alerts(trends: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Return a list of {field, message, severity} alert dicts based on trend rules.
    """
    alerts = []
    for field, direction, threshold, message in TREND_RULES:
        info = trends.get(field)
        if not info or info["pct_change"] is None:
            continue
        pct = info["pct_change"]

        triggered = False
        if direction == "up"   and pct >  threshold:
            triggered = True
        elif direction == "down" and pct < -threshold:
            triggered = True

        if triggered:
            severity = "HIGH" if abs(pct) > threshold * 2 else "MEDIUM"
            alerts.append({
                "field":    field,
                "message":  message,
                "pct":      f"{pct:+.1f}%",
                "severity": severity,
            })
    return alerts


# ── Per-field progress summary ─────────────────────────────────────────────────
def build_field_progress_summary(trends: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build a list of per-field progress cards for the Monitoring UI.

    Returns
    -------
    list of dicts: {metric, label, latest, unit, progress, pct_change}
    """
    _UNITS = {
        "trestbps":   "mmHg",
        "chol":       "mg/dL",
        "thalach":    "bpm",
        "oldpeak":    "mm",
        "ca":         "vessels",
        "fbs":        "",
        "exang":      "",
        "risk_score": "%",
    }
    display_fields = ["risk_score", "trestbps", "chol", "thalach", "oldpeak", "ca", "exang", "fbs"]
    result = []
    for metric in display_fields:
        info = trends.get(metric)
        if not info:
            continue
        result.append({
            "metric":     metric,
            "label":      METRIC_LABELS.get(metric, metric),
            "latest":     info["latest"],
            "unit":       _UNITS.get(metric, ""),
            "progress":   info["progress"],
            "pct_change": info["pct_change"],
        })
    return result


# ── Progress summary (for Agent 4) ────────────────────────────────────────────
def build_progress_summary(
    df: pd.DataFrame,
    trends: Dict[str, Any],
    alerts: List[Dict[str, str]],
) -> str:
    """
    Build a concise text summary of the patient's health trajectory for
    consumption by reco_agent (Agent 4).
    """
    if df.empty:
        return "No historical data available."

    latest = df.iloc[-1]
    n_records = len(df)

    lines = [
        f"Patient has {n_records} recorded assessments.",
        f"Latest risk score: {latest.get('risk_score', 0):.1f}%  "
        f"[{latest.get('risk_label', 'N/A')}]",
        f"Latest metrics: BP={latest.get('trestbps', 'N/A')} mmHg, "
        f"Chol={latest.get('chol', 'N/A')} mg/dL, "
        f"MaxHR={latest.get('thalach', 'N/A')} bpm.",
    ]

    # Per-field progress for context
    improved = []
    worsened = []
    for metric, info in trends.items():
        prog = info.get("progress", {})
        label = METRIC_LABELS.get(metric, metric)
        if prog.get("status") == "improved":
            improved.append(label)
        elif prog.get("status") == "worsened":
            worsened.append(f"{label} ({info['pct_change']:+.1f}%)")

    if improved:
        lines.append(f"Improving fields: {', '.join(improved)}.")
    if worsened:
        lines.append(f"Worsening fields: {', '.join(worsened)}.")

    if alerts:
        lines.append("\nConcerning trends:")
        for a in alerts:
            lines.append(f"  ⚠ {a['message']} ({a['pct']})")

    return "\n".join(lines)


# ── AI alerts enhancement ──────────────────────────────────────────────────────
def _ai_enhance_alerts(progress_summary: str) -> str:
    system = (
        "You are a preventive cardiology AI assistant. "
        "Given a patient's health trend summary, write 2-3 concise, empathetic sentences "
        "highlighting the most important health concerns and motivating the patient to act. "
        "Never diagnose or prescribe. Keep under 120 words."
    )
    return call_groq(
        system_prompt=system,
        user_prompt=progress_summary,
        fallback=(
            "Your health trends have been analysed. "
            "Please review the alerts above and consult your doctor."
        ),
        max_tokens=200,
    )


# ── Build chart data ───────────────────────────────────────────────────────────
def build_comparison_chart_data(latest_row: pd.Series) -> List[Dict[str, Any]]:
    """
    For each metric in SAFE_RANGES, return current value and safe-range bounds.
    Used by the Plotly comparison chart in the dashboard.
    """
    chart_data = []
    for metric, (lo, hi, unit) in SAFE_RANGES.items():
        val = latest_row.get(metric)
        if val is None:
            continue
        chart_data.append({
            "metric":  METRIC_LABELS.get(metric, metric),
            "value":   float(val),
            "safe_lo": lo,
            "safe_hi": hi,
            "unit":    unit,
        })
    return chart_data


# ── Agent entry-point ─────────────────────────────────────────────────────────
def run_monitor_agent(
    supabase,
    user_id: str,
    use_groq: bool = True,
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Agent 3 full pipeline (Phase 2).

    Returns
    -------
    dict with keys:
        history             : pd.DataFrame
        trends              : dict
        alerts              : list[dict]
        progress_summary    : str
        ai_summary          : str
        chart_data          : list[dict]
        field_progress      : list[dict]  — NEW: per-field improved/worsened cards
        has_history         : bool
    """
    try:
        df = fetch_history(supabase, user_id, limit=limit)
    except Exception as exc:
        logger.error("fetch_history failed: %s", exc)
        df = pd.DataFrame()

    if df.empty:
        return {
            "history":          df,
            "trends":           {},
            "alerts":           [],
            "progress_summary": "No historical records found.",
            "ai_summary":       "Start your first assessment to track your heart health over time.",
            "chart_data":       [],
            "field_progress":   [],
            "has_history":      False,
        }

    try:
        trends   = compute_trends(df)
        alerts   = detect_alerts(trends)
        summary  = build_progress_summary(df, trends, alerts)
        field_progress = build_field_progress_summary(trends)
    except Exception as exc:
        logger.error("Trend computation failed: %s", exc)
        trends, alerts, summary, field_progress = {}, [], "Unable to compute trends.", []

    ai_sum = ""
    if use_groq:
        try:
            ai_sum = _ai_enhance_alerts(summary)
        except Exception as exc:
            logger.error("AI summary failed: %s", exc)

    try:
        chart_data = build_comparison_chart_data(df.iloc[-1])
    except Exception as exc:
        logger.error("Chart data build failed: %s", exc)
        chart_data = []

    return {
        "history":          df,
        "trends":           trends,
        "alerts":           alerts,
        "progress_summary": summary,
        "ai_summary":       ai_sum,
        "chart_data":       chart_data,
        "field_progress":   field_progress,
        "has_history":      True,
    }
