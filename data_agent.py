"""
data_agent.py  –  Agent 1
Validates user-submitted health data and saves it to Supabase.
Also computes deltas compared with the user's most recent record.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Medical validation ranges ──────────────────────────────────────────────────
FIELD_RANGES = {
    "age":      (1,   120,  "Age must be between 1 and 120 years."),
    "sex":      (0,   1,    "Sex must be 0 (Female) or 1 (Male)."),
    "cp":       (0,   3,    "Chest pain type must be 0–3."),
    "trestbps": (60,  250,  "Resting BP must be 60–250 mmHg."),
    "chol":     (1,   700,  "Cholesterol must be 1–700 mg/dL."),
    "fbs":      (0,   1,    "Fasting BS must be 0 (≤120 mg/dL) or 1 (>120 mg/dL)."),
    "restecg":  (0,   2,    "Resting ECG must be 0, 1, or 2."),
    "thalach":  (1,   300,  "Max heart rate must be 1–300 bpm."),
    "exang":    (0,   1,    "Exercise angina must be 0 (No) or 1 (Yes)."),
    "oldpeak":  (0.0, 10.0, "ST depression must be 0.0–10.0."),
    "slope":    (0,   2,    "ST slope must be 0, 1, or 2."),
    "ca":       (0,   4,    "Number of major vessels must be 0–4."),
    "thal":     (0,   3,    "Thal must be 0 (Normal), 1 (Fixed), 2 (Reversible), or 3."),
}

FEATURE_COLS = list(FIELD_RANGES.keys())

# Clinically "safe" reference ranges for delta comparison chart
SAFE_RANGES = {
    "trestbps": (90, 120,  "mmHg"),
    "chol":     (0,  200,  "mg/dL"),
    "thalach":  (60, 100,  "bpm"),  # safe reference range (chart display only)
    "oldpeak":  (0,  1.0,  "mm"),
    "age":      (0,  50,   "years"),
}


# ── Validation ────────────────────────────────────────────────────────────────
def validate_fields(data: Dict[str, Any]) -> List[str]:
    """
    Validate all 13 health fields against medical ranges.

    Parameters
    ----------
    data : dict
        Keys are feature names, values are numeric values.

    Returns
    -------
    list[str]
        List of human-readable error messages (empty = all valid).
    """
    errors: List[str] = []

    for field, (lo, hi, msg) in FIELD_RANGES.items():
        val = data.get(field)
        if val is None:
            errors.append(f"'{field}' is required.")
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            errors.append(f"'{field}' must be a number.")
            continue
        if not (lo <= v <= hi):
            errors.append(msg)

    return errors


# ── Delta calculation ─────────────────────────────────────────────────────────
def compute_delta(
    current: Dict[str, Any],
    previous: Optional[Dict[str, Any]],
) -> Dict[str, Optional[float]]:
    """
    Compute the percent change between the current and previous records
    for each numeric feature.

    Returns
    -------
    dict
        {field: percent_change_or_None}
    """
    if previous is None:
        return {f: None for f in FEATURE_COLS}

    deltas: Dict[str, Optional[float]] = {}
    for field in FEATURE_COLS:
        cur = current.get(field)
        prv = previous.get(field)
        try:
            c, p = float(cur), float(prv)
            if p == 0:
                deltas[field] = None
            else:
                deltas[field] = round(((c - p) / abs(p)) * 100, 2)
        except (TypeError, ValueError):
            deltas[field] = None

    return deltas


# ── Supabase helpers ──────────────────────────────────────────────────────────
def _fetch_latest_record(supabase, user_id: str) -> Optional[Dict]:
    """Return the most recent health_record for *user_id*, or None."""
    try:
        res = (
            supabase.table("health_records")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return res.data[0] if res.data else None
    except Exception as exc:
        logger.error("Error fetching latest record: %s", exc)
        return None


def save_health_data(
    supabase,
    user_id: str,
    health_data: Dict[str, Any],
    risk_score: float,
    risk_label: str,
    source: str = "manual",
) -> Tuple[bool, str]:
    """
    Validate and persist health data to Supabase.

    Parameters
    ----------
    supabase       : Supabase client instance
    user_id        : UUID of the authenticated user
    health_data    : Dict with 13 feature keys
    risk_score     : Model output probability (0–1)
    risk_label     : "LOW" | "MEDIUM" | "HIGH"
    source         : "manual" or "pdf"

    Returns
    -------
    (success: bool, message: str)
    """
    # Validate
    errors = validate_fields(health_data)
    if errors:
        return False, " | ".join(errors)

    # Integer columns in the DB schema
    _INT_COLS = {"age", "sex", "cp", "trestbps", "chol", "fbs",
                 "restecg", "thalach", "exang", "slope", "ca", "thal"}

    record = {
        "user_id":    user_id,
        "risk_score": round(float(risk_score), 6),
        "risk_label": risk_label,
        "source":     source,
    }
    for col in FEATURE_COLS:
        val = health_data[col]
        if col in _INT_COLS:
            record[col] = int(round(float(val)))   # must be integer for Supabase
        else:
            record[col] = round(float(val), 4)     # oldpeak → float


    try:
        resp = supabase.table("health_records").insert(record).execute()
        # supabase-py v2 raises on error; v1 returns empty data on failure
        if hasattr(resp, "data") and resp.data is None:
            return False, "Supabase returned no data — possible RLS or schema issue."
        return True, "Health data saved successfully."
    except Exception as exc:
        detail = str(exc)
        logger.error("Supabase insert error: %s", detail)
        return False, f"Database error: {detail}"



# ── Agent entry-point ─────────────────────────────────────────────────────────
def run_data_agent(
    supabase,
    user_id: str,
    health_data: Dict[str, Any],
    risk_score: float = 0.0,
    risk_label: str = "UNKNOWN",
    source: str = "manual",
) -> Dict[str, Any]:
    """
    Full pipeline for Agent 1:
      1. Validate inputs
      2. Fetch previous record
      3. Compute delta
      4. Save to Supabase

    Returns
    -------
    dict with keys:
        success     : bool
        message     : str
        errors      : list[str]
        delta       : dict
        previous    : dict | None
    """
    errors = validate_fields(health_data)
    previous = _fetch_latest_record(supabase, user_id)
    delta    = compute_delta(health_data, previous)

    if errors:
        return {
            "success":  False,
            "message":  "Validation failed.",
            "errors":   errors,
            "delta":    delta,
            "previous": previous,
        }

    ok, msg = save_health_data(
        supabase, user_id, health_data, risk_score, risk_label, source
    )
    return {
        "success":  ok,
        "message":  msg,
        "errors":   [],
        "delta":    delta,
        "previous": previous,
    }
