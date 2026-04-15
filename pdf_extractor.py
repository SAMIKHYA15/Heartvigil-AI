"""
pdf_extractor.py
Extracts 13 UCI Heart Disease features from uploaded PDF medical reports.
Uses pdfplumber for text extraction and regex for field parsing.
"""

import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Field extraction patterns ──────────────────────────────────────────────────
PATTERNS: Dict[str, list] = {
    "age": [
        r"(?:age|patient age)[:\s]*(\d{1,3})\s*(?:years?|yrs?)?",
        r"\bage[:\s]+(\d{2,3})\b",
    ],
    "sex": [
        r"(?:sex|gender)[:\s]*(male|female|m|f)\b",
    ],
    "cp": [
        r"chest\s*pain\s*type[:\s]*(\d)",
        r"cp[:\s]*(\d)",
        r"(?:typical angina|atypical angina|non[- ]?anginal|asymptomatic)",
    ],
    "trestbps": [
        r"(?:resting\s*)?(?:blood\s*pressure|BP)[:\s]*(\d{2,3})\s*/\s*\d+",
        r"(?:resting\s*)?(?:blood\s*pressure|BP)[:\s]*(\d{2,3})\b",
        r"systolic[:\s]*(\d{2,3})",
    ],
    "chol": [
        r"(?:serum\s*)?cholesterol[:\s]*(\d{3,4})\s*(?:mg/dl)?",
        r"total\s*cholesterol[:\s]*(\d{3,4})",
        r"chol[:\s]*(\d{3,4})",
    ],
    "fbs": [
        r"fasting\s*(?:blood\s*)?(?:sugar|glucose)[:\s]*(\d+(?:\.\d+)?)\s*(?:mg/dl)?",
    ],
    "restecg": [
        r"(?:resting\s*)?ecg(?:\s*result)?[:\s]*(\d)",
        r"restecg[:\s]*(\d)",
    ],
    "thalach": [
        r"(?:max(?:imum)?\s*)?(?:heart\s*rate|HR)[:\s]*(\d{2,3})\s*(?:bpm)?",
        r"thalach[:\s]*(\d{2,3})",
    ],
    "exang": [
        r"exercise\s*(?:induced\s*)?angina[:\s]*(yes|no|1|0)",
        r"exang[:\s]*(yes|no|1|0)",
    ],
    "oldpeak": [
        r"(?:ST\s*)?(?:depression|oldpeak)[:\s]*(\d+(?:\.\d+)?)",
        r"oldpeak[:\s]*(\d+(?:\.\d+)?)",
    ],
    "slope": [
        r"(?:peak\s*exercise\s*)?ST\s*slope[:\s]*(\d)",
        r"slope[:\s]*(\d)",
    ],
    "ca": [
        r"(?:number\s*of\s*)?(?:major\s*)?vessels?[:\s]*(\d)\s*(?:colored|fluoroscopy)?",
        r"\bca[:\s]*(\d)\b",
    ],
    "thal": [
        r"thal(?:assemia)?[:\s]*(\d|normal|fixed|reversible)",
        r"thal[:\s]*(\d)",
    ],
}

# ── Normalisation helpers ──────────────────────────────────────────────────────
_SEX_MAP   = {"male": 1, "m": 1, "female": 0, "f": 0}
_YESNO_MAP = {"yes": 1, "1": 1, "no": 0, "0": 0}
_CP_KEYWORDS = {
    "typical angina": 0,
    "atypical angina": 1,
    "non-anginal": 2,
    "non anginal": 2,
    "asymptomatic": 3,
}
_THAL_MAP = {"normal": 1, "fixed": 2, "reversible": 3}


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Return raw text extracted from a PDF byte stream."""
    try:
        import pdfplumber, io
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        return ""
    except Exception as exc:
        logger.error("PDF extraction error: %s", exc)
        return ""


def _search_patterns(text: str, patterns: list) -> Optional[str]:
    """Return the first regex match from a list of patterns (case-insensitive)."""
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1) if m.lastindex else m.group(0)
    return None


def extract_features_from_pdf(file_bytes: bytes) -> Dict[str, Optional[float]]:
    """
    Extract all 13 UCI Heart Disease features from a PDF.

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the uploaded PDF file.

    Returns
    -------
    dict
        Keys are feature names; values are extracted floats (or None if not found).
    """
    text = _extract_text_from_pdf(file_bytes)
    if not text:
        return {}

    result: Dict[str, Optional[float]] = {}

    for field, patterns in PATTERNS.items():
        raw = _search_patterns(text, patterns)
        if raw is None:
            result[field] = None
            continue

        raw = raw.strip().lower()

        # Field-specific normalisation
        if field == "sex":
            result[field] = float(_SEX_MAP.get(raw, 1))

        elif field == "cp":
            # Try keyword first
            matched = next(
                (v for k, v in _CP_KEYWORDS.items() if k in raw), None
            )
            if matched is not None:
                result[field] = float(matched)
            else:
                try:
                    result[field] = float(raw)
                except ValueError:
                    result[field] = None

        elif field == "fbs":
            try:
                glucose = float(raw)
                result[field] = 1.0 if glucose > 120 else 0.0
            except ValueError:
                result[field] = None

        elif field in ("exang",):
            result[field] = float(_YESNO_MAP.get(raw, 0))

        elif field == "thal":
            if raw in _THAL_MAP:
                result[field] = float(_THAL_MAP[raw])
            else:
                try:
                    result[field] = float(raw)
                except ValueError:
                    result[field] = None

        else:
            try:
                result[field] = float(raw)
            except ValueError:
                result[field] = None

    n_found = sum(1 for v in result.values() if v is not None)
    logger.info("PDF extractor found %d / %d fields.", n_found, len(PATTERNS))
    return result
