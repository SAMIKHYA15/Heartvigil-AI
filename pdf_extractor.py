"""
pdf_extractor.py
Extracts 13 UCI Heart Disease features from uploaded PDF medical reports.
Uses pdfplumber for text extraction and regex for field parsing.
"""

import re
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)
from ai_helper import call_groq

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


def _extract_with_llm(text: str) -> Dict[str, Optional[float]]:
    """Use Groq LLM to extract 13 UCI features from text in JSON format."""
    system_prompt = (
        "You are a medical data extraction expert. Extract the 13 UCI Heart Disease features from the provided medical report text. "
        "Return ONLY a JSON object with the following keys and map clinical findings to the specific numeric codes provided:\n\n"
        "1. 'age': numeric\n"
        "2. 'sex': 1 for male, 0 for female\n"
        "3. 'cp' (chest pain type): 0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic\n"
        "4. 'trestbps' (resting blood pressure): numeric systolic value (e.g., 120 from 120/80)\n"
        "5. 'chol' (serum cholesterol): numeric mg/dL\n"
        "6. 'fbs' (fasting blood sugar): 1 if > 120 mg/dL, else 0\n"
        "7. 'restecg' (resting ECG): 0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy\n"
        "8. 'thalach' (max heart rate): numeric bpm\n"
        "9. 'exang' (exercise induced angina): 1 for yes/present, 0 for no/absent\n"
        "10. 'oldpeak' (ST depression): numeric value (e.g., 1.5)\n"
        "11. 'slope' (ST slope): 0: Upsloping, 1: Flat, 2: Downsloping\n"
        "12. 'ca' (major vessels): 0-4 numeric\n"
        "13. 'thal' (perfusion/thalassemia): 1: Normal, 2: Fixed defect, 3: Reversible defect\n\n"
        "If a value is not found, use null. Provide ONLY valid JSON. No conversational filler."
    )
    
    import json
    response = call_groq(
        system_prompt=system_prompt,
        user_prompt=f"Medical Report Text:\n{text}",
        fallback="{}"
    )
    
    try:
        # Strip potential markdown formatting
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        data = json.loads(response)
        # Ensure all values are floats or None
        return {k: float(v) if v is not None else None for k, v in data.items() if k in PATTERNS}
    except Exception as exc:
        logger.warning("LLM extraction or JSON parsing failed: %s", exc)
        return {}


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

    # Attempt LLM extraction first
    result = _extract_with_llm(text)
    n_llm = sum(1 for v in result.values() if v is not None)
    
    # If LLM missed fields, attempt regex fallback for those specific fields
    if n_llm < len(PATTERNS):
        logger.info("LLM missed %d fields, trying regex fallback...", len(PATTERNS) - n_llm)
        for field, patterns in PATTERNS.items():
            if result.get(field) is not None:
                continue
            
            raw = _search_patterns(text, patterns)
            if raw is None:
                result[field] = None
                continue

            raw = raw.strip().lower()

            # Field-specific normalisation (rest of the logic remains same)
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