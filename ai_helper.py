"""
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
        "Summarise the patient\'s health trends in 3-4 clear, empathetic sentences. "
        "Do NOT diagnose or prescribe. Use simple, non-technical language. "
        "If trends are improving, be encouraging. If worsening, gently urge physician consultation."
    )
    return call_groq(
        system_prompt=system,
        user_prompt=f"Health trend data:\n{trends_text}",
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
