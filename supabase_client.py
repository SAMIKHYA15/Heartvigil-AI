"""
supabase_client.py
Initializes and returns a Supabase client.
Supports both local .env files and Streamlit Cloud secrets.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required on Streamlit Cloud

from supabase import create_client, Client


def get_supabase() -> Client:
    """
    Returns an initialised Supabase client.
    Reads credentials from environment variables (local) or
    st.secrets (Streamlit Cloud).
    """
    # Try environment variables first
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")

    # Fall back to Streamlit secrets if running on Streamlit Cloud
    if not url or not key:
        try:
            import streamlit as st
            url = st.secrets.get("SUPABASE_URL", url)
            key = st.secrets.get("SUPABASE_KEY", key)
        except Exception:
            pass

    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_KEY must be set in environment "
            "variables or Streamlit secrets."
        )

    return create_client(url, key)
