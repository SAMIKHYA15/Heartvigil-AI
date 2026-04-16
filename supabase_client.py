"""
supabase_client.py
Initializes Supabase clients:
  - get_supabase()       : anon/publishable key  (for normal reads)
  - get_admin_supabase() : service role key       (bypasses RLS for user upserts)
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required on Streamlit Cloud

from supabase import create_client, Client


def _get_credentials(service_role: bool = False):
    """Read Supabase credentials from env or Streamlit secrets."""
    url  = os.environ.get("SUPABASE_URL", "")
    key  = os.environ.get(
        "SUPABASE_SERVICE_KEY" if service_role else "SUPABASE_KEY", ""
    )

    # Fall back to Streamlit secrets when running on Streamlit Cloud
    if not url or not key:
        try:
            import streamlit as st
            url = st.secrets.get("SUPABASE_URL", url)
            key = st.secrets.get(
                "SUPABASE_SERVICE_KEY" if service_role else "SUPABASE_KEY", key
            )
        except Exception:
            pass

    # If service key not configured, fall back to anon key
    if service_role and not key:
        key = os.environ.get("SUPABASE_KEY", "")
        if not key:
            try:
                import streamlit as st
                key = st.secrets.get("SUPABASE_KEY", "")
            except Exception:
                pass

    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_KEY must be set in environment "
            "variables or Streamlit secrets."
        )
    return url, key


def get_supabase() -> Client:
    """Returns an anon Supabase client (respects RLS)."""
    url, key = _get_credentials(service_role=False)
    return create_client(url, key)


def get_admin_supabase() -> Client:
    """
    Returns a service-role Supabase client that BYPASSES RLS.
    Use ONLY for trusted server-side operations (user creation, admin writes).
    Reads SUPABASE_SERVICE_KEY from .env / st.secrets.
    Falls back to SUPABASE_KEY if service key is not configured.
    """
    url, key = _get_credentials(service_role=True)
    return create_client(url, key)