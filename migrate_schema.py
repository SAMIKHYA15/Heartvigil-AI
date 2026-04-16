"""
migrate_schema.py -- Adds missing columns to the health_records table
Run once: python migrate_schema.py
"""
import os, pathlib, requests, json
from dotenv import load_dotenv

load_dotenv(dotenv_path=str(pathlib.Path(".env").absolute()), override=True)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SERVICE_KEY  = os.environ["SUPABASE_SERVICE_KEY"]

# Use the Supabase REST + special SQL path via service key header
HEADERS = {
    "apikey":        SERVICE_KEY,
    "Authorization": f"Bearer {SERVICE_KEY}",
    "Content-Type":  "application/json",
}

SQL = """
-- ============================================================
-- HeartVigil AI: health_records schema migration
-- Run this once in Supabase SQL Editor or via this script
-- ============================================================

-- 1. Make sure the table exists with correct base schema
CREATE TABLE IF NOT EXISTS public.health_records (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Add 13 UCI feature columns (safe - IF NOT EXISTS)
ALTER TABLE public.health_records
  ADD COLUMN IF NOT EXISTS age       INTEGER,
  ADD COLUMN IF NOT EXISTS sex       INTEGER,
  ADD COLUMN IF NOT EXISTS cp        INTEGER,
  ADD COLUMN IF NOT EXISTS trestbps  INTEGER,
  ADD COLUMN IF NOT EXISTS chol      INTEGER,
  ADD COLUMN IF NOT EXISTS fbs       INTEGER,
  ADD COLUMN IF NOT EXISTS restecg   INTEGER,
  ADD COLUMN IF NOT EXISTS thalach   INTEGER,
  ADD COLUMN IF NOT EXISTS exang     INTEGER,
  ADD COLUMN IF NOT EXISTS oldpeak   FLOAT,
  ADD COLUMN IF NOT EXISTS slope     INTEGER,
  ADD COLUMN IF NOT EXISTS ca        INTEGER,
  ADD COLUMN IF NOT EXISTS thal      INTEGER;

-- 3. Add risk + metadata columns
ALTER TABLE public.health_records
  ADD COLUMN IF NOT EXISTS risk_score FLOAT,
  ADD COLUMN IF NOT EXISTS risk_label TEXT,
  ADD COLUMN IF NOT EXISTS source     TEXT DEFAULT 'manual';

-- 4. Enable RLS
ALTER TABLE public.health_records ENABLE ROW LEVEL SECURITY;

-- 5. RLS Policies (drop & recreate so they're always correct)
DROP POLICY IF EXISTS "Users read own records"   ON public.health_records;
DROP POLICY IF EXISTS "Users insert own records" ON public.health_records;
DROP POLICY IF EXISTS "Service role full access" ON public.health_records;

CREATE POLICY "Users read own records"
  ON public.health_records FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users insert own records"
  ON public.health_records FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Allow service role to do everything (for server-side saves)
CREATE POLICY "Service role full access"
  ON public.health_records FOR ALL
  USING (TRUE)
  WITH CHECK (TRUE);

-- 6. Refresh Supabase schema cache
NOTIFY pgrst, 'reload schema';
"""

# Try via the Management API query endpoint
project_ref = SUPABASE_URL.split("//")[1].split(".")[0]
mgmt_url    = f"https://api.supabase.com/v1/projects/{project_ref}/database/query"

print("Attempting migration via Management API...")
resp = requests.post(mgmt_url, headers={
    "Authorization": f"Bearer {SERVICE_KEY}",
    "Content-Type":  "application/json",
}, json={"query": SQL}, timeout=30)

if resp.status_code == 200:
    print("SUCCESS via Management API!")
    print(resp.json())
else:
    print(f"Management API failed ({resp.status_code}): {resp.text[:300]}")
    print()
    print("=" * 60)
    print("MANUAL STEP REQUIRED")
    print("=" * 60)
    print("1. Go to: https://supabase.com/dashboard/project/" + project_ref + "/sql/new")
    print("2. Paste and run the following SQL:")
    print()
    print(SQL)
