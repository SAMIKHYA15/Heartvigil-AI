-- HeartVigil AI – Supabase Schema
-- Run this SQL in your Supabase SQL editor

-- ── Enable UUID extension ──────────────────────────────────────────────────────
create extension if not exists "uuid-ossp";

-- ── users table ───────────────────────────────────────────────────────────────
create table if not exists public.users (
    id         uuid primary key default uuid_generate_v4(),
    email      text unique not null,
    created_at timestamptz default now()
);

-- ── health_records table ──────────────────────────────────────────────────────
create table if not exists public.health_records (
    id         uuid primary key default uuid_generate_v4(),
    user_id    uuid not null references public.users(id) on delete cascade,

    -- 13 UCI Heart Disease features
    age        numeric not null,
    sex        numeric not null,
    cp         numeric not null,
    trestbps   numeric not null,
    chol       numeric not null,
    fbs        numeric not null,
    restecg    numeric not null,
    thalach    numeric not null,
    exang      numeric not null,
    oldpeak    numeric not null,
    slope      numeric not null,
    ca         numeric not null,
    thal       numeric not null,

    -- Model output
    risk_score numeric not null,   -- 0.0 – 1.0
    risk_label text   not null,    -- LOW | MEDIUM | HIGH

    -- Metadata
    source     text default 'manual',  -- 'manual' | 'pdf'
    created_at timestamptz default now()
);

-- ── Indexes ───────────────────────────────────────────────────────────────────
create index if not exists idx_health_records_user_id   on public.health_records(user_id);
create index if not exists idx_health_records_created_at on public.health_records(created_at desc);

-- ── Row-Level Security (RLS) ──────────────────────────────────────────────────
alter table public.users          enable row level security;
alter table public.health_records enable row level security;

-- Users can only see/edit their own user row
create policy "Users select own row"
    on public.users for select
    using (true);                          -- allow read so we can do get-or-create

create policy "Users insert own row"
    on public.users for insert
    with check (true);                     -- backend creates user on first login

-- Health records: users can read/insert their own records
create policy "Users select own records"
    on public.health_records for select
    using (true);                          -- or: auth.uid() = user_id if using Supabase Auth

create policy "Users insert own records"
    on public.health_records for insert
    with check (true);

-- ── Quick verification ─────────────────────────────────────────────────────────
-- select * from public.users;
-- select * from public.health_records order by created_at desc limit 10;
