# HeartVigil AI

> AI-powered Heart Disease Risk Prediction  
> Multi-Agent Architecture • OTP Email Auth • Streamlit

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

### 3. Set up Supabase

1. Create a free project at [supabase.com](https://supabase.com)
2. Open the SQL Editor and run **schema.sql**
3. Copy your project URL and anon key into `.env`

### 4. Download the dataset

Download **heart.csv** (UCI Heart Disease) from:  
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

Place `heart.csv` in the project root.

### 5. Train the ML model

```bash
python train.py
```

This creates `model.joblib`, `scaler.joblib`, `feature_names.joblib`.

### 6. Run the app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
heartvigil-ai/
├── app.py               # Main Streamlit application
├── train.py             # ML model training (RF + XGBoost ensemble)
├── data_agent.py        # Agent 1: Validation & data persistence
├── risk_agent.py        # Agent 2: Risk prediction & explanation
├── monitor_agent.py     # Agent 3: Trend monitoring & alerts
├── reco_agent.py        # Agent 4: Personalised recommendations
├── supabase_client.py   # Supabase client initialization
├── pdf_extractor.py     # PDF medical report parsing
├── ai_helper.py         # Groq API helper utilities
├── schema.sql           # Supabase database schema
├── requirements.txt     # Python dependencies
└── .env.example         # Environment variable template
```

---

## ⚙️ Environment Variables

| Variable          | Description                        |
|-------------------|------------------------------------|
| `SUPABASE_URL`    | Supabase project URL               |
| `SUPABASE_KEY`    | Supabase anon/service key          |
| `SMTP_SERVER`     | SMTP server (e.g. smtp.gmail.com)  |
| `SMTP_PORT`       | SMTP port (587 for TLS)            |
| `SENDER_EMAIL`    | Sender email address               |
| `SENDER_PASSWORD` | Email app password                 |
| `GROQ_API_KEY`    | Groq API key (llama-3.3-70b)       |

---

## ☁️ Streamlit Cloud Deployment

1. Push this repo to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Add all environment variables under **Secrets** (TOML format):

```toml
SUPABASE_URL = "https://..."
SUPABASE_KEY = "..."
SMTP_SERVER  = "smtp.gmail.com"
SMTP_PORT    = "587"
SENDER_EMAIL = "..."
SENDER_PASSWORD = "..."
GROQ_API_KEY = "..."
```

4. Upload `model.joblib`, `scaler.joblib`, `feature_names.joblib` to the repo  
   (or retrain on first deploy by calling `python train.py` in the startup command)

---

## 🤖 Multi-Agent Architecture

```
User Input
    │
    ▼
Agent 1 (data_agent)   ─── Validate + Save to Supabase
    │
    ▼
Agent 2 (risk_agent)   ─── ML Prediction + Clinical Rules + Groq Explanation
    │
    ▼
Agent 3 (monitor_agent) ── Trend Analysis + Alerts + Progress Summary
    │
    ▼
Agent 4 (reco_agent)   ─── Personalised Recommendations (Groq / Rule-based)
```

---

## ⚠️ Disclaimer

This application is for **educational purposes only**.  
It is not a medical device and should not be used for clinical diagnosis.  
Always consult a qualified healthcare professional.
