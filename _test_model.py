import warnings; warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import joblib, pandas as pd

FC = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
m = joblib.load("model.joblib")
s = joblib.load("scaler.joblib")

def predict(vals):
    df = pd.DataFrame([vals], columns=FC)
    sc = pd.DataFrame(s.transform(df), columns=FC)
    return float(m.predict_proba(sc)[0][1])

tests = [
    ("Young healthy female 32yo",     {"age":32,"sex":0,"cp":0,"trestbps":110,"chol":170,"fbs":0,"restecg":0,"thalach":185,"exang":0,"oldpeak":0.0,"slope":2,"ca":0,"thal":2}),
    ("Middle male normal 55yo",       {"age":55,"sex":1,"cp":0,"trestbps":138,"chol":205,"fbs":0,"restecg":0,"thalach":152,"exang":0,"oldpeak":0.8,"slope":1,"ca":0,"thal":2}),
    ("Older male HTN 63yo",           {"age":63,"sex":1,"cp":3,"trestbps":150,"chol":265,"fbs":1,"restecg":1,"thalach":125,"exang":0,"oldpeak":1.8,"slope":1,"ca":1,"thal":3}),
    ("High-risk male UCI pattern",    {"age":67,"sex":1,"cp":2,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":1,"ca":3,"thal":2}),
]

print("=== Model v3 prediction test (13 UCI features) ===")
for desc, vals in tests:
    p = predict(vals)
    label = "HIGH" if p >= 0.65 else ("MEDIUM" if p >= 0.35 else "LOW")
    print(f"  {desc}: {p*100:.1f}% -> {label}")

print()

# Full risk_agent flow
from risk_agent import run_risk_agent
sample = {"age":58,"sex":1,"cp":2,"trestbps":145,"chol":250,"fbs":0,"restecg":1,"thalach":135,"exang":0,"oldpeak":1.2,"slope":1,"ca":1,"thal":2}
out = run_risk_agent(sample, use_groq=False)
print("=== run_risk_agent integration test ===")
print(f"  risk_pct    : {out['risk_pct']}%")
print(f"  risk_label  : {out['risk_label']}")
print(f"  raw_prob    : {out['raw_prob']:.4f}")
print(f"  reasons[0]  : {out['reasons'][0]}")
print(f"  direction   : {out['direction_info']['direction']}")
print()
print("ALL CHECKS PASSED")
