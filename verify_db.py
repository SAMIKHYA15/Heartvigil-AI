"""
verify_db.py -- Verify write to health_records works end-to-end
Run after running the SQL migration in Supabase
"""
import os, pathlib
from dotenv import load_dotenv
load_dotenv(dotenv_path=str(pathlib.Path(".env").absolute()), override=True)
from supabase import create_client

url = os.environ["SUPABASE_URL"]
svc = os.environ["SUPABASE_SERVICE_KEY"]
sb  = create_client(url, svc)

# Get user
users = sb.table("users").select("*").eq("email", "samikhya79@gmail.com").execute()
if not users.data:
    print("ERROR: User not found")
    exit(1)

uid = users.data[0]["id"]
print(f"User found: {uid}")

# Test insert
test = {
    "user_id": uid, "age": 45, "sex": 1, "cp": 2,
    "trestbps": 120, "chol": 200, "fbs": 0, "restecg": 0,
    "thalach": 150, "exang": 0, "oldpeak": 0.5, "slope": 1,
    "ca": 0, "thal": 2, "risk_score": 0.32,
    "risk_label": "LOW", "source": "test"
}
res = sb.table("health_records").insert(test).execute()
if res.data:
    print("INSERT SUCCESS:", res.data[0]["id"])
    # Cleanup
    sb.table("health_records").delete().eq("id", res.data[0]["id"]).execute()
    print("Test record cleaned up. DATABASE IS WORKING!")
else:
    print("INSERT FAILED:", res)
