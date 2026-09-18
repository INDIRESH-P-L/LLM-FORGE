import time
import requests
import urllib3
urllib3.disable_warnings()

print("--- 1. Testing Greeting ('hi') latency ---")
t0 = time.time()
r_hi = requests.post(
    "https://127.0.0.1:8443/api/chat",
    json={"query": "hi"},
    verify=False,
    timeout=10
)
lat_hi = time.time() - t0
print(f"Greeting Status: {r_hi.status_code}, Latency: {lat_hi:.4f}s")
if r_hi.status_code == 200:
    print("Response:", r_hi.json().get("assistant_message", {}).get("content", "")[:120])

print("\n--- 2. Testing /api/moot/interject latency ---")
t0 = time.time()
r_moot = requests.post(
    "https://127.0.0.1:8443/api/moot/interject",
    json={
        "coram": "criminal",
        "temperament": "inquisitive",
        "round_num": 1,
        "argument": "My Lord, the applicant seeks bail under Section 439 CrPC as the twin conditions of Section 45 PMLA are unconstitutional and arbitrary.",
        "case_topic": "Bail under PMLA",
        "counsel_side": "petitioner",
        "history": []
    },
    verify=False,
    timeout=60
)
lat_moot = time.time() - t0
print(f"Moot Court Status: {r_moot.status_code}, Latency: {lat_moot:.2f}s")
if r_moot.status_code == 200:
    data = r_moot.json()
    print("Bench Interjection:", data.get("judicial_interjection", "")[:120])
    print("Score:", data.get("scorecard", {}).get("overall_score") or data.get("scorecard", {}).get("total_score"))
