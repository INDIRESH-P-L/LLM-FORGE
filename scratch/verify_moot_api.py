"""
scratch/verify_moot_api.py
Direct verification script testing Moot Court endpoints over HTTPS.
"""
import urllib.request
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# 1. Test /api/moot/problems
req1 = urllib.request.Request("https://127.0.0.1:8443/api/moot/problems")
with urllib.request.urlopen(req1, context=ctx, timeout=10) as r:
    data1 = json.loads(r.read().decode())
    print("1. PROBLEMS COUNT:", len(data1.get("problems", [])))

# 2. Test /api/moot/corams
req2 = urllib.request.Request("https://127.0.0.1:8443/api/moot/corams")
with urllib.request.urlopen(req2, context=ctx, timeout=10) as r:
    data2 = json.loads(r.read().decode())
    print("2. CORAMS:", list(data2.get("corams", {}).keys()))

# 3. Test /api/moot/export
export_payload = {
    "bench_type": "constitutional",
    "temperament": "inquisitive",
    "counsel_side": "petitioner",
    "case_title": "PUDR v. Union of India",
    "rounds": [
        {"counsel": "Submissions on Article 21 and privacy.", "bench": "Bench challenge on four-prong proportionality."}
    ],
    "scorecard": {
        "overall_score": 82.0,
        "readiness_rating": "Competent Submission",
        "constitutional_grounding": 85,
        "statutory_precision": 78,
        "precedent_authority": 84,
        "judicial_persuasion": 80,
        "rebuttal_tip": "Focus on least restrictive alternative test under Puttaswamy."
    }
}
req3 = urllib.request.Request(
    "https://127.0.0.1:8443/api/moot/export",
    data=json.dumps(export_payload).encode(),
    headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req3, context=ctx, timeout=10) as r:
    data3 = json.loads(r.read().decode())
    print("3. EXPORT SUCCESS:", data3.get("success"))
    print("   SAMPLE ORDER SHEET:")
    print("\n".join(data3.get("record", "").split("\n")[:18]))

print("\nALL VERIFICATIONS COMPLETED SUCCESSFULLY!")
