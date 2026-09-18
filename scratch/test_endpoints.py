#!/usr/bin/env python3
import json
import ssl
import sys
import urllib.request
import urllib.error

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

BASE = "https://127.0.0.1:8443"

tests = [
    ("GET", "/health", None),
    ("GET", "/stats", None),
    ("GET", "/api/stats", None),
    ("GET", "/api/conversations?limit=5", None),
    ("GET", "/api/activity?limit=5&offset=0", None),
    ("GET", "/api/precedents/landmarks", None),
    ("GET", "/api/precedents/graph?case=Kesavananda%20Bharati%20v.%20State%20of%20Kerala&depth=1", None),
    ("GET", "/api/moot/cases", None),
    ("POST", "/api/drafter/generate", {"document_type": "legal_notice_138_ni", "parameters": {}}),
    ("POST", "/api/drafter/review", {"contract_text": "Sample text for non-compete clause."}),
    ("POST", "/api/temporal/convert", {"offense_date": "2024-08-01", "act": "IPC", "sections": ["302", "420"]}),
    ("POST", "/api/bail/assess", {"offense_category": "Cheating / Financial Crime", "is_special_act": False, "custody_days": 45, "chargesheet_filed": True}),
    ("POST", "/api/fir/audit", {"fir_text": "FIR No 12/2024 dated 01-08-2024 u/s 420 IPC registered at PS Connaught Place.", "arrest_made": False, "is_pmla": False}),
    ("POST", "/api/citations/validate", {"text": "In Kesavananda Bharati v. State of Kerala, AIR 1973 SC 1461, the court held..."}),
    ("POST", "/api/limitation/calculate", {"suit_type": "money_recovery", "cause_of_action_date": "2023-01-01", "exclude_days": 0}),
    ("POST", "/api/pleading/format", {"pleading_type": "criminal_bail", "raw_text": "Accused is falsely implicated."}),
    ("POST", "/api/dossier/generate", {"case_title": "State v. Sharma", "legal_domain": "criminal_defense", "facts": "Arrested without section 41A notice.", "target_relief": "Regular Bail"}),
]

print(f"Testing {len(tests)} LegalMind AI endpoints against {BASE}...\n")
failed = 0
passed = 0

for method, path, body in tests:
    url = BASE + path
    data = json.dumps(body).encode("utf-8") if body else None
    headers = {"Content-Type": "application/json"} if body else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            status = resp.status
            content = resp.read().decode("utf-8")
            print(f"✅ [{status}] {method:4} {path[:45]:45} -> {len(content)} bytes")
            passed += 1
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")[:120]
        print(f"❌ [{e.code}] {method:4} {path[:45]:45} -> {err_body}")
        failed += 1
    except Exception as e:
        print(f"❌ [ERR ] {method:4} {path[:45]:45} -> {e}")
        failed += 1

print(f"\nResults: {passed} passed, {failed} failed.")
sys.exit(1 if failed else 0)
