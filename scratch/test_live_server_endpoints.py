import urllib.request
import ssl
import json

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

print("1. Testing GET /api/audio/stream on live server...")
req = urllib.request.Request(
    "https://127.0.0.1:8443/api/audio/stream?text=Testing+speech+synthesis+for+LegalMindAI",
    headers={"User-Agent": "LegalMindTest/1.0"}
)
with urllib.request.urlopen(req, context=ctx, timeout=15) as res:
    print(f"Status: {res.status}")
    print(f"Content-Type: {res.headers.get('Content-Type')}")
    data = res.read()
    print(f"Received audio bytes: {len(data)}")
    assert res.status == 200
    assert "audio/mpeg" in res.headers.get("Content-Type")
    assert len(data) > 1000

print("\n2. Testing /health endpoint...")
req_health = urllib.request.Request("https://127.0.0.1:8443/health")
with urllib.request.urlopen(req_health, context=ctx, timeout=5) as res:
    health_data = json.loads(res.read().decode())
    print("Health response:", health_data)
    assert health_data.get("status") == "healthy"

print("\nALL LIVE ENDPOINT TESTS PASSED SUCCESSFULLY!")
