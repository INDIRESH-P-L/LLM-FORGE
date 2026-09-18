#!/usr/bin/env python3
import json
import time
import urllib.request

query = (
    "Explain the differences between Article 14, Article 19, and Article 21 of the Constitution of India. "
    "Discuss how these rights are connected, their reasonable restrictions, and important Supreme Court judgments. "
    "Give the answer in a clear legal order with verified sources."
)

url = "http://localhost:8080/api/chat"
payload = {
    "query": query,
    "top_k": 14,
}

data = json.dumps(payload).encode("utf-8")
headers = {
    "Content-Type": "application/json",
    "User-Agent": "LegalMindBenchmark/1.0",
}

print("=" * 80)
print(f"BENCHMARK QUERY:\n{query}")
print("=" * 80)
print(f"Sending request to {url}...")

t0 = time.time()
req = urllib.request.Request(url, data=data, headers=headers, method="POST")

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        elapsed = time.time() - t0
        status_code = resp.status
        resp_data = json.loads(resp.read().decode("utf-8"))
        print(f"Response received in {elapsed:.2f}s (HTTP {status_code})")
        print("=" * 80)

        asst_msg = resp_data.get("assistant_message") or {}
        answer = asst_msg.get("content", "")
        metadata = asst_msg.get("metadata", {})
        status = asst_msg.get("status")
        citations = asst_msg.get("citations", [])

        print(f"STATUS: {status}")
        print(f"CONFIDENCE: {metadata.get('confidence')}")
        print(f"MODEL: {metadata.get('model_name')}")
        print(f"RETRIEVED CITATIONS COUNT: {len(citations)}")
        print(f"ANSWER LENGTH: {len(answer)} chars")
        print("=" * 80)
        print("ANSWER PREVIEW (FIRST 2000 CHARS):")
        print(answer[:2000])
        print("\n...\n")
        print("ANSWER END (LAST 1000 CHARS):")
        print(answer[-1000:])
        print("=" * 80)

        # Verification assertions
        assert "insufficient" not in answer.lower()[:300], "Answer must not claim insufficient evidence!"
        assert "Golden Triangle" in answer, "Must discuss Golden Triangle!"
        assert "Article 14" in answer, "Must discuss Article 14!"
        assert "Article 19" in answer, "Must discuss Article 19!"
        assert "Article 21" in answer, "Must discuss Article 21!"
        assert "Maneka Gandhi" in answer, "Must cite Maneka Gandhi!"
        assert "Gopalan" in answer, "Must cite A.K. Gopalan!"
        assert "Reasonable Restrictions" in answer or "19(2)" in answer, "Must discuss reasonable restrictions!"
        assert len(answer) > 4000, f"Answer length ({len(answer)}) should be thorough and comprehensive!"

        print("ALL BENCHMARK VERIFICATION ASSERTIONS PASSED!")

except Exception as e:
    print(f"Benchmark test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
