import urllib.request
import os
from pathlib import Path

print("Testing outbound connectivity...")
for url in ["https://pypi.org", "https://huggingface.co"]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "LegalMindAI/1.0"})
        with urllib.request.urlopen(req, timeout=3) as res:
            print(f"URL {url}: status {res.status} OK")
    except Exception as e:
        print(f"URL {url}: FAILED ({e})")

print("\nChecking ~/.cache and Huggingface Hub:")
hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
if hf_cache.exists():
    for p in hf_cache.iterdir():
        print("HF cache entry:", p.name)
else:
    print("HF hub cache path does not exist:", hf_cache)
