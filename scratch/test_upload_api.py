import io
import httpx

client = httpx.Client(base_url="http://127.0.0.1:8080", timeout=30.0)

# 1. Test health
res = client.get("/health")
print(f"/health status: {res.status_code}, data: {res.json()}")

# 2. Test text file upload
sample_txt = b"Supreme Court of India\nWrit Petition No. 123 of 2024\nArticle 21 guarantees life and personal liberty."
files = {"file": ("sample_judgment.txt", io.BytesIO(sample_txt), "text/plain")}
res = client.post("/api/upload", files=files)
print(f"/api/upload (txt) status: {res.status_code}")
print("Response:", res.json())
assert res.status_code == 200
assert res.json()["status"] == "success"
assert "Article 21" in res.json()["extracted_text"]

# 3. Test frontend page
res = client.get("/")
assert res.status_code == 200
assert "mode-selector" in res.text
assert "upload-btn" in res.text
assert "voice-btn" in res.text
print("Frontend HTML contains all new features successfully!")
print("ALL WEB EXTENSION TESTS PASSED!")
