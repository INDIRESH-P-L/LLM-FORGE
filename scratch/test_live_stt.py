import urllib.request
import ssl
import json
import io
import mimetypes
from gtts import gTTS

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Generate audio sample
sample_text = "What is the basic structure doctrine in Indian constitutional law"
tts = gTTS(text=sample_text, lang="en", tld="co.in")
buf = io.BytesIO()
tts.write_to_fp(buf)
audio_bytes = buf.getvalue()

boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
body = io.BytesIO()
body.write(f"--{boundary}\r\n".encode())
body.write(b'Content-Disposition: form-data; name="file"; filename="voice_query.mp3"\r\n')
body.write(b"Content-Type: audio/mpeg\r\n\r\n")
body.write(audio_bytes)
body.write(b"\r\n")
body.write(f"--{boundary}\r\n".encode())
body.write(b'Content-Disposition: form-data; name="language"\r\n\r\n')
body.write(b"en\r\n")
body.write(f"--{boundary}--\r\n".encode())
payload = body.getvalue()

req = urllib.request.Request(
    "https://127.0.0.1:8443/api/speech-to-text",
    data=payload,
    headers={
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(payload)),
    },
    method="POST",
)

with urllib.request.urlopen(req, context=ctx, timeout=20) as res:
    print("Status:", res.status)
    resp_data = json.loads(res.read().decode())
    print("STT Live Response:", resp_data)
    assert resp_data["success"] is True
    assert "basic structure doctrine" in resp_data["text"].lower()
    print("LIVE ENDPOINT TRANSCRIPTION VERIFIED SUCCESSFULLY!")
