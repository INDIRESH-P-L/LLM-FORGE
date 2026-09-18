import io
import time
from faster_whisper import WhisperModel
from gtts import gTTS

print("1. Testing gTTS generation...")
tts = gTTS(text="Article 21 guarantees the right to life and personal liberty.", lang="en", tld="co.in")
audio_fp = io.BytesIO()
tts.write_to_fp(audio_fp)
audio_bytes = audio_fp.getvalue()
print(f"Generated TTS audio: {len(audio_bytes)} bytes")

print("2. Testing faster-whisper loading...")
t0 = time.time()
# Load tiny model with cpu
model = WhisperModel("tiny", device="cpu", compute_type="int8")
print(f"WhisperModel loaded in {time.time() - t0:.2f}s")

print("3. Testing transcription on generated audio...")
import tempfile
with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
    f.write(audio_bytes)
    tmp_path = f.name

segments, info = model.transcribe(tmp_path, language="en")
text = " ".join([seg.text for seg in segments]).strip()
print(f"Detected language: {info.language} (prob: {info.language_probability:.2f})")
print(f"Transcribed text: '{text}'")

import os
os.remove(tmp_path)
print("ALL TESTS PASSED SUCCESSFULLY!")
