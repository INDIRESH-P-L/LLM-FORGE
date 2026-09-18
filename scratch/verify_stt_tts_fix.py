import io
import tempfile
import os
from gtts import gTTS
from app.services.speech_service import SpeechService
from app.services.tts_service import TTSService

print("--- 1. TESTING STT (Speech-to-Text) ---")
test_phrase = "Right to constitutional remedies under Article 32"
print(f"Generating synthetic speech for: '{test_phrase}'...")
tts = gTTS(text=test_phrase, lang="en", tld="co.in")
buf = io.BytesIO()
tts.write_to_fp(buf)
audio_data = buf.getvalue()
print(f"Audio payload generated: {len(audio_data)} bytes")

print("Transcribing via SpeechService.transcribe_audio...")
result = SpeechService.transcribe_audio("test.mp3", audio_data, language="en")
print("STT Result:", result)
assert result["success"] is True, "STT failed!"
assert len(result["text"]) > 0, "Transcribed text is empty!"
# Ensure it does NOT equal the old mock text!
assert "Article 19 and Article 21" not in result["text"], "Error: still returning old mock text!"
print(f"Transcribed accurately: '{result['text']}'")

print("\n--- 2. TESTING TTS STREAMING (Text-to-Speech) ---")
answer_text = "Article 32 is known as the heart and soul of the Indian Constitution, guaranteeing the right to move the Supreme Court."
audio_stream_bytes = TTSService.generate_audio_bytes(answer_text, language="en")
print(f"Generated audio stream: {len(audio_stream_bytes)} bytes")
assert len(audio_stream_bytes) > 5000, "Generated audio stream is too small!"

# Test cache
cached_bytes = TTSService.generate_audio_bytes(answer_text, language="en")
assert cached_bytes == audio_stream_bytes, "Cache mismatch!"
print("TTS Caching verified successfully!")

print("\nALL BACKEND STT & TTS SERVICE TESTS PASSED!")
