"""Text-to-speech (TTS) output module for LegalMindAI CLI."""

from typing import Optional
from cli.renderer import console, render_info, render_warning


class SpeechOutputHandler:
    """Manages text-to-speech synthesis and audio playback."""

    @staticmethod
    def is_available() -> bool:
        """Check if TTS synthesis engines and audio devices are present."""
        try:
            import pyttsx3  # type: ignore
            engine = pyttsx3.init()
            return True
        except Exception:
            return False

    @staticmethod
    def speak_text(text: str, summary_only: bool = False, language: str = "en") -> None:
        """Synthesize and speak the provided legal answer."""
        if not SpeechOutputHandler.is_available():
            render_info("[INFO] Text-to-speech is not currently available.")
            return

        clean_text = text.strip()
        if not clean_text:
            render_warning("[WARNING] No answer available to read aloud.")
            return

        if summary_only and len(clean_text) > 300:
            clean_text = clean_text[:300] + "... (summary concluded)"

        try:
            import pyttsx3  # type: ignore
            console.print("[bold cyan]● Speaking answer...[/bold cyan]")
            engine = pyttsx3.init()
            engine.say(clean_text)
            engine.runAndWait()
        except Exception as e:
            render_info(f"[INFO] Text-to-speech playback failed: {e}")
