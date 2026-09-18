"""Shortcut dispatcher and command handlers for interactive LegalMindAI session."""

import os
from typing import Optional

from cli.client import LegalMindClient
from cli.config import config
from cli.image_input import ImageAnalyzer
from cli.pdf_input import DocumentParser
from cli.renderer import (
    console,
    render_banner,
    render_error,
    render_info,
    render_shortcuts,
    render_sources_list,
    render_status,
    render_warning,
)
from cli.speech_output import SpeechOutputHandler
from cli.validators import ValidationError
from cli.voice_input import VoiceInputHandler


class ShortcutManager:
    """Processes interactive slash commands and shortcut keys."""

    def __init__(self, client: LegalMindClient):
        self.client = client

    def handle(self, command_line: str) -> bool:
        """Process a shortcut command.
        
        Returns:
            True if the command was recognized and handled; False otherwise.
        """
        raw = command_line.strip()
        if not raw:
            return False

        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        # Shortcuts mapping
        if cmd in ("?", "/shortcuts", "/shortcut"):
            render_shortcuts()
            return True

        if cmd in ("/help", "help"):
            self._handle_help()
            return True

        if cmd in ("/clear", "clear"):
            console.clear()
            health = self.client.check_health()
            stats = self.client.get_stats()
            render_banner(health, stats)
            return True

        if cmd in ("/reset", "/new"):
            config.active_conversation_id = None
            config.active_document_id = None
            config.active_document_name = None
            config.active_document_type = None
            config.active_document_text = None
            config.last_response = None
            config.last_citations = []
            config.last_confidence = "UNKNOWN"
            config.last_latency_ms = 0.0
            render_info("[INFO] Conversation context reset. Starting fresh session.")
            return True

        if cmd in ("/source", "/sources"):
            render_sources_list(config.last_citations)
            return True

        if cmd == "/status":
            health = self.client.check_health()
            stats = self.client.get_stats()
            render_status(
                client_health=health,
                stats=stats,
                voice_available=VoiceInputHandler.is_available(),
                ocr_available=True,
                tts_available=SpeechOutputHandler.is_available(),
            )
            return True

        if cmd == "/model":
            self._handle_model()
            return True

        if cmd == "/voice":
            lang = arg.lower() if arg in ("en", "ta", "hi") else "en"
            transcript = VoiceInputHandler.capture_and_transcribe(language=lang)
            if transcript:
                from cli.commands import ask_question
                ask_question(self.client, transcript)
            return True

        if cmd == "/upload":
            if not arg:
                render_warning("[WARNING] Usage: /upload <path-to-file>")
            else:
                self._handle_upload(arg)
            return True

        if cmd == "/ocr":
            self._handle_ocr()
            return True

        if cmd == "/json":
            config.json_mode = not config.json_mode
            status_str = "ENABLED" if config.json_mode else "DISABLED"
            render_info(f"[INFO] JSON output mode: {status_str}")
            return True

        if cmd == "/stream":
            config.stream = not config.stream
            status_str = "ENABLED" if config.stream else "DISABLED"
            render_info(f"[INFO] Streaming mode: {status_str}")
            return True

        if cmd == "/summary":
            self._handle_summary()
            return True

        if cmd == "/speak":
            if config.last_response and config.last_response.get("answer"):
                SpeechOutputHandler.speak_text(config.last_response["answer"])
            else:
                render_warning("[WARNING] No answer available to read aloud.")
            return True

        if cmd == "/history":
            self._handle_history()
            return True

        if cmd in ("/exit", "/quit", ":q", "exit", "quit"):
            console.print("\n[grey70]Exiting LegalMindAI CLI. Goodbye![/grey70]")
            import sys
            sys.exit(0)

        # Unrecognized slash command
        if cmd.startswith("/"):
            render_error(f"[ERROR] Unknown command: {cmd}. Type ? for available shortcuts.")
            return True

        return False

    def _handle_help(self) -> None:
        console.print()
        console.print("[bold cyan]LegalMindAI CLI Help & Usage[/bold cyan]")
        console.print("[grey70]Ask any Indian legal question or upload legal documents for analysis.[/grey70]\n")
        console.print("[bold white]Examples:[/bold white]")
        console.print("  > Explain Article 21 of the Constitution of India")
        console.print("  > What are the reasonable restrictions under Article 19(2)?")
        console.print("  > What is the Golden Triangle doctrine connecting Articles 14, 19, and 21?")
        console.print("  > /upload ./sample_judgment.pdf")
        console.print("  > Explain paragraph 3 of the uploaded document\n")
        console.print("[bold white]Interactive Commands:[/bold white]")
        render_shortcuts()

    def _handle_model(self) -> None:
        stats = self.client.get_stats()
        model_data = stats.get("model", {})
        gpu_data = stats.get("gpu", {})
        
        console.print()
        console.print("[bold cyan]Active Model Architecture[/bold cyan]")
        console.print(f"  [bold white]Model Name:[/bold white]    {model_data.get('name', config.default_model)}")
        console.print(f"  [bold white]Model Path:[/bold white]    {model_data.get('path', 'models/Qwen3.6-35B-A3B')}")
        if gpu_data:
            console.print(f"  [bold white]GPU Device:[/bold white]    {gpu_data.get('device', 'NVIDIA DGX')}")
            console.print(f"  [bold white]VRAM Usage:[/bold white]    {gpu_data.get('memory_used', 'N/A')}")
        console.print(f"  [bold white]RAG Fusion:[/bold white]    Hybrid BM25 + FAISS (BGE-M3) + RRF Reranking")
        console.print()

    def _handle_upload(self, file_path: str) -> None:
        try:
            from cli.commands import analyze_document
            analyze_document(self.client, file_path)
        except ValidationError as e:
            render_error(str(e))
        except Exception as e:
            render_error(f"[ERROR] Failed to process document: {e}")

    def _handle_ocr(self) -> None:
        if not config.active_document_text:
            render_info("[INFO] No document or image has been uploaded in this session.")
            return

        console.print()
        console.print(f"[bold cyan]Extracted Text for {config.active_document_name or 'Document'}[/bold cyan]")
        console.print("─" * 60)
        console.print(config.active_document_text)
        console.print("─" * 60)
        console.print()

    def _handle_summary(self) -> None:
        if not config.last_response or not config.last_response.get("answer"):
            render_warning("[WARNING] No answer available to summarize.")
            return

        answer = config.last_response["answer"]
        from cli.commands import ask_question
        summary_query = f"Please summarize the previous legal answer into 3 key takeaways:\n\n{answer[:3000]}"
        ask_question(self.client, summary_query)

    def _handle_history(self) -> None:
        convs = self.client.list_conversations(limit=10)
        console.print()
        console.print("[bold cyan]Recent Conversations (Backend Store)[/bold cyan]")
        console.print("─" * 60)
        if not convs:
            render_info("[INFO] No previous conversations found.")
            console.print("─" * 60)
            return

        for idx, c in enumerate(convs, start=1):
            cid = c.get("id") or c.get("conversation_id", "")
            title = c.get("title", "Untitled Conversation")
            count = c.get("message_count", 0)
            marker = " (Active)" if cid == config.active_conversation_id else ""
            console.print(f"  [{idx}] [bold white]{title}[/bold white]{marker}")
            console.print(f"      [grey60]ID: {cid} | Messages: {count}[/grey60]")
        console.print("─" * 60)
        console.print()
