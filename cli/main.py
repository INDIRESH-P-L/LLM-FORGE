"""Command-line entrypoint and argument parser for LegalMindAI CLI."""

import argparse
import logging
import sys
from typing import List, Optional

from cli import __app_name__, __version__
from cli.client import LegalMindClient
from cli.commands import (
    analyze_document_command,
    analyze_image_command,
    ask_question,
    show_sources_command,
    show_status_command,
    speak_command,
    voice_command,
)
from cli.config import config
from cli.interactive import start_interactive_session
from cli.renderer import console, render_error, render_shortcuts


def create_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="legalmind",
        description="LegalMindAI: Indian Legal Research Assistant CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"{__app_name__} {__version__}",
        help="Show program version and exit.",
    )
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help="Specify LegalMindAI backend URL (e.g. http://127.0.0.1:8080).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose diagnostic debug logging.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Render answers as structured JSON responses.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: chat
    subparsers.add_parser("chat", help="Start full-screen interactive legal chat session.")

    # Command: ask
    ask_parser = subparsers.add_parser("ask", help="Ask a single legal question.")
    ask_parser.add_argument("query", type=str, help="Question to ask LegalMindAI.")
    ask_parser.add_argument(
        "--mode",
        type=str,
        choices=["simple", "detailed", "student", "case-analysis"],
        default=None,
        help="Response depth and perspective mode.",
    )
    ask_parser.add_argument("--json", action="store_true", help="Output raw JSON response.")
    ask_parser.add_argument("--no-stream", action="store_true", help="Disable streaming output.")

    # Command: image
    img_parser = subparsers.add_parser("image", help="Upload and analyze a legal image/notice.")
    img_parser.add_argument("file_path", type=str, help="Path to image file (PNG, JPG, WEBP).")
    img_parser.add_argument("question", nargs="?", default=None, help="Optional question about the image.")
    img_parser.add_argument("--show-ocr", action="store_true", help="Display extracted OCR text.")

    # Command: pdf
    pdf_parser = subparsers.add_parser("pdf", help="Upload and question a legal PDF.")
    pdf_parser.add_argument("file_path", type=str, help="Path to PDF document.")
    pdf_parser.add_argument("question", nargs="?", default=None, help="Optional question about the PDF.")

    # Command: analyze
    analyze_parser = subparsers.add_parser("analyze", help="Analyze and summarize a judgment or legal document.")
    analyze_parser.add_argument("file_path", type=str, help="Path to document (PDF, DOCX, TXT, PNG).")

    # Command: voice
    voice_parser = subparsers.add_parser("voice", help="Record and query via voice typing.")
    voice_parser.add_argument(
        "--language",
        type=str,
        choices=["en", "ta", "hi"],
        default="en",
        help="Spoken language code (en=English, ta=Tamil, hi=Hindi).",
    )

    # Command: sources
    subparsers.add_parser("sources", help="Display retrieved sources from the latest answer.")

    # Command: status
    subparsers.add_parser("status", help="Show backend connectivity and component status.")

    # Command: speak
    speak_parser = subparsers.add_parser("speak", help="Read latest answer aloud using TTS.")
    speak_parser.add_argument("--summary", action="store_true", help="Read short summary only.")

    # Command: help
    subparsers.add_parser("help", help="Show CLI shortcuts and help instructions.")

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """CLI execution entrypoint."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    # Global flags
    if parsed.server:
        config.server_url = parsed.server
    if parsed.debug:
        config.debug = True
        logging.basicConfig(level=logging.DEBUG)
    if parsed.json:
        config.json_mode = True

    client = LegalMindClient(server_url=config.server_url)

    # Default to interactive chat if no command provided
    if parsed.command in (None, "chat"):
        start_interactive_session(client)
        return 0

    if parsed.command == "ask":
        ask_question(
            client=client,
            query=parsed.query,
            mode=parsed.mode,
            json_output=parsed.json,
            stream=not parsed.no_stream,
        )
        return 0

    if parsed.command == "image":
        analyze_image_command(
            client=client,
            file_path=parsed.file_path,
            question=parsed.question,
            show_ocr=parsed.show_ocr,
        )
        return 0

    if parsed.command == "pdf":
        analyze_document_command(
            client=client,
            file_path=parsed.file_path,
            question=parsed.question,
        )
        return 0

    if parsed.command == "analyze":
        analyze_document_command(
            client=client,
            file_path=parsed.file_path,
        )
        return 0

    if parsed.command == "voice":
        voice_command(client=client, language=parsed.language)
        return 0

    if parsed.command == "sources":
        show_sources_command(client)
        return 0

    if parsed.command == "status":
        show_status_command(client)
        return 0

    if parsed.command == "speak":
        speak_command(summary=parsed.summary)
        return 0

    if parsed.command == "help":
        parser.print_help()
        render_shortcuts()
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
