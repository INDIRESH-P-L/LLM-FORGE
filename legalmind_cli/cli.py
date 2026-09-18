"""
legalmind_cli/cli.py
====================
Main command-line interface entry point for LegalMindAI CLI.
Supports direct Q&A, document analysis, and the complete Legal Intelligence Suite.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from legalmind_cli.client import LegalMindClient
from legalmind_cli.commands import (
    handle_ask,
    handle_chat,
    handle_document,
    handle_dossier,
    handle_draft,
    handle_health,
    handle_image,
    handle_moot,
    handle_precedent,
    handle_redline,
    handle_sources,
    handle_version,
    handle_voice,
)
from legalmind_cli.config import config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legalmind",
        description="LegalMindAI Terminal CLI — Indian Legal Intelligence Suite",
    )
    parser.add_argument("--server", default=config.server_url, help=f"LegalMindAI backend URL (default: {config.server_url})")
    parser.add_argument("--timeout", type=int, default=config.timeout_seconds, help="Timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Output raw JSON response")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # ask
    ask_p = subparsers.add_parser("ask", help="Ask a direct legal research question")
    ask_p.add_argument("query", help="The legal question to research")
    ask_p.add_argument("--mode", default="detailed", choices=["detailed", "simple", "student", "case-analysis"], help="Answer mode")
    ask_p.add_argument("--stream", action="store_true", default=True, help="Stream answer tokens")
    ask_p.add_argument("--no-stream", action="store_false", dest="stream", help="Disable streaming output")
    ask_p.add_argument("--sources", action="store_true", default=True, help="Display verified legal citations")
    ask_p.add_argument("--no-sources", action="store_false", dest="sources", help="Hide legal citations")

    # draft
    draft_p = subparsers.add_parser("draft", help="Generate court-ready legal notices, bail petitions, or NDAs")
    draft_p.add_argument("--type", default="legal_notice_138_ni", choices=["legal_notice_138_ni", "bail_application", "commercial_nda", "consumer_complaint"], help="Document type")
    draft_p.add_argument("--out", default=None, help="Output file path to save draft")

    # redline
    redline_p = subparsers.add_parser("redline", help="Audit contract clause for statutory risk and get safe revision")
    redline_p.add_argument("text", help="Contract clause or agreement text to audit")

    # precedent
    prec_p = subparsers.add_parser("precedent", help="Trace landmark precedent lineage and treatment graph")
    prec_p.add_argument("case", help="Case name to trace (e.g. 'Kesavananda Bharati', 'Maneka Gandhi')")

    # moot
    moot_p = subparsers.add_parser("moot", help="Simulate courtroom judicial cross-examination with Advocacy Scorecard")
    moot_p.add_argument("argument", help="Counsel's submission or legal proposition")
    moot_p.add_argument("--bench", default="constitutional", choices=["constitutional", "criminal", "commercial"], help="Judicial Bench archetype")

    # dossier
    dos_p = subparsers.add_parser("dossier", help="Generate structured court memorandum and export A4 PDF brief")
    dos_p.add_argument("--title", required=True, help="Cause title or case name")
    dos_p.add_argument("--query", required=True, help="Core legal question or proposition")
    dos_p.add_argument("--court", default="IN THE SUPREME COURT OF INDIA", help="Target court or forum")

    # image
    img_p = subparsers.add_parser("image", help="Upload and query an image document")
    img_p.add_argument("path", help="Path to image file (JPG, PNG, WEBP)")
    img_p.add_argument("question", nargs="?", default=None, help="Question regarding the image")

    # pdf
    pdf_p = subparsers.add_parser("pdf", help="Upload and query a PDF judgment or document")
    pdf_p.add_argument("path", help="Path to PDF file")
    pdf_p.add_argument("question", nargs="?", default=None, help="Question regarding the PDF")

    # document
    doc_p = subparsers.add_parser("document", help="Upload and query any legal document (PDF, DOCX, TXT)")
    doc_p.add_argument("path", help="Path to document file")
    doc_p.add_argument("question", nargs="?", default=None, help="Question regarding the document")

    # voice
    subparsers.add_parser("voice", help="Interactive voice typing input")

    # sources
    src_p = subparsers.add_parser("sources", help="Inspect verified legal sources for a topic")
    src_p.add_argument("query", help="Legal topic or query")

    # health
    subparsers.add_parser("health", help="Check backend service health and model status")

    # version
    subparsers.add_parser("version", help="Show CLI version")

    # chat
    subparsers.add_parser("chat", help="Launch interactive terminal chat session")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        # Default to interactive chat if no subcommand passed
        client = LegalMindClient(base_url=args.server, timeout=args.timeout)
        handle_chat(client)
        return 0

    client = LegalMindClient(base_url=args.server, timeout=args.timeout)

    if args.command == "ask":
        handle_ask(
            client=client,
            query=args.query,
            mode=args.mode,
            stream=args.stream,
            as_json=args.json,
            show_sources=args.sources,
        )
    elif args.command == "draft":
        handle_draft(client=client, doc_type=args.type, save_file=args.out)
    elif args.command == "redline":
        handle_redline(client=client, contract_text=args.text)
    elif args.command == "precedent":
        handle_precedent(client=client, case_name=args.case)
    elif args.command == "moot":
        handle_moot(client=client, argument=args.argument, bench_type=args.bench)
    elif args.command == "dossier":
        handle_dossier(client=client, case_title=args.title, query=args.query, court=args.court)
    elif args.command == "image":
        handle_image(client=client, image_path_str=args.path, question=args.question, as_json=args.json)
    elif args.command in ("pdf", "document"):
        handle_document(client=client, doc_path_str=args.path, question=args.question, as_json=args.json)
    elif args.command == "voice":
        handle_voice(client=client)
    elif args.command == "sources":
        handle_sources(client=client, query=args.query)
    elif args.command == "health":
        handle_health(client=client)
    elif args.command == "version":
        handle_version()
    elif args.command == "chat":
        handle_chat(client=client)
    else:
        parser.print_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
