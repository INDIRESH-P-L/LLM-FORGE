"""
legalmind_cli/commands.py
=========================
Command implementations for LegalMindAI CLI.
Supports direct Q&A, document analysis, and the complete Legal Intelligence Suite.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from legalmind_cli import __version__
from legalmind_cli.client import ClientError, LegalMindClient
from legalmind_cli.renderer import (
    console,
    render_answer_response,
    render_banner,
    render_dossier_summary,
    render_draft_preview,
    render_error,
    render_help_menu,
    render_info,
    render_input_box_bottom,
    render_input_box_top,
    render_moot_exchange,
    render_precedent_tree,
    render_redline_results,
    render_warning,
)


def handle_ask(
    client: LegalMindClient,
    query: str,
    mode: str = "detailed",
    stream: bool = True,
    as_json: bool = False,
    show_sources: bool = True,
) -> None:
    """Executes a direct legal research question."""
    if not query.strip():
        render_error("Question cannot be empty.")
        return

    try:
        render_info(f"Researching: [italic]{query}[/italic]...")
        res = client.query_legal(query, mode=mode)
        render_answer_response(res, as_json=as_json, show_sources=show_sources, stream=stream)
    except ClientError as e:
        render_error(e.message)


def handle_draft(
    client: LegalMindClient,
    doc_type: str = "legal_notice_138_ni",
    params: Optional[Dict[str, Any]] = None,
    save_file: Optional[str] = None,
) -> None:
    """Generates a court-ready legal document draft."""
    try:
        render_info(f"Generating draft for [bold white]{doc_type}[/bold white]...")
        res = client.generate_draft(document_type=doc_type, parameters=params or {})
        
        saved_path = None
        if save_file or True:  # Default save locally for convenience
            filename = save_file or f"LegalMind_Draft_{doc_type}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(res.get("draft_text", ""))
            saved_path = os.path.abspath(filename)

        render_draft_preview(res, saved_path=saved_path)
    except ClientError as e:
        render_error(e.message)


def handle_redline(
    client: LegalMindClient,
    contract_text: str,
) -> None:
    """Audits contract clauses for Indian statutory risk."""
    if not contract_text.strip():
        render_error("Contract clause text cannot be empty.")
        return
    try:
        render_info("Auditing contract clause against Indian Contract Act & Supreme Court precedents...")
        res = client.review_contract(contract_text=contract_text)
        render_redline_results(res)
    except ClientError as e:
        render_error(e.message)


def handle_precedent(
    client: LegalMindClient,
    case_name: str,
) -> None:
    """Visualizes precedent relationships and treatment lineage."""
    if not case_name.strip():
        render_error("Case name cannot be empty.")
        return
    try:
        render_info(f"Tracing precedent lineage for: [italic]{case_name}[/italic]...")
        res = client.get_precedent_graph(case_name=case_name)
        render_precedent_tree(res)
    except ClientError as e:
        render_error(e.message)


def handle_moot(
    client: LegalMindClient,
    argument: str,
    bench_type: str = "constitutional",
    round_num: int = 1,
) -> None:
    """Simulates active judicial cross-examination with Advocacy Scorecard."""
    if not argument.strip():
        render_error("Counsel submission cannot be empty.")
        return
    try:
        render_info("Submitting proposition to the simulated Supreme Court Bench...")
        res = client.moot_interject(argument=argument, bench_type=bench_type, round_num=round_num)
        render_moot_exchange(res)
    except ClientError as e:
        render_error(e.message)


def handle_dossier(
    client: LegalMindClient,
    case_title: str,
    query: str,
    answer: Optional[str] = None,
    court: str = "IN THE SUPREME COURT OF INDIA",
    export_pdf: bool = True,
) -> None:
    """Generates structured advocate bench dossier and downloads official PDF."""
    try:
        render_info(f"Compiling advocate case dossier for [bold white]{case_title}[/bold white]...")
        ans_text = answer or query
        res = client.generate_dossier(case_title=case_title, query=query, answer=ans_text, court=court)

        pdf_path = None
        if export_pdf:
            render_info("Compiling ReportLab A4 court-formatted brief...")
            pdf_bytes = client.export_dossier_pdf(case_title=case_title, query=query, answer=ans_text, court=court)
            clean_title = case_title.replace(" ", "_").replace("/", "_")[:30]
            pdf_filename = f"Court_Brief_{clean_title}.pdf"
            with open(pdf_filename, "wb") as f:
                f.write(pdf_bytes)
            pdf_path = os.path.abspath(pdf_filename)

        render_dossier_summary(res.get("dossier", {}), pdf_path=pdf_path)
    except ClientError as e:
        render_error(e.message)


def handle_image(
    client: LegalMindClient,
    image_path_str: str,
    question: Optional[str] = None,
    as_json: bool = False,
) -> None:
    """Uploads an image, performs OCR and entity extraction, and answers questions."""
    path = Path(image_path_str)
    if not path.is_file():
        render_error(f"Image file not found: {image_path_str}")
        return

    try:
        render_info(f"Uploading image: {path.name}...")
        up_res = client.upload_image(path)
        img_id = up_res.get("image_id")
        render_info(f"Image processed successfully. Assigned ID: [bold]{img_id}[/bold]")

        user_q = question or "Provide an overview of this legal document image."
        render_info(f"Analyzing image content for: [italic]{user_q}[/italic]...")
        ans_res = client.ask_image(img_id, user_q)
        render_answer_response(ans_res, as_json=as_json, show_sources=True, stream=True)
    except ClientError as e:
        render_error(e.message)


def handle_document(
    client: LegalMindClient,
    doc_path_str: str,
    question: Optional[str] = None,
    as_json: bool = False,
) -> None:
    """Uploads and analyzes a PDF, DOCX, or TXT file."""
    path = Path(doc_path_str)
    if not path.is_file():
        render_error(f"Document file not found: {doc_path_str}")
        return

    try:
        render_info(f"Uploading document: {path.name}...")
        up_res = client.upload_document(path)
        doc_id = up_res.get("document_id")
        render_info(f"Document registered. Context ID: [bold]{doc_id}[/bold]")

        if question:
            render_info(f"Querying document: [italic]{question}[/italic]...")
            ans_res = client.ask_document(doc_id, question)
            render_answer_response(ans_res, as_json=as_json, show_sources=True, stream=True)
        else:
            render_info("Generating structured legal breakdown...")
            analysis = client.analyze_document(doc_id)
            if as_json:
                print(json.dumps(analysis, indent=2))
            else:
                render_answer_response({
                    "answer": f"### Document Analysis: {path.name}\n\n**Genre:** {analysis.get('document_type')}\n\n" +
                              f"**Court:** {analysis.get('entities', {}).get('court') or 'Not specified'}\n\n" +
                              f"**Case Number:** {analysis.get('entities', {}).get('case_number') or 'Not specified'}\n\n" +
                              f"**Parties:** {', '.join(analysis.get('entities', {}).get('parties', [])) or 'Not specified'}\n\n" +
                              f"**Provisions Cited:** {', '.join(analysis.get('entities', {}).get('sections', [])) or 'None'}\n\n" +
                              f"**Summary:**\n{analysis.get('extracted_text', '')[:1000]}...",
                    "citations": analysis.get("entities", {}).get("acts", []),
                    "confidence": analysis.get("confidence", "medium"),
                    "evidence_coverage": 0.9,
                }, show_sources=True, stream=True)
    except ClientError as e:
        render_error(e.message)


def handle_voice(client: LegalMindClient) -> None:
    """Interactive voice typing flow with speech transcription."""
    from cli.voice_input import VoiceInputHandler
    transcription = VoiceInputHandler.capture_and_transcribe()
    if transcription:
        handle_ask(client, transcription)


def handle_sources(client: LegalMindClient, query: str) -> None:
    """Retrieves and inspects verified sources for a given query."""
    if not query.strip():
        render_error("Query cannot be empty.")
        return
    try:
        render_info(f"Retrieving verified sources for: [italic]{query}[/italic]...")
        res = client.query_legal(query)
        citations = res.get("citations", [])
        if not citations:
            render_warning("No verified sources directly matched.")
            return
        console.print("\n[bold cyan]Verified Statutory & Precedent Sources:[/bold cyan]")
        for idx, c in enumerate(citations, 1):
            console.print(f"  [bold][{idx}][/bold] {c}")
        console.print()
    except ClientError as e:
        render_error(e.message)


def handle_health(client: LegalMindClient) -> None:
    """Checks backend server health and displays telemetry."""
    try:
        h = client.check_health()
        render_info(f"LegalMindAI Backend: [bold green]{h.get('status', 'ok').upper()}[/bold green]")
        console.print(f"  Model Loaded:     {'Yes' if h.get('model_loaded') else 'No'}")
        console.print(f"  Retriever Loaded: {'Yes' if h.get('retriever_loaded') else 'No'}")
        console.print(f"  LoRA Adapter:     [bold green]{h.get('lora_adapter', 'lora-legal-v2')}[/bold green] ({h.get('lora_status', 'ACTIVE')})")
        console.print(f"  Queries Served:   {h.get('queries_served', 0)}")
        console.print(f"  Server Uptime:    {h.get('uptime_s', 0):.1f} seconds")
    except ClientError as e:
        render_error(e.message)


def handle_version() -> None:
    console.print(f"[bold cyan]LegalMindAI CLI[/bold cyan] version [bold white]{__version__}[/bold white]")

def handle_temporal(client: LegalMindClient, sections: str) -> None:
    try:
        render_info(f"Checking temporal application for sections: [bold]{sections}[/bold]...")
        res = client._post("/api/temporal/convert", {"offense_date": "2024-08-01", "sections": [s.strip() for s in sections.split(",")], "act": "auto"})
        console.print(f"\n[bold cyan]Temporal Assessment:[/bold cyan] {res.get('regime')}")
        console.print(f"[dim]{res.get('regime_rationale')}[/dim]")
        for m in res.get('section_mappings', []):
            color = "green" if m['status'] == "mapped" else "red"
            console.print(f"  • {m['original_section']} -> [{color}]{m['new_section']}[/{color}] ({m['change_summary']})")
    except ClientError as e:
        render_error(e.message)

def handle_bail(client: LegalMindClient, cat: str) -> None:
    try:
        render_info(f"Assessing bail feasibility for: [bold]{cat}[/bold]...")
        res = client._post("/api/bail/assess", {"offense_category": cat, "is_special_act": False, "custody_days": 100, "chargesheet_filed": True})
        d = res.get('assessment', {})
        console.print(f"\n[bold cyan]Bail Verdict:[/bold cyan] {d.get('verdict')} (Score: {d.get('bail_score')}/100)")
        for f in d.get('positive_factors', []):
            console.print(f"  [green]✓[/green] {f}")
    except ClientError as e:
        render_error(e.message)

def handle_fir(client: LegalMindClient, text: str) -> None:
    try:
        render_info("Auditing FIR for flaws...")
        res = client._post("/api/fir/audit", {"fir_text": text, "arrest_made": True, "is_pmla": False})
        d = res.get('audit_report', {})
        console.print(f"\n[bold cyan]Quashing Feasibility:[/bold cyan] {d.get('verdict')} ({d.get('quashing_feasibility_score')}/100)")
        for f in d.get('fatal_flaws', []):
            console.print(f"  [red]✗[/red] {f}")
    except ClientError as e:
        render_error(e.message)

def handle_citcheck(client: LegalMindClient, text: str) -> None:
    try:
        render_info("Validating citations...")
        res = client._post("/api/citations/validate", {"text": text})
        for v in res.get('validations', []):
            color = "red" if "OVERRULED" in v['status'] else "green"
            console.print(f"  • {v['citation_text']}: [{color}]{v['status']}[/{color}] - {v['treatment_note']}")
    except ClientError as e:
        render_error(e.message)

def handle_pleading(client: LegalMindClient, text: str) -> None:
    try:
        render_info("Formatting pleading from dictation...")
        res = client._post("/api/pleading/format", {"raw_text": text, "pleading_type": "bail"})
        console.print(f"\n[bold cyan]Formatted Pleading Draft:[/bold cyan]\n")
        console.print(res.get('pleading_draft', {}).get('formatted_draft'))
    except ClientError as e:
        render_error(e.message)


def handle_chat(client: LegalMindClient) -> None:
    """Interactive terminal chat cockpit with framed input box."""
    try:
        import readline
        COMMANDS = [
            "/help", "/draft notice", "/draft bail", "/draft nda", "/draft consumer",
            "/redline", "/precedent", "/moot", "/dossier", "/doc", "/image",
            "/lang en", "/lang hi", "/lang ta", "/health", "/stats", "/clear", "exit", "quit"
        ]
        def completer(text: str, state: int) -> Optional[str]:
            options = [c for c in COMMANDS if c.startswith(text)]
            if state < len(options):
                return options[state]
            return None
        readline.set_completer(completer)
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass

    client.ensure_service_ready(wait_seconds=5)
    render_banner()

    active_doc_id: Optional[str] = None
    moot_round = 1

    while True:
        try:
            render_input_box_top(active_doc_id=active_doc_id)
            doc_badge = f" [{active_doc_id}]" if active_doc_id else ""
            prompt_str = f"│ ⚖️  LegalMind{doc_badge} ❯ "
            user_input = input(prompt_str).strip()
            render_input_box_bottom()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", ":q"):
                console.print("[dim]Exiting LegalMindAI CLI. Session closed.[/dim]\n")
                break

            # Help Command
            if user_input.lower() in ("/help", "help", "?"):
                render_help_menu()
                continue

            # Clear Command
            if user_input.lower() == "/clear":
                active_doc_id = None
                os.system("clear" if os.name == "posix" else "cls")
                render_banner()
                continue

            # Health Command
            if user_input.lower() in ("/health", "/stats"):
                handle_health(client)
                continue

            # Language Command
            if user_input.startswith("/lang"):
                parts = user_input.split()
                lang = parts[1] if len(parts) > 1 else "en"
                render_info(f"Language set to: [bold white]{lang}[/bold white]")
                continue

            # Document Drafter Command
            if user_input.startswith("/draft"):
                parts = user_input.split(maxsplit=1)
                sub_type = parts[1].strip() if len(parts) > 1 else "notice"
                if "bail" in sub_type.lower():
                    doc_type = "bail_application"
                    params = {"accused_name": "Petitioner Accused", "fir_details": "Crime FIR 102/2024", "sections": "S. 316, 318 BNS", "grounds": "Investigation completed, no flight risk"}
                elif "nda" in sub_type.lower():
                    doc_type = "commercial_nda"
                    params = {"disclosing_party": "Alpha Systems", "receiving_party": "Beta Solutions", "purpose": "Strategic commercial integration", "term": "3 Years"}
                elif "consumer" in sub_type.lower():
                    doc_type = "consumer_complaint"
                    params = {"complainant": "Consumer", "opposite_party": "Company Ltd", "deficiency": "Deficient service delivery", "relief": "Refund with interest"}
                else:
                    doc_type = "legal_notice_138_ni"
                    params = {"sender_name": "Complainant Payee", "receiver_name": "Drawer Accused", "cheque_number": "482019", "amount": "₹ 10,00,000/-", "bank_name": "State Bank of India"}
                handle_draft(client, doc_type=doc_type, params=params)
                continue

            # Contract Redline Command
            if user_input.startswith("/redline"):
                clause = user_input[8:].strip()
                if not clause:
                    clause = "The Vendor shall unconditionally defend, indemnify, and hold harmless the Company from and against any and all losses, claims, and damages whatsoever, without any limitation of liability or cap."
                handle_redline(client, contract_text=clause)
                continue

            # Precedent Lineage Command
            if user_input.startswith("/precedent"):
                cname = user_input[10:].strip()
                if not cname:
                    cname = "Kesavananda Bharati v. State of Kerala"
                handle_precedent(client, case_name=cname)
                continue

            # Moot Court Command
            if user_input.startswith("/moot"):
                arg = user_input[5:].strip()
                if not arg:
                    arg = "Article 21 incorporates procedural fairness and proportionality under Maneka Gandhi."
                handle_moot(client, argument=arg, round_num=moot_round)
                moot_round += 1
                continue

            # Case Dossier Command
            if user_input.startswith("/dossier"):
                title = user_input[8:].strip() or "Sharma v. State of Maharashtra"
                handle_dossier(client, case_title=title, query="Right to speedy trial and bail under Article 21")
                continue

            # Temporal Law Bridge Command
            if user_input.startswith("/temporal"):
                sec = user_input[9:].strip() or "302, 34"
                handle_temporal(client, sections=sec)
                continue

            # Bail Matrix Command
            if user_input.startswith("/bail"):
                cat = user_input[5:].strip() or "Economic Offence"
                handle_bail(client, cat=cat)
                continue

            # FIR Auditor Command
            if user_input.startswith("/fir"):
                text = user_input[4:].strip() or "The accused committed breach of contract after a gap of 2 years."
                handle_fir(client, text=text)
                continue

            # Citation Check Command
            if user_input.startswith("/citcheck"):
                text = user_input[9:].strip() or "ADM Jabalpur v. Shivkant Shukla"
                handle_citcheck(client, text=text)
                continue


            # Voice Pleading Command
            if user_input.startswith("/pleading"):
                text = user_input[9:].strip() or "The applicant is innocent. FIR registered after 5 days."
                handle_pleading(client, text=text)
                continue

            # Document Attach Command
            if user_input.startswith("/doc "):
                doc_path = user_input[5:].strip()
                path = Path(doc_path)
                if not path.is_file():
                    render_error(f"File not found: {doc_path}")
                    continue
                render_info(f"Loading document context: {path.name}...")
                up_res = client.upload_document(path)
                active_doc_id = up_res.get("document_id")
                render_info(f"Active document context set to: [bold]{active_doc_id}[/bold]")
                continue

            # Image Attach Command
            if user_input.startswith("/image "):
                img_path = user_input[7:].strip()
                path = Path(img_path)
                if not path.is_file():
                    render_error(f"File not found: {img_path}")
                    continue
                render_info(f"Loading image context: {path.name}...")
                up_res = client.upload_image(path)
                active_doc_id = up_res.get("image_id")
                render_info(f"Active image context set to: [bold]{active_doc_id}[/bold]")
                continue

            # Sanitize input: if user typed `ask "..."` or `ask ...` in interactive mode
            clean_query = user_input.strip()
            if clean_query.lower().startswith("ask "):
                clean_query = clean_query[4:].strip().strip("\"'")

            # Route to document context or general legal research
            if active_doc_id:
                if active_doc_id.startswith("img_"):
                    ans_res = client.ask_image(active_doc_id, clean_query)
                else:
                    ans_res = client.ask_document(active_doc_id, clean_query)
                render_answer_response(ans_res, show_sources=True, stream=True)
            else:
                handle_ask(client, clean_query)

        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Session terminated.[/dim]\n")
            break
