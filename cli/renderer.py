"""Terminal UI and Rich-based formatting engine for LegalMindAI CLI."""

import json
import shutil
import sys
from typing import Any, Dict, List, Optional

from rich.box import ROUNDED, SIMPLE
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.config import config

# Primary console
console = Console(highlight=False, force_terminal=True)


def get_terminal_width() -> int:
    """Get the current terminal width with a safe minimum."""
    try:
        width = shutil.get_terminal_size((80, 24)).columns
        return max(width, 70)
    except Exception:
        return 80


def print_separator(char: str = "─", color: str = "grey50") -> None:
    """Print a horizontal separator across the terminal width."""
    width = get_terminal_width()
    console.print(f"[{color}]{char * width}[/{color}]")


def render_banner(health: Optional[Dict[str, Any]] = None, stats: Optional[Dict[str, Any]] = None) -> None:
    """Render the official LegalMindAI startup banner."""
    width = get_terminal_width()
    backend_ok = health.get("backend_connected", False) if health else False
    model_ok = health.get("model_loaded", False) if health else False
    retriever_ok = health.get("retriever_loaded", False) if health else False

    # Extract model name
    model_name = config.default_model
    if stats and "model" in stats and isinstance(stats["model"], dict):
        model_name = stats["model"].get("name", model_name)

    print_separator("─", "grey50")
    
    # Title centered
    title_text = Text("LEGALMIND AI", style="bold cyan")
    console.print()
    console.print(title_text, justify="center")
    console.print()

    # Meta Info
    console.print("[bold white]LegalMindAI CLI 1.0.0[/bold white]")
    console.print("[grey70]Indian Legal Research Assistant[/grey70]")
    console.print()
    console.print(f"[grey60]User:[/grey60]    [white]{config.user_name}[/white]")
    console.print(f"[grey60]Model:[/grey60]   [white]{model_name}[/white]")
    console.print(f"[grey60]Backend:[/grey60] [white]{config.server_url}[/white]")
    console.print()

    # Health Checks
    console.print("[bold white]Status[/bold white]")
    if backend_ok:
        console.print("  [bold green]✓[/bold green] [white]Backend connected[/white]")
    else:
        console.print("  [bold red]✗[/bold red] [red]Backend disconnected[/red]")

    if model_ok:
        console.print("  [bold green]✓[/bold green] [white]Model available[/white]")
    else:
        console.print("  [bold red]✗[/bold red] [yellow]Model loading / unavailable[/yellow]")

    if retriever_ok:
        console.print("  [bold green]✓[/bold green] [white]RAG retrieval enabled[/white]")
    else:
        console.print("  [bold red]✗[/bold red] [yellow]RAG retriever inactive[/yellow]")

    console.print("  [bold green]✓[/bold green] [white]Citation verification enabled[/white]")
    console.print("  [bold green]✓[/bold green] [white]Conversation context enabled[/white]")

    console.print()
    print_separator("─", "grey50")


def render_answer_box(
    answer: str,
    citations: Optional[List[Any]] = None,
    confidence: str = "HIGH",
    evidence_coverage: float = 0.0,
    warnings: Optional[List[str]] = None,
    latency_ms: float = 0.0,
    status: str = "completed",
    raw_response: Optional[Dict[str, Any]] = None,
) -> None:
    """Render a legal answer inside a structured, terminal-width-aware card."""
    if config.json_mode and raw_response:
        console.print_json(data=raw_response)
        return

    # Check for interrupted generation
    is_interrupted = status == "partial" or "interrupted" in answer.lower()
    
    # Title Box
    header = Text("LegalMindAI Answer", style="bold cyan")
    panel = Panel(header, box=ROUNDED, border_style="cyan", padding=(0, 1))
    console.print()
    console.print(panel)
    console.print()

    # Markdown rendered answer
    if answer:
        try:
            md = Markdown(answer)
            console.print(md)
        except Exception:
            console.print(answer)
    else:
        console.print("[italic grey70]No textual answer provided.[/italic grey70]")

    console.print()

    # Citations / Sources Section
    if citations:
        console.print("[bold cyan]Sources:[/bold cyan]")
        for idx, src in enumerate(citations, start=1):
            if isinstance(src, dict):
                title = src.get("title") or src.get("act_name") or src.get("source") or f"Source {idx}"
                sec = src.get("section") or src.get("section_number") or src.get("article_number")
                url = src.get("url") or src.get("source_url")
                
                label = f"[{idx}] {title}"
                if sec:
                    label += f" — Section/Article {sec}"
                if url:
                    label += f" ({url})"
                console.print(f"[grey85]{label}[/grey85]")
            elif isinstance(src, str):
                console.print(f"[grey85][{idx}] {src}[/grey85]")
        console.print()
    else:
        console.print("[grey60]Sources: None verified for this turn[/grey60]")
        console.print()

    # Confidence & Metrics Bar
    conf_color = "green" if confidence.upper() == "HIGH" else ("yellow" if confidence.upper() == "MEDIUM" else "red")
    coverage_str = f"{evidence_coverage:.0f}%" if evidence_coverage else "N/A"
    
    metrics_line = (
        f"[bold grey70]Confidence:[/bold grey70] [{conf_color}]{confidence.upper()}[/{conf_color}]   "
        f"[bold grey70]Evidence Coverage:[/bold grey70] [white]{coverage_str}[/white]   "
    )
    if latency_ms > 0:
        metrics_line += f"[bold grey70]Latency:[/bold grey70] [white]{latency_ms:.0f}ms[/white]"

    console.print(metrics_line)

    # Warnings
    if warnings:
        for w in warnings:
            console.print(f"[bold yellow][WARNING][/bold yellow] [yellow]{w}[/yellow]")
    else:
        console.print("[bold grey70]Warnings:[/bold grey70] [white]None[/white]")

    if is_interrupted:
        console.print("[bold yellow][WARNING] Answer generation was interrupted.[/bold yellow]")
        console.print("[bold cyan][INFO] The displayed answer may be incomplete.[/bold cyan]")

    console.print()
    print_separator("─", "grey50")


def render_document_analysis(
    file_name: str,
    doc_type: str,
    extracted_text: str,
    answer: str,
    confidence: str = "MEDIUM",
    sources: Optional[List[Any]] = None,
    warnings: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Render structured document/image analysis output."""
    console.print()
    console.print("[bold cyan]Document Analysis[/bold cyan]")
    print_separator("─", "grey50")
    console.print()

    console.print("[bold white]File:[/bold white]")
    console.print(f"  {file_name}")
    console.print()

    console.print("[bold white]Document Type:[/bold white]")
    console.print(f"  [cyan]{doc_type}[/cyan]")
    console.print()

    if metadata:
        for k, v in metadata.items():
            if v:
                console.print(f"[bold white]{k}:[/bold white]")
                console.print(f"  {v}")
        console.print()

    console.print("[bold white]Extracted Text Summary:[/bold white]")
    snippet = extracted_text.strip()
    if len(snippet) > 400:
        snippet = snippet[:400] + "\n... [truncated]"
    console.print(Panel(snippet, box=ROUNDED, border_style="grey50", padding=(0, 1)))
    console.print()

    console.print("[bold white]Answer / Analysis:[/bold white]")
    try:
        console.print(Markdown(answer))
    except Exception:
        console.print(answer)
    console.print()

    conf_color = "green" if confidence.upper() == "HIGH" else "yellow"
    console.print(f"[bold white]Confidence:[/bold white] [{conf_color}]{confidence.upper()}[/{conf_color}]")
    console.print()

    console.print("[bold white]Sources:[/bold white]")
    if sources:
        for s in sources:
            console.print(f"  • {s}")
    else:
        console.print("  • Extracted from uploaded document")
    console.print()

    console.print("[bold white]Warnings:[/bold white]")
    if warnings:
        for w in warnings:
            console.print(f"  [yellow]• {w}[/yellow]")
    else:
        console.print("  • None")
    console.print()
    print_separator("─", "grey50")


def render_shortcuts() -> None:
    """Render the shortcuts help cheatsheet."""
    console.print()
    console.print("[bold cyan]Shortcuts[/bold cyan]")
    print_separator("─", "grey50")
    
    table = Table(box=None, padding=(0, 2), show_header=False)
    table.add_column("Command", style="bold green")
    table.add_column("Description", style="white")

    shortcuts = [
        ("?", "Show shortcuts"),
        ("/help", "Show detailed help"),
        ("/clear", "Clear the terminal and reprint banner"),
        ("/reset", "Reset current conversation context"),
        ("/source", "Show sources from the latest answer"),
        ("/sources", "Show all available sources in current session"),
        ("/status", "Show comprehensive backend status"),
        ("/model", "Show active model and backend architecture"),
        ("/voice", "Start interactive voice typing"),
        ("/upload", "Upload an image or document into conversation"),
        ("/ocr", "Show extracted text from the active document/image"),
        ("/json", "Toggle structured JSON response mode"),
        ("/stream", "Toggle streaming mode (where supported)"),
        ("/summary", "Summarize the latest answer"),
        ("/speak", "Read the latest answer aloud (TTS)"),
        ("/history", "View recent conversations from backend"),
        ("/exit", "Exit the interactive CLI"),
    ]

    for cmd, desc in shortcuts:
        table.add_row(cmd, desc)

    console.print(table)
    console.print()
    print_separator("─", "grey50")


def render_status(
    client_health: Dict[str, Any],
    stats: Dict[str, Any],
    voice_available: bool,
    ocr_available: bool,
    tts_available: bool,
) -> None:
    """Render comprehensive diagnostic status dashboard."""
    console.print()
    console.print("[bold cyan]LegalMindAI System Status[/bold cyan]")
    print_separator("─", "grey50")
    console.print()

    table = Table(box=SIMPLE, border_style="grey50", show_header=True)
    table.add_column("Component / Parameter", style="bold white")
    table.add_column("Value / State", style="cyan")

    table.add_row("Backend URL", config.server_url)
    table.add_row(
        "Backend Availability",
        "[bold green]Connected[/bold green]" if client_health.get("backend_connected") else "[bold red]Disconnected[/bold red]"
    )
    table.add_row(
        "Model Loading Status",
        "[bold green]Loaded[/bold green]" if client_health.get("model_loaded") else "[bold red]Unavailable[/bold red]"
    )
    table.add_row("Active Model", config.default_model)
    table.add_row(
        "Retriever Status",
        "[bold green]Active[/bold green]" if client_health.get("retriever_loaded") else "[bold yellow]Inactive[/bold yellow]"
    )
    table.add_row("Citation Verification", "[bold green]Available[/bold green]")
    table.add_row("Active Conversation ID", str(config.active_conversation_id or "None (New)"))
    table.add_row("Active Document", str(config.active_document_name or "None"))
    table.add_row("Last Response Latency", f"{config.last_latency_ms:.0f} ms" if config.last_latency_ms else "N/A")
    table.add_row("Last Confidence", str(config.last_confidence))
    table.add_row("Streaming Enabled", "Yes" if config.stream else "No")
    table.add_row("JSON Mode", "Enabled" if config.json_mode else "Disabled")
    table.add_row("Voice Input Support", "[bold green]Available[/bold green]" if voice_available else "[bold yellow]Fallback Only[/bold yellow]")
    table.add_row("OCR Pipeline", "[bold green]Available[/bold green]" if ocr_available else "[bold yellow]Fallback / Uncalibrated[/bold yellow]")
    table.add_row("Text-to-Speech", "[bold green]Available[/bold green]" if tts_available else "[bold yellow]Unavailable[/bold yellow]")

    console.print(table)
    console.print()
    print_separator("─", "grey50")


def render_sources_list(citations: List[Any]) -> None:
    """Render a dedicated sources breakdown."""
    console.print()
    console.print("[bold cyan]Retrieved Legal Sources[/bold cyan]")
    print_separator("─", "grey50")
    console.print()

    if not citations:
        console.print("[bold cyan][INFO][/bold cyan] No verified sources were returned for the latest question.")
        console.print()
        print_separator("─", "grey50")
        return

    for idx, c in enumerate(citations, start=1):
        if isinstance(c, dict):
            title = c.get("title") or c.get("act_name") or c.get("source") or f"Source {idx}"
            sec = c.get("section") or c.get("section_number") or c.get("article_number") or ""
            text = c.get("text") or c.get("content") or ""
            url = c.get("url") or c.get("source_url") or ""
            doc_type = c.get("doc_type") or c.get("document_type") or "Legislation"

            console.print(f"[bold white][{idx}] {title}[/bold white] [grey70]({doc_type})[/grey70]")
            if sec:
                console.print(f"    [cyan]Section/Article:[/cyan] {sec}")
            if url:
                console.print(f"    [grey60]Source Link:[/grey60] {url}")
            if text:
                snippet = text.strip()[:250].replace("\n", " ")
                console.print(f"    [italic grey80]\"{snippet}...\"[/italic grey80]")
            console.print()
        else:
            console.print(f"[{idx}] {c}")
            console.print()

    print_separator("─", "grey50")


def render_error(message: str) -> None:
    """Print a standardized error message."""
    if not message.startswith("[ERROR]"):
        message = f"[ERROR] {message}"
    console.print(f"[bold red]{message}[/bold red]")


def render_warning(message: str) -> None:
    """Print a standardized warning message."""
    if not message.startswith("[WARNING]"):
        message = f"[WARNING] {message}"
    console.print(f"[bold yellow]{message}[/bold yellow]")


def render_info(message: str) -> None:
    """Print a standardized info message."""
    if not message.startswith("[INFO]"):
        message = f"[INFO] {message}"
    console.print(f"[bold cyan]{message}[/bold cyan]")
