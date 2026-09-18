"""
legalmind_cli/renderer.py
=========================
Advanced Terminal UI & Rendering Engine for LegalMindAI CLI.
Designed with rich aesthetics, glass-like terminal panels, and visual hierarchy.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, List, Optional

try:
    from rich import box
    from rich.columns import Columns
    from rich.console import Console, Group
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.tree import Tree
    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class SimpleConsole:
        def print(self, *args, **kwargs):
            text = " ".join(str(a) for a in args)
            print(text)
    console = SimpleConsole()  # type: ignore


ASCII_LOGO = """[bold cyan]
██╗     ███████╗ ██████╗  █████╗ ██╗     ███╗   ███╗██╗███╗   ██╗██████╗ 
██║     ██╔════╝██╔════╝ ██╔══██╗██║     ████╗ ████║██║████╗  ██║██╔══██╗
██║     █████╗  ██║  ███╗███████║██║     ██╔████╔██║██║██╔██╗ ██║██║  ██║
██║     ██╔══╝  ██║   ██║██╔══██║██║     ██║╚██╔╝██║██║██║╚██╗██║██║  ██║
███████╗███████╗╚██████╔╝██║  ██║███████╗██║ ╚═╝ ██║██║██║ ╚████║██████╔╝
╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝ [/bold cyan]
[bold white]      ⚖️   I N D I A N   L E G A L   I N T E L L I G E N C E   S U I T E[/bold white]
"""


def render_banner(
    model: str = "Qwen3.6-35B-A3B (BF16)",
    adapter: str = "lora-legal-v2 (Active)",
    corpus: str = "Constitution • 15,057 Acts • 11,040 SC Judgments",
    server_url: str = "https://localhost:8443",
) -> None:
    """Prints the state-of-the-art LegalMind AI terminal dashboard."""
    if not HAS_RICH:
        print("\nLegalMind AI CLI — Indian Legal Intelligence Suite\n")
        return

    console.print(ASCII_LOGO)

    # Status Grid
    status_table = Table(box=None, padding=(0, 2), show_header=False, expand=True)
    status_table.add_column("Key", style="bold cyan", width=14)
    status_table.add_column("Val", style="bold white")
    status_table.add_column("Key2", style="bold cyan", width=14)
    status_table.add_column("Val2", style="bold white")

    status_table.add_row("🧠 Model:", f"[white]{model}[/white] on [bold green]cuda:2[/bold green]", "⚡ Adapter:", f"[green]{adapter}[/green]")
    status_table.add_row("🔍 Retrieval:", "Hybrid BGE-M3 + BM25 on [bold green]cuda:4[/bold green]", "📚 Corpus:", f"[white]{corpus}[/white]")
    status_table.add_row("🌐 Server:", f"[bright_blue]{server_url}[/bright_blue] [dim](Online)[/dim]", "🏛️ Suite:", "[yellow]Drafter • Precedent Graph • Moot • Dossier[/yellow]")

    panel = Panel(
        status_table,
        title="[bold bright_blue]SYSTEM TELEMETRY & HARDWARE COCKPIT[/bold bright_blue]",
        border_style="bright_blue",
        box=box.ROUNDED,
        subtitle="[dim]Type [bold white]/help[/bold white] for command suite • [bold white]exit[/bold white] to quit[/dim]",
    )
    console.print(panel)
    console.print()


def stream_answer_text(text: str, delay: float = 0.004) -> None:
    """Simulates streaming answer output for realistic interactive CLI experience."""
    for word in text.split(" "):
        sys.stdout.write(word + " ")
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()


def render_answer_response(
    response: Dict[str, Any],
    as_json: bool = False,
    show_sources: bool = True,
    stream: bool = False,
) -> None:
    """Renders the complete answer with confidence, citations, and metadata."""
    if as_json:
        print(json.dumps(response, indent=2))
        return

    if not HAS_RICH:
        print(response.get("answer", ""))
        return

    answer = response.get("answer", "")
    citations = response.get("citations", [])
    confidence = (response.get("confidence") or "medium").upper()
    coverage = response.get("evidence_coverage", 0.0)
    warnings = response.get("warnings", []) or response.get("legal_warnings", [])
    latency = response.get("total_latency", 0.0)

    conf_color = "green" if confidence == "HIGH" else ("yellow" if confidence == "MEDIUM" else "red")

    # Main Analysis Panel
    try:
        md = Markdown(answer)
        answer_renderable = md
    except Exception:
        answer_renderable = Text(answer)

    console.print()
    answer_panel = Panel(
        answer_renderable,
        title="[bold cyan]⚖️  JUDICIAL ANALYSIS & RATIO DECIDENDI[/bold cyan]",
        subtitle=f"[{conf_color}]● {confidence} CONFIDENCE[/{conf_color}] • [dim]Evidence Coverage: {round(coverage * 100)}%[/dim] • [dim]Latency: {latency:.2f}s[/dim]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print(answer_panel)

    # Warnings
    if warnings:
        for w in warnings:
            console.print(f"[bold yellow]⚠️  STATUTORY NOTICE:[/bold yellow] [yellow]{w}[/yellow]")

    # Table of Authorities
    if show_sources and citations:
        sources_table = Table(
            title="[bold white]📚 Verified Statutory Authorities & Supreme Court Precedents[/bold white]",
            box=box.SIMPLE_HEAVY,
            header_style="bold cyan",
            expand=True,
        )
        sources_table.add_column("#", style="dim cyan", width=4, justify="center")
        sources_table.add_column("Authority / Case Law", style="bold white")
        sources_table.add_column("Type / Citation", style="green")

        for idx, cit in enumerate(citations, 1):
            parts = str(cit).split(" — ")
            title = parts[0].strip()
            detail = " — ".join(parts[1:]) if len(parts) > 1 else "Binding Statutory Provision"
            sources_table.add_row(str(idx), title, detail)

        console.print(sources_table)
    console.print()


def render_draft_preview(draft: Dict[str, Any], saved_path: Optional[str] = None) -> None:
    """Renders a generated legal document draft."""
    if not HAS_RICH:
        print(draft.get("draft_text", ""))
        return

    text = draft.get("draft_text", "")
    title = draft.get("title", "Statutory Draft")
    statute = draft.get("statute", "Indian Law")
    deadline = draft.get("statutory_deadline", "As per statute")

    meta_text = f"[bold yellow]Governing Statute:[/bold yellow] {statute}  |  [bold yellow]Statutory Window:[/bold yellow] {deadline}"
    if saved_path:
        meta_text += f"\n[bold green]Saved to Disk:[/bold green] {saved_path}"

    panel = Panel(
        Text(text, style="white"),
        title=f"[bold green]📜 {title.upper()}[/bold green]",
        subtitle=meta_text,
        border_style="green",
        box=box.DOUBLE,
        padding=(1, 2),
    )
    console.print()
    console.print(panel)
    console.print()


def render_redline_results(results: Dict[str, Any]) -> None:
    """Renders a contract risk audit scorecard with safe revision."""
    if not HAS_RICH:
        print(results)
        return

    score = results.get("risk_score", 75)
    level = (results.get("risk_level") or results.get("risk_rating") or "HIGH").upper()
    level_color = "red" if "HIGH" in level or "CRITICAL" in level else ("yellow" if "MED" in level else "green")
    issues = results.get("issues", []) or results.get("findings", [])
    safe_draft = results.get("safe_draft", "")

    console.print()
    # Summary Header
    header_table = Table(box=box.ROUNDED, border_style=level_color, expand=True)
    header_table.add_column("Risk Category", style="bold white")
    header_table.add_column("Score", justify="center", style=f"bold {level_color}")
    header_table.add_column("Rating", justify="center", style=f"bold {level_color}")

    header_table.add_row("Indian Contract Act & Precedent Audit", f"{score}/100", f"● {level} RISK")
    console.print(header_table)

    # Issues Table
    if issues:
        issues_table = Table(
            title="[bold red]⚠️  Identified Statutory Vulnerabilities[/bold red]",
            box=box.SIMPLE_HEAVY,
            expand=True,
        )
        issues_table.add_column("Category", style="bold yellow", width=22)
        issues_table.add_column("Vulnerability Analysis", style="white")
        issues_table.add_column("Governing Indian Law / Precedent", style="cyan", width=34)

        for iss in issues:
            cat = iss.get("category", "Statutory Risk")
            desc = iss.get("description", "")
            auth = iss.get("precedent_or_statute") or iss.get("indian_law") or "Indian Contract Act, 1872"
            issues_table.add_row(cat, desc, auth)

        console.print(issues_table)

    # Safe Revision
    if safe_draft:
        rev_panel = Panel(
            Text(safe_draft, style="bright_white"),
            title="[bold green]🛡️  RECOMMENDED ENFORCEABLE SAFE REVISION[/bold green]",
            subtitle="[dim]Tailored for balanced protection and Indian law compliance[/dim]",
            border_style="green",
            box=box.ROUNDED,
            padding=(1, 2),
        )
        console.print(rev_panel)
    console.print()


def render_precedent_tree(graph_data: Dict[str, Any]) -> None:
    """Renders an interactive tree of precedent relationships."""
    if not HAS_RICH:
        print(graph_data)
        return

    root_name = graph_data.get("root", "Landmark Case")
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    tree = Tree(f"[bold yellow]🏛️  {root_name}[/bold yellow] [dim](Landmark Root)[/dim]")

    for n in nodes:
        if n.get("id") == root_name:
            continue
        case_id = n.get("id", "")
        # Find edge
        edge = next((e for e in edges if e.get("target") == case_id or e.get("to") == case_id), {})
        rel = (edge.get("relation") or edge.get("type") or "cited").lower()

        if "overrule" in rel:
            badge = "[bold red]🔴 OVERRULED[/bold red]"
        elif "follow" in rel:
            badge = "[bold green]🟢 FOLLOWED[/bold green]"
        elif "distinguish" in rel:
            badge = "[bold yellow]🟡 DISTINGUISHED[/bold yellow]"
        else:
            badge = "[bold blue]🔵 RELIED UPON[/bold blue]"

        court_year = f"[dim]({n.get('court', 'SC')}, {n.get('year', '')})[/dim]"
        tree.add(f"{badge} [bold white]{case_id}[/bold white] {court_year}")

    panel = Panel(
        tree,
        title="[bold cyan]⚖️  PRECEDENT RELATIONSHIP & DOCTRINE LINEAGE GRAPH[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print()
    console.print(panel)
    console.print()


def render_moot_exchange(moot_res: Dict[str, Any]) -> None:
    """Renders simulated judicial cross-examination with Advocacy Scorecard."""
    if not HAS_RICH:
        print(moot_res)
        return

    bench_title = moot_res.get("bench_title", "Supreme Court Bench")
    judge_name = moot_res.get("presiding_judge", "Hon'ble Judge")
    interjection = moot_res.get("judicial_interjection", "")
    scorecard = moot_res.get("scorecard", {})
    rebuttal_tip = moot_res.get("rebuttal_tip", "")
    counter_prec = moot_res.get("counter_precedent", "")

    # Bench Dialogue Panel
    bench_panel = Panel(
        Text(f'"{interjection}"', style="italic bright_white"),
        title=f"[bold yellow]🏛️  THE BENCH ({judge_name.upper()})[/bold yellow]",
        subtitle=f"[dim]{bench_title}[/dim]",
        border_style="yellow",
        box=box.DOUBLE,
        padding=(1, 2),
    )
    console.print()
    console.print(bench_panel)

    # Scorecard Table
    score_table = Table(
        title="[bold cyan]📊 Real-Time Advocacy Scorecard[/bold cyan]",
        box=box.ROUNDED,
        border_style="cyan",
        expand=True,
    )
    score_table.add_column("Evaluation Dimension", style="bold white")
    score_table.add_column("Score", justify="center", style="bold green")
    score_table.add_column("Readiness & Bench Disposition", justify="center", style="bold yellow")

    overall = scorecard.get("overall_score", 78)
    rating = scorecard.get("readiness_rating", "Competent Submission")

    score_table.add_row("Constitutional Grounding", f"{scorecard.get('constitutional_grounding', 80)}%", "Threshold Examined")
    score_table.add_row("Statutory Precision", f"{scorecard.get('statutory_precision', 75)}%", "Anchors Verified")
    score_table.add_row("Precedent Authority", f"{scorecard.get('precedent_authority', 78)}%", "Binding Rulings Tested")
    score_table.add_row("[bold cyan]Composite Advocacy Score[/bold cyan]", f"[bold green]{overall}/100[/bold green]", f"[bold yellow]{rating}[/bold yellow]")

    console.print(score_table)

    # Strategy Hint
    if rebuttal_tip:
        strat_panel = Panel(
            f"[bold yellow]Strategic Rebuttal:[/bold yellow] {rebuttal_tip}\n[bold cyan]Counter Authority:[/bold cyan] {counter_prec}",
            title="[bold yellow]💡 BENCH COUNTER-STRATEGY[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
        )
        console.print(strat_panel)
    console.print()


def render_dossier_summary(dossier: Dict[str, Any], pdf_path: Optional[str] = None) -> None:
    """Renders advocate case dossier summary."""
    if not HAS_RICH:
        print(dossier)
        return

    title = dossier.get("case_title", "Cause Title")
    court = dossier.get("court", "Supreme Court of India")
    questions = dossier.get("issues_framed", [])
    authorities = dossier.get("table_of_authorities", [])

    q_text = "\n".join(f"  {idx}. {q}" for idx, q in enumerate(questions, 1))
    auth_text = "\n".join(f"  • [bold]{a.get('case_name', 'Precedent')}[/bold] ({a.get('citation', '')}): {a.get('ratio', '')}" for a in authorities[:5])

    content = f"[bold cyan]Court:[/bold cyan] {court}\n\n[bold yellow]Questions of Law Framed:[/bold yellow]\n{q_text}\n\n[bold green]Table of Authorities Cited:[/bold green]\n{auth_text}"
    if pdf_path:
        content += f"\n\n[bold green]✅ Official Court-Ready PDF Saved:[/bold green] [bold white underline]{pdf_path}[/bold white underline]"

    panel = Panel(
        content,
        title=f"[bold bright_blue]📁 ADVOCATE'S BENCH MEMORANDUM: {title.upper()}[/bold bright_blue]",
        border_style="bright_blue",
        box=box.ROUNDED,
        padding=(1, 2),
    )
    console.print()
    console.print(panel)
    console.print()


def render_help_menu() -> None:
    """Displays the comprehensive interactive CLI help menu."""
    table = Table(
        title="[bold cyan]⚖️  LegalMindAI Interactive CLI Command Suite[/bold cyan]",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Command / Shortcut", style="bold white", width=26)
    table.add_column("Description & Usage", style="white")

    table.add_row("ask <question>", "Direct legal research with evidence-backed citations")
    table.add_row("/draft [notice|bail|nda]", "Draft statutory demand notice, bail petition, or commercial NDA")
    table.add_row("/redline <clause>", "Audit contract clauses for Indian statutory risk & get safe revision")
    table.add_row("/precedent <case>", "Explore landmark Supreme Court precedent graph and treatment history")
    table.add_row("/moot <argument>", "Simulate judicial cross-examination before Supreme Court Bench")
    table.add_row("/dossier <case_title>", "Generate structured court memorandum and export A4 PDF brief")
    table.add_row("/doc <path>", "Attach PDF, DOCX, or TXT document context")
    table.add_row("/image <path>", "Attach legal notice or judgment image context")
    table.add_row("/lang <en|hi|ta>", "Switch assistant language (English, Hindi, Tamil)")
    table.add_row("/health", "Display DGX model status, LoRA adapter, and server telemetry")
    table.add_row("/clear", "Clear document context and redraw cockpit banner")
    table.add_row("exit / quit / :q", "Exit the CLI session")

    console.print()
    console.print(table)
    console.print()


def render_input_box_top(active_doc_id: Optional[str] = None) -> None:
    """Renders the top rounded border and header for the Counsel Input Box."""
    if not HAS_RICH:
        return
    term_width = console.width or 88
    target_width = max(min(term_width - 2, 92), 50)
    title = " 💬 Counsel Input Box "
    if active_doc_id:
        hint = f"[Context: {active_doc_id[:20]}] "
    else:
        hint = "[Type query, /draft, /moot, /help] "

    header_len = 3 + len(title) + len(hint) + 1
    fill = max(target_width - header_len, 4)
    console.print(f"\n[bold cyan]╭──[bold white]{title}[/bold white][dim]{hint}[/dim]{'─' * fill}╮[/bold cyan]")


def render_input_box_bottom() -> None:
    """Renders the bottom rounded border and action subtitle for the Counsel Input Box."""
    if not HAS_RICH:
        return
    term_width = console.width or 88
    target_width = max(min(term_width - 2, 92), 50)
    sub = " Enter to submit • /help for commands • exit to quit "
    bot_len = 4 + len(sub) + 1
    fill = max(target_width - bot_len, 4)
    console.print(f"[bold cyan]╰───[dim]{sub}[/dim]{'─' * fill}╯[/bold cyan]\n")


def render_error(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[bold red]❌ ERROR:[/bold red] {msg}")
    else:
        print(f"[ERROR] {msg}")


def render_warning(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[bold yellow]⚠️  WARNING:[/bold yellow] {msg}")
    else:
        print(f"[WARNING] {msg}")


def render_info(msg: str) -> None:
    if HAS_RICH:
        console.print(f"[bold cyan]ℹ️  {msg}[/bold cyan]")
    else:
        print(f"[INFO] {msg}")
