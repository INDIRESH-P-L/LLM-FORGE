"""Full-screen interactive REPL shell for LegalMindAI CLI."""

import sys
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from cli.client import LegalMindClient
from cli.commands import ask_question
from cli.config import config
from cli.renderer import (
    console,
    get_terminal_width,
    print_separator,
    render_banner,
    render_error,
    render_info,
)
from cli.shortcuts import ShortcutManager

# Prompt-toolkit styling
pt_style = Style.from_dict({
    "prompt": "bold cyan",
    "bottom-toolbar": "bg:#1e293b #94a3b8",
})


def bottom_toolbar():
    """Render bottom status line matching reference CLI design."""
    width = get_terminal_width()
    doc_str = f" | Doc: {config.active_document_name}" if config.active_document_name else ""
    left_text = f"? for shortcuts{doc_str}"
    
    model_name = config.default_model.split("/")[-1]
    right_text = model_name

    padding = max(2, width - len(left_text) - len(right_text) - 4)
    return HTML(f" <b>{left_text}</b>{' ' * padding}<b>{right_text}</b> ")


def start_interactive_session(client: Optional[LegalMindClient] = None) -> None:
    """Launch the full-screen interactive LegalMindAI terminal session."""
    if client is None:
        client = LegalMindClient()

    # Probe backend health
    health = client.check_health()
    stats = client.get_stats()

    # Clear screen and display startup banner
    console.clear()
    render_banner(health, stats)

    shortcut_mgr = ShortcutManager(client)
    history = InMemoryHistory()
    session: PromptSession = PromptSession(history=history, style=pt_style)

    while True:
        try:
            # Interactive prompt
            user_input = session.prompt(
                "> ",
                bottom_toolbar=bottom_toolbar,
            ).strip()

            if not user_input:
                continue

            # Check if this input is a shortcut or slash command
            if shortcut_mgr.handle(user_input):
                continue

            # Normal continuous conversation question
            ask_question(client, user_input)

        except KeyboardInterrupt:
            # Ctrl+C resets line rather than aborting session
            console.print("\n[grey60]Input cancelled. Type /exit or Ctrl+D to quit.[/grey60]")
            continue
        except EOFError:
            # Ctrl+D clean exit
            console.print("\n[grey70]Exiting LegalMindAI CLI. Goodbye![/grey70]")
            break
        except Exception as e:
            render_error(f"[ERROR] Session error: {e}")
