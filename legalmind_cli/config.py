"""
legalmind_cli/config.py
=======================
Configuration and environment options for LegalMindAI CLI.
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CLIConfig:
    server_url: str = os.environ.get("LEGALMIND_SERVER_URL", "https://localhost:8443")
    timeout_seconds: int = int(os.environ.get("LEGALMIND_TIMEOUT", "180"))
    model_name: str = "Qwen3.6-35B-A3B"
    default_mode: str = "detailed"
    stream_output: bool = True
    json_output: bool = False
    show_sources: bool = True
    history_file: Path = Path.home() / ".legalmind_history"
    supported_image_exts: tuple = (".jpg", ".jpeg", ".png", ".webp")
    supported_doc_exts: tuple = (".pdf", ".docx", ".txt")


config = CLIConfig()
