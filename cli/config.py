"""Configuration settings for the LegalMindAI CLI assistant."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CLIConfig:
    """Runtime configuration for LegalMindAI CLI."""

    server_url: str = os.getenv("LEGALMIND_SERVER", "http://127.0.0.1:8080")
    stream: bool = True
    json_mode: bool = False
    debug: bool = False
    timeout: float = 180.0
    connect_timeout: float = 10.0
    user_name: str = os.getenv("USER", "Local User")
    default_model: str = "Qwen/Qwen3.6-35B-A3B"

    # Context state
    active_conversation_id: Optional[str] = None
    active_document_id: Optional[str] = None
    active_document_name: Optional[str] = None
    active_document_type: Optional[str] = None
    active_document_text: Optional[str] = None
    last_response: Optional[dict] = None
    last_citations: list = field(default_factory=list)
    last_confidence: str = "UNKNOWN"
    last_latency_ms: float = 0.0

    # Limits
    max_image_size_bytes: int = 15 * 1024 * 1024  # 15 MB
    max_doc_size_bytes: int = 25 * 1024 * 1024    # 25 MB

    # Supported formats
    supported_image_exts: tuple = (".png", ".jpg", ".jpeg", ".webp")
    supported_doc_exts: tuple = (".pdf", ".docx", ".txt")

    # Legal Disclaimer
    legal_disclaimer: str = (
        "This analysis is based on the uploaded document and retrieved legal sources. "
        "It is informational and is not a substitute for advice from a qualified legal professional."
    )


# Global default configuration instance
config = CLIConfig()
