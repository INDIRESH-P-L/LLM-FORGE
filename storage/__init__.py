"""
storage
=======
Durable persistence for LegalMind AI.

`chat_store` owns the chat database (conversations, messages, authorities,
full-text search). `exporters` renders a stored conversation for download.
Nothing here imports the model, the retriever, or anything from `scripts/`.
"""

from storage import chat_store, exporters  # noqa: F401

__all__ = ["chat_store", "exporters"]
