"""
app/api/__init__.py
===================
API routers for multimodal LegalMindAI endpoints and legal engineering suite.
"""

from app.api.speech import router as speech_router
from app.api.images import router as images_router
from app.api.documents import router as documents_router
from app.api.audio import router as audio_router
from app.api.precedents import router as precedents_router
from app.api.drafter import router as drafter_router
from app.api.moot import router as moot_router
from app.api.dossier import router as dossier_router
__all__ = [
    "speech_router",
    "images_router",
    "documents_router",
    "audio_router",
    "precedents_router",
    "drafter_router",
    "moot_router",
    "dossier_router",
]
