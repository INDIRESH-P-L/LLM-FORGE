"""
legalmind_cli/client.py
=======================
Backend API Client for LegalMindAI CLI.
Handles HTTP requests, timeouts, retries, and network errors gracefully.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from legalmind_cli.config import config


class ClientError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class LegalMindClient:
    """Client for communicating with LegalMindAI REST endpoints."""

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[int] = None):
        self.session = requests.Session()
        self.session.verify = False
        self.timeout = timeout or config.timeout_seconds

        resolved_url = (base_url or config.server_url).rstrip("/")
        # If user didn't explicitly specify a URL or env var, auto-detect active server (8443 vs 8080)
        if not base_url and "LEGALMIND_SERVER_URL" not in os.environ:
            for candidate in ("https://localhost:8443", "https://127.0.0.1:8443", "http://127.0.0.1:8443", "http://localhost:8443", "http://127.0.0.1:8080", "http://localhost:8080"):
                try:
                    r = self.session.get(f"{candidate}/health", timeout=1.0)
                    if r.status_code == 200:
                        resolved_url = candidate
                        break
                except Exception:
                    pass
        self.base_url = resolved_url

    def ensure_service_ready(self, wait_seconds: int = 30) -> bool:
        """
        Ensures the background LegalMindAI service is running and ready.
        If inactive or stopped, attempts to start it via systemd user service.
        """
        # 1. Quick probe
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=1.5)
            if r.status_code == 200 and r.json().get("model_loaded"):
                return True
        except Exception:
            pass

        # 2. Check systemd service status
        try:
            chk = subprocess.run(
                ["systemctl", "--user", "is-active", "legalmind.service"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            status = chk.stdout.strip()
            if status != "active":
                # Attempt auto-start
                subprocess.run(
                    ["systemctl", "--user", "start", "legalmind.service"],
                    capture_output=True,
                    timeout=5,
                )
        except Exception:
            pass

        # 3. Poll /health until model is loaded
        start_time = time.time()
        while time.time() - start_time < wait_seconds:
            try:
                r = self.session.get(f"{self.base_url}/health", timeout=2.0)
                if r.status_code == 200 and r.json().get("model_loaded"):
                    return True
            except Exception:
                pass
            time.sleep(1.5)
        return False

    def check_health(self) -> Dict[str, Any]:
        """Checks API server connectivity and model status."""
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            raise ClientError(f"Server returned status {r.status_code}", status_code=r.status_code)
        except requests.exceptions.ConnectionError:
            # Try auto-recovery once
            if self.ensure_service_ready(wait_seconds=15):
                return self.session.get(f"{self.base_url}/health", timeout=self.timeout).json()
            raise ClientError(f"Cannot connect to LegalMindAI backend at {self.base_url}. Is the server running?")
        except requests.exceptions.Timeout:
            raise ClientError(f"Request timed out connecting to {self.base_url}")
        except Exception as e:
            if isinstance(e, ClientError):
                raise
            raise ClientError(f"Health check failed: {e}")

    def query_legal(self, query: str, top_k: int = 7, mode: str = "detailed") -> Dict[str, Any]:
        """Dispatches a legal research query to the backend with auto-retry."""
        url = f"{self.base_url}/query"
        payload = {"query": query, "top_k": top_k}

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                r = self.session.post(url, json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (502, 503, 504) and attempt < max_attempts:
                    time.sleep(2.0)
                    continue
                err_msg = r.text
                try:
                    err_msg = r.json().get("detail", err_msg)
                except Exception:
                    pass
                raise ClientError(f"Backend returned error ({r.status_code}): {err_msg}", status_code=r.status_code)
            except requests.exceptions.ConnectionError:
                if attempt < max_attempts:
                    # Check if service needs to be booted
                    self.ensure_service_ready(wait_seconds=10)
                    time.sleep(1.5)
                    continue
                raise ClientError(f"Cannot connect to {self.base_url}. Ensure the LegalMindAI server is online.")
            except requests.exceptions.Timeout:
                raise ClientError(f"Query request timed out after {self.timeout}s.")
            except Exception as e:
                if isinstance(e, ClientError):
                    raise
                raise ClientError(f"Query execution failed: {e}")

    def upload_image(self, file_path: Path) -> Dict[str, Any]:
        """Uploads an image file to /images/upload."""
        url = f"{self.base_url}/images/upload"
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "image/jpeg")}
            try:
                r = self.session.post(url, files=files, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                raise ClientError(f"Image upload failed ({r.status_code}): {r.text}", status_code=r.status_code)
            except Exception as e:
                if isinstance(e, ClientError):
                    raise
                raise ClientError(f"Image upload error: {e}")

    def ask_image(self, image_id: str, question: str) -> Dict[str, Any]:
        """Asks a question regarding an uploaded image."""
        url = f"{self.base_url}/images/ask"
        payload = {"image_id": image_id, "question": question}
        try:
            r = self.session.post(url, json=payload, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            raise ClientError(f"Image Q&A failed ({r.status_code}): {r.text}", status_code=r.status_code)
        except Exception as e:
            if isinstance(e, ClientError):
                raise
            raise ClientError(f"Image Q&A error: {e}")

    def upload_document(self, file_path: Path) -> Dict[str, Any]:
        """Uploads a PDF, DOCX, or TXT file to /documents/upload."""
        url = f"{self.base_url}/documents/upload"
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f, "application/octet-stream")}
            try:
                r = self.session.post(url, files=files, timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                raise ClientError(f"Document upload failed ({r.status_code}): {r.text}", status_code=r.status_code)
            except Exception as e:
                if isinstance(e, ClientError):
                    raise
                raise ClientError(f"Document upload error: {e}")

    def analyze_document(self, document_id: str) -> Dict[str, Any]:
        """Requests structured analysis of an uploaded document."""
        url = f"{self.base_url}/documents/analyze"
        try:
            r = self.session.post(url, json={"document_id": document_id}, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            raise ClientError(f"Document analysis failed ({r.status_code}): {r.text}", status_code=r.status_code)
        except Exception as e:
            if isinstance(e, ClientError):
                raise
            raise ClientError(f"Document analysis error: {e}")

    def ask_document(self, document_id: str, question: str) -> Dict[str, Any]:
        """Asks a question regarding an uploaded document."""
        url = f"{self.base_url}/documents/ask"
        payload = {"document_id": document_id, "question": question}
        try:
            r = self.session.post(url, json=payload, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            raise ClientError(f"Document Q&A failed ({r.status_code}): {r.text}", status_code=r.status_code)
        except Exception as e:
            if isinstance(e, ClientError):
                raise
            raise ClientError(f"Document Q&A error: {e}")

    def generate_draft(self, document_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generates tailored legal notices, bail applications, or NDAs."""
        url = f"{self.base_url}/api/drafter/generate"
        try:
            r = self.session.post(url, json={"document_type": document_type, "parameters": parameters}, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            raise ClientError(f"Draft generation failed ({r.status_code}): {r.text}", status_code=r.status_code)
        except Exception as e:
            if isinstance(e, ClientError):
                raise
            raise ClientError(f"Drafting error: {e}")

    def review_contract(self, contract_text: str) -> Dict[str, Any]:
        """Audits contract clause against Indian statutory bars."""
        url = f"{self.base_url}/api/drafter/review"
        try:
            r = self.session.post(url, json={"contract_text": contract_text}, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            raise ClientError(f"Contract review failed ({r.status_code}): {r.text}", status_code=r.status_code)
        except Exception as e:
            if isinstance(e, ClientError):
                raise
            raise ClientError(f"Contract audit error: {e}")

    def get_precedent_graph(self, case_name: str) -> Dict[str, Any]:
        """Retrieves precedent network relationships."""
        url = f"{self.base_url}/api/precedents/graph"
        try:
            r = self.session.get(url, params={"case": case_name, "depth": 1}, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            raise ClientError(f"Precedent graph lookup failed ({r.status_code}): {r.text}", status_code=r.status_code)
        except Exception as e:
            if isinstance(e, ClientError):
                raise
            raise ClientError(f"Precedent error: {e}")

    def moot_interject(self, argument: str, bench_type: str = "constitutional", round_num: int = 1, history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Submits argument to simulated Supreme Court bench."""
        url = f"{self.base_url}/api/moot/interject"
        payload = {
            "argument": argument,
            "bench_type": bench_type,
            "round_num": round_num,
            "history": history or [],
        }
        try:
            r = self.session.post(url, json=payload, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            raise ClientError(f"Moot simulation failed ({r.status_code}): {r.text}", status_code=r.status_code)
        except Exception as e:
            if isinstance(e, ClientError):
                raise
            raise ClientError(f"Moot court error: {e}")

    def generate_dossier(self, case_title: str, query: str, answer: str, court: str = "IN THE SUPREME COURT OF INDIA") -> Dict[str, Any]:
        """Synthesizes advocate bench dossier."""
        url = f"{self.base_url}/api/dossier/generate"
        payload = {"case_title": case_title, "query": query, "answer": answer, "court": court}
        try:
            r = self.session.post(url, json=payload, timeout=self.timeout)
            if r.status_code == 200:
                return r.json()
            raise ClientError(f"Dossier generation failed ({r.status_code}): {r.text}", status_code=r.status_code)
        except Exception as e:
            if isinstance(e, ClientError):
                raise
            raise ClientError(f"Dossier error: {e}")

    def export_dossier_pdf(self, case_title: str, query: str, answer: str, court: str = "IN THE SUPREME COURT OF INDIA") -> bytes:
        """Downloads court-ready PDF file."""
        url = f"{self.base_url}/api/dossier/export-pdf"
        payload = {"case_title": case_title, "query": query, "answer": answer, "court": court}
        try:
            r = self.session.post(url, json=payload, timeout=self.timeout)
            if r.status_code == 200:
                return r.content
            raise ClientError(f"PDF export failed ({r.status_code}): {r.text}", status_code=r.status_code)
        except Exception as e:
            if isinstance(e, ClientError):
                raise
            raise ClientError(f"PDF export error: {e}")

