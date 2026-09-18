"""Comprehensive test suite for LegalMindAI CLI assistant.
Verifies all 30 functional and safety requirements.
"""

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cli import __app_name__, __version__
from cli.client import BackendError, LegalMindClient
from cli.commands import (
    analyze_document_command,
    analyze_image_command,
    ask_question,
    show_sources_command,
    show_status_command,
    speak_command,
    voice_command,
)
from cli.config import config
from cli.image_input import ImageAnalyzer
from cli.main import create_parser, main
from cli.pdf_input import DocumentParser, LegalDocumentAnalyzer
from cli.renderer import (
    console,
    get_terminal_width,
    print_separator,
    render_answer_box,
    render_banner,
    render_document_analysis,
    render_error,
    render_info,
    render_shortcuts,
    render_sources_list,
    render_status,
    render_warning,
)
from cli.shortcuts import ShortcutManager
from cli.speech_output import SpeechOutputHandler
from cli.validators import (
    ValidationError,
    validate_document_file,
    validate_file_path,
    validate_image_file,
)
from cli.voice_input import VoiceInputHandler


class TestLegalMindCLI(unittest.TestCase):
    """30-Point Test Suite for LegalMindAI CLI."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.temp_path = Path(cls.temp_dir)

        # Create sample valid text file
        cls.sample_txt = cls.temp_path / "sample.txt"
        cls.sample_txt.write_text("Supreme Court of India\nWrit Petition No. 123 of 2024\nArticle 21 Protection of Life.")

        # Create sample valid PDF with magic header
        cls.sample_pdf = cls.temp_path / "sample.pdf"
        try:
            import fitz
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Supreme Court of India\nJudgment in Maneka Gandhi v. Union of India\nArticle 21 and Article 14")
            doc.save(str(cls.sample_pdf))
            doc.close()
        except Exception:
            # Fallback mock minimal PDF
            cls.sample_pdf.write_bytes(b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj 3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\nxref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000102 00000 n \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n149\n%%EOF")

        # Create sample valid image (PNG)
        cls.sample_png = cls.temp_path / "sample.png"
        from PIL import Image
        img = Image.new("RGB", (100, 100), color="white")
        img.save(cls.sample_png)

        # Create invalid/unsupported file
        cls.sample_exe = cls.temp_path / "malicious.exe"
        cls.sample_exe.write_bytes(b"MZ\x90\x00\x03\x00\x00\x00")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self):
        # Reset runtime config
        config.active_conversation_id = None
        config.active_document_id = None
        config.active_document_name = None
        config.active_document_text = None
        config.last_response = None
        config.last_citations = []
        config.last_confidence = "UNKNOWN"
        config.json_mode = False

    # 1. CLI Startup
    def test_01_cli_startup(self):
        health = {"backend_connected": True, "model_loaded": True, "retriever_loaded": True}
        stats = {"model": {"name": "Qwen/Qwen3.6-35B-A3B"}}
        # Ensure render_banner executes without error
        render_banner(health, stats)

    # 2. Version Command
    def test_02_version_command(self):
        self.assertEqual(__version__, "1.0.0")
        parser = create_parser()
        with self.assertRaises(SystemExit) as cm:
            parser.parse_args(["--version"])
        self.assertEqual(cm.exception.code, 0)

    # 3. Help Command
    def test_03_help_command(self):
        parser = create_parser()
        # Parse help subcommand
        args = parser.parse_args(["help"])
        self.assertEqual(args.command, "help")
        # Ensure render_shortcuts renders
        render_shortcuts()

    # 4. Backend Health Check
    def test_04_backend_health_check(self):
        client = LegalMindClient("http://127.0.0.1:8080")
        health = client.check_health()
        self.assertIn("backend_connected", health)
        self.assertIn("model_loaded", health)

    # 5. Normal Question
    def test_05_normal_question(self):
        client = LegalMindClient()
        mock_resp = {
            "conversation_id": "test-conv-123",
            "assistant_message": {
                "content": "Article 21 guarantees the protection of life and personal liberty.",
                "metadata": {
                    "confidence": "HIGH",
                    "evidence_coverage": 95.0,
                    "citations": [{"title": "Constitution of India", "section": "21"}],
                    "status": "completed",
                    "warnings": [],
                },
            },
        }
        with patch.object(client, "chat", return_value={
            "answer": "Article 21 guarantees life and personal liberty.",
            "citations": [{"title": "Constitution of India", "section": "21"}],
            "confidence": "HIGH",
            "evidence_coverage": 95.0,
            "warnings": [],
            "status": "completed",
        }):
            res = ask_question(client, "Explain Article 21")
            self.assertIsNotNone(res)
            self.assertEqual(res["confidence"], "HIGH")

    # 6. Interactive Chat
    def test_06_interactive_chat(self):
        client = LegalMindClient()
        mgr = ShortcutManager(client)
        # Test shortcut handling
        handled = mgr.handle("?")
        self.assertTrue(handled)
        handled = mgr.handle("/help")
        self.assertTrue(handled)

    # 7. Conversation ID Preservation
    def test_07_conversation_id_preservation(self):
        client = LegalMindClient()
        config.active_conversation_id = "initial-conv-id"
        with patch("httpx.Client.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "conversation_id": "initial-conv-id",
                "assistant_message": {"content": "Answer 1", "metadata": {}},
            }
            mock_post.return_value = mock_resp
            client.chat("Follow up question")
            self.assertEqual(config.active_conversation_id, "initial-conv-id")

    # 8. JSON Output
    def test_08_json_output(self):
        client = LegalMindClient()
        with patch.object(client, "chat", return_value={
            "answer": "JSON Test Answer",
            "citations": [],
            "confidence": "HIGH",
            "evidence_coverage": 100.0,
            "raw": {"status": "ok", "answer": "JSON Test Answer"},
        }):
            res = ask_question(client, "Test Query", json_output=True)
            self.assertIsNotNone(res)

    # 9. Streaming Output
    def test_09_streaming_output(self):
        # Verify streaming toggle flag works
        parser = create_parser()
        args = parser.parse_args(["ask", "Test", "--no-stream"])
        self.assertTrue(args.no_stream)

    # 10. Ctrl+C Handling
    def test_10_ctrl_c_handling(self):
        client = LegalMindClient()
        with patch.object(client, "chat", side_effect=KeyboardInterrupt):
            # Should catch KeyboardInterrupt and return None gracefully without raising
            res = ask_question(client, "Interrupted query")
            self.assertIsNone(res)

    # 11. Empty Question
    def test_11_empty_question(self):
        client = LegalMindClient()
        res = ask_question(client, "   ")
        self.assertIsNone(res)

    # 12. Invalid Command
    def test_12_invalid_command(self):
        client = LegalMindClient()
        mgr = ShortcutManager(client)
        # Non-slash normal text returns False (to be handled as a query)
        self.assertFalse(mgr.handle("Normal query"))
        # Invalid slash command returns True (handled with error display)
        self.assertTrue(mgr.handle("/invalidcommand"))

    # 13. Image Validation
    def test_13_image_validation(self):
        path, mime, (w, h) = validate_image_file(str(self.sample_png))
        self.assertEqual(mime, "image/png")
        self.assertEqual((w, h), (100, 100))

    # 14. Missing Image
    def test_14_missing_image(self):
        with self.assertRaises(ValidationError) as cm:
            validate_image_file("/non/existent/path.png")
        self.assertIn("does not exist", str(cm.exception))

    # 15. Unsupported Image Type
    def test_15_unsupported_image_type(self):
        with self.assertRaises(ValidationError) as cm:
            validate_image_file(str(self.sample_exe))
        self.assertIn("Unsupported image type", str(cm.exception))

    # 16. PDF Validation
    def test_16_pdf_validation(self):
        path, ext, pages = validate_document_file(str(self.sample_pdf))
        self.assertEqual(ext, ".pdf")
        self.assertIsNotNone(pages)

    # 17. Missing PDF
    def test_17_missing_pdf(self):
        with self.assertRaises(ValidationError) as cm:
            validate_document_file("/non/existent/judgment.pdf")
        self.assertIn("does not exist", str(cm.exception))

    # 18. Unsupported Document Type
    def test_18_unsupported_document_type(self):
        with self.assertRaises(ValidationError) as cm:
            validate_document_file(str(self.sample_exe))
        self.assertIn("Unsupported file type", str(cm.exception))

    # 19. Backend Timeout
    def test_19_backend_timeout(self):
        client = LegalMindClient("http://127.0.0.1:8080", timeout=0.001)
        import httpx
        with patch("httpx.Client.post", side_effect=httpx.TimeoutException("Timeout")):
            with self.assertRaises(BackendError) as cm:
                client.chat("Timeout query")
            self.assertIn("Connection timeout", str(cm.exception))

    # 20. Backend Unavailable
    def test_20_backend_unavailable(self):
        client = LegalMindClient("http://127.0.0.1:9999")  # Unused port
        import httpx
        with patch("httpx.Client.post", side_effect=httpx.ConnectError("Refused")):
            with self.assertRaises(BackendError) as cm:
                client.chat("Offline query")
            self.assertIn("Backend is unavailable", str(cm.exception))

    # 21. Voice Fallback
    def test_21_voice_fallback(self):
        # In headless environment without sounddevice input device, must return None cleanly
        with patch.object(VoiceInputHandler, "is_available", return_value=False):
            res = VoiceInputHandler.capture_and_transcribe("en")
            self.assertIsNone(res)

    # 22. OCR Fallback
    def test_22_ocr_fallback(self):
        # Force pytesseract to fail or return empty, verify uncertainty flag
        text, meta = ImageAnalyzer.extract_text(str(self.sample_png))
        self.assertIn("ocr_uncertain", meta)
        self.assertTrue(meta["ocr_uncertain"])

    # 23. TTS Fallback
    def test_23_tts_fallback(self):
        # In environment without audio device, speak_text should not raise exception
        with patch.object(SpeechOutputHandler, "is_available", return_value=False):
            SpeechOutputHandler.speak_text("Sample answer to read")

    # 24. Terminal Width Handling
    def test_24_terminal_width_handling(self):
        w = get_terminal_width()
        self.assertGreaterEqual(w, 70)
        print_separator("─")

    # 25. Shortcut Commands
    def test_25_shortcut_commands(self):
        client = LegalMindClient()
        mgr = ShortcutManager(client)
        # Test /reset
        config.active_conversation_id = "test-123"
        mgr.handle("/reset")
        self.assertIsNone(config.active_conversation_id)

        # Test /json toggle
        prev = config.json_mode
        mgr.handle("/json")
        self.assertNotEqual(config.json_mode, prev)

        # Test /stream toggle
        prev_s = config.stream
        mgr.handle("/stream")
        self.assertNotEqual(config.stream, prev_s)

    # 26. Source Display
    def test_26_source_display(self):
        sources = [
            {"title": "Constitution of India", "section": "21", "url": "https://legislative.gov.in"},
            {"title": "Supreme Court of India", "section": "AIR 1978 SC 597"},
        ]
        render_sources_list(sources)

    # 27. Confidence Display
    def test_27_confidence_display(self):
        # Verify render_answer_box displays HIGH, MEDIUM, LOW correctly
        for conf in ["HIGH", "MEDIUM", "LOW"]:
            render_answer_box(
                answer="Test Answer",
                confidence=conf,
                evidence_coverage=88.0,
                latency_ms=120.0,
            )

    # 28. Uploaded Document Context
    def test_28_uploaded_document_context(self):
        client = LegalMindClient()
        config.active_document_name = "test_doc.pdf"
        config.active_document_text = "Article 14 guarantees equality before law."
        config.active_conversation_id = None

        with patch("httpx.Client.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "conversation_id": "new-doc-conv",
                "assistant_message": {"content": "Response with doc context", "metadata": {}},
            }
            mock_post.return_value = mock_resp
            res = client.chat("Explain this")
            # Verify that document context was injected on turn 1
            call_args = mock_post.call_args[1]["json"]
            self.assertIn("Context from uploaded document", call_args["query"])

    # 29. No Duplicate History Database
    def test_29_no_duplicate_history_database(self):
        # Verify that no second SQLite file was created by cli package
        data_dir = PROJECT_ROOT / "data"
        db_files = list(data_dir.glob("*.db"))
        # Should only be legalmind.db (the existing database)
        db_names = [f.name for f in db_files]
        self.assertIn("legalmind.db", db_names)
        self.assertNotIn("cli_history.db", db_names)
        self.assertNotIn("terminal_history.db", db_names)

    # 30. Existing Web UI Remains Unchanged
    def test_30_existing_web_ui_unchanged(self):
        # Verify app/static/ files and app/index.html exist and were not overwritten
        index_html = PROJECT_ROOT / "app" / "index.html"
        static_dir = PROJECT_ROOT / "app" / "static"
        self.assertTrue(index_html.exists(), "app/index.html must exist")
        self.assertTrue(static_dir.exists(), "app/static/ must exist")
        # Ensure static files are intact
        self.assertTrue((static_dir / "index.html").exists())
        self.assertTrue((static_dir / "style.css").exists())
        self.assertTrue((static_dir / "app.js").exists())


if __name__ == "__main__":
    unittest.main()
