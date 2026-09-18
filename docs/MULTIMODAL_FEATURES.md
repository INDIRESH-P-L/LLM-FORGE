# LegalMindAI — Multimodal Services & Legal Document Intelligence

LegalMindAI extends beyond text RAG to support full multimodal document analysis: PDF agreements, scanned court orders, official stamps, audio voice typing, and neural speech synthesis.

---

## 1. Document Parsing & Analysis (`app/services/document_service.py`)

The document service parses high-volume legal filings asynchronously:
- **Supported Formats**: `.pdf`, `.docx`, `.doc`, `.txt`, `.rtf`.
- **PDF Engine**: Uses `pypdf` with fallback to text extraction streams.
- **Section Extraction**: Automatically identifies legal document subdivisions:
  - Title & Cause Title
  - Parties (Petitioner vs Respondent)
  - Operative Clauses & Sub-Sections
  - Prayer / Relief Sought
  - Verification & Signature Block
- **Context Attachment**: Attaches parsed text to the active conversation context (`app/services/multimodal_context.py`), enabling follow-up questions without re-uploading.

---

## 2. Optical Character Confusion Correction (`app/services/ocr_service.py`)

Scanned Indian court orders and stamp papers frequently suffer from scan degradation and OCR character confusion:
- Letter `O` confused with digit `0` (e.g. `Sec. 3O2` $\to$ `Section 302`).
- Digit `1` confused with letter `l` or `I` (e.g. `Artic1e 21` $\to$ `Article 21`).
- Digit `5` confused with letter `S` (e.g. `5ection 438` $\to$ `Section 438`).
- Digit `8` confused with letter `B` (e.g. `65B` vs `658`).

### Non-Destructive Correction:
`ocr_service.py` detects these patterns without destroying valid text, cross-referencing against an Indian legal lexicon of known acts and section numbers.

---

## 3. Legal Information Extraction (`app/services/legal_extraction_service.py`)

Extracts structured legal entities from raw document text:
- **Acts & Codes**: Detects all referenced statutes (e.g., *Arbitration and Conciliation Act, 1996*, *Companies Act, 2013*).
- **Sections**: Gathers all specific section numbers and clauses.
- **Courts & Tribunals**: Identifies forum (Supreme Court, Delhi High Court, NCLT, NGT).
- **Citations**: Finds official law reporters (AIR, SCC, SCR).

---

## 4. Vision QA & Document Element Analysis (`app/services/vision_service.py`)

Analyzes document images:
- **Stamp & Seal Detection**: Identifies official court seals, notary stamps, and revenue stamps.
- **Signature Verification**: Flags presence and location of signature blocks.
- **Header Analysis**: Extracts court cause lists and filing numbers.

---

## 5. Voice Typing & Text-to-Speech

### Voice Typing (`app/services/speech_service.py`):
- Transcribes audio streams using local Whisper ASR.
- Audio formats supported: WAV, MP3, OGG, WebM.
- Converts spoken legal phrasing (e.g., *"section three zero two of the IPC"*) into standardized text.

### Text-to-Speech (`app/services/tts_service.py`):
- Generates speech audio streams from synthesized legal answers for hands-free audio playback.
