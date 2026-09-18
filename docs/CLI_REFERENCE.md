# LegalMindAI — Terminal CLI Command Reference

The **LegalMind CLI** provides a terminal interface modeled after developer assistants like Antigravity CLI, with rich formatting, real-time citation rendering, and multimodal subcommands.

Executable Path: `bin/legalmind` or `.venv/bin/python cli/main.py`

---

## 1. Subcommands

### `ask` — Single-Turn Legal Query
Ask any statutory, constitutional, or case law question:
```bash
bin/legalmind ask "What is the Golden Triangle under the Constitution of India?"
```
Flags:
- `--json`: Output raw structured JSON response.
- `--stream`: Stream answer tokens live to stdout.
- `--conv-id <id>`: Attach inquiry to a persistent conversation context.

### `image` — Image Upload & Question Answering
Analyze legal documents, court seals, or affidavits from an image:
```bash
bin/legalmind image /path/to/affidavit.png --prompt "Is this document notarized?"
```

### `pdf` / `document` — Document Analysis
Parse and query a PDF, DOCX, or text agreement:
```bash
bin/legalmind pdf /path/to/contract.pdf
bin/legalmind document /path/to/chargesheet.docx --prompt "What offences are alleged?"
```

### `voice` — Interactive Voice Typing
Start microphone recording and transcribes spoken legal queries into the assistant:
```bash
bin/legalmind voice
```

### `sources` — Active Evidence & Source Inspection
Display citations, documents, and evidence chunks retrieved during the most recent query:
```bash
bin/legalmind sources
```

### `health` — Backend Connection & Health Check
Verify API connectivity, active model, and vector database status:
```bash
bin/legalmind health
```

### `version` — Version Information
Show LegalMindAI CLI and backend version:
```bash
bin/legalmind version
```

### `chat` — Full-Screen Interactive Assistant
Launch the interactive terminal session:
```bash
bin/legalmind chat
```

---

## 2. Interactive Slash Commands

Inside the interactive chat interface (`bin/legalmind chat`), the following slash commands are supported:

| Command | Action | Description |
|---|---|---|
| `/help` | Show Help | Displays command usage and sample legal queries. |
| `/clear` | Clear Screen | Clears terminal scrollback while preserving history. |
| `/status` | System Status | Shows backend health, active vector indexes, and latency. |
| `/model` | Model Info | Shows active LLM architecture (Qwen3.6-35B-A3B) and precision. |
| `/voice` | Voice Typing | Starts interactive audio recording and speech-to-text. |
| `/upload` | Upload File | Prompts for document/image path to load into conversation. |
| `/ocr` | Inspect OCR | Displays extracted text and metadata from the active document. |
| `/json` | Toggle JSON | Toggles structured JSON output mode on or off. |
| `/stream` | Toggle Stream | Toggles live token streaming. |
| `/summary`| Summarize | Generates a 3-bullet executive summary of the latest answer. |
| `/speak` | Read Aloud | Synthesizes neural TTS audio of the latest answer. |
| `/history`| View History | Displays recent exchanges from SQLite WAL storage. |
| `/exit` | Exit | Closes the CLI session gracefully. |
