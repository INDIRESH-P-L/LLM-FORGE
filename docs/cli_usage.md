# LegalMindAI Terminal CLI User Guide

## Overview

The **LegalMindAI CLI** is a terminal-based Indian Legal Research Assistant designed for high-performance interactive research, statutory retrieval, judgment analysis, and multimodal document inquiry directly within your terminal.

It communicates with the live LegalMindAI backend (`http://127.0.0.1:8080`), preserving conversation threads, citation grounding, and confidence scoring.

---

## 1. Installation & Environment Setup

The CLI is included in the `cli/` module of the LegalMindAI repository. To launch using the approved environment:

```bash
cd ~/LegalMindAI
source /opt/llm-training/bin/activate
```

You can run the CLI directly with Python:
```bash
python -m cli.main
```
or via the executable launcher:
```bash
./bin/legalmind
```

To create a global system alias or symlink in your user environment:
```bash
alias legalmind="~/LegalMindAI/bin/legalmind"
```

---

## 2. Starting the CLI

### Interactive Full-Screen Shell
```bash
legalmind
# or
legalmind chat
```

This launches the full-screen terminal interface with:
- Centered **LEGALMIND AI** startup banner
- Live subsystem health checks (`Backend connected`, `Model available`, `RAG retrieval enabled`)
- Dynamic prompt `>`
- Status bar displaying `? for shortcuts` and active model (`Qwen3.6-35B-A3B`)

---

## 3. Asking Questions

### Direct One-Shot Question
```bash
legalmind ask "Explain Article 21 of the Constitution of India"
```

### Depth and Perspective Modes
Tailor the complexity and target audience using `--mode`:
```bash
# Plain language explanation for clients or laypersons
legalmind ask "Explain Article 21" --mode simple

# In-depth statutory doctrine with exhaustive jurisprudence
legalmind ask "Explain Article 21" --mode detailed

# Exam-oriented synthesis with definitions and principles
legalmind ask "Explain Article 21" --mode student

# Case analysis focusing on facts, issues, and ratio decidendi
legalmind ask "Explain Article 21" --mode case-analysis
```

### Raw JSON Output
Render complete API responses including citation metadata, latencies, and token counts:
```bash
legalmind ask "What is Section 302 IPC?" --json
```

### Streaming Control
Streaming is enabled by default where supported by the server. Disable with:
```bash
legalmind ask "Explain the Golden Triangle doctrine" --no-stream
```

---

## 4. Multimodal Document & Image Analysis

### Image Questioning & OCR
Upload legal notices, FIR copies, or court order photos (`.png`, `.jpg`, `.jpeg`, `.webp`):

```bash
# Ask a specific question about an image
legalmind image ./notice.png "What is the reply deadline mentioned in this notice?"

# Analyze an image and produce a structured breakdown
legalmind image ./notice.png

# View extracted OCR text before generation
legalmind image ./notice.png --show-ocr
```

If OCR text is uncertain or noisy, LegalMindAI informs you:
```
[WARNING] Some text may have been incorrectly recognized.
[INFO] Please verify the extracted text before relying on it.
```

### PDF & Document Analysis
Support for judgments, statutory drafts, agreements, and contracts (`.pdf`, `.docx`, `.txt`):

```bash
# Summarize a judgment with 13-point structured breakdown
legalmind analyze ./judgment.pdf

# Ask a specific inquiry about a document
legalmind pdf ./judgment.pdf "What was the Supreme Court's decision on Article 14?"
```

#### 13-Point Judgment Breakdown Structure
When analyzing judgments, the CLI organizes output into:
1. Case Name & Parties
2. Court & Bench
3. Date / Year
4. Material Facts
5. Legal Issues
6. Arguments of the Parties
7. Relevant Provisions (Constitution, Acts, Sections)
8. Court Decision / Order
9. Ratio Decidendi & Legal Principles
10. Legal Significance & Impact
11. Related / Connected Judgments
12. Verified Sources
13. Limitations & Disclaimers

---

## 5. Multi-Turn Follow-Up Questions

In interactive mode (`legalmind chat`), conversations maintain multi-turn context and document state:

```text
> /upload ./sample_notice.png
[Document Analysis displayed...]

> Who is the sender of this notice?
[Answer identifies sender from notice context]

> What section of the Negotiable Instruments Act is cited?
[Answer identifies Section 138]

> What are the required legal steps to respond?
[Answer provides actionable legal next steps]
```

To reset the session and start a new conversation:
```text
> /reset
```

---

## 6. Voice Typing & Speech-to-Text

```bash
legalmind voice
```
or inside interactive mode:
```text
> /voice
```

### Supported Languages
- English (`en`)
- Tamil (`ta`)
- Hindi (`hi`)

Example:
```bash
legalmind voice --language ta
```

### Interactive Flow:
1. Terminal prompts: `Press Enter to start recording. Press Ctrl+C to cancel.`
2. Capture speech input.
3. Display transcript:
   ```
   Transcript:
     Explain Article 21 of the Constitution of India in simple language.
   ```
4. Confirmation prompt: `Send this question? [Y/n]`
5. If confirmed, sends query to the backend and renders the answer.

*Note*: If recording hardware or audio dependencies are unavailable on headless servers, the CLI provides a clean notification:
```
[WARNING] Voice input is unavailable.
[INFO] Please use normal text input or install the approved speech-to-text dependency.
```

---

## 7. Text-to-Speech (TTS)

Read the latest generated answer aloud:
```bash
legalmind speak
```
or inside interactive mode:
```text
> /speak
```

To read only a short executive summary:
```bash
legalmind speak --summary
```

---

## 8. Interactive Shortcuts & Commands

In interactive mode, type `?` to display the cheatsheet:

| Command | Description |
|---|---|
| `?` | Show available shortcuts |
| `/help` | Display detailed help instructions and examples |
| `/clear` | Clear terminal screen and reprint startup banner |
| `/reset` | Reset conversation thread and start fresh context |
| `/source` | Show verified sources from the latest answer |
| `/sources` | Show all sources retrieved in the current session |
| `/status` | Show comprehensive backend and subsystem status |
| `/model` | Show active LLM parameters and architecture |
| `/voice` | Start interactive voice typing |
| `/upload <path>` | Upload and analyze a document or image |
| `/ocr` | View extracted OCR text of the currently loaded document |
| `/json` | Toggle raw JSON output mode on/off |
| `/stream` | Toggle streaming response mode on/off |
| `/summary` | Request a concise 3-point takeaway of the latest answer |
| `/speak` | Read the latest answer aloud via TTS |
| `/history` | View recent conversations from the backend database |
| `/exit`, `exit`, `quit`, `:q` | Exit the interactive session |

---

## 9. Configuration & Backend Management

### Specifying Backend URL
By default, the CLI connects to `http://127.0.0.1:8080`. You can override this using:
```bash
legalmind --server http://192.168.1.100:8080 chat
```
or by setting an environment variable:
```bash
export LEGALMIND_SERVER="http://127.0.0.1:8080"
```

### Enabling Debug Logs
```bash
legalmind --debug ask "Explain Article 14"
```

---

## 10. Legal Safety & Disclaimer

LegalMindAI displays authoritative legal research and citations. In accordance with legal safety guidelines:
> **Disclaimer**: This analysis is based on uploaded documents and retrieved legal sources. It is informational and is not a substitute for advice from a qualified legal professional.
