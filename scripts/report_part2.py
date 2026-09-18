"""
scripts/report_part2.py
Chapters 6 through 10 of LegalMind AI Academic Project Report:
- Chapter 6: Technologies and Tools Used
- Chapter 7: Dataset and Knowledge Base
- Chapter 8: Methodology
- Chapter 9: Model Development and Fine-Tuning
- Chapter 10: Implementation
"""

CHAPTER_6 = {
    "number": 6,
    "title": "TECHNOLOGIES AND TOOLS USED",
    "sections": [
        {
            "heading": "6.1 Overview of System Technology Stack",
            "content": (
                "LegalMind AI integrates a modern, high-performance open-source technology stack spanning deep learning frameworks, "
                "vector databases, web protocols, and client-side web technologies. Every architectural component was selected after "
                "rigorous technical evaluation of stability, performance on NVIDIA enterprise hardware, open-source community governance, "
                "and native support for modern AI standards."
            )
        },
        {
            "heading": "6.2 Deep Dive into Core Technologies",
            "content": (
                "1. Python (Version 3.12):\n"
                "- What it is: The premier high-level programming language for scientific computing, deep learning, and asynchronous backend development.\n"
                "- Selection Rationale: Provides mature ecosystem support for PyTorch, Hugging Face, FAISS, and modern typing constructs.\n"
                "- Usage in LegalMind AI: Core development language across data preprocessing, vector indexing, neural inference, and API routing.\n"
                "- Advantages: Unmatched ecosystem, dynamic scripting, clean asynchronous syntax (async/await).\n"
                "- Limitations: Global Interpreter Lock (GIL) limits multi-core CPU throughput; mitigated by offloading neural workloads to CUDA accelerators.\n\n"

                "2. FastAPI (Version 0.115+):\n"
                "- What it is: A modern, high-performance web framework for building APIs with Python 3.8+ based on standard Python type hints.\n"
                "- Selection Rationale: Native asynchronous architecture, automatic OpenAPI/Swagger documentation, and Pydantic data validation.\n"
                "- Usage in LegalMind AI: Exposes primary REST endpoints (/query, /health, /stats, /temporal/check, /precedents) and manages model lifespan.\n"
                "- Advantages: Near-zero serialization overhead, clean async syntax, robust dependency injection.\n"
                "- Limitations: Requires strict type annotation compliance across all incoming and outgoing data contracts.\n\n"

                "3. Uvicorn:\n"
                "- What it is: A lightning-fast ASGI (Asynchronous Server Gateway Interface) web server implementation for Python.\n"
                "- Selection Rationale: Production-grade asynchronous throughput, native SSL/TLS termination, and HTTP/1.1 and WebSockets support.\n"
                "- Usage in LegalMind AI: Serves the FastAPI application on HTTPS port 8443 and HTTP port 8080.\n"
                "- Advantages: Minimal CPU overhead, rock-solid stability under sustained concurrent connections.\n"
                "- Limitations: Single-process by default; requires multiple worker processes or reverse proxying for multi-core scaling.\n\n"

                "4. HTML5, Vanilla CSS3, and JavaScript (ES2022):\n"
                "- What it is: The native foundational triad of the World Wide Web.\n"
                "- Selection Rationale: Eliminates reliance on heavy third-party UI frameworks (React, Vue, Angular), ensuring instantaneous loading (<100ms).\n"
                "- Usage in LegalMind AI: Power the Apple-inspired Liquid Glass user interface, dual-feed sidebar history, PDF viewer, and audio controls.\n"
                "- Advantages: Zero build step, maximum flexibility, direct DOM manipulation, superior responsiveness.\n"
                "- Limitations: Manual state management; mitigated through clean object-oriented client architectures.\n\n"

                "5. SQLite (Version 3 with WAL Mode):\n"
                "- What it is: A self-contained, serverless, zero-configuration, transactional SQL database engine.\n"
                "- Selection Rationale: Embedded persistence, zero background process overhead, and ACID compliance for local conversational history.\n"
                "- Usage in LegalMind AI: Persists conversations, message histories, latency telemetry, and FTS5 full-text search indexes.\n"
                "- Advantages: Single-file database durability (data/legalmind.db), non-blocking reads under Write-Ahead Logging (WAL).\n"
                "- Limitations: Not designed for high-concurrency multi-node distributed writing; perfectly suited for local enterprise deployments.\n\n"

                "6. PyTorch (Version 2.x with CUDA 12.x):\n"
                "- What it is: The premier open-source machine learning and deep learning tensor library developed by Meta AI.\n"
                "- Selection Rationale: Native acceleration on NVIDIA DGX B200 accelerators, native support for Bfloat16, and dynamic computation graphs.\n"
                "- Usage in LegalMind AI: Underpins transformer tensor operations, neural cross-encoder inference, and device memory allocations.\n"
                "- Advantages: Industry standard, comprehensive operator coverage, seamless CUDA memory management.\n"
                "- Limitations: Large library binary footprint; requires exact CUDA version matching.\n\n"

                "7. Hugging Face Transformers & PEFT:\n"
                "- What it is: The preeminent library for state-of-the-art transformer architectures and Parameter-Efficient Fine-Tuning.\n"
                "- Selection Rationale: Provides native loading for Qwen3.6 MoE models, safetensors integration, and modular LoRA adapter attachment.\n"
                "- Usage in LegalMind AI: Manages AutoProcessor, Qwen3_5MoeForConditionalGeneration, and PeftModel runtime adapter switching.\n"
                "- Advantages: Modular, transparent, unified API across diverse foundation architectures.\n"
                "- Limitations: High memory footprint when loading large 35B models; requires careful device mapping.\n\n"

                "8. Qwen3.6-35B-A3B (Alibaba Cloud):\n"
                "- What it is: A 35-billion-parameter Mixture-of-Experts (MoE) foundation model with ~3 billion active parameters per token.\n"
                "- Selection Rationale: Outstanding legal reasoning, structured instruction following, high coding proficiency, and native BF16 stability.\n"
                "- Usage in LegalMind AI: Acts as the primary generative reasoning core, synthesizing evidence-grounded legal answers.\n"
                "- Advantages: Dense model representation quality with the computational efficiency and latency of a 3B model.\n"
                "- Limitations: Demands ~70 GB VRAM for native BF16 weights; relies on reference PyTorch fallback when custom attention kernels are absent.\n\n"

                "9. Low-Rank Adaptation (LoRA) and PEFT:\n"
                "- What it is: A parameter-efficient fine-tuning methodology decomposing weight update matrices into low-rank representations (r=16, alpha=32).\n"
                "- Selection Rationale: Enables domain adaptation to Indian legal syntax without altering frozen base model weights or risking catastrophic forgetting.\n"
                "- Usage in LegalMind AI: Applied to self-attention projections (q, k, v, o) in lora-legal-v2 (3,440,640 parameters).\n"
                "- Advantages: Extremely lightweight (14 MB safetensors), rapid loading, zero risk of modifying base model weights.\n"
                "- Limitations: Must be audited for gradient stability; poorly tuned training can induce weight NaNs as seen in legacy lora-legal-v1.\n\n"

                "10. Brain Floating Point (BF16 Precision):\n"
                "- What it is: A 16-bit floating-point numeric format featuring an 8-bit exponent and 7-bit mantissa (identical exponent range to FP32).\n"
                "- Selection Rationale: Eliminates gradient underflow/overflow common in FP16, preventing NaN generation during extended legal inference.\n"
                "- Usage in LegalMind AI: Native precision for Qwen3.6-35B-A3B model weights, activations, and LoRA adapter matrices.\n"
                "- Advantages: 50% memory reduction compared to FP32 with zero loss of dynamic numerical range.\n"
                "- Limitations: Requires modern GPU architectures (Ampere, Hopper, Blackwell) supporting native BF16 arithmetic.\n\n"

                "11. NVIDIA CUDA Platform (CUDA_HOME=/usr/local/cuda):\n"
                "- What it is: NVIDIA's parallel computing platform and programming model for GPU hardware acceleration.\n"
                "- Selection Rationale: Direct hardware acceleration on the NVIDIA DGX B200 enterprise server.\n"
                "- Usage in LegalMind AI: Powers accelerated tensor math on GPU 1 (LLM inference) and GPU 4 (dense embeddings and reranker).\n"
                "- Advantages: Unrivaled high-performance compute bandwidth and hardware-accelerated matrix multiplication.\n"
                "- Limitations: Vendor lock-in to NVIDIA hardware ecosystems.\n\n"

                "12. BAAI/bge-m3 Embedding Model:\n"
                "- What it is: Multi-Lingual, Multi-Functionality, Multi-Granularity dense text embedding model developed by Beijing Academy of AI.\n"
                "- Selection Rationale: Encodes 1024-dimensional continuous vectors, excels at long legal passages (up to 8192 tokens), and multi-lingual law.\n"
                "- Usage in LegalMind AI: Generates dense semantic embeddings for 26,097 legal chunks and real-time user query vectors.\n"
                "- Advantages: Superior semantic recall on conceptual queries compared to traditional Sentence-BERT models.\n"
                "- Limitations: Bi-encoder design misses word-level granular interactions; requires downstream cross-encoder reranking.\n\n"

                "13. BAAI/bge-reranker-v2-m3:\n"
                "- What it is: A cross-encoder neural reranking model scoring query-document relevance through full bidirectional cross-attention.\n"
                "- Selection Rationale: Dramatically elevates precision by evaluating fine-grained token alignments between question and statutory text.\n"
                "- Usage in LegalMind AI: Rescores top candidates from hybrid retrieval to select the top-k highest-relevance evidence chunks.\n"
                "- Advantages: Significantly higher retrieval accuracy and noise reduction than vector similarity alone.\n"
                "- Limitations: Higher computational cost per candidate pair; restricted to reranking top 15–30 candidates.\n\n"

                "14. Okapi BM25 (Rank-BM25):\n"
                "- What it is: A non-parametric probabilistic information retrieval algorithm scoring documents based on term frequency and inverse document frequency.\n"
                "- Selection Rationale: Industry gold standard for exact statutory alphanumeric matching (e.g., 'Section 302', 'Article 21').\n"
                "- Usage in LegalMind AI: Provides sparse lexical candidate retrieval across 15,057 statutory chunks.\n"
                "- Advantages: Instantaneous in-memory querying, zero GPU requirement, immune to semantic drift.\n"
                "- Limitations: Zero comprehension of semantic synonyms, paraphrased questions, or linguistic context.\n\n"

                "15. FAISS (Facebook AI Similarity Search):\n"
                "- What it is: A high-performance library for dense vector clustering and similarity search developed by Meta Research.\n"
                "- Selection Rationale: Provides blazing-fast inner-product (IndexFlatIP) and L2 similarity search over tens of thousands of high-dimensional vectors.\n"
                "- Usage in LegalMind AI: Indexes and searches 26,097 1024-dimensional BGE-M3 vectors representing Indian legislation and case law.\n"
                "- Advantages: Microsecond search latency, exceptional vector compression and clustering efficiency.\n"
                "- Limitations: Requires careful dimension alignment and normalized vectors for inner-product cosine equivalence.\n\n"

                "16. PDF.js (Mozilla):\n"
                "- What it is: A standards-compliant, HTML5-based technology platform for parsing and rendering Portable Document Format (PDF) files.\n"
                "- Selection Rationale: Enables client-side, zero-plugin legal document inspection directly within the web browser.\n"
                "- Usage in LegalMind AI: Displays statutory acts, petitions, and legal briefs in an interactive preview pane beside the chat stream.\n"
                "- Advantages: Completely client-side, highly secure, zero server-side rendering latency.\n"
                "- Limitations: Memory-intensive for massive 500-page scanned court files.\n\n"

                "17. Web Speech API (Browser Native):\n"
                "- What it is: A native browser API providing speech-to-text recognition through speech recognition engines.\n"
                "- Selection Rationale: Enables hands-free legal query entry for advocates and pro se users without requiring external cloud audio services.\n"
                "- Usage in LegalMind AI: Powers the voice-input microphone button in the chat interface.\n"
                "- Advantages: Real-time interim transcriptions, zero server audio processing overhead.\n"
                "- Limitations: Strictly enforced to HTTPS 'Secure Contexts'; fails with 'not-allowed' over insecure plain HTTP."
            )
        }
    ]
}

CHAPTER_7 = {
    "number": 7,
    "title": "DATASET AND KNOWLEDGE BASE",
    "sections": [
        {
            "heading": "7.1 Legal-Domain Data Characteristics and Requirements",
            "content": (
                "Legal data possesses distinctive characteristics that fundamentally distinguish it from open-domain web data:\n"
                "1. Hierarchical Authority: Constitutional provisions override Union statutes; Union enactments override state rules; Supreme Court rulings "
                "bind all lower courts under Article 141 of the Constitution.\n"
                "2. Extreme Syntactic Precision: A single word—such as 'may' versus 'shall'—drastically alters legal obligation, requiring exact token preservation.\n"
                "3. Inter-Document Cross-Referencing: Statutes rarely operate in isolation; sections cross-reference definitions, exceptions, and procedural "
                "enactments across separate acts.\n"
                "4. Temporal Sensitivity: Laws are frequently amended, repealed, or renumbered, creating severe obsolescence hazards if temporal metadata is ignored."
            )
        },
        {
            "heading": "7.2 Corpus Composition: Statutes, Constitution, and Precedents",
            "content": (
                "The LegalMind AI knowledge base was curated from authoritative public domain repositories, including India Code (legislative.gov.in) "
                "and official Supreme Court judgment archives (judgments.ecourts.gov.in):\n"
                "- The Constitution of India: Ingested in its entirety, including the Preamble, all 448 Articles across Parts I to XXII, and Schedules 1 to 12.\n"
                "- Core Penal and Procedural Enactments: Complete statutory text of the Indian Penal Code (1860), Bharatiya Nyaya Sanhita (2023), "
                "Code of Criminal Procedure (1973), Bharatiya Nagarik Suraksha Sanhita (2023), Indian Evidence Act (1872), and Bharatiya Sakshya Adhiniyam (2023).\n"
                "- Substantive Commercial and Civil Enactments: Indian Contract Act (1872), Companies Act (1956 and 2013), Arbitration and Conciliation Act (1996), "
                "Information Technology Act (2000), Specific Relief Act (1963), and Limitation Act (1963).\n"
                "- Landmark Supreme Court Precedents: High-impact rulings establishing core constitutional doctrines, including Kesavananda Bharati v. State of Kerala "
                "(1973, Basic Structure doctrine), Maneka Gandhi v. Union of India (1978, Substantive Due Process under Article 21), E.P. Royappa v. State of Tamil Nadu "
                "(1974, Arbitrariness test under Article 14), and A.K. Gopalan v. State of Madras (1950)."
            )
        },
        {
            "heading": "7.3 Preprocessing, Normalization, and Chunking Strategy",
            "content": (
                "Raw statutory texts were processed through a multi-stage data engineering pipeline:\n"
                "1. Text Normalization: Removed scanning artifacts, fixed hyphenation across line breaks, normalized Unicode quotes and dashes, and expanded "
                "standard legal abbreviations (e.g., 'Sec.' -> 'Section', 'Art.' -> 'Article', 'Cr.P.C.' -> 'Code of Criminal Procedure').\n\n"
                "2. Provision-Aware Chunking: Traditional RAG pipelines utilize arbitrary fixed character or token windows (e.g., 500 characters with 50-character "
                "overlap). In statutory law, arbitrary splitting breaks cohesive sub-clauses, severing definitions from operative punishments. LegalMind AI "
                "implemented a Provision-Aware Chunking Strategy:\n"
                "- Each statutory section or constitutional article was segmented as an atomic baseline chunk;\n"
                "- Long provisions (e.g., Section 438 CrPC or Article 371) were segmented along structural sub-clause boundaries (Sub-section (1), Proviso, Explanation);\n"
                "- Target chunk size was maintained between 400 and 800 tokens, preserving complete statutory context while fitting comfortably within LLM context budgets.\n\n"
                "3. Metadata Schema Design: Every chunk was tagged with rich structured metadata:\n"
                "{\n"
                "  \"chunk_id\": \"const_art_021_01\",\n"
                "  \"act_name\": \"Constitution of India\",\n"
                "  \"document_type\": \"constitution\",\n"
                "  \"article\": \"21\",\n"
                "  \"section\": null,\n"
                "  \"title\": \"Protection of life and personal liberty\",\n"
                "  \"year\": 1950,\n"
                "  \"status\": \"in_force\"\n"
                "}"
            )
        },
        {
            "heading": "7.4 Technical Significance of Corpus Scale: 15,057 Chunks and 26,097 Vectors",
            "content": (
                "A critical empirical characteristic of LegalMind AI is the scale of its indexed knowledge base:\n"
                "- 15,057 Indexed Statutory Chunks: This figure represents the complete set of segmented legislative provisions stored in "
                "data/chunks/legislation/chunks.jsonl. Each chunk represents a discrete, legally coherent provision of Indian statutory law, "
                "guaranteeing exhaustive coverage of all major Union acts without missing sections.\n"
                "- 26,097 FAISS Vectors: The vector index (vector_db/legislation/index.faiss) contains 26,097 1024-dimensional dense embeddings. "
                "The divergence between chunk count (15,057) and vector count (26,097) reflects multi-scale representation engineering: key statutory "
                "provisions, landmark precedent headnotes, and constitutional provisions were embedded using dual granularity—both at the individual clause "
                "level and at the full provision summary level. This multi-vector representation ensures that both high-level thematic queries and fine-grained "
                "clause-specific inquiries match with exceptional semantic fidelity."
            )
        },
        {
            "heading": "7.5 Data Quality, Deduplication, and Copyright Considerations",
            "content": (
                "To ensure corpus integrity, automated deduplication removed redundant statutory amendments, maintaining the latest in-force version of every enactment. "
                "Missing metadata fields were repaired using regex pattern extractors (scripts/patch_chunk_metadata.py). Repealed acts were explicitly tagged to prevent "
                "anachronistic legal citations.\n\n"
                "Under Indian law, Section 52(1)(q) of the Copyright Act, 1957 expressly provides that the reproduction or publication of any Act of the Legislature, "
                "or any judgment or order of a court, tribunal, or other judicial authority, does not constitute an infringement of copyright. The entire knowledge base "
                "of LegalMind AI consists strictly of open, public domain statutory and judicial authorities."
            )
        }
    ]
}

CHAPTER_8 = {
    "number": 8,
    "title": "METHODOLOGY",
    "sections": [
        {
            "heading": "8.1 Comprehensive 18-Stage Execution Workflow",
            "content": (
                "The end-to-end operational methodology of LegalMind AI follows an 18-stage deterministic pipeline from initial user prompt ingestion "
                "to client-side visualization:\n\n"
                "Stage 01 (User Query Ingestion): The user enters a legal inquiry via the Liquid Glass web interface or the terminal CLI.\n\n"
                "Stage 02 (Input Sanitization & Validation): The query is validated for length, non-emptiness, and character encoding, and sanitized against prompt injection patterns.\n\n"
                "Stage 03 (Query Preprocessing & Legal Expansion): Common abbreviations ('Art. 21', 'Sec 302 IPC') are expanded into canonical statutory terms ('Article 21 Constitution of India', 'Section 302 Indian Penal Code').\n\n"
                "Stage 04 (Constitutional & Golden Triangle Router): Queries concerning fundamental rights are routed through ConstitutionalRetriever to guarantee balanced multi-article evidence.\n\n"
                "Stage 05 (BM25 Lexical Retrieval): The preprocessed query searches the 15,057-chunk BM25 index, retrieving the top-30 lexical candidates.\n\n"
                "Stage 06 (Dense Vector Retrieval): The query is embedded via BAAI/bge-m3 on GPU 4 and searched against the 26,097-vector FAISS index, retrieving top-30 semantic candidates.\n\n"
                "Stage 07 (Reciprocal Rank Fusion): Candidate lists from BM25 and FAISS are merged using RRF scoring (k=60), producing a unified top-30 candidate pool.\n\n"
                "Stage 08 (Cross-Encoder Neural Reranking): Candidate pairs (Query, Passage) are evaluated by BAAI/bge-reranker-v2-m3 on GPU 4, computing deep cross-attention relevance scores.\n\n"
                "Stage 09 (Dynamic Top-K Selection): The top-k highest-scoring chunks (default top_k=3 to 7) are selected, eliminating low-relevance distractor passages.\n\n"
                "Stage 10 (Context Assembly & Citation Formatting): Selected chunks are formatted into numbered reference blocks ([1], [2], [3]) with exact act names and provision numbers.\n\n"
                "Stage 11 (Prompt Engineering): The RAG prompt is assembled with strict system instructions requiring IRAC structure, factual grounding, and thinking token suppression.\n\n"
                "Stage 12 (Tokenization & Inference): Qwen3.6-35B-A3B processes the prompt on GPU 1 in native BF16 precision, applying chat templates.\n\n"
                "Stage 13 (LoRA Attention Modulation): The lora-legal-v2 adapter modulates query, key, value, and output projections across all transformer layers.\n\n"
                "Stage 14 (Generative Synthesis): The model generates the structured legal explanation with temperature=0.2 and top_p=0.9.\n\n"
                "Stage 15 (Thinking Suppression & Post-Processing): Internal <think>...</think> chain-of-thought blocks are stripped to ensure clean, professional output.\n\n"
                "Stage 16 (Citation Cross-Verification): Regex citation extractors parse all cited provisions and verify them against retrieved chunks, calculating hallucination scores.\n\n"
                "Stage 17 (Temporal Status Check): The temporal law engine checks for repealed acts and appends warnings regarding modern BNS/BNSS/BSA equivalents.\n\n"
                "Stage 18 (Client-Side Rendering): The structured JSON response is transmitted over HTTPS to the frontend, rendering markdown, citations, and conversation history."
            )
        }
    ]
}

CHAPTER_9 = {
    "number": 9,
    "title": "MODEL DEVELOPMENT AND FINE-TUNING",
    "sections": [
        {
            "heading": "9.1 Base Model Selection: Qwen3.6-35B-A3B",
            "content": (
                "Selecting the optimal foundation model required balancing computational efficiency with sophisticated legal reasoning. "
                "Qwen3.6-35B-A3B was chosen because of its cutting-edge Mixture-of-Experts (MoE) architecture. The model features 35 billion total parameters, "
                "of which approximately 3 billion active parameters are dynamically routed per token via learned gating networks. This sparse activation "
                "provides the broad semantic representation capacity of an enterprise-scale foundation model while executing with the throughput, "
                "memory bandwidth efficiency, and latency profile of a compact 3B model.\n\n"
                "The model was deployed in native Brain Floating Point (BF16) precision on GPU 1 of the NVIDIA DGX B200 server. Operating in native BF16 "
                "eliminates quantization artifacts common in 4-bit/8-bit compression (QLoRA, AWQ, GPTQ), which are known to degrade long-tail statutory "
                "reasoning and induce numerical instability in MoE gating routers."
            )
        },
        {
            "heading": "9.2 The Low-Rank Adaptation (LoRA) Paradigm",
            "content": (
                "While foundation models exhibit broad language fluency, legal analysis demands rigorous deductive argumentation, specialized terminology, "
                "and strict IRAC formatting. Full parameter fine-tuning of a 35B model would require multi-node GPU clusters and carries severe risks "
                "of catastrophic forgetting. Low-Rank Adaptation (LoRA) overcomes these limitations by freezing the pre-trained weight matrix W_0 in R^{d x k} "
                "and parameterizing the weight update Delta W with two low-rank matrices B in R^{d x r} and A in R^{r x k}, where rank r << min(d, k):\n\n"
                "W = W_0 + (alpha / r) * (B * A)\n\n"
                "During inference, the low-rank branch operates in parallel with the frozen weights, introducing zero inference latency overhead while "
                "enabling domain adaptation using less than 0.01% of the original parameter count."
            )
        },
        {
            "heading": "9.3 LoRA Adapter Architecture and Hyperparameters (lora-legal-v2)",
            "content": (
                "The domain adapter lora-legal-v2 was trained using standard BF16 LoRA with the following verified architectural parameters:\n"
                "- Target Modules: Self-attention projections: q_proj, k_proj, v_proj, o_proj across all 40 transformer layers;\n"
                "- Rank (r): 16;\n"
                "- Alpha scaling factor (alpha): 32 (effective scaling factor = alpha/r = 2.0);\n"
                "- Dropout rate: 0.05;\n"
                "- Total Adapter Parameters: 3,440,640 (representing 14 MB in safetensors format);\n"
                "- Training Hardware: NVIDIA DGX B200 GPU cluster;\n"
                "- Optimization Regime: AdamW optimizer, learning_rate=2e-5, max_grad_norm=0.3, warmup_ratio=0.05, gradient_accumulation_steps=8, "
                "gradient checkpointing enabled, standard LoRA in native BF16 (strictly zero quantization)."
            )
        },
        {
            "heading": "9.4 Forensic Tensor Safety Audit: lora-legal-v1 vs. lora-legal-v2",
            "content": (
                "A foundational contribution of this research was the discovery, forensic isolation, and resolution of numerical corruption "
                "in fine-tuned adapters. An earlier training run, designated lora-legal-v1, exhibited catastrophic degeneration during preliminary testing:\n\n"
                "FORENSIC AUDIT OF LORA-LEGAL-V1 (CORRUPT):\n"
                "- File: fine_tuning/adapters/lora-legal-v1/adapter_model.safetensors\n"
                "- Total Tensors: 320\n"
                "- NaN Tensors: 320 (100% of all adapter weight matrices contained NaN values)\n"
                "- Inf Tensors: 0\n"
                "- Root Cause Analysis: The v1 training script lacked gradient clipping safeguards and targeted both self-attention and all MoE MLP "
                "shared expert projections (gate_proj, up_proj, down_proj) without per-expert gradient normalization, triggering catastrophic gradient "
                "explosions that corrupted all weights into NaNs.\n"
                "- Resolution: lora-legal-v1 was permanently quarantined and blocked from deployment.\n\n"
                "FORENSIC AUDIT OF LORA-LEGAL-V2 (VALID & PRODUCTION-READY):\n"
                "- File: fine_tuning/adapters/lora-legal-v2/adapter_model.safetensors\n"
                "- Total Tensors: 80 (self-attention projections: q, k, v, o across 40 layers)\n"
                "- Total Parameters: 3,440,640\n"
                "- NaN Tensors: 0 (100% clean)\n"
                "- Inf Tensors: 0 (100% clean)\n"
                "- Status: VALID\n"
                "- In-Memory VRAM Audit: Pinned to GPU 1 via PEFT; zero NaNs and zero Infs verified in VRAM buffers."
            )
        },
        {
            "heading": "9.5 Dynamic Adapter Attachment and Fail-Safe Fallback Logic",
            "content": (
                "To guarantee system resilience, LegalMindModel in scripts/05_rag.py implements a dynamic adapter loader with fail-safe fallback:\n"
                "1. Pre-Load Inspection: Before loading weights into VRAM, inspect_safetensors_file performs CPU-level tensor verification. If any NaN/Inf is "
                "detected or if files are missing, the loader aborts adapter attachment and logs a critical error.\n"
                "2. Automatic Fallback: The model automatically operates using base Qwen3.6-35B weights + RAG. The system never crashes or raises unhandled exceptions.\n"
                "3. In-Memory Verification: Following PEFT attachment, in-memory model parameters are scanned. If corruption is found, model.unload() reverts "
                "the model to base weights."
            )
        }
    ]
}

CHAPTER_10 = {
    "number": 10,
    "title": "IMPLEMENTATION",
    "sections": [
        {
            "heading": "10.1 Modular Project Directory Organization",
            "content": (
                "The LegalMind AI codebase is organized into clean, decoupled modules adhering to strict separation of concerns:\n"
                "LegalMindAI/\n"
                "├── app/                     # FastAPI web application and endpoints\n"
                "│   ├── main.py              # Application entrypoint, lifespan, /health, /query\n"
                "│   ├── chat_api.py          # Chat persistence and conversation endpoints\n"
                "│   ├── static/              # Web client: index.html, style.css, app.js, PDF.js\n"
                "├── scripts/                 # Core pipeline and retrieval services\n"
                "│   ├── 05_rag.py            # LegalMindModel singleton, RAG prompt, generation\n"
                "│   ├── retriever.py         # HybridRetriever (BM25 + FAISS + Reranker)\n"
                "│   ├── constitutional_retrieval.py # Specialized Golden Triangle retrieval router\n"
                "│   ├── temporal_law.py      # Indian statutory transition engine (IPC to BNS)\n"
                "│   ├── citation_verifier.py # Automated citation cross-verification service\n"
                "├── fine_tuning/             # Parameter-efficient LoRA fine-tuning\n"
                "│   ├── validate_adapter.py  # Forensic safetensors NaN/Inf safety validator\n"
                "│   ├── validate_and_compare_v2.py # 10-category comparative benchmark suite\n"
                "│   └── adapters/            # lora-legal-v1 (quarantined), lora-legal-v2 (active)\n"
                "├── data/                    # Chunks, SQLite database (legalmind.db), manifests\n"
                "├── vector_db/               # FAISS vector indexes (index.faiss, chunk_ids.json)\n"
                "├── tests/                   # Automated unit and integration test suites (171 tests)\n"
                "└── evaluation/              # Benchmark data JSONs and validation markdown reports"
            )
        },
        {
            "heading": "10.2 Backend Implementation and Key Endpoints",
            "content": (
                "The backend is driven by app/main.py, exposing two critical endpoints:\n\n"
                "1. GET /health (System Telemetry & Health Check):\n"
                "- Operational Function: Performs real-time diagnostics on model memory, retriever state, and LoRA adapter status.\n"
                "- Response Contract:\n"
                "{\n"
                "  \"status\": \"ok\",\n"
                "  \"model_loaded\": true,\n"
                "  \"retriever_loaded\": true,\n"
                "  \"lora_adapter\": \"lora-legal-v2\",\n"
                "  \"lora_status\": \"ACTIVE\",\n"
                "  \"lora_active\": true,\n"
                "  \"uptime_s\": 59227.4,\n"
                "  \"queries_served\": 12\n"
                "}\n\n"
                "2. POST /query (Primary Legal Research Endpoint):\n"
                "- Operational Function: Receives QueryRequest (query, top_k, conversation_id), coordinates hybrid retrieval, executes cross-encoder reranking, "
                "generates the evidence-grounded IRAC answer, extracts citations, evaluates temporal validity, commits messages to SQLite, and returns QueryResponse.\n"
                "- Response Contract:\n"
                "{\n"
                "  \"query\": \"What does Article 21 guarantee?\",\n"
                "  \"answer\": \"## Answer\\nArticle 21 guarantees that no person shall be deprived of his life or personal liberty...\",\n"
                "  \"legal_provisions\": [\"Article 21\", \"Article 359(1)\"],\n"
                "  \"judgments\": [\"Maneka Gandhi v. Union of India (1978)\"],\n"
                "  \"citations\": [\"[1] Constitution of India — Article 21\"],\n"
                "  \"confidence\": \"high\",\n"
                "  \"confidence_score\": 0.85,\n"
                "  \"retrieval_latency\": 0.73,\n"
                "  \"generation_latency\": 18.45,\n"
                "  \"total_latency\": 19.18\n"
                "}"
            )
        },
        {
            "heading": "10.3 HTTPS Server Startup and GPU Process Management",
            "content": (
                "The server runs on the enterprise DGX B200 platform. HTTPS termination is enforced on port 8443 using self-signed TLS certificates "
                "(certs/legalmind.crt, certs/legalmind.key) to enable browser Web Speech API permissions, alongside port 8080 for HTTP auto-redirect traffic. Long-running training "
                "processes and inference servers are partitioned onto separate GPU instances (LLM on GPU 1, Retriever on GPU 4) to ensure that background "
                "training never starves active production inference memory."
            )
        },
        {
            "heading": "10.4 Implementation of Multimodal Services Subsystem",
            "content": (
                "The multimodal service layer (`app/services/*`) enables multi-format document parsing and interactive voice/vision intelligence:\n\n"
                "1. Document Processing Engine (`document_service.py`):\n"
                "- Multi-Format Support: Asynchronously ingests PDF, DOCX, and plain text files. Multi-page PDFs are extracted using PyPDF2 / pdfplumber.\n"
                "- Clause Breakdown: Documents are segmented into structural clauses (e.g., Definitions, Covenants, Indemnities, Governing Law).\n"
                "- Context Store Integration: Ingested document paragraphs are registered in `multimodal_context.py` for paragraph-targeted follow-up Q&A.\n\n"
                "2. Image Vision & OCR Correction (`vision_service.py`):\n"
                "- Image Processing: Handles degraded legal scans, image briefs, and court orders.\n"
                "- OCR Confusion Correction: Applies non-destructive regex confusion matrices to fix common Tesseract OCR errors (e.g., '1PC' -> 'IPC', 'Sect1on' -> 'Section', 'Art1cle' -> 'Article').\n"
                "- Visual Elements: Detects court seals, registrar stamps, and signatures.\n\n"
                "3. Speech-to-Text Audio Engine (`speech_service.py`):\n"
                "- Audio Validation: Enforces format checks (WAV, MP3, WEBM, OGG) and file size bounds (<25MB).\n"
                "- Whisper Adapter: Converts spoken user queries into text using local Faster-Whisper, providing seamless hands-free voice typing."
            )
        },
        {
            "heading": "10.5 Implementation of Antigravity Developer Terminal CLI",
            "content": (
                "The Antigravity Terminal CLI (`bin/legalmind`, `legalmind_cli/*`) provides command-line power users with a full-screen interactive interface:\n\n"
                "1. Interactive Token Streaming & Animated UI:\n"
                "- Powered by Python Rich and Prompt-Toolkit, delivering smooth token-by-token streaming, custom ASCII header banners, and real-time status spinners.\n\n"
                "2. Full Subcommand Suite:\n"
                "- `ask`: One-shot natural language legal query with formatted IRAC markdown output;\n"
                "- `image`: Analyzes legal document images and extracts text via OCR;\n"
                "- `pdf` / `document`: Parses multi-page PDF/DOCX files and performs context-aware legal analysis;\n"
                "- `voice`: Records or transcribes spoken audio queries;\n"
                "- `sources`: Inspects underlying statutory and judgment citations;\n"
                "- `health`: Displays real-time server and GPU VRAM diagnostics;\n"
                "- `chat`: Enters full-screen interactive REPL session with slash command support (`/voice`, `/upload`, `/ocr`, `/speak`)."
            )
        },
        {
            "heading": "10.6 Implementation of Apple Space Black Web UI & Dynamic Island",
            "content": (
                "The primary web application (`app/static/index.html`, `app/static/style.css`, `app/static/app.js`) delivers a state-of-the-art user experience:\n\n"
                "1. Design System & Liquid Glass Aesthetics:\n"
                "- Styled with Vanilla CSS using an Apple macOS Space Black dark mode color scheme (`#0B0C10`, `#1F2833`, `#C5C6C7`, `#66FCF1`).\n"
                "- Implements fluid iOS 18 glassmorphism panels with backdrop-blur filters, dynamic hover effects, and subtle glow animations.\n\n"
                "2. Dynamic Island Navigation Bar:\n"
                "- Features a floating top island rendering live status badges (e.g., 'Qwen3.6-35B Active', 'GPU 1: 38.4 GB', 'Voice Capture Ready').\n\n"
                "3. Workbench Features:\n"
                "- Tabbed Document Workspace: Simultaneous legal chat feed and embedded PDF.js document viewer;\n"
                "- Sidebar History & Isolation: Tabbed feed for global team searches and user-isolated conversation history;\n"
                "- Real-Time Voice Typing: Browser Web Speech API integration with microphone recording controls."
            )
        },
        {
            "heading": "10.7 Precedent Relationship Graph & Temporal Law Transition Engine",
            "content": (
                "To support statutory reforms and case-law analysis, two specialized scripts provide dedicated domain logic:\n\n"
                "1. Precedent Relationship Graph (`scripts/precedent_graph.py`):\n"
                "- Models landmark Supreme Court judgment networks, categorizing ratios decidendi into `FOLLOWED`, `OVERRULED`, `DISTINGUISHED`, `AFFIRMED`, and `DOUBTED`.\n"
                "- Graph traversal ensures that overruled cases (e.g., *Kameshwar Singh* overruled by 44th Amendment) trigger explicit warnings.\n\n"
                "2. Temporal Statutory Law Engine (`scripts/temporal_law.py`):\n"
                "- Intercepts queries citing repealed penal statutes (IPC 1860, CrPC 1973, Evidence Act 1872).\n"
                "- Automatically maps historical sections to modern equivalents under Bharatiya Nyaya Sanhita (BNS 2023), BNSS 2023, and BSA 2023, injecting statutory transition warnings."
            )
        }
    ]
}
