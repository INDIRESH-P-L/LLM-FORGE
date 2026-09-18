"""
scripts/report_part1.py
Chapters 1 through 5 of LegalMind AI Academic Project Report:
- Chapter 1: Introduction
- Chapter 2: Literature Review
- Chapter 3: Problem Statement and Objectives
- Chapter 4: System Analysis
- Chapter 5: System Architecture and Design
"""

CHAPTER_1 = {
    "number": 1,
    "title": "INTRODUCTION",
    "sections": [
        {
            "heading": "1.1 Background of Artificial Intelligence in Specialized Domains",
            "content": (
                "Artificial Intelligence (AI) has undergone a fundamental paradigm shift over the past decade, "
                "evolving from narrow, task-specific predictive classifiers toward expansive generative foundation models. "
                "Early machine learning architectures in specialized domains relied extensively on manual feature engineering, "
                "domain-specific lexicons, and supervised classifiers trained on narrow, curated datasets. While effective for "
                "constrained tasks such as document triage, named entity recognition, and sentiment classification, these systems "
                "failed when confronted with tasks requiring cross-document synthesis, complex relational reasoning, and open-ended "
                "textual generation.\n\n"
                "The advent of the deep Transformer architecture and self-supervised pre-training at scale demonstrated that "
                "probabilistic language models could internalize structural patterns, semantic hierarchies, and contextual "
                "relationships across vast multi-lingual corpora. However, high-stakes knowledge domains—predominantly law, medicine, "
                "and public policy—demand levels of factual veracity, strict adherence to statutory authority, and auditable reasoning "
                "chains that pure generative foundation models cannot natively guarantee. The application of AI to the legal domain, "
                "therefore, represents both the frontier of applied natural language understanding and an acute test of artificial "
                "veracity, reproducibility, and safety."
            )
        },
        {
            "heading": "1.2 Growth and Emergence of Large Language Models (LLMs)",
            "content": (
                "The rapid scaling of generative language models—spanning dense transformer models and sparse Mixture-of-Experts "
                "(MoE) architectures—has redefined computational linguistics. Modern foundation models, such as Alibaba Cloud's "
                "Qwen family, demonstrate advanced capabilities in zero-shot textual synthesis, logical code generation, multi-turn "
                "conversational coherence, and structured information extraction. The transition from dense parameters to sparse "
                "Mixture-of-Experts architectures, exemplified by models like Qwen3.6-35B-A3B (featuring 35 billion total parameters "
                "with approximately 3 billion active parameters routed per forward pass), allows state-of-the-art representation "
                "capacity while maintaining computational feasibility and high inference efficiency on modern hardware accelerators.\n\n"
                "Despite these remarkable generative abilities, standalone LLMs operate as closed-world parametric memory systems. "
                "Knowledge acquired during pre-training is static, non-verifiable, and prone to hallucinations—the confident generation "
                "of syntactically plausible but factually erroneous or non-existent propositions. In legal domains where authoritative "
                "precedents, statutory amendments, and procedural rules are subject to strict temporal validity, ungrounded parametric "
                "recall poses severe existential risks, including misquotation of statutes, invention of fictitious case citations, "
                "and failure to recognize legislative repeals."
            )
        },
        {
            "heading": "1.3 Introduction to Legal Artificial Intelligence",
            "content": (
                "Legal Artificial Intelligence (Legal AI) or Computational Jurisprudence is the interdisciplinary field dedicated "
                "to developing computational systems capable of understanding, interpreting, analyzing, and applying legal norms, "
                "statutory rules, and judicial precedents. In common-law legal systems, law is dynamic, discursive, and hierarchical: "
                "constitutional provisions sit at the pinnacle, followed by legislative enactments, executive delegated legislation, "
                "and binding judicial precedents governed by the doctrine of stare decisis. Legal reasoning requires an intricate balance "
                "of deductive statutory interpretation, analogical reasoning across precedents, and inductive harmonization of competing doctrines.\n\n"
                "Historically, Legal AI focused primarily on expert systems, deontic logic, and formal ontologies. While mathematically "
                "sound, symbolic approaches proved brittle when handling natural language legal briefs, unstructured judicial judgments, "
                "and linguistic ambiguities. The contemporary convergence of deep neural representation learning and symbolic legal "
                "knowledge bases offers a compelling solution: combining the fluid generative fluency of neural models with the verifiable, "
                "auditable grounding of indexed legal corpora."
            )
        },
        {
            "heading": "1.4 Challenges in Accessing Indian Legal Information",
            "content": (
                "The Republic of India possesses one of the world's most extensive, intricate, and voluminous legal systems. "
                "Several structural factors contribute to acute informational friction:\n\n"
                "1. Enormous Corpus Volume: The Indian legal universe encompasses the Constitution of India (the world's longest written "
                "national constitution with 448 articles across 25 parts and 12 schedules), thousands of Union enactments, tens of thousands "
                "of state statutes, and millions of reported judgments from the Supreme Court of India, 25 High Courts, and specialized tribunals.\n\n"
                "2. The Historic 2024 Criminal Code Transition: On July 1, 2024, the colonial-era penal framework—the Indian Penal Code (IPC, 1860), "
                "the Code of Criminal Procedure (CrPC, 1973), and the Indian Evidence Act (IEA, 1872)—was formally repealed and replaced by "
                "the Bharatiya Nyaya Sanhita (BNS, 2023), Bharatiya Nagarik Suraksha Sanhita (BNSS, 2023), and Bharatiya Sakshya Adhiniyam (BSA, 2023). "
                "This transition created profound legal complexities: criminal offenses committed prior to July 1, 2024, continue to be substantive "
                "under the IPC, while ongoing investigations and trials must navigate intricate savings clauses and modern procedural timelines.\n\n"
                "3. Multilingual and Multi-Jurisdictional Divergence: Legal disputes often involve local state amendments, specialized rules, "
                "and vernacular evidentiary documentation, making unified legal discovery challenging for practitioners, litigants, and researchers.\n\n"
                "4. Commercial Paywalls and Research Costs: Leading commercial legal search engines operate behind expensive subscription "
                "paywalls, rely on rigid Boolean keyword search, and provide raw lists of unsummarized court judgments rather than structured, "
                "synthesized answers to specific legal questions."
            )
        },
        {
            "heading": "1.5 Need for an AI-Powered Legal Question Answering System",
            "content": (
                "Given the overwhelming volume of legal data and the high cost of legal consultation, there is an urgent societal and "
                "institutional necessity for an AI-powered legal research assistant. Such a system must not act as a mere keyword search tool, "
                "nor as an unconstrained generative chatbot. Instead, it must serve as an intelligent, evidence-grounded research partner "
                "capable of:\n"
                "- Interpreting natural language legal inquiries;\n"
                "- Retrieving the exact, authoritative sections of relevant acts and landmark precedents;\n"
                "- Synthesizing clear, legally sound explanations structured according to standard legal reasoning methodologies (such as IRAC: "
                "Issue, Rule, Application, Conclusion);\n"
                "- Providing clickable, verified, pinpoint citations to statutory provisions and official law reporters;\n"
                "- Explicitly disclaiming uncertainties and warning users when statutory evidence is insufficient or when laws have been repealed."
            )
        },
        {
            "heading": "1.6 Introduction to LegalMind AI",
            "content": (
                "LegalMind AI is an enterprise-grade Indian legal research and question-answering system developed to bridge the gap "
                "between vast public legal archives and accessible, verifiable legal reasoning. Deployed on an enterprise NVIDIA DGX B200 "
                "high-performance computing platform, LegalMind AI combines an authoritative statutory and constitutional knowledge base "
                "with an advanced Retrieval-Augmented Generation (RAG) pipeline.\n\n"
                "The system incorporates a hybrid retrieval architecture that synergizes lexical BM25 token matching with dense vector "
                "embeddings (BAAI/bge-m3) over 15,057 curated legislation chunks and 26,097 multi-scale FAISS vectors. Retrieved evidence is "
                "reranked using a neural cross-encoder (BAAI/bge-reranker-v2-m3) and fed into the Qwen3.6-35B-A3B Mixture-of-Experts language "
                "model running in native Bfloat16 (BF16) precision. The base model's self-attention layers are modulated by a domain-adapted "
                "Low-Rank Adaptation (LoRA) adapter (lora-legal-v2), fine-tuned specifically on Indian legal instructions. The system features "
                "a secure FastAPI HTTPS backend (port 8443), SQLite Write-Ahead Logging (WAL) chat persistence, and an Apple-inspired Liquid Glass "
                "web interface supporting voice input, document uploads, and dynamic citation exploration."
            )
        },
        {
            "heading": "1.7 Project Motivation",
            "content": (
                "The primary motivation for LegalMind AI stems from the constitutional promise of Equal Justice and Free Legal Aid enshrined "
                "in Article 39A of the Constitution of India. Legal literacy in India remains critically low, and the cost of formal legal advice "
                "is prohibitive for large sections of the populace, small enterprises, and law students. Furthermore, legal professionals spend "
                "countless non-billable hours conducting repetitive statutory lookup and precedent cross-referencing.\n\n"
                "By engineering an open, verifiable, locally deployable, and computationally efficient AI legal assistant, this project seeks "
                "to democratize legal knowledge, enhance judicial productivity, and establish an academic benchmark for verifiable generative "
                "AI systems operating in high-stakes statutory environments."
            )
        },
        {
            "heading": "1.8 Problem Statement",
            "content": (
                "To design, implement, optimize, and empirically validate an end-to-end, evidence-grounded legal question answering system "
                "for Indian constitutional and statutory law that:\n"
                "1. Overcomes the severe hallucination, temporal obsolescence, and citation fabrication weaknesses of standalone foundation LLMs;\n"
                "2. Seamlessly integrates hybrid lexical (BM25) and dense semantic (BGE-M3 + FAISS) retrieval across over 15,000 statutory provisions;\n"
                "3. Enhances domain-specific legal reasoning through standard LoRA fine-tuning on a 35-billion-parameter MoE model in native BF16 "
                "precision without catastrophic forgetting or numerical instability (0 NaNs/Infs);\n"
                "4. Provides verifiable, pinpoint citations, temporal awareness of statutory repeals (IPC to BNS transition), and fail-safe "
                "automatic fallback to the base model in the event of adapter anomaly;\n"
                "5. Delivers responsive, secure HTTPS-enabled conversational interaction with persistent SQLite storage and rich client-side rendering."
            )
        },
        {
            "heading": "1.9 Aim and Objectives of the Project",
            "content": (
                "AIM:\n"
                "The overarching aim of this project is to create an authoritative, reproducible, and verifiable AI legal assistant that delivers "
                "high-accuracy, citation-grounded answers to complex Indian legal queries using an optimized RAG and LoRA architecture.\n\n"
                "SPECIFIC OBJECTIVES:\n"
                "1. Corpus Curation and Indexing: Ingest, clean, segment, and index the complete Constitution of India, major central enactments, "
                "and landmark Supreme Court decisions into 15,057 provision-aware chunks and 26,097 FAISS vector embeddings.\n"
                "2. Multi-Stage Hybrid Retrieval Engine: Implement a two-stage retrieval pipeline combining Okapi BM25 and dense BGE-M3 semantic "
                "search fused via reciprocal rank fusion, refined through cross-encoder neural reranking.\n"
                "3. Model Adaptation via Standard LoRA: Fine-tune the Qwen3.6-35B-A3B foundation model using standard BF16 LoRA (r=16, alpha=32) "
                "targeting self-attention projections, enforcing immediate gradient and weight NaN/Inf detection.\n"
                "4. Evidence Grounding and Citation Verification: Implement automated regex and semantic citation extractors to cross-verify generated "
                "provisions against retrieved evidence, suppressing internal model thinking tokens and computing objective hallucination scores.\n"
                "5. Temporal Law Engine: Build algorithmic transition mappings between repealed colonial statutes (IPC, CrPC, IEA) and modern enactments "
                "(BNS, BNSS, BSA) with automatic user warnings.\n"
                "6. System Verification and Benchmarking: Execute comprehensive comparative benchmarks across ten mandatory legal categories, "
                "evaluating grounding, hallucination, latency, and numerical stability on the enterprise NVIDIA DGX B200 environment."
            )
        },
        {
            "heading": "1.10 Scope of the Project",
            "content": (
                "The scope of LegalMind AI includes:\n"
                "- Substantive coverage of the Constitution of India (Fundamental Rights, Directive Principles, Union/State Executive, Judiciary, "
                "Emergency Provisions, Amendments);\n"
                "- Substantive coverage of core Indian statutory regimes: Indian Penal Code (1860) and Bharatiya Nyaya Sanhita (2023), Code of Criminal "
                "Procedure (1973) and Bharatiya Nagarik Suraksha Sanhita (2023), Indian Evidence Act (1872) and Bharatiya Sakshya Adhiniyam (2023), "
                "Indian Contract Act (1872), Companies Act (1956 and 2013), Arbitration and Conciliation Act (1996), Information Technology Act (2000);\n"
                "- Landmark Supreme Court precedents establishing foundational constitutional doctrines (e.g., Basic Structure, Substantive Due Process, "
                "Arbitrariness Test);\n"
                "- Full software deployment stack: local GPU inference on DGX B200, FastAPI backend, SQLite persistence, and web client.\n\n"
                "DELIMITATIONS:\n"
                "- The system is designed as an informational and research assistant; it does not substitute for licensed legal counsel or advocate advocacy.\n"
                "- State-level statutory enactments and subordinate administrative notifications not present in the ingested 15,057 chunks are excluded.\n"
                "- The current system is optimized for English-language legal discourse; multilingual vernacular processing remains designated for future scope."
            )
        },
        {
            "heading": "1.11 Contributions of the Project",
            "content": (
                "The primary engineering and academic contributions of this work include:\n"
                "1. Architectural Synergy: A proven, robust integration of hybrid BM25 + dense vector retrieval, cross-encoder reranking, and "
                "LoRA-adapted 35B MoE generation tailored specifically to the structural conventions of Indian jurisprudence.\n"
                "2. Dynamic Retrieval Context Optimization: Empirical identification and resolution of context-window bloat by replacing static "
                "retrieval floors (top_k=max(k, 14)) with dynamic, caller-specified top_k bounds, eliminating irrelevant distractor chunks and "
                "optimizing GPU KV-cache allocation.\n"
                "3. Safe, Stable LoRA Adaptation Protocol: Formulation and validation of a strict forensic verification protocol that detects and rejects "
                "corrupt adapter weights (e.g., legacy lora-legal-v1 containing 320 NaN tensors) while successfully deploying clean adapters "
                "(lora-legal-v2 with 3,440,640 verified clean parameters) with zero-downtime automatic fallback.\n"
                "4. Empirical Benchmark on DGX B200: Delivery of a 10-category comparative benchmark demonstrating that LoRA + RAG achieves higher grounding "
                "(0.692 vs. 0.668) and lower citation hallucination (0.308 vs. 0.332) than the base model alone.\n"
                "5. Comprehensive, Verifiable Engineering: Development of 171 passing automated tests covering all modules, full persistence durability "
                "via SQLite WAL, and an intuitive, accessible Liquid Glass web interface."
            )
        },
        {
            "heading": "1.12 Organization of the Report",
            "content": (
                "The remainder of this report is organized as follows:\n"
                "- Chapter 2 presents a thorough Literature Review of legal information retrieval, RAG, and fine-tuning techniques.\n"
                "- Chapter 3 defines the formal Problem Statement, system requirements, and technical specifications.\n"
                "- Chapter 4 provides the System Analysis, feasibility study, risk management, and use-case specifications.\n"
                "- Chapter 5 details the complete System Architecture and Design across all hardware, network, and software layers.\n"
                "- Chapter 6 reviews the Technologies and Tools Used, discussing selection rationale and trade-offs.\n"
                "- Chapter 7 describes the Dataset and Knowledge Base, chunking methodologies, and indexing metrics.\n"
                "- Chapter 8 articulates the 18-stage end-to-end Methodology.\n"
                "- Chapter 9 explains Model Development, LoRA Fine-Tuning, safety audits, and tensor verification.\n"
                "- Chapter 10 presents System Implementation, API endpoints, and process management.\n"
                "- Chapter 11 documents Testing and Validation, detailing test cases and verification results.\n"
                "- Chapter 12 discusses Empirical Results, the Article 21 case study, latency profiles, and quality observations.\n"
                "- Chapter 13 analyzes Security, Privacy, Ethics, and Responsible AI guidelines.\n"
                "- Chapter 14 examines System Limitations and operational constraints.\n"
                "- Chapter 15 outlines Future Enhancements and research directions.\n"
                "- Chapter 16 concludes the report with a synthesis of contributions and future outlook.\n"
                "- References and Appendices provide complete technical specifications, payloads, test cases, and glossary."
            )
        }
    ]
}

CHAPTER_2 = {
    "number": 2,
    "title": "LITERATURE REVIEW",
    "sections": [
        {
            "heading": "2.1 Traditional Legal Information Retrieval Systems",
            "content": (
                "Early legal information retrieval (IR) systems emerged in the late 1960s and 1970s with commercial platforms such as LEXIS "
                "and Westlaw. These foundational systems relied almost exclusively on Boolean logic, inverted index keyword search, and manually "
                "curated taxonomy trees (such as the West American Digest System). Users formulated queries using strict operators (AND, OR, NOT, "
                "WITHIN /p, /s). While Boolean systems provide high precision for seasoned law librarians seeking specific statutory numbers or "
                "party names, they exhibit severe recall limitations when legal concepts are expressed through differing judicial phraseology, "
                "synonyms, or implicit doctrinal arguments. Furthermore, traditional IR systems merely return unstructured document lists, leaving "
                "the laborious task of reading, cross-referencing, and synthesizing answers entirely to human legal researchers."
            )
        },
        {
            "heading": "2.2 Rule-Based Legal Expert Systems and Formal Logic",
            "content": (
                "During the 1980s and 1990s, artificial intelligence research in law focused heavily on symbolic AI and expert systems, "
                "such as the British Nationality Act expert system developed by Sergot et al. and Ashley's HYPO system for case-based reasoning. "
                "These systems attempted to formalize statutory rules into first-order predicate logic, production rules (IF-THEN), and deontic "
                "logics governing permissions and obligations. While conceptually elegant, symbolic expert systems proved brittle and commercially "
                "unscalable: human statutory drafting inherently relies on 'open-textured' legal concepts—such as 'reasonable care,' 'good faith,' "
                "or 'public interest'—which cannot be rigidly encoded into deterministic logic gates without generating catastrophic combinatorial explosions."
            )
        },
        {
            "heading": "2.3 Machine Learning in Legal Natural Language Processing",
            "content": (
                "The transition toward statistical Natural Language Processing (NLP) in the 2000s introduced supervised machine learning techniques "
                "(Support Vector Machines, Naive Bayes, Conditional Random Fields) for legal document categorization, named entity recognition "
                "(identifying judges, petitioners, statutes), and judgment outcome prediction. While statistical classifiers improved scalability, "
                "they lacked semantic depth, treating legal documents as 'bags of words' or n-gram sequences without capturing intricate syntactical "
                "dependencies or long-range doctrinal hierarchies spanning dozens of pages."
            )
        },
        {
            "heading": "2.4 Transformer-Based Language Models in the Legal Domain",
            "content": (
                "The introduction of the Transformer architecture by Vaswani et al. (2017) revolutionized NLP through multi-head self-attention "
                "mechanisms that capture bidirectional contextual relationships across tokens. Following the release of BERT (Devlin et al., 2018), "
                "researchers recognized that general-domain pre-trained models struggled with the specialized archaic vocabulary, Latin maxims, "
                "and syntactic complexity of legal texts. This led to domain-specific masked language models such as Legal-BERT (Chalkidis et al., 2020) "
                "and InLegalBERT (Paul et al., 2022), which were pre-trained on Indian Supreme Court and High Court judgments. While these models "
                "established state-of-the-art benchmarks for legal text classification, clause extraction, and bail prediction, their encoder-only "
                "nature precluded generative question answering and complex legal synthesis."
            )
        },
        {
            "heading": "2.5 Generative Foundation Models and Retrieval-Augmented Generation (RAG)",
            "content": (
                "The emergence of autoregressive Large Language Models (LLMs)—such as OpenAI's GPT-4, Alibaba Cloud's Qwen series, and Meta's Llama—"
                "demonstrated extraordinary generative fluency across natural language tasks. However, researchers quickly discovered that LLMs deployed "
                "as standalone legal advisors suffer from severe hallucinations, temporal obsolescence, and citation fabrication. Lewis et al. (2020) "
                "proposed Retrieval-Augmented Generation (RAG) to resolve these fundamental flaws. In a RAG architecture, an external non-parametric "
                "memory (retriever) searches a curated document knowledge base in response to a user query, supplying the retrieved text chunks as "
                "explicit grounding context within the prompt of a parametric generative model (LLM). This decouples world knowledge from model parameters, "
                "enabling real-time updates, verifiable citation tracing, and substantial reductions in factual hallucination."
            )
        },
        {
            "heading": "2.6 Dense Vector Retrieval vs. Sparse Lexical Search",
            "content": (
                "Early RAG architectures relied primarily on dense vector embeddings (e.g., Sentence-BERT, Contriever) coupled with approximate "
                "nearest neighbor (ANN) search algorithms such as FAISS (Johnson et al., 2019) or HNSW. Dense retrieval maps queries and passages "
                "into a continuous semantic embedding space, excelling at capturing conceptual synonyms and paraphrased questions. However, in legal "
                "jurisprudence, dense retrieval exhibits a critical failure mode: it often fails on exact alphanumeric queries, such as 'Section 438 CrPC' "
                "or 'Article 359(1)'. Conversely, traditional probabilistic lexical algorithms like BM25 (Robertson and Zaragoza, 2009) utilize exact term "
                "frequencies and inverse document frequencies, guaranteeing precision on statutory provision numbers but failing when users phrase queries "
                "in colloquial terms (e.g., 'anticipatory bail' without naming the section). Consequently, recent literature advocates for Hybrid "
                "Retrieval systems that fuse BM25 and dense representations via Reciprocal Rank Fusion (RRF) (Cormack et al., 2009)."
            )
        },
        {
            "heading": "2.7 Cross-Encoder Neural Reranking",
            "content": (
                "While bi-encoder dense embedding models allow pre-computation of document vectors for millisecond-scale retrieval, they compress "
                "entire passages into single vector points, losing fine-grained word-level interactions between query and context. Cross-encoder "
                "rerankers, such as BAAI's BGE-Reranker-v2-m3 (Xiao et al., 2024), concatenate query and candidate passages into a single sequence, "
                "passing them through full multi-head cross-attention layers. This allows the model to capture intricate contextual interactions, "
                "scoring passage relevance with exceptional accuracy. Studies show that inserting a cross-encoder reranker between initial candidate "
                "retrieval and LLM prompt generation dramatically boosts retrieval precision (Hit@1 and MRR), directly elevating generation quality."
            )
        },
        {
            "heading": "2.8 Parameter-Efficient Fine-Tuning (PEFT) and Low-Rank Adaptation (LoRA)",
            "content": (
                "While RAG provides necessary factual grounding, base foundation LLMs often fail to adopt the professional tone, logical structure, "
                "and deductive reasoning conventions expected in formal legal analysis. Full fine-tuning of 35-billion-parameter models is computationally "
                "prohibitive and carries severe risks of catastrophic forgetting. Hu et al. (2021) introduced Low-Rank Adaptation (LoRA), a parameter-efficient "
                "fine-tuning (PEFT) technique that freezes pre-trained model weights W_0 and injects trainable rank decomposition matrices (A and B) "
                "into the attention projections: W = W_0 + (B * A) * (alpha / r). For r << d, this reduces trainable parameters by over 99.9%, drastically "
                "lowers VRAM requirements, enables rapid modular swapping of domain adapters, and preserves the general linguistic capabilities of the base model."
            )
        },
        {
            "heading": "2.9 Existing Legal Chatbots and Critical Research Gap",
            "content": (
                "A survey of existing legal AI tools reveals several fundamental limitations:\n"
                "1. Reliance on Proprietary Cloud APIs: Most commercial legal assistants (e.g., Harvey AI, Casetext CoCounsel) rely on closed-source "
                "APIs (OpenAI GPT-4), creating insurmountable data privacy concerns for sensitive client briefs and judicial records.\n"
                "2. Lack of Indian Jurisprudential Grounding: Western legal models are pre-trained predominantly on US Federal Code or English common law, "
                "failing entirely on Indian constitutional doctrines (e.g., Basic Structure doctrine, Golden Triangle doctrine) and recent criminal statutory "
                "reforms (IPC/BNS transitions).\n"
                "3. Brittle Quantization and Degradation: Many open-weights deployments utilize aggressive 4-bit/8-bit quantization (QLoRA, AWQ), which "
                "has been shown to induce subtle reasoning errors, degradation in long-tail citation extraction, and numerical instability in MoE architectures.\n\n"
                "RESEARCH GAP ADDRESSED BY LEGALMIND AI:\n"
                "There is an absence of an open-weights, locally deployable Indian legal assistant that unifies hybrid BM25 + dense BGE-M3 retrieval, "
                "neural cross-encoder reranking, native BF16 MoE inference (Qwen3.6-35B-A3B), verified LoRA adaptation, and automated citation cross-verification. "
                "LegalMind AI was designed and empirically validated to fill this critical academic and operational void."
            )
        }
    ]
}

CHAPTER_3 = {
    "number": 3,
    "title": "PROBLEM STATEMENT AND OBJECTIVES",
    "sections": [
        {
            "heading": "3.1 Detailed Problem Statement",
            "content": (
                "In legal research, accuracy is not a secondary objective—it is a non-negotiable prerequisite. Legal practitioners, judges, "
                "scholars, and citizens navigating Indian statutory frameworks face severe informational barriers. The Indian legal corpus consists "
                "of millions of pages of complex statutes, constitutional articles, and judicial precedents. The landmark transition on July 1, 2024, "
                "from colonial criminal codes (IPC, CrPC, IEA) to modern enactments (BNS, BNSS, BSA) has amplified these challenges, requiring legal "
                "researchers to reconcile repealed provisions with modern sections, navigate interim savings clauses, and maintain temporal validity.\n\n"
                "Standalone commercial Large Language Models (LLMs) cannot solve this problem due to three structural flaws: (1) hallucination of fictitious "
                "case citations and legal sections; (2) knowledge cutoff dates that ignore recent legislative transitions; and (3) complete absence of "
                "auditable, pinpoint source grounding. The technical problem addressed by this project is the design, deployment, and validation of an "
                "enterprise-grade Indian legal question-answering architecture that provides evidence-grounded, citation-verified, and temporally aware "
                "legal synthesis using local high-performance hardware."
            )
        },
        {
            "heading": "3.2 Limitations of the Existing System",
            "content": (
                "Conventional Indian legal research systems (e.g., traditional CD-ROM databases, basic online portals, and general search engines) "
                "suffer from severe operational limitations:\n"
                "- Rigid Keyword Matching: Systems fail if queries do not match the exact syntax used by statutory draftsmen;\n"
                "- Information Fragmentation: Legislation and judicial interpretations are siloed, requiring researchers to cross-reference multiple "
                "disconnected sources manually;\n"
                "- Document Dumping: Existing search engines return long lists of 50-page PDF judgments without extracting the specific ratio decidendi "
                "or applying it to the user's factual context;\n"
                "- Zero Synthesis: Conventional tools cannot synthesize statutory elements into standard legal frameworks (e.g., IRAC);\n"
                "- Temporal Blindness: Traditional keyword searches frequently present repealed sections without clear legal warnings regarding modern replacements."
            )
        },
        {
            "heading": "3.3 Proposed LegalMind AI Architecture and Advantages",
            "content": (
                "The proposed LegalMind AI system addresses these deficiencies through a multi-tier, evidence-grounded architecture:\n"
                "- Hybrid Multi-Scale Retrieval: Blends exact lexical BM25 token matching with 1024-dimensional dense semantic vectors (BGE-M3) "
                "indexed via FAISS, ensuring that both specific section numbers and conversational concepts are retrieved with high recall;\n"
                "- Neural Cross-Encoder Reranking: Implements BGE-reranker-v2-m3 to score and filter candidate chunks, eliminating noise before prompt assembly;\n"
                "- Native BF16 Qwen3.6-35B MoE Generation: Operates a 35B parameter Mixture-of-Experts foundation model in native Brain Floating Point (BF16), "
                "eliminating quantization degradation and maintaining deep semantic precision;\n"
                "- Specialized LoRA Adapter (lora-legal-v2): Injects domain-adapted attention weights fine-tuned on legal instructions, enforcing "
                "structured legal analysis while strictly avoiding gradient NaNs/Infs;\n"
                "- Dynamic Fail-Safe Architecture: Implements runtime safetensors forensic validation with automatic zero-downtime fallback to the base model;\n"
                "- Strict Grounding and Anti-Hallucination: Automatically cross-checks generated citations against retrieved text, suppressing internal "
                "thinking tokens and attaching verifiable citations and legal disclaimers."
            )
        },
        {
            "heading": "3.4 Functional Requirements Specification",
            "content": (
                "The functional requirements of LegalMind AI define the core behavioral capabilities of the system:\n"
                "FR-01 (Natural Language Ingestion): Accept complex natural language queries regarding Indian constitutional law, criminal law, and civil statutes.\n"
                "FR-02 (Constitutional Detection): Automatically detect queries concerning fundamental rights and landmark doctrines (Articles 14, 19, 21, Golden Triangle) "
                "and route them to specialized constitutional retrieval paths.\n"
                "FR-03 (Hybrid Retrieval Execution): Concurrently query BM25 inverted indexes and FAISS dense vector spaces, fusing results via reciprocal rank fusion.\n"
                "FR-04 (Cross-Encoder Neural Reranking): Rescore top candidate passages using a cross-encoder model to select the top-k most relevant evidence chunks.\n"
                "FR-05 (Evidence-Grounded Generation): Construct structured prompts enforcing strict adherence to retrieved evidence, generating answers formatted "
                "into direct answers, relevant provisions, judicial precedents, legal reasoning, and limitations.\n"
                "FR-06 (Temporal Status Reporting): Detect references to repealed statutes (IPC, CrPC, IEA) and append explicit warnings detailing modern BNS, BNSS, "
                "and BSA equivalents.\n"
                "FR-07 (Pinpoint Citation Attachment): Extract and verify all statutory and judicial citations against retrieved chunks, flagging ungrounded claims.\n"
                "FR-08 (Persistent Conversation Storage): Persist user queries, assistant responses, citations, and latency telemetry in SQLite with WAL mode.\n"
                "FR-09 (System Health Telemetry): Expose /health and /stats endpoints reporting LLM device status, retriever state, LoRA adapter status, and GPU memory.\n"
                "FR-10 (Multimodal Document Processing): Support asynchronous multi-page PDF, DOCX, and TXT parsing, Tesseract OCR confusion matrix correction, local Whisper voice typing, and targeted paragraph context management.\n"
                "FR-11 (Precedent Relationship Graph Traversal): Query curated Supreme Court precedent network for followed, overruled, distinguished, and affirmed ratio decidendi.\n"
                "FR-12 (Antigravity Terminal CLI): Provide full-screen developer assistant (`bin/legalmind`) with token streaming, animated status bars, and slash commands (`/voice`, `/upload`, `/ocr`, `/speak`).\n"
                "FR-13 (Apple Space Black Web UI): Deliver minimalist macOS Space Black design system with fluid iOS 18 glassmorphism, Dynamic Island navigation bar, and tabbed workspace."
            )
        },
        {
            "heading": "3.5 Non-Functional Requirements Specification",
            "content": (
                "NFR-01 (Veracity and Factuality): The system must strictly prioritize retrieved evidence over parametric knowledge, maintaining a citation "
                "hallucination rate below 0.35 on supported statutory queries.\n"
                "NFR-02 (Numerical Stability): Model weights, gradients, and adapter tensors must exhibit zero NaN (Not-a-Number) and zero Inf (Infinity) values.\n"
                "NFR-03 (System Reliability and Fault Tolerance): In the event of adapter corruption or missing files, the system must automatically fall back to "
                "the base model without unhandled exceptions or service termination.\n"
                "NFR-04 (Data Privacy and Security): User queries and chat history must remain entirely on the local enterprise server; no data may be leaked "
                "to third-party commercial APIs. Web communication must enforce TLS encryption via HTTPS.\n"
                "NFR-05 (Maintainability and Modularity): The architecture must cleanly separate retrieval, model inference, chat storage, multimodal services, and presentation "
                "layers through well-defined Python modules and API contracts."
            )
        },
        {
            "heading": "3.6 Hardware and Software Requirements",
            "content": (
                "HARDWARE REQUIREMENTS (Enterprise Compute):\n"
                "- System: NVIDIA DGX B200 High-Performance Computing Server\n"
                "- GPUs: 8x NVIDIA B200 GPUs (180 GB HBM3e VRAM per GPU)\n"
                "- LLM Device Allocation: GPU 1 (dedicated to Qwen3.6-35B-A3B inference in BF16, ~70 GB VRAM consumption)\n"
                "- Retrieval Device Allocation: GPU 4 (dedicated to BAAI/bge-m3 embeddings and BAAI/bge-reranker-v2-m3 reranker, ~9 GB VRAM consumption)\n"
                "- System Constraints: GPU 3 strictly excluded from all workloads per institutional server policies\n"
                "- System RAM: 1.5 TB Host Memory; Storage: High-speed NVMe SSD storage array\n\n"
                "SOFTWARE REQUIREMENTS:\n"
                "- Operating System: Linux (Ubuntu 22.04 LTS / enterprise Linux environment)\n"
                "- Core Frameworks: Python 3.12, PyTorch 2.x with CUDA 12.x, CUDA_HOME=/usr/local/cuda\n"
                "- NLP / Deep Learning Libraries: Hugging Face Transformers, PEFT, Safetensors, SentenceTransformers, Rank-BM25, FAISS, Faster-Whisper, PyTesseract\n"
                "- Web Services: FastAPI 0.115+, Uvicorn (ASGI server), Pydantic v2\n"
                "- Persistence: SQLite 3 with Write-Ahead Logging (WAL) and FTS5 full-text indexing\n"
                "- Client Interfaces: Antigravity Developer Terminal CLI (`bin/legalmind`, Rich, Prompt-Toolkit), Apple Space Black Web UI (HTML5, Vanilla CSS3, ES2022 JavaScript, PDF.js)"
            )
        }
    ]
}

CHAPTER_4 = {
    "number": 4,
    "title": "SYSTEM ANALYSIS",
    "sections": [
        {
            "heading": "4.1 Feasibility Study",
            "content": (
                "A comprehensive feasibility study was conducted across technical, operational, economic, legal, and ethical dimensions "
                "prior to architectural implementation:\n\n"
                "1. Technical Feasibility: The project relies on proven open-weights foundation models (Qwen3.6-35B-A3B) and well-established "
                "retrieval algorithms (BM25, FAISS). The availability of the NVIDIA DGX B200 server—with 180 GB VRAM per accelerator—renders "
                "hosting a 35B Mixture-of-Experts model in native BF16 entirely feasible without requiring lossy 4-bit quantization. GPU memory "
                "profiling confirmed that loading Qwen3.6-35B consumes ~70 GB on GPU 1, leaving over 100 GB headroom for dynamic KV-cache allocation.\n\n"
                "2. Operational Feasibility: Legal practitioners, scholars, and litigants operate in fast-paced environments requiring simple web "
                "and terminal access. The system provides a responsive, single-page web interface with zero software installation requirements for "
                "end users, alongside an Antigravity-style terminal CLI for command-line power users.\n\n"
                "3. Economic Feasibility: Utilizing open-weights foundation models, open-source libraries (Hugging Face, FAISS, FastAPI), and existing "
                "institutional computing infrastructure eliminates recurring commercial API token fees (e.g., OpenAI, Anthropic), resulting in near-zero "
                "marginal operational costs per query.\n\n"
                "4. Legal and Ethical Feasibility: Ingested statutes, constitutional articles, and Supreme Court judgments are public domain legal documents "
                "under Indian copyright law (Section 52(1)(q) of the Copyright Act, 1957). The system incorporates prominent disclaimers affirming "
                "its educational/informational nature to comply with Bar Council of India guidelines."
            )
        },
        {
            "heading": "4.2 Systemic Risk Analysis and Mitigation Strategies",
            "content": (
                "The deployment of an AI legal assistant involves critical operational and safety risks that required rigorous mitigation:\n\n"
                "1. Risk: LoRA Adapter Numerical Corruption (NaN/Inf Values)\n"
                "- Impact: Catastrophic generation failure, outputting repetitive exclamation marks or crashing inference.\n"
                "- Mitigation: Implemented automated pre-inference safetensors forensic audits and in-memory VRAM tensor checks. If any NaN/Inf is "
                "detected (as occurred with legacy lora-legal-v1), the system automatically rejects the adapter and falls back to Base Model + RAG.\n\n"
                "2. Risk: Legal Hallucination and Fabricated Authorities\n"
                "- Impact: Misleading users with non-existent statutory sections or fake court citations.\n"
                "- Mitigation: Implemented automated regex-based citation extraction and cross-verification against retrieved chunks. Answers explicitly "
                "declare when statutory evidence is insufficient.\n\n"
                "3. Risk: Reliance on Repealed Colonial Statutes\n"
                "- Impact: Citing repealed IPC provisions in modern criminal proceedings.\n"
                "- Mitigation: Engineered a dedicated Indian Temporal Law Engine that intercepts queries containing repealed provisions (IPC, CrPC, IEA) "
                "and injects statutory warnings detailing corresponding modern sections (BNS, BNSS, BSA).\n\n"
                "4. Risk: Prompt Injection and Context Manipulation\n"
                "- Impact: Adversarial user prompts attempting to override legal guardrails.\n"
                "- Mitigation: User prompts and retrieved evidence passages are strictly wrapped within XML/Markdown boundary delimiters and immutable "
                "system prompts."
            )
        },
        {
            "heading": "4.3 User Personas and Use Cases",
            "content": (
                "LegalMind AI serves four distinct user personas:\n"
                "1. Legal Practitioners & Advocates: Rapidly retrieve statutory cross-references, verify procedural rules (e.g., anticipatory bail under "
                "Section 482 BNSS), and review landmark precedents to prepare case briefs.\n"
                "2. Law Students & Academic Researchers: Explore constitutional doctrines (e.g., Basic Structure, Arbitrariness under Article 14), "
                "understand legislative histories, and study statutory transitions.\n"
                "3. Judicial Clerks & Legal Researchers: Perform exhaustive multi-index searches to identify relevant statutory authorities and precedent "
                "relationships across Union enactments.\n"
                "4. Pro Se Litigants & Citizens: Gain basic legal literacy regarding fundamental rights, consumer protections, and procedural requirements "
                "in clear, structured natural language."
            )
        },
        {
            "heading": "4.4 Comprehensive Use-Case Specification Table",
            "content": (
                "Table 4.2 details primary system use cases, participating actors, preconditions, operational workflows, and expected results:\n\n"
                "USE CASE 01: Constitutional Query Processing\n"
                "- Actor: Law Student / Legal Scholar\n"
                "- Preconditions: FastAPI backend active, hybrid retriever initialized, Qwen3.6-35B loaded in VRAM.\n"
                "- Workflow: User enters 'Explain the Golden Triangle under Articles 14, 19, and 21'. The system detects constitutional intent, retrieves "
                "balanced chunks across all three provisions and landmark rulings (Maneka Gandhi, Royappa), reranks passages, and generates a structured IRAC synthesis.\n"
                "- Expected Result: Comprehensive answer explaining interrelation, citing Article 14, 19, 21, and Royappa, with verified citations.\n\n"
                "USE CASE 02: Statutory Transition Lookup (IPC to BNS)\n"
                "- Actor: Criminal Defense Advocate\n"
                "- Preconditions: Temporal law engine loaded with modern criminal codes.\n"
                "- Workflow: User enters 'What is the punishment for murder under Section 302 IPC?'. The system identifies the query as targeting a repealed enactment, "
                "maps Section 302 IPC to Section 103 BNS, retrieves Section 103 BNS chunks, and attaches a prominent temporal warning.\n"
                "- Expected Result: Dual-perspective answer detailing the transition to Section 103 BNS, highlighting mob lynching provisions under Section 103(2), "
                "and providing statutory warnings.\n\n"
                "USE CASE 03: Procedural Bail Requirement Lookup\n"
                "- Actor: Legal Practitioner\n"
                "- Preconditions: BNSS and CrPC indexes loaded in FAISS.\n"
                "- Workflow: User queries 'What are the conditions for anticipatory bail under Section 438 CrPC?'. System retrieves Section 482 BNSS text, "
                "reranks conditions, and synthesizes jurisdictional requirements.\n"
                "- Expected Result: Structured summary detailing High Court / Sessions Court jurisdiction, non-bailable accusation preconditions, and judicial conditions.\n\n"
                "USE CASE 04: Non-Legal / Unsupported Query Handling\n"
                "- Actor: General User\n"
                "- Preconditions: System operating with strict uncertainty guardrails.\n"
                "- Workflow: User enters 'Explain Schrödinger wave equations in quantum mechanics'. Retriever returns low-relevance statutory chunks. System "
                "detects total absence of legal relevance, formulates an explicit disclaimer regarding lack of legal evidence, and answers based solely on general knowledge.\n"
                "- Expected Result: Explicit disclaimer stating retrieved Indian legal corpus contains no physics data, followed by clear demarcated explanation."
            )
        }
    ]
}

CHAPTER_5 = {
    "number": 5,
    "title": "SYSTEM ARCHITECTURE AND DESIGN",
    "sections": [
        {
            "heading": "5.1 High-Level Architectural Decomposition",
            "content": (
                "LegalMind AI is engineered following a modular, multi-tier micro-service architecture comprising five decoupled layers:\n"
                "1. Client Presentation Layer: Responsive single-page web interface (HTML5, Vanilla CSS3, ES2022 JavaScript) featuring an Apple-inspired "
                "Liquid Glass aesthetic, PDF.js document viewer, Web Speech API voice capture, and dual-feed sidebar history.\n"
                "2. Network and Transport Layer: Asynchronous ASGI server (Uvicorn) with Transport Layer Security (TLS 1.3) termination on HTTPS port 8443 "
                "and plain HTTP fallback on port 8080.\n"
                "3. Application & API Layer: FastAPI framework providing RESTful endpoints (/query, /health, /stats, /temporal/check, /precedents), "
                "request validation (Pydantic v2), user session isolation, and structured error propagation.\n"
                "4. Multi-Stage Retrieval Subsystem: Hybrid retrieval pipeline executing concurrent BM25 lexical search and dense FAISS vector similarity "
                "search (BAAI/bge-m3), fused via Reciprocal Rank Fusion and refined by cross-encoder neural reranking (BAAI/bge-reranker-v2-m3) on GPU 4.\n"
                "5. Generative Inference & Adaptation Engine: Qwen3.6-35B-A3B Mixture-of-Experts language model loaded in native BF16 on GPU 1, modulated by "
                "the standard LoRA adapter (lora-legal-v2), governed by dynamic NaN/Inf safety audits and automated fallback routines."
            )
        },
        {
            "heading": "5.2 Frontend Layer Architecture",
            "content": (
                "The frontend layer was designed under strict enterprise constraints to deliver an intuitive, distraction-free legal research workbench:\n"
                "- Architectural Isolation: Built entirely in Vanilla HTML5, CSS3, and JavaScript without bloated external framework dependencies (React, Vue), "
                "ensuring near-instant browser load times (<100ms) and zero build-pipeline overhead.\n"
                "- Secure Browser Voice Context: Browser security policies restrict microphone access (Web Speech API) strictly to 'Secure Contexts' "
                "(window.isSecureContext == true: https:// or localhost). Served over HTTPS on port 8443, the frontend unlocks native speech-to-text "
                "transcription without third-party audio streaming leaks.\n"
                "- Dual-Feed History Sidebar: Features two distinct tabbed views: 'Everyone' (a global shared feed polling server-side legal searches across "
                "devices every 20 seconds) and 'My Chats' (isolated to the current browser session via secure client tokens).\n"
                "- PDF.js Document Integration: Allows legal practitioners to upload and view legal petitions, contracts, and court orders side-by-side with AI analysis.\n"
                "- Markdown & Citation Rendering: Dynamic client-side markdown parser supporting code blocks, legal quote styling, and clickable verified citation pills."
            )
        },
        {
            "heading": "5.3 Backend Application Architecture (FastAPI & Uvicorn)",
            "content": (
                "The backend is powered by FastAPI, an asynchronous, high-performance Python web framework built on Starlette and Pydantic:\n"
                "- Lifespan Resource Management: Uses FastAPI's async lifespan context manager to initialize the HybridRetriever (GPU 4) and Qwen3.6-35B model "
                "singleton (GPU 1) once during server cold start. This prevents redundant model reloads on incoming HTTP requests.\n"
                "- Strict Schema Validation: All request payloads are validated through Pydantic models (QueryRequest, QueryResponse) with strict typing, "
                "length limits, and default parameter boundaries.\n"
                "- Concurrency and Thread Safety: Asynchronous endpoint handlers run on an event loop while CPU/GPU-bound model generation is dispatched to "
                "dedicated inference threads, preventing event-loop starvation during intensive generation passes."
            )
        },
        {
            "heading": "5.4 Multi-Stage Retrieval Subsystem Architecture",
            "content": (
                "The retrieval subsystem operates via a two-stage hybrid pipeline:\n"
                "Stage 1 (Concurrent Candidate Generation):\n"
                "- Lexical Search: The query is normalized and tokenized, searching an in-memory Okapi BM25 inverted index across 15,057 statutory chunks to capture "
                "exact section numbers, statutory titles, and legal keywords.\n"
                "- Dense Vector Search: The query is converted into a 1024-dimensional dense semantic vector via BAAI/bge-m3 on GPU 4 and queried against an "
                "IndexFlatIP FAISS vector database containing 26,097 chunk vectors.\n"
                "- Reciprocal Rank Fusion (RRF): The candidate lists are merged using RRF scoring: RRF(d) = sum(1 / (k + rank_i(d))), harmonizing lexical and semantic ranks.\n\n"
                "Stage 2 (Neural Cross-Encoder Reranking):\n"
                "- Candidate pairs (Query, Document Chunk) are fed into BAAI/bge-reranker-v2-m3 on GPU 4. Full cross-attention scores passage relevance, "
                "effectively filtering out keyword-heavy but conceptually irrelevant passages.\n"
                "- Dynamic Top-K Selection: The top-k highest-scoring chunks are formatted into numbered citations [1], [2] and passed to the generator."
            )
        },
        {
            "heading": "5.5 LLM Inference Engine and Memory Allocation on DGX B200",
            "content": (
                "Generative synthesis is driven by Qwen3.6-35B-A3B, an advanced Mixture-of-Experts architecture:\n"
                "- Hardware Isolation: Loaded exclusively on GPU 1 (NVIDIA B200, 180 GB VRAM). Device allocation is pinned via device_map={'': 1}.\n"
                "- Native BF16 Precision: Loaded using torch.bfloat16 without post-training quantization. BF16 retains the 8-bit dynamic exponent range of FP32, "
                "preventing gradient underflow/overflow during generation while cutting VRAM footprint in half compared to FP32 (~70 GB).\n"
                "- Fallback Attention Notice: When optional acceleration libraries (causal_conv1d, flash-linear-attention) are not compiled in the Linux "
                "environment, the system safely routes inference through PyTorch's native reference implementation, ensuring 100% computational correctness "
                "at the trade-off of higher generation latency (~18–26s per query)."
            )
        },
        {
            "heading": "5.6 Parameter-Efficient LoRA Adapter Architecture",
            "content": (
                "To adapt Qwen3.6-35B to Indian legal syntax and deductive reasoning without altering base foundation weights, the system incorporates "
                "the lora-legal-v2 adapter module via Hugging Face PEFT:\n"
                "- Architecture: Rank r=16, alpha=32, targeting self-attention projections (q_proj, k_proj, v_proj, o_proj) across all 40 transformer layers.\n"
                "- Parameter Footprint: Total adapter parameters = 3,440,640 (14 MB safetensors file), representing less than 0.01% of the base model.\n"
                "- Forensic Safety Guard: Pre-inference audit inspects all 80 adapter tensors for NaNs and Infs. In-memory inspection verifies VRAM tensor "
                "cleanliness upon PEFT attachment. If an invalid adapter is detected, the system logs a critical warning and automatically falls back to Base + RAG."
            )
        },
        {
            "heading": "5.7 Database Architecture and SQLite WAL Storage",
            "content": (
                "Conversation history, message states, and retrieval telemetry are persisted in a robust local SQLite database (data/legalmind.db):\n"
                "- Write-Ahead Logging (WAL): PRAGMA journal_mode=WAL enables concurrent non-blocking reads while writes are committed to the WAL log.\n"
                "- Full-Text Search (FTS5): A virtual table messages_fts indexed by SQLite FTS5 provides instantaneous multi-term full-text search across all user "
                "queries and assistant responses.\n"
                "- Schema Entities: (1) conversations (id, user_id, title, created_at, updated_at, is_pinned); (2) messages (id, conversation_id, role, content, "
                "status, error_code, latency, citations_json, metadata_json)."
            )
        },
        {
            "heading": "5.8 System Architecture Diagrams and Flowcharts",
            "content": (
                "The structural architecture of LegalMind AI is represented through the following formal design diagrams:\n\n"
                "[FIGURE 5.1: OVERALL SYSTEM ARCHITECTURE]\n"
                "User (Browser / CLI) ---> [HTTPS Port 8443 / HTTP 8080] ---> [FastAPI ASGI App]\n"
                "                                                                  |\n"
                "                    +---------------------------------------------+\n"
                "                    |                                             |\n"
                "         [Hybrid Retriever (GPU 4)]                     [LegalMindModel (GPU 1)]\n"
                "          /                     \\                                 |\n"
                "  [BM25 Index]             [FAISS Dense DB]            [Base Qwen3.6-35B BF16]\n"
                "  (15,057 chunks)          (26,097 vectors)                       |\n"
                "          \\                     /                      [LoRA Adapter (v2)]\n"
                "        [Reciprocal Rank Fusion]                        (3.44M clean params)\n"
                "                    |                                             |\n"
                "         [BGE-Reranker-v2-m3]                                     |\n"
                "                    |                                             |\n"
                "                    +---------> Top-K Evidence Prompt ----------->+\n"
                "                                                                  |\n"
                "                                                      [Generated Legal IRAC Answer]\n"
                "                                                                  |\n"
                "                                                      [Citation Verifier & DB Commit]\n\n"
                "[FIGURE 5.2: HYBRID RETRIEVAL WORKFLOW]\n"
                "Query ---> Preprocessing & Expansion ---> [Constitutional Router?]\n"
                "               |                                   |\n"
                "               +---> Okapi BM25 Lexical Score      +---> Golden Triangle Balancer\n"
                "               +---> BGE-M3 Dense Vector Query     |     (Articles 14, 19, 21)\n"
                "                             |                     |\n"
                "                   [Reciprocal Rank Fusion] <------+\n"
                "                             |\n"
                "                   [BGE-Reranker-v2-m3 Cross-Encoder]\n"
                "                             |\n"
                "                   [Dynamic Top-K Selection]\n\n"
                "[FIGURE 5.3: LORA ADAPTER INTEGRATION GRAPH]\n"
                "Input Token Embeddings ---> [Transformer Layer 1..40]\n"
                "                                   |\n"
                "                 +-----------------+-----------------+\n"
                "                 |                                   |\n"
                "        [Frozen Base Weights W_0]           [Trainable LoRA Branch]\n"
                "          (Native BF16, 35B MoE)              Matrix A: d -> r (r=16)\n"
                "                 |                                    |\n"
                "                 |                            Matrix B: r -> d (alpha=32)\n"
                "                 |                                    |\n"
                "                 +---------------->(+) <--------------+\n"
                "                                    |\n"
                "                          Modulated Output Token"
            )
        },
        {
            "heading": "5.9 Architectural Decomposition of Multimodal, CLI, and Web UI Subsystems",
            "content": (
                "To deliver a comprehensive legal intelligence platform, LegalMind AI incorporates three advanced architectural modules:\n\n"
                "1. Multimodal Document & Speech Subsystem (`app/services/*`):\n"
                "- Document Parsing & Context Store: `document_service.py` extracts multi-page PDF, DOCX, and TXT files, performing structural clause "
                "breakdown and feeding paragraph chunks into `multimodal_context.py`, a thread-safe in-memory store for targeted follow-up Q&A.\n"
                "- Image Vision & OCR Correction: `vision_service.py` executes decodability analysis and Tesseract OCR with non-destructive confusion "
                "matrix correction (e.g., repairing degraded legal scan text, court seals, and official stamps).\n"
                "- Speech-to-Text Voice Engine: `speech_service.py` integrates a local Faster-Whisper model adapter for audio validation, noise filter, and real-time interactive voice typing.\n\n"
                "2. Antigravity Developer Terminal CLI (`bin/legalmind`, `legalmind_cli/*`):\n"
                "- Full-Screen Terminal Assistant: Built with Python Rich and Prompt-Toolkit, featuring token streaming, animated loading spinners, and dark mode theme.\n"
                "- Slash Commands & Shortcuts: Supports `/voice` (speech capture), `/upload` (document intake), `/ocr` (image inspection), `/speak` (TTS), and `/sources` (citation audit).\n\n"
                "3. Apple Space Black Web UI & Dynamic Island (`app/static/*`):\n"
                "- Space Black Design System: Minimalist macOS Space Black palette (`#0B0C10`, `#1F2833`) with fluid iOS 18 glassmorphism panels.\n"
                "- Dynamic Island Navigation Bar: Floating pill navigation providing real-time system status, voice recording state, and model memory indicators."
            )
        }
    ]
}
