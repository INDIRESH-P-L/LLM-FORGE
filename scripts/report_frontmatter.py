"""
scripts/report_frontmatter.py
Front matter content for LegalMind AI Project Report:
- Cover Page details
- Certificate of Approval
- Candidate Declaration
- Acknowledgements
- Abstract
- List of Abbreviations
- List of Figures
- List of Tables
"""

FRONT_MATTER = {
    "title": "LegalMind AI – An AI-Powered Legal Question Answering System Using Retrieval-Augmented Generation, Qwen LLM, Hybrid Search, Reranking, and LoRA Fine-Tuning",
    "subtitle": "A Comprehensive University Project Report Submitted in Partial Fulfillment of the Requirements for the Degree of Master of Technology in Computer Science and Engineering",
    "author": "SECE 2026 AI Research Team",
    "institution": "Department of Computer Science and Engineering\nNvidia Computing High-Performance AI Research Facility",
    "academic_year": "Academic Year 2025–2026",
    
    "certificate": {
        "heading": "CERTIFICATE OF APPROVAL",
        "body": (
            "This is to certify that the project report entitled \"LegalMind AI – An AI-Powered "
            "Legal Question Answering System Using Retrieval-Augmented Generation, Qwen LLM, "
            "Hybrid Search, Reranking, and LoRA Fine-Tuning\" is a bona fide record of technical "
            "work carried out by the candidate under our supervision and guidance during the academic "
            "year 2025–2026. The technical contributions, empirical evaluations, architectural "
            "implementations, and research findings presented in this report have reached the "
            "requisite standard of academic excellence and technical rigor for submission in partial "
            "fulfillment of the degree requirements.\n\n"
            "This work was executed on the enterprise NVIDIA DGX B200 computing facility and "
            "faithfully implements, evaluates, and verifies a hybrid retrieval-augmented generation "
            "pipeline, local parameter-efficient LoRA fine-tuning, cross-encoder neural reranking, "
            "and an evidence-grounded legal reasoning architecture dedicated to Indian constitutional "
            "and statutory jurisprudence."
        ),
        "signatories": [
            ("Project Guide & Mentor", "Department of Computer Science & Engineering"),
            ("Head of Department", "Faculty of Engineering & Technology"),
            ("External Examiner", "Board of Technical Examiners"),
        ]
    },
    
    "declaration": {
        "heading": "CANDIDATE DECLARATION",
        "body": (
            "I hereby declare that the project entitled \"LegalMind AI – An AI-Powered Legal Question "
            "Answering System Using Retrieval-Augmented Generation, Qwen LLM, Hybrid Search, "
            "Reranking, and LoRA Fine-Tuning\" submitted to the Department of Computer Science "
            "and Engineering is an authentic record of original research and engineering work conducted "
            "by me. No part of this project report has been submitted previously for the award of any "
            "other degree, diploma, fellowship, or professional associate-ship at this or any other "
            "institution of higher education.\n\n"
            "All algorithms, code structures, empirical benchmark results, and analytical observations "
            "presented herein reflect the actual operating characteristics of the system deployed on "
            "the DGX B200 computing environment. All published literature, statutory references, judicial "
            "precedents, and open-source software libraries utilized during the development of this work "
            "have been cited and acknowledged in full compliance with academic ethics."
        )
    },
    
    "acknowledgements": {
        "heading": "ACKNOWLEDGEMENTS",
        "body": (
            "The realization of this research project has been made possible through the profound guidance, "
            "unwavering technical support, and critical encouragement of numerous individuals and academic "
            "bodies. I express my deepest gratitude to my project supervisor, whose deep domain insight into "
            "Natural Language Processing, information retrieval, and computational linguistics shaped the "
            "architectural trajectory of LegalMind AI.\n\n"
            "I extend my sincere appreciation to the Department of Computer Science and Engineering and the "
            "administrative authorities of the institution for granting access to the high-performance "
            "NVIDIA DGX B200 computing infrastructure. Without this enterprise platform—featuring high-bandwidth "
            "B200 GPUs and unified memory architecture—deploying a 35-billion-parameter Mixture-of-Experts "
            "(MoE) foundation model in native BF16 precision would have been practically infeasible.\n\n"
            "I am indebted to the open-source artificial intelligence community, particularly the maintainers "
            "of Hugging Face Transformers, PEFT, PyTorch, FAISS, FastAPI, and the Qwen research team at Alibaba "
            "Cloud, whose robust open-weights foundation models and toolkits enabled the development of domain-specialized "
            "systems. Finally, I thank my peers, family, and colleagues for their constant intellectual dialogue, "
            "constructive critiques, and moral support throughout this demanding academic endeavor."
        )
    },
    
    "abstract": {
        "heading": "ABSTRACT",
        "body": (
            "Access to legal knowledge in India is characterized by acute informational friction, driven by a vast, "
            "multi-tiered statutory corpus, over seven decades of constitutional jurisprudence, and historic transitions "
            "such as the replacement of colonial criminal codes with the Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik "
            "Suraksha Sanhita (BNSS), and Bharatiya Sakshya Adhiniyam (BSA). While general-purpose Large Language Models (LLMs) "
            "exhibit impressive conversational fluency, they suffer from unacceptable hallucination rates, temporal obsolescence, "
            "and lack of verifiable citation grounding when applied to specialized legal queries. This project introduces "
            "LegalMind AI, an enterprise-grade, evidence-grounded legal question-answering system engineered specifically "
            "for Indian constitutional and statutory jurisprudence.\n\n"
            "LegalMind AI implements a multi-stage Retrieval-Augmented Generation (RAG) architecture running on an NVIDIA DGX B200 "
            "computing environment. The knowledge base indexes approximately 15,057 statutory chunks spanning the Constitution "
            "of India, major central enactments, and 26,097 multi-scale dense vector representations generated using BAAI/bge-m3. "
            "The retrieval subsystem operates via a hybrid paradigm, fusing sparse BM25 lexical token matching with dense FAISS "
            "vector similarity search through reciprocal rank fusion, followed by cross-encoder neural reranking using BAAI/bge-reranker-v2-m3. "
            "The generative reasoning layer leverages the Qwen3.6-35B-A3B Mixture-of-Experts architecture in native Bfloat16 (BF16) "
            "precision, modulated by a domain-adapted Low-Rank Adaptation (LoRA) adapter (lora-legal-v2) targeting self-attention projections.\n\n"
            "Empirical validation across ten mandated legal categories—including Article 14, Article 21, Fundamental Rights, "
            "Criminal Procedure, Contract Law, and Statutory Transitions—demonstrates that the LoRA-adapted RAG pipeline achieves "
            "an average grounding score of 0.692 (a 2.4% increase over the base model) and reduces the citation hallucination rate "
            "to 0.308 (a 2.4% reduction). The system achieves 100% test pass rates across 171 automated unit and integration tests, "
            "implements automatic zero-downtime fallback to the base model if adapter corruption is detected, and exposes a secure "
            "FastAPI HTTPS service on port 8443 with an Apple-inspired Liquid Glass frontend. Detailed qualitative evaluations "
            "on landmark queries (e.g., Article 21) demonstrate strong constitutional reasoning while highlighting critical "
            "operational realities: inference latencies averaging ~18 to 26 seconds under reference PyTorch fallback execution, "
            "the emergence of tangential context distractors, and the necessity of strict automated citation grounding. LegalMind AI "
            "establishes a robust, reproducible, and verifiable foundation for responsible AI in the Indian justice ecosystem."
        )
    },
    
    "abbreviations": [
        ("AI", "Artificial Intelligence"),
        ("ANN", "Approximate Nearest Neighbor"),
        ("API", "Application Programming Interface"),
        ("AIR", "All India Reporter (Judicial Citations)"),
        ("BF16", "Brain Floating Point (16-bit)"),
        ("BGE", "BAAI General Embedding"),
        ("BM25", "Best Matching 25 (Probabilistic Lexical Ranking)"),
        ("BNS", "Bharatiya Nyaya Sanhita, 2023 (Replaced IPC)"),
        ("BNSS", "Bharatiya Nagarik Suraksha Sanhita, 2023 (Replaced CrPC)"),
        ("BSA", "Bharatiya Sakshya Adhiniyam, 2023 (Replaced IEA)"),
        ("CLI", "Command-Line Interface"),
        ("CrPC", "Code of Criminal Procedure, 1973 (Repealed)"),
        ("CUDA", "Compute Unified Device Architecture"),
        ("DGX", "Deep Learning GPU Enterprise Server (NVIDIA)"),
        ("FAISS", "Facebook AI Similarity Search"),
        ("FTS5", "Full-Text Search Engine 5 (SQLite Module)"),
        ("GPU", "Graphics Processing Unit"),
        ("HTML", "HyperText Markup Language"),
        ("HTTP", "Hypertext Transfer Protocol"),
        ("HTTPS", "Hypertext Transfer Protocol Secure (TLS/SSL)"),
        ("IEA", "Indian Evidence Act, 1872 (Repealed)"),
        ("IPC", "Indian Penal Code, 1860 (Repealed)"),
        ("IR", "Information Retrieval"),
        ("IRAC", "Issue, Rule, Application, Conclusion (Legal Methodology)"),
        ("JSON", "JavaScript Object Notation"),
        ("LLM", "Large Language Model"),
        ("LoRA", "Low-Rank Adaptation of Large Language Models"),
        ("MoE", "Mixture of Experts Architecture"),
        ("MRR", "Mean Reciprocal Rank"),
        ("NDCG", "Normalized Discounted Cumulative Gain"),
        ("NLP", "Natural Language Processing"),
        ("OCR", "Optical Character Recognition"),
        ("PEFT", "Parameter-Efficient Fine-Tuning"),
        ("PDF", "Portable Document Format"),
        ("RAG", "Retrieval-Augmented Generation"),
        ("RRF", "Reciprocal Rank Fusion"),
        ("SCC", "Supreme Court Cases (Judicial Reporter)"),
        ("SCR", "Supreme Court Reports (Official Reporter)"),
        ("SQL", "Structured Query Language"),
        ("TLS", "Transport Layer Security"),
        ("TTS", "Text-to-Speech"),
        ("UI", "User Interface"),
        ("VRAM", "Video Random-Access Memory"),
        ("WAL", "Write-Ahead Logging (SQLite Concurrency Protocol)"),
    ],
    
    "figures": [
        ("Figure 5.1", "High-Level System Architecture of LegalMind AI", "Page 15"),
        ("Figure 5.2", "Multi-Stage Hybrid Retrieval & Cross-Encoder Reranking Pipeline", "Page 17"),
        ("Figure 5.3", "Parameter-Efficient LoRA Adapter (lora-legal-v2) Integration Graph", "Page 19"),
        ("Figure 5.4", "End-to-End User Query Processing, Validation, and Grounding Flow", "Page 21"),
        ("Figure 5.5", "SQLite Chat Store Entity-Relationship (ER) Architecture", "Page 23"),
        ("Figure 5.6", "Secure HTTPS / TLS 1.3 Client-Server Transport Architecture", "Page 25"),
        ("Figure 8.1", "Detailed 18-Stage Execution Methodology Workflow", "Page 31"),
        ("Figure 10.1", "LegalMind AI Modular Repository Directory Organization", "Page 37"),
        ("Figure 12.1", "Comparative Grounding vs. Hallucination Profile (Base vs. LoRA)", "Page 44"),
    ],
    
    "tables": [
        ("Table 3.1", "Functional Requirements Specification Matrix", "Page 10"),
        ("Table 3.2", "NVIDIA DGX B200 Hardware & VRAM Allocation Scheme", "Page 12"),
        ("Table 4.1", "System Feasibility Analysis Summary", "Page 13"),
        ("Table 4.2", "Comprehensive Use-Case Specification Table", "Page 14"),
        ("Table 6.1", "Comprehensive Technology Stack Analysis & Trade-Off Matrix", "Page 26"),
        ("Table 7.1", "Corpus Composition, Chunk Distribution, and FAISS Vector Statistics", "Page 29"),
        ("Table 9.1", "LoRA Adapter Hyperparameter Configuration (lora-legal-v2)", "Page 34"),
        ("Table 9.2", "Forensic Tensor Safety Audit: lora-legal-v1 vs. lora-legal-v2", "Page 35"),
        ("Table 11.1", "Structured Verification & Validation Test Case Matrix", "Page 40"),
        ("Table 12.1", "10-Category Side-by-Side Benchmark: Base LLM vs. LoRA-Adapted LLM", "Page 42"),
        ("Table 12.2", "Retrieval Context Optimization: Impact of top_k Code Modification", "Page 45"),
    ]
}
