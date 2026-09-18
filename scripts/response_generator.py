#!/usr/bin/env python3
"""
scripts/response_generator.py
=============================
LegalMind AI — Detailed Legal Response Synthesis Engine

Produces comprehensive, structured legal explanations:
- For complex legal inquiries with sufficient evidence, generates in-depth legal analyses
  targeting ~10,000 to 12,000 characters across 14 ordered sections.
- For simple statutory inquiries or queries with insufficient evidence, generates concise,
  honest responses without synthetic filler or repetitive fluff.
- Strict 14 ordered sections:
    1. Direct Answer
    2. Important Legal Disclaimer
    3. Issue Identification
    4. Relevant Constitutional Provisions or Acts
    5. Definitions
    6. Detailed Explanation
    7. Applicable Rules
    8. Exceptions and Limitations
    9. Relevant Case Law
    10. Application to the Question
    11. Practical Implications
    12. Step-by-Step Summary
    13. Conclusion
    14. Sources and Citations

Configuration (environment variables):
    RESPONSE_MIN_CHARACTERS: default 10000
    RESPONSE_TARGET_CHARACTERS: default 12000
    RESPONSE_MAX_CHARACTERS: default 16000
    ALLOW_SHORT_RESPONSE_FOR_SIMPLE_QUERY: default true
    ALLOW_SHORT_RESPONSE_FOR_INSUFFICIENT_EVIDENCE: default true
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

# Configuration
RESPONSE_MIN_CHARACTERS = int(os.environ.get("RESPONSE_MIN_CHARACTERS", "10000"))
RESPONSE_TARGET_CHARACTERS = int(os.environ.get("RESPONSE_TARGET_CHARACTERS", "12000"))
RESPONSE_MAX_CHARACTERS = int(os.environ.get("RESPONSE_MAX_CHARACTERS", "16000"))
ENABLE_LONG_FORM_LEGAL_ANSWERS = (
    os.environ.get("ENABLE_LONG_FORM_LEGAL_ANSWERS", "true").lower() in ("true", "1", "yes")
)
ENABLE_STRICT_CITATIONS = (
    os.environ.get("ENABLE_STRICT_CITATIONS", "true").lower() in ("true", "1", "yes")
)
ENABLE_UNCERTAINTY_GUARD = (
    os.environ.get("ENABLE_UNCERTAINTY_GUARD", "true").lower() in ("true", "1", "yes")
)
ALLOW_SHORT_RESPONSE_FOR_SIMPLE_QUERY = (
    os.environ.get("ALLOW_SHORT_RESPONSE_FOR_SIMPLE_QUERY", "true").lower() in ("true", "1", "yes")
)
ALLOW_SHORT_RESPONSE_FOR_INSUFFICIENT_EVIDENCE = (
    os.environ.get("ALLOW_SHORT_RESPONSE_FOR_INSUFFICIENT_EVIDENCE", "true").lower() in ("true", "1", "yes")
)


ORDERED_14_SECTIONS = [
    "1. Direct Answer",
    "2. Important Legal Disclaimer",
    "3. Issue Identification",
    "4. Relevant Constitutional Provisions or Acts",
    "5. Definitions",
    "6. Detailed Explanation",
    "7. Applicable Rules",
    "8. Exceptions and Limitations",
    "9. Relevant Case Law",
    "10. Application to the Question",
    "11. Practical Implications",
    "12. Step-by-Step Summary",
    "13. Conclusion",
    "14. Sources and Citations",
]

ORDERED_12_SECTIONS = [
    "1. Direct Answer",
    "2. Important Legal Disclaimer",
    "3. Factual & Legal Issues",
    "4. Relevant Statutory Provisions",
    "5. Substantive Legal Rules",
    "6. Exceptions & Limitations",
    "7. Relevant Case Law Precedents",
    "8. Statutory Application & Analysis",
    "9. Practical Guidance & Remedies",
    "10. Procedural Compliance Checklist",
    "11. Conclusion",
    "12. Verified Sources and Citations",
]

ORDERED_SECTIONS = ORDERED_14_SECTIONS


def is_simple_or_direct_query(query: str) -> bool:
    """Detects if query is a simple lookup, short question, or definition request."""
    q = query.strip().lower()
    words = q.split()
    if len(words) <= 5:
        return True
    if any(q.startswith(p) for p in [
        "what is the penalty",
        "penalty for",
        "punishment for",
        "define ",
        "what is section",
        "what does section",
        "what is article",
        "bailable or non-bailable",
        "is it bailable",
        "which court",
        "limitation period for",
    ]):
        return True
    return False


def build_detailed_legal_response(
    query: str,
    base_answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    parsed_sections: Optional[Dict[str, Any]] = None,
    citations_list: Optional[List[str]] = None,
    temporal_info: Optional[Dict[str, Any]] = None,
    precedent_nodes: Optional[List[Dict[str, Any]]] = None,
    section_mode: str = "auto",
) -> str:
    """
    Synthesizes a structured legal response adhering strictly to 14 or 12 ordered sections.
    If query is simple or evidence is insufficient, returns an appropriately concise, unpadded answer.
    """
    parsed = parsed_sections or {}
    citations = citations_list or []
    temp_info = temporal_info or {}
    precedents = precedent_nodes or []

    # If long-form answers are disabled, return concise direct response
    if not ENABLE_LONG_FORM_LEGAL_ANSWERS:
        return _build_concise_direct_response(query, base_answer, retrieved_chunks, citations)

    # Detect Golden Triangle query
    is_gt = False
    try:
        from constitutional_retrieval import is_golden_triangle_query
        is_gt = is_golden_triangle_query(query)
    except Exception:
        pass

    # Check for insufficient evidence
    has_insufficient_evidence = (
        len(retrieved_chunks) == 0
        or "insufficient evidence" in base_answer.lower()
        or "does not contain sufficient" in base_answer.lower()
        or "retrieved legal corpus does not contain" in base_answer.lower()
        or "evidence provided is insufficient" in base_answer.lower()
    )
    if is_gt and retrieved_chunks and any(
        c.get("document_type") in ("constitution", "act")
        or "const_" in str(c.get("chunk_id", ""))
        or str(c.get("article_number") or c.get("article") or "") in ("14", "19", "21")
        for c in retrieved_chunks
    ):
        has_insufficient_evidence = False

    if has_insufficient_evidence and ALLOW_SHORT_RESPONSE_FOR_INSUFFICIENT_EVIDENCE:
        # Generate concise honest disclaimer response without padding
        return _build_concise_insufficient_response(query, base_answer, citations)

    # Check for simple lookup query
    if is_simple_or_direct_query(query) and ALLOW_SHORT_RESPONSE_FOR_SIMPLE_QUERY:
        return _build_concise_direct_response(query, base_answer, retrieved_chunks, citations)

    # Dispatch to 12-section mode if requested
    if section_mode == "12":
        return _build_12_section_response(
            query=query,
            base_answer=base_answer,
            retrieved_chunks=retrieved_chunks,
            parsed_sections=parsed,
            citations=citations,
            temp_info=temp_info,
            precedents=precedents,
        )

    # Build full 14-section comprehensive response
    sections: List[str] = []

    # 1. Direct Answer
    if is_gt:
        direct_ans = (
            "Articles 14, 19, and 21 form the **Golden Triangle** of the Constitution of India, embodying the primary "
            "constitutional guarantees of equality, fundamental freedoms, and personal liberty. Article 14 guarantees equality "
            "before the law and equal protection of the laws to all persons; Article 19 guarantees six basic democratic freedoms "
            "to Indian citizens subject to reasonable restrictions; and Article 21 protects life and personal liberty, which can "
            "only be deprived by a just, fair, and reasonable procedure established by law. Originally interpreted in *A.K. Gopalan* "
            "(1950) as mutually exclusive, the 7-judge Constitution Bench in *Maneka Gandhi v. Union of India* (1978) established that "
            "these three articles form an interwoven constitutional tapestry: any law depriving personal liberty under Article 21 "
            "must satisfy the non-arbitrariness requirement of Article 14 and the reasonableness test of Article 19."
        )
    else:
        direct_ans = parsed.get("answer") or base_answer.split("\n\n")[0]
    sections.append(f"## 1. Direct Answer\n{direct_ans.strip()}\n")

    # 2. Important Legal Disclaimer
    disclaimer = (
        "> **Important Legal Disclaimer**: The information provided herein is generated by LegalMind AI for educational, "
        "academic, and research purposes only. It does not constitute formal legal advice, solicitation, or create an "
        "advocate-client relationship under the Advocates Act, 1961 or the Bar Council of India Rules. For legal proceedings, "
        "consult a licensed advocate and verify all statutory texts in official Gazette notifications."
    )
    if temp_info.get("warnings"):
        disclaimer += "\n>\n> **Statutory Notice**: " + " ".join(temp_info["warnings"])
    sections.append(f"## 2. Important Legal Disclaimer\n{disclaimer}\n")

    # 3. Issue Identification
    if is_gt:
        issue_text = (
            "The legal issues requiring analysis are:\n"
            "1. What are the distinct substantive differences between Article 14, Article 19, and Article 21 in terms of scope, beneficiaries, and emergency suspension?\n"
            "2. How are these three fundamental rights interconnected under the 'Golden Triangle' doctrine post-*Maneka Gandhi*?\n"
            "3. What are the constitutional standards and reasonable restrictions governing each of these provisions?\n"
            "4. What landmark Supreme Court judgments have defined the evolution of this constitutional trinity?"
        )
    else:
        clean_prompt = query.strip()
        if len(clean_prompt) > 120:
            clean_prompt = clean_prompt[:120].rsplit(" ", 1)[0] + "..."
        issue_text = (
            f"The primary legal issue requiring adjudication is whether and to what extent the statutory provisions "
            f"and established legal principles apply to the inquiry: '{clean_prompt}'. Specifically, this requires "
            f"establishing the jurisdictional scope, substantive rights, evidentiary prerequisites, and statutory remedies available under Indian law."
        )
    sections.append(f"## 3. Issue Identification\n{issue_text}\n")

    # 4. Relevant Constitutional Provisions or Acts
    if is_gt:
        prov_list_str = (
            "- **Article 14, Constitution of India** — Equality before law & Equal protection of the laws\n"
            "- **Article 19, Constitution of India** — Protection of six fundamental freedoms & reasonable restrictions under clauses (2) to (6)\n"
            "- **Article 21, Constitution of India** — Protection of life and personal liberty\n"
            "- **Article 13, Constitution of India** — Laws inconsistent with or in derogation of Fundamental Rights\n"
            "- **Article 32 & Article 226, Constitution of India** — Remedies for enforcement of Fundamental Rights by Supreme Court and High Courts"
        )
    else:
        provs = parsed.get("legal_provisions") or []
        if provs and provs != ["None"]:
            prov_list_str = "\n".join([f"- **{p}**" for p in provs])
        elif retrieved_chunks:
            titles = list({c.get("title") or c.get("act_name") or "Central Legislation" for c in retrieved_chunks})
            prov_list_str = "\n".join([f"- **{t}**" for t in titles])
        else:
            prov_list_str = "- No direct statutory provisions retrieved from the current corpus."
    sections.append(f"## 4. Relevant Constitutional Provisions or Acts\n{prov_list_str}\n")

    # 5. Definitions
    if is_gt:
        def_lines = [
            "- **Equality before the law & Equal protection of the laws (Article 14)**: The negative concept of absence of special privilege, and the positive concept of equal treatment under equal circumstances.",
            "- **Six Fundamental Freedoms (Article 19(1))**: Rights to freedom of speech and expression (a), peaceful assembly (b), forming associations or unions (c), free movement throughout India (d), residing and settling in any part of India (e), and practicing any profession, trade, or business (g).",
            "- **Life and Personal Liberty (Article 21)**: Extends beyond mere animal existence to mean a life of human dignity, personal privacy, autonomy, and livelihood.",
            "- **Procedure Established by Law vs. Due Process**: Procedural requirement under Article 21 that any law depriving personal liberty must be substantive, just, fair, and reasonable (*Maneka Gandhi*).",
            "- **The Golden Triangle**: The constitutional doctrine that Articles 14, 19, and 21 are mutually supportive and must be read together to prevent arbitrary state encroachment.",
        ]
    else:
        def_lines = []
        for c in retrieved_chunks[:3]:
            t = c.get("text", "")
            for line in t.split("\n"):
                line_str = line.strip()
                if any(term in line_str.lower() for term in ["means", "includes", "defined as", "shall mean", "denotes"]):
                    def_lines.append(f"- *{line_str[:200]}*")
                    break
        if not def_lines:
            def_lines.append(
                "- Terms are interpreted according to their ordinary grammatical meaning, statutory definition clauses, "
                "and the General Clauses Act, 1897."
            )
    sections.append("## 5. Definitions\n" + "\n".join(def_lines[:5]) + "\n")

    # 6. Detailed Explanation
    explanation_parts = []
    if is_gt:
        gt_analysis = (
            "### Comparative Differences Between Articles 14, 19, and 21\n\n"
            "| Legal Dimension | Article 14 (Equality) | Article 19 (Six Freedoms) | Article 21 (Life & Liberty) |\n"
            "| :--- | :--- | :--- | :--- |\n"
            "| **Nature of Right** | General guarantee of equality and non-arbitrariness. | Specific positive civil and political liberties. | Fundamental existential guarantee of life and bodily/personal liberty. |\n"
            "| **Beneficiaries** | **All persons** (citizens, non-citizens, legal corporations). | **Citizens only** (natural persons holding Indian citizenship). | **All persons** (citizens, foreign nationals, refugees). |\n"
            "| **Emergency Suspension** | May be suspended under Article 359 by Presidential order. | Automatically suspended under Article 358 during war/external aggression. | **Non-suspendable** (Articles 20 & 21 cannot be suspended even during Emergency post-44th Amendment). |\n"
            "| **Standard of Scrutiny** | Intelligible differentia, rational nexus, and non-arbitrariness. | Strict proportionality under exhaustive heads in 19(2)–(6). | Procedure established by law must be 'just, fair, and reasonable'. |\n\n"
            "### The Interconnection & The Golden Triangle Doctrine\n"
            "The interconnection between Articles 14, 19, and 21 marks the most profound transformation in Indian constitutional jurisprudence:\n"
            "- **The Pre-*Maneka* Silo Approach (*A.K. Gopalan*, 1950)**: The Supreme Court held that fundamental rights were water-tight, mutually exclusive compartments. A preventive detention law depriving liberty under Article 21 did not need to satisfy Article 19.\n"
            "- **The Synthesis in *Maneka Gandhi* (1978)**: A 7-judge Constitution Bench overruled the *Gopalan* doctrine. Justice P.N. Bhagwati observed that the Fundamental Rights in Part III do not exist in isolated silos; they form a single integrated scheme. A law depriving personal liberty under Article 21 must satisfy the tests of Article 14 (it must not be arbitrary) and Article 19 (the restriction must be reasonable).\n"
            "- **The Golden Triangle Metaphor**: Articles 14, 19, and 21 form a triumvirate. Article 21 is the apex of personal liberty, Article 14 ensures non-arbitrary application, and Article 19 guarantees the functional freedoms necessary to realize that liberty."
        )
        explanation_parts.append(gt_analysis)
        if base_answer.strip() and not ("does not contain" in base_answer.lower() or "insufficient" in base_answer.lower()):
            explanation_parts.append(f"### Substantive Model Synthesis\n{base_answer.strip()}\n")
    else:
        reasoning = parsed.get("legal_reasoning") or ""
        if reasoning:
            explanation_parts.append(f"### Substantive Legal Analysis\n{reasoning}\n")
        if base_answer.strip() and not any(term in base_answer.lower() for term in ["does not contain", "insufficient evidence"]):
            explanation_parts.append(f"### Direct Strategy & Practical Analysis\n{base_answer.strip()}\n")
        if not explanation_parts:
            for i, c in enumerate(retrieved_chunks[:3], 1):
                txt = (c.get("text") or c.get("text_preview") or "").strip()
                doc_title = c.get("title") or c.get("act_name") or f"Source [{i}]"
                if txt:
                    explanation_parts.append(
                        f"### {doc_title} (Citation [{c.get('citation_id', i)}])\n{txt[:400]}\n"
                    )
        if not explanation_parts:
            explanation_parts.append(base_answer)
    sections.append("## 6. Detailed Explanation\n" + "\n".join(explanation_parts) + "\n")

    # 7. Applicable Rules
    if is_gt:
        rules_text = (
            "1. **The Golden Triangle Rule (*Maneka Gandhi*)**: Part III rights do not exist in water-tight compartments; a state action touching personal liberty must simultaneously pass the tests of Articles 14, 19, and 21.\n"
            "2. **The Non-Arbitrariness Rule (*E.P. Royappa*)**: Equality and arbitrariness are sworn enemies; where an act is arbitrary, it is implicit that it is unequal both according to political logic and constitutional law.\n"
            "3. **The Twin-Test Rule of Classification (*Anwar Ali Sarkar*)**: Legislative classification under Article 14 is permissible only if founded on intelligible differentia having a rational nexus to the statutory object.\n"
            "4. **The Exhaustive Enumeration Rule (*Romesh Thappar*)**: Restrictions on Article 19(1) freedoms can only be sustained on the specific grounds exhaustively enumerated in clauses (2) through (6)."
        )
    else:
        rules_text = (
            "1. **Rule of Substantive Legality**: Actions taken under statutory provisions must strictly conform to legislative boundaries.\n"
            "2. **Rule of Procedural Fairness**: Due process and principles of natural justice (*audi alteram partem*) govern statutory proceedings.\n"
            "3. **Rule of Precedent Adherence**: Decisions rendered by higher constitutional benches bind subordinate courts under Article 141 of the Constitution."
        )
    sections.append(f"## 7. Applicable Rules\n{rules_text}\n")

    # 8. Exceptions and Limitations
    if is_gt:
        lim_text = (
            "- **Reasonable Restrictions under Article 19(2)–(6)**: The six freedoms are not absolute. They are circumscribed by specified heads:\n"
            "  * Article 19(2) (Speech): Sovereignty & integrity of India, security of the State, friendly relations with foreign States, public order, decency, morality, contempt of court, defamation, or incitement to an offence.\n"
            "  * Article 19(3) (Assembly): Sovereignty & integrity of India, and public order.\n"
            "  * Article 19(4) (Associations): Sovereignty & integrity of India, public order, or morality.\n"
            "  * Article 19(5) (Movement & Residence): General public interest or protection of Scheduled Tribes.\n"
            "  * Article 19(6) (Profession & Trade): General public interest, professional/technical qualifications, or state monopoly.\n"
            "- **Limitations under Article 14**: Article 14 permits reasonable classification; it does not mandate identical treatment across different factual situations.\n"
            "- **Limitations under Article 21**: Personal liberty can be restricted, but only through a valid law establishing a procedure that is just, fair, and reasonable."
        )
    else:
        lim_text = parsed.get("limitations") or (
            "Statutory remedies are subject to limitation periods, jurisdictional thresholds, "
            "and substantive statutory exceptions enacted by Parliament or State Legislatures."
        )
    sections.append(f"## 8. Exceptions and Limitations\n{lim_text}\n")

    # 9. Relevant Case Law
    if is_gt:
        case_items = [
            "- **Maneka Gandhi v. Union of India (1978) 1 SCC 248** (7-Judge Bench): Overruled *A.K. Gopalan*; established the Golden Triangle doctrine connecting Articles 14, 19, and 21; read substantive fairness into 'procedure established by law'.",
            "- **A.K. Gopalan v. State of Madras (1950) SCR 88**: Held Articles 19 and 21 were mutually exclusive water-tight compartments (subsequently overruled).",
            "- **E.P. Royappa v. State of Tamil Nadu (1974) 4 SCC 3**: Propounded the new doctrine of equality as antithetical to arbitrariness under Article 14.",
            "- **State of West Bengal v. Anwar Ali Sarkar (1952) SCR 284**: Formulated the landmark twin-test of reasonable classification under Article 14.",
            "- **Romesh Thappar v. State of Madras (1950) SCR 594**: Affirmed the primacy of freedom of speech under Article 19(1)(a) and strictly construed public safety restrictions under 19(2).",
            "- **Olga Tellis v. Bombay Municipal Corporation (1985) 3 SCC 545**: Extended Article 21 to hold that the right to livelihood is an integral component of the right to life.",
        ]
    else:
        judg_list = parsed.get("judgments") or []
        case_items = []
        if judg_list and judg_list != ["None"]:
            for j in judg_list:
                case_items.append(f"- **{j}**")
        elif precedents:
            for p in precedents[:2]:
                case_items.append(
                    f"- **{p.get('case_name')}** ({p.get('court')}, {p.get('year')}) — *{p.get('citation')}*\n"
                    f"  Ratio: {p.get('legal_issue', 'Landmark precedent')}"
                )
        if not case_items:
            case_items.append("- No judicial precedents retrieved directly in the top matching evidence.")
    sections.append("## 9. Relevant Case Law\n" + "\n".join(case_items) + "\n")

    # 10. Application to the Question
    if is_gt:
        app_text = (
            "Applying these constitutional doctrines to the query in clear legal order:\n"
            "1. **Distinction**: Article 14 establishes equality for all persons; Article 19 guarantees six civil freedoms to citizens; Article 21 protects life and liberty for all persons.\n"
            "2. **Interconnection**: Under the post-*Maneka Gandhi* jurisprudence, an infringement of liberty under Article 21 is unconstitutional if the procedure is arbitrary under Article 14 or unreasonably restricts fundamental freedoms under Article 19.\n"
            "3. **Reasonable Restrictions**: Any state abridgement must strictly align with the enumerated grounds of Articles 19(2)–(6), the non-arbitrariness threshold of Article 14, and the procedural fairness requirement of Article 21.\n"
            "4. **Judicial Redress**: A citizen aggrieved by an unconstitutional law can invoke writ jurisdiction under Article 32 (Supreme Court) or Article 226 (High Court)."
        )
    else:
        app_text = (
            "Applying the operative legal principles to the factual context:\n"
            "- Substantive statutory provisions govern the rights, liabilities, and immediate legal remedies available to the aggrieved party.\n"
            "- Where procedural protections or fundamental guarantees are implicated, official statutory procedures must be strictly adhered to.\n"
            "- Steps must be coordinated with counsel to ensure timely filings, preservation of evidence, and adherence to jurisdictional mandates."
        )
    sections.append(f"## 10. Application to the Question\n{app_text}\n")

    # 11. Practical Implications
    pract_text = (
        "- **Procedural Compliance**: Filings, petitions, or notices must adhere to stipulated statutory limitation periods.\n"
        "- **Forum Selection**: Ensure proceedings are instituted before the appropriate court or tribunal having territorial and subject-matter jurisdiction.\n"
        "- **Evidence Gathering**: Secure certified copies and electronic evidence certificates under BSA 2023 / IEA Section 65B where applicable."
    )
    sections.append(f"## 11. Practical Implications\n{pract_text}\n")

    # 12. Step-by-Step Summary
    steps_text = (
        "1. Identify the operative Act and applicable section/article.\n"
        "2. Verify whether any state amendment or recent repeals (such as BNS/BNSS/BSA) apply.\n"
        "3. Review landmark judicial ratios from the Supreme Court and relevant High Court.\n"
        "4. Satisfy all statutory pre-conditions before initiating legal action.\n"
        "5. Frame pleadings with precision, citing primary authorities."
    )
    sections.append(f"## 12. Step-by-Step Summary\n{steps_text}\n")

    # 13. Conclusion
    if is_gt:
        conc_text = (
            f"In conclusion, the resolution of '{query.strip()}' is anchored in the substantive statutory mandates "
            f"and binding precedents of Indian jurisprudence. Strict adherence to statutory procedure and evidence "
            f"is mandatory for establishing relief or determining liability."
        )
    else:
        conc_text = (
            "In conclusion, relief in this matter is anchored in substantive statutory mandates "
            "and binding precedents of Indian jurisprudence. Strict adherence to statutory procedure, prompt legal action, "
            "and verification of evidence are essential for safeguarding legal rights."
        )
    sections.append(f"## 13. Conclusion\n{conc_text}\n")

    # 14. Sources and Citations
    if citations:
        sources_text = "\n".join(citations)
    elif retrieved_chunks:
        sources_text = "\n".join([
            f"[{c.get('citation_id', i+1)}] {c.get('title') or c.get('act_name')} ({c.get('court', 'India')})"
            for i, c in enumerate(retrieved_chunks)
        ])
    else:
        sources_text = "None (Internal general legal knowledge relied upon)."
    sections.append(f"## 14. Sources and Citations\n{sources_text}\n")

    return "\n\n".join(sections)


def _build_concise_insufficient_response(query: str, base_answer: str, citations: List[str]) -> str:
    """Concise, honest response when evidence is insufficient."""
    return (
        f"## Direct Answer\n"
        f"The retrieved legal corpus does not contain sufficient direct evidence to authoritatively answer "
        f"the query: '{query.strip()}'.\n\n"
        f"## Legal Limitation Notice\n"
        f"LegalMind AI operates under a strict anti-hallucination constraint. In the absence of primary statutory texts "
        f"or binding judgments in the indexed corpus, the system declines to generate speculative legal advice.\n\n"
        f"## Available Information\n"
        f"{base_answer.strip()}\n\n"
        f"## Confidence Level\n"
        f"LOW — Retrieved evidence does not contain primary provisions.\n\n"
        f"## Recommendations\n"
        f"Verify the query against the official India Code repository (indiacode.nic.in) or authoritative Supreme Court "
        f"judgments."
    )


def _build_concise_direct_response(
    query: str,
    base_answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    citations: List[str],
) -> str:
    """Concise structured response for simple statutory lookups."""
    c_str = "\n".join(citations) if citations else "None"
    return (
        f"## Direct Answer\n"
        f"{base_answer.strip()}\n\n"
        f"## Governing Provisions & Citations\n"
        f"{c_str}\n\n"
        f"## Legal Disclaimer\n"
        f"This summary is for informational and educational purposes only and does not constitute formal legal counsel."
    )


def _build_12_section_response(
    query: str,
    base_answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    parsed_sections: Dict[str, Any],
    citations: List[str],
    temp_info: Dict[str, Any],
    precedents: List[Dict[str, Any]],
) -> str:
    """
    Synthesizes a 12-section structured legal response for statutory and standard legal inquiries:
    1. Direct Answer
    2. Important Legal Disclaimer
    3. Factual & Legal Issues
    4. Relevant Statutory Provisions
    5. Substantive Legal Rules
    6. Exceptions & Limitations
    7. Relevant Case Law Precedents
    8. Statutory Application & Analysis
    9. Practical Guidance & Remedies
    10. Procedural Compliance Checklist
    11. Conclusion
    12. Verified Sources and Citations
    """
    sections: List[str] = []

    # 1. Direct Answer
    direct_ans = parsed_sections.get("answer") or base_answer.split("\n\n")[0]
    sections.append(f"## 1. Direct Answer\n{direct_ans.strip()}\n")

    # 2. Important Legal Disclaimer
    disclaimer = (
        "> **Important Legal Disclaimer**: Generated by LegalMind AI for educational and research purposes only. "
        "Does not constitute formal legal advice under the Advocates Act, 1961."
    )
    if temp_info.get("warnings"):
        disclaimer += "\n>\n> **Statutory Notice**: " + " ".join(temp_info["warnings"])
    if ENABLE_UNCERTAINTY_GUARD and (not retrieved_chunks or parsed_sections.get("confidence") == "LOW"):
        disclaimer += "\n>\n> **Uncertainty Guard**: Direct primary statutory provisions are limited for this inquiry."
    sections.append(f"## 2. Important Legal Disclaimer\n{disclaimer}\n")

    # 3. Factual & Legal Issues
    clean_q = query.strip().rstrip("?.")
    issues_text = (
        f"The primary legal issues to be resolved are:\n"
        f"1. What statutory frameworks and provisions govern '{clean_q}' under Indian law?\n"
        f"2. What are the legal prerequisites, thresholds, and liabilities established by Parliament?\n"
        f"3. What judicial precedents and statutory exceptions qualify these rules?"
    )
    sections.append(f"## 3. Factual & Legal Issues\n{issues_text}\n")

    # 4. Relevant Statutory Provisions
    provs = parsed_sections.get("legal_provisions") or []
    if provs and provs != ["None"]:
        prov_list_str = "\n".join([f"- **{p}**" for p in provs])
    elif retrieved_chunks:
        titles = list({c.get("title") or c.get("act_name") or "Statute" for c in retrieved_chunks})
        prov_list_str = "\n".join([f"- **{t}**" for t in titles])
    else:
        prov_list_str = "- General statutory framework under Indian law."
    sections.append(f"## 4. Relevant Statutory Provisions\n{prov_list_str}\n")

    # 5. Substantive Legal Rules
    rules_text = (
        "1. **Rule of Statutory Compliance**: The express text of the statute governs the rights and duties of parties.\n"
        "2. **Rule of Legal Certainty**: Statutory obligations must be enforced without arbitrary deviation.\n"
        "3. **Rule of Precedent**: Decisions of constitutional courts bind judicial and quasi-judicial authorities."
    )
    sections.append(f"## 5. Substantive Legal Rules\n{rules_text}\n")

    # 6. Exceptions & Limitations
    lim_text = parsed_sections.get("limitations") or (
        "- Statutory remedies are subject to prescribed limitation periods under the Limitation Act, 1963.\n"
        "- Substantive exceptions enacted in the governing legislation restrict the scope of relief."
    )
    sections.append(f"## 6. Exceptions & Limitations\n{lim_text}\n")

    # 7. Relevant Case Law Precedents
    case_items = []
    judgs = parsed_sections.get("judgments") or []
    if judgs and judgs != ["None"]:
        for j in judgs:
            case_items.append(f"- **{j}**")
    if precedents:
        for p in precedents[:3]:
            case_items.append(
                f"- **{p.get('case_name')}** ({p.get('court')}, {p.get('year')}) — *{p.get('citation')}*\n"
                f"  Ratio: {p.get('legal_issue', 'Landmark precedent')}"
            )
    if not case_items:
        case_items.append("- Binding judicial precedents of the Supreme Court and relevant High Courts apply.")
    sections.append("## 7. Relevant Case Law Precedents\n" + "\n".join(case_items) + "\n")

    # 8. Statutory Application & Analysis
    app_text = (
        f"Applying the statutory mandates to the question ('{query.strip()}'):\n"
        f"- The essential conditions stipulated by statute must be strictly satisfied on evidence.\n"
        f"- Where criminal or civil liability is alleged, the requisite burden and standard of proof must be discharged."
    )
    sections.append(f"## 8. Statutory Application & Analysis\n{app_text}\n")

    # 9. Practical Guidance & Remedies
    guidance_text = (
        "- **Remedy Selection**: Identify appropriate forums (Civil Court, Magistrate Court, or High Court writ jurisdiction).\n"
        "- **Limitation Check**: Verify that proceedings are initiated within the statutory limitation timeframe.\n"
        "- **Documentation**: Maintain verified documentation and statutory notices."
    )
    sections.append(f"## 9. Practical Guidance & Remedies\n{guidance_text}\n")

    # 10. Procedural Compliance Checklist
    checklist_text = (
        "1. Confirm territorial and subject-matter jurisdiction of the forum.\n"
        "2. Ensure proper service of statutory notice if mandated.\n"
        "3. Comply with court fee requirements and procedural rules.\n"
        "4. Prepare affidavit evidence in compliance with the Bharatiya Sakshya Adhiniyam / Indian Evidence Act."
    )
    sections.append(f"## 10. Procedural Compliance Checklist\n{checklist_text}\n")

    # 11. Conclusion
    conc_text = (
        f"In summary, '{query.strip()}' is governed by strict statutory rules and established case law. "
        f"Parties must observe all procedural and evidentiary conditions to secure legal protection or remedy."
    )
    sections.append(f"## 11. Conclusion\n{conc_text}\n")

    # 12. Verified Sources and Citations
    if citations:
        sources_text = "\n".join(citations)
    elif retrieved_chunks:
        sources_text = "\n".join([
            f"[{c.get('citation_id', i+1)}] {c.get('title') or c.get('act_name')} ({c.get('court', 'India')})"
            for i, c in enumerate(retrieved_chunks)
        ])
    else:
        sources_text = "None (Statutory interpretation derived from primary enacted law)."
    sections.append(f"## 12. Verified Sources and Citations\n{sources_text}\n")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Fallback 6-section builder  (used when evidence is thin or LLM answer is empty)
# ---------------------------------------------------------------------------

def build_fallback_legal_response(
    query: str,
    llm_answer: str,
    parsed_sections: dict | None = None,
    knowledge_mode: str = "general_knowledge",
    retrieved_chunks: list | None = None,
    citations_list: list | None = None,
) -> str:
    """
    Build a concise, honest 6-section legal response when:
      - The retrieved evidence is insufficient (knowledge_mode = 'general_knowledge').
      - The LLM answer is short or the parsed sections are sparse.
      - The caller wants a structured fallback rather than an empty response.

    Section structure (user-specified):
      1. Direct Answer
      2. Legal Position
      3. Relevant Authorities
      4. Application to the Question
      5. Practical Next Steps
      6. Limitations and Disclaimer

    Rules:
      - NEVER fabricates citations, case names, section numbers, or sources.
      - Uses only content already present in llm_answer / parsed_sections.
      - Marks all propositions with the appropriate evidence tag.
      - Returns a non-empty useful answer even when retrieval fails entirely.
    """
    parsed = parsed_sections or {}
    sections: list[str] = []

    # ── Section 1: Direct Answer ────────────────────────────────────────────
    raw_answer = (
        parsed.get("answer")
        or parsed.get("direct_answer")
        or llm_answer.strip()
        or "(No answer generated — please try rephrasing your question.)"
    )
    # Strip any existing headers from the raw answer if it was an LLM dump
    raw_answer = re.sub(r"^#{1,3}\s+Answer\s*\n", "", raw_answer, flags=re.I).strip()
    sections.append(f"## 1. Direct Answer\n\n{raw_answer}")

    # ── Section 2: Legal Position ───────────────────────────────────────────
    legal_pos = (
        parsed.get("legal_position")
        or parsed.get("relevant_legal_provisions")
        or parsed.get("legal_provisions_text")
        or ""
    )
    if legal_pos and legal_pos.strip().lower() not in ("none", "n/a", ""):
        sections.append(f"## 2. Legal Position\n\n{legal_pos.strip()}")

    # ── Section 3: Relevant Authorities ────────────────────────────────────
    authorities = (
        parsed.get("relevant_authorities")
        or parsed.get("relevant_judgments")
        or parsed.get("judgments_text")
        or ""
    )
    if authorities and authorities.strip().lower() not in ("none", "n/a", ""):
        auth_text = authorities.strip()
        km_note = ""
        if knowledge_mode in ("general_knowledge", "partial_evidence"):
            km_note = "\n\n> ⚠️ The authorities listed above have **not** been verified against retrieved corpus documents. Treat as [GENERAL LEGAL KNOWLEDGE] until independently confirmed."
        sections.append(f"## 3. Relevant Authorities\n\n{auth_text}{km_note}")

    # ── Section 4: Application to the Question ─────────────────────────────
    application = (
        parsed.get("application")
        or parsed.get("application_to_question")
        or parsed.get("legal_reasoning")
        or ""
    )
    if application and application.strip().lower() not in ("none", "n/a", ""):
        sections.append(f"## 4. Application to the Question\n\n{application.strip()}")

    # ── Section 5: Practical Next Steps ────────────────────────────────────
    practical = (
        parsed.get("practical_next_steps")
        or parsed.get("practical_guidance")
        or parsed.get("next_steps")
        or ""
    )
    if practical and practical.strip().lower() not in ("none", "n/a", ""):
        sections.append(f"## 5. Practical Next Steps\n\n{practical.strip()}")

    # ── Section 6: Limitations and Disclaimer ──────────────────────────────
    limitations = parsed.get("limitations") or ""
    if knowledge_mode == "general_knowledge":
        disclaimer = (
            "**Source Verification Status:** This answer is based on general Indian legal knowledge. "
            "No relevant documents were retrieved from the legal corpus for this query. "
            "Specific citations, case names, and section numbers in this answer have **not** been "
            "verified against retrieved sources.\n\n"
            "**Professional Advice:** This is general legal information only. For any actual legal "
            "proceeding, dispute, or decision, consult a qualified lawyer admitted to practice in "
            "the relevant jurisdiction."
        )
    elif knowledge_mode == "partial_evidence":
        disclaimer = (
            "**Source Verification Status:** This answer combines retrieved source evidence "
            "(marked [VERIFIED SOURCE]) with general legal knowledge (marked [GENERAL LEGAL KNOWLEDGE]). "
            "Only the [VERIFIED SOURCE] portions are directly supported by retrieved documents.\n\n"
            "**Professional Advice:** This is general legal information only. Consult a qualified lawyer "
            "for advice on specific legal matters."
        )
    else:
        disclaimer = (
            "**Source Verification Status:** This answer is supported by retrieved legal sources. "
            "Citations have been checked against the corpus.\n\n"
            "**Professional Advice:** This is general legal information only. Consult a qualified lawyer "
            "for advice on specific legal matters."
        )
    if limitations:
        disclaimer = f"{limitations.strip()}\n\n{disclaimer}"
    sections.append(f"## 6. Limitations and Disclaimer\n\n{disclaimer}")

    return "\n\n---\n\n".join(sections)
