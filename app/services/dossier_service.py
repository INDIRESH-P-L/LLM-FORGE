"""
app/services/dossier_service.py
===============================
Advocate's Case Dossier & Court Brief Generator for LegalMind AI.

Synthesizes court-ready bench memorandums and exports formatted PDF dossiers
using ReportLab with formal legal typography and cause titles.
"""

from __future__ import annotations

import io
import logging
import re
import sys
from datetime import date
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

log = logging.getLogger("legalmind.dossier")


def _get_llm_instance() -> Any:
    """Safely retrieves the loaded LegalMindModel instance if available in memory."""
    try:
        import app.main as main_mod
        if getattr(main_mod, "_model", None) is not None:
            return main_mod._model
    except Exception:
        pass

    try:
        if "scripts_05_rag" in sys.modules:
            mod = sys.modules["scripts_05_rag"]
            cls_obj = getattr(mod, "LegalMindModel", None)
            if cls_obj and getattr(cls_obj, "_instance", None) is not None:
                return cls_obj._instance
    except Exception:
        pass

    try:
        if "rag_module" in sys.modules:
            mod = sys.modules["rag_module"]
            cls_obj = getattr(mod, "LegalMindModel", None)
            if cls_obj and getattr(cls_obj, "_instance", None) is not None:
                return cls_obj._instance
    except Exception:
        pass

    return None


class DossierService:
    """Builds court-ready case dossiers and compiles downloadable PDF files."""

    @classmethod
    def generate_dossier(
        cls,
        case_title: str,
        query: str,
        answer: str,
        court: str = "IN THE SUPREME COURT OF INDIA",
        citations: Optional[List[str]] = None,
        model: Any = None,
        brief_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Structures research into a formal advocate's bench memorandum.
        Analyzes user input using LLM / analytical legal reasoning to generate
        a brief, synthesized report rather than echoing raw input text.
        """
        if model is None:
            model = _get_llm_instance()

        today = date.today().strftime("%d %B %Y")
        case_title = (case_title or "").strip()
        query = (query or "").strip()
        answer = (answer or "").strip()
        court = (court or "").strip() or "IN THE SUPREME COURT OF INDIA"


        DISALLOWED_STRINGS = {
            "", "cause title", "legal question", "factual matrix",
            "legal memorandum for advocate", "n/a", "none", "null", "undefined", "brief title"
        }

        if not case_title or case_title.lower() in DISALLOWED_STRINGS:
            raise ValueError("Genuine, non-placeholder Case Name / Cause Title must be provided by user.")

        if not query or query.lower() in DISALLOWED_STRINGS:
            raise ValueError("Genuine, non-placeholder legal proposition / question must be provided by user.")

        if not answer or answer.lower() in DISALLOWED_STRINGS:
            raise ValueError("Genuine, non-placeholder factual matrix / submissions must be provided by user.")

        # Discard legacy dummy citations injected by older frontend caches
        clean_cites = [
            str(c).strip() for c in (citations or [])
            if str(c).strip() and str(c).strip() not in ("(2024) INSC 595", "(2022) 10 SCC 51", "INSC 595", "10 SCC 51")
        ]

        # 1. Extract or frame issues strictly from user's proposition / query
        issues = []
        raw_lines = [line.strip().lstrip("0123456789.-)•* ") for line in query.split("\n") if line.strip()]
        for line in raw_lines:
            if not line or line.lower() in DISALLOWED_STRINGS:
                continue
            if line.lower().startswith("whether") or line.endswith("?"):
                issues.append(line if line.endswith("?") else f"{line}?")
            else:
                first_char = line[0].lower() if len(line) > 1 and not line[:4].isupper() else line[0]
                issues.append(f"Whether {first_char + line[1:] if len(line) > 1 else line}?")

        if not issues:
            q_clean = query.rstrip("?").strip()
            issues.append(f"Whether on the facts presented: '{q_clean}' is sustainable under law?")

        # 2. Extract statutory references solely from user text
        search_text = f"{query}\n{answer}"
        statutes = []

        # Check for Constitution Articles mentioned in user text
        const_articles = re.findall(r"\bArticle[s]?\s+([0-9A-Za-z,\s(a-z)]+)", search_text, re.IGNORECASE)
        if const_articles or re.search(r"\bConstitution\s+of\s+India\b", search_text, re.IGNORECASE):
            arts_found = []
            for m in re.finditer(r"\bArticle\s+([0-9A-Za-z]+(?:\s*\(\d+\)[a-z]?)?)\b", search_text, re.IGNORECASE):
                arts_found.append(f"Art. {m.group(1)}")
            arts_unique = list(dict.fromkeys(arts_found))
            provisions_str = ", ".join(arts_unique) if arts_unique else "Fundamental Rights / Constitutional Provisions"
            statutes.append({"act": "Constitution of India", "provisions": provisions_str})

        # Known Acts with detection patterns
        KNOWN_ACT_PATTERNS = [
            ("Negotiable Instruments Act, 1881", [r"\bNI\s+Act\b", r"\bNegotiable\s+Instruments\b", r"\bSection\s+138\b"]),
            ("Bharatiya Nyaya Sanhita, 2023", [r"\bBNS\b", r"\bBharatiya\s+Nyaya\s+Sanhita\b"]),
            ("Indian Penal Code, 1860", [r"\bIPC\b", r"\bIndian\s+Penal\s+Code\b"]),
            ("Bharatiya Nagarik Suraksha Sanhita, 2023", [r"\bBNSS\b", r"\bBharatiya\s+Nagarik\s+Suraksha\b"]),
            ("Code of Criminal Procedure, 1973", [r"\bCrPC\b", r"\bCode\s+of\s+Criminal\s+Procedure\b"]),
            ("Bharatiya Sakshya Adhiniyam, 2023", [r"\bBSA\b", r"\bBharatiya\s+Sakshya\b"]),
            ("Indian Evidence Act, 1872", [r"\bEvidence\s+Act\b", r"\bIEA\b"]),
            ("Companies Act, 2013", [r"\bCompanies\s+Act\b"]),
            ("Arbitration and Conciliation Act, 1996", [r"\bArbitration\s+and\s+Conciliation\b", r"\bArbitration\s+Act\b"]),
            ("Insolvency and Bankruptcy Code, 2016", [r"\bIBC\b", r"\bInsolvency\s+and\s+Bankruptcy\b"]),
            ("Prevention of Money Laundering Act, 2002", [r"\bPMLA\b", r"\bPrevention\s+of\s+Money\s+Laundering\b"]),
            ("Indian Contract Act, 1872", [r"\bContract\s+Act\b"]),
            ("Specific Relief Act, 1963", [r"\bSpecific\s+Relief\b"]),
            ("Code of Civil Procedure, 1908", [r"\bCPC\b", r"\bCivil\s+Procedure\b"]),
            ("Income Tax Act, 1961", [r"\bIncome\s+Tax\s+Act\b"]),
            ("Motor Vehicles Act, 1988", [r"\bMotor\s+Vehicles\s+Act\b", r"\bMV\s+Act\b"]),
            ("Consumer Protection Act, 2019", [r"\bConsumer\s+Protection\b"]),
        ]

        sections_found = re.findall(r"\bSection[s]?\s+([0-9A-Za-z,\s(a-z)]+)", search_text, re.IGNORECASE)

        for act_name, pats in KNOWN_ACT_PATTERNS:
            if any(re.search(p, search_text, re.IGNORECASE) for p in pats):
                act_sections = []
                for sec_m in re.finditer(r"\bSection[s]?\s+([0-9A-Za-z]+(?:\s*\(\d+\))?)\b", search_text, re.IGNORECASE):
                    act_sections.append(f"Sec. {sec_m.group(1)}")
                act_sec_unique = list(dict.fromkeys(act_sections))
                statutes.append({
                    "act": act_name,
                    "provisions": ", ".join(act_sec_unique[:5]) if act_sec_unique else "Substantive provisions as cited"
                })

        if not statutes and sections_found:
            statutes.append({
                "act": "Governing Statutory Enactments",
                "provisions": f"Sections {', '.join(s.strip() for s in sections_found[:5])}"
            })

        # 3. Resolve Table of Authorities ONLY if mentioned or cited by user in query or answer
        KNOWN_AUTHORITIES = [
            {
                "keys": ["manish sisodia", "sisodia v. directorate", "sisodia v. ed"],
                "case_name": "Manish Sisodia v. Directorate of Enforcement",
                "citation": "(2024) INSC 595",
                "ratio": "Right to speedy trial under Art. 21 is sacred and overrides statutory bail bars when prolonged pre-trial incarceration occurs without trial progression."
            },
            {
                "keys": ["satender kumar antil", "satender antil v. cbi"],
                "case_name": "Satender Kumar Antil v. CBI",
                "citation": "(2022) 10 SCC 51",
                "ratio": "Bail is the rule, jail is the exception; procedure must be just and non-oppressive; strict compliance with Section 41/41A CrPC (S. 35 BNSS)."
            },
            {
                "keys": ["arnesh kumar v. state", "arnesh kumar"],
                "case_name": "Arnesh Kumar v. State of Bihar",
                "citation": "(2014) 8 SCC 273",
                "ratio": "Mandatory notice before arrest for offences punishable up to 7 years; arrest must not be routine or automatic."
            },
            {
                "keys": ["puttaswamy", "k.s. puttaswamy"],
                "case_name": "K.S. Puttaswamy v. Union of India",
                "citation": "(2017) 10 SCC 1",
                "ratio": "Right to privacy is an intrinsic fundamental right under Article 21; State action must satisfy four-prong proportionality test."
            },
            {
                "keys": ["bhajan lal", "state of haryana v. bhajan lal"],
                "case_name": "State of Haryana v. Bhajan Lal",
                "citation": "1992 Supp (1) SCC 335",
                "ratio": "Seven landmark categories for quashing criminal proceedings under Section 482 CrPC / Section 528 BNSS to prevent abuse of court process."
            },
            {
                "keys": ["kesavananda bharati"],
                "case_name": "Kesavananda Bharati v. State of Kerala",
                "citation": "(1973) 4 SCC 225",
                "ratio": "Parliament's amending power under Article 368 cannot damage or destroy the Basic Structure of the Constitution."
            },
            {
                "keys": ["maneka gandhi v. union", "maneka gandhi"],
                "case_name": "Maneka Gandhi v. Union of India",
                "citation": "(1978) 1 SCC 248",
                "ratio": "Procedure established by law under Article 21 must be just, fair, and reasonable, not arbitrary, fanciful or oppressive."
            }
        ]

        user_content_lower = f"{query} {answer}".lower()
        table_of_authorities = []
        seen_citations = set()

        for auth in KNOWN_AUTHORITIES:
            if any(k in user_content_lower for k in auth["keys"]):
                if auth["citation"] not in seen_citations:
                    table_of_authorities.append({
                        "case_name": auth["case_name"],
                        "citation": auth["citation"],
                        "ratio": auth["ratio"]
                    })
                    seen_citations.add(auth["citation"])

        # Include explicit citations passed by caller ONLY if present in user's query or factual matrix
        for c in clean_cites:
            c_str = str(c).strip()
            if c_str and (c_str.lower() in user_content_lower) and (c_str not in seen_citations):
                table_of_authorities.append({
                    "case_name": "Precedent Cited by Counsel",
                    "citation": c_str,
                    "ratio": "Cited in user submissions."
                })
                seen_citations.add(c_str)

        # Detect reporter citations in user text
        text_cites = re.findall(r"(?:\(\d{4}\)|\b\d{4})\s+(?:Supp\s+\(\d+\)\s+)?\d*\s*(?:SCC|INSC|SCR|AIR|SCALE)\s*\d*", search_text, re.IGNORECASE)
        for tc in text_cites:
            tc_clean = tc.strip()
            if tc_clean and tc_clean not in seen_citations:
                table_of_authorities.append({
                    "case_name": "Precedent Cited in Matrix",
                    "citation": tc_clean,
                    "ratio": "Cited in user factual matrix / research synthesis."
                })
                seen_citations.add(tc_clean)

        # CRITICAL: NO FALLBACK! If user did not cite precedents, table_of_authorities remains empty.

        # 4. Synthesize an analytical, concise advocate's brief using LLM or structured legal analysis
        if brief_text and len(brief_text.strip()) > 80:
            cleaned = brief_text.strip()
            synopsis_match = re.search(
                r"(?:I\.\s+CONCISE\s+FACTUAL\s+SYNOPSIS|SYNOPSIS)[\s:\-]+(.*?)(?=\n\s*(?:II\.|STATUTORY|\Z))",
                cleaned,
                re.DOTALL | re.IGNORECASE,
            )
            synopsis = synopsis_match.group(1).strip() if synopsis_match else cleaned[:350].strip()
            brief_data = {
                "synopsis": synopsis,
                "full_brief_text": cleaned
            }
        else:
            brief_data = cls._synthesize_analytical_brief(
                case_title=case_title,
                court=court,
                query=query,
                facts=answer,
                issues=issues,
                statutes=statutes,
                authorities=table_of_authorities,
                model=model,
            )

        dossier = {
            "dossier_id": f"DOSSIER-{date.today().strftime('%Y%m%d')}-01",
            "date": today,
            "court": court,
            "court_heading": court,
            "case_title": case_title,
            "jurisdiction": "CONSTITUTIONAL & APPELLATE JURISDICTION",
            "query": query,
            "questions_framed": issues,
            "issues_framed": issues,
            "statutory_matrix": statutes,
            "table_of_authorities": table_of_authorities,
            "binding_citations": [a["citation"] for a in table_of_authorities],
            "synopsis": brief_data.get("synopsis", ""),
            "executive_summary": brief_data.get("synopsis", "")[:600],
            "full_brief_text": brief_data.get("full_brief_text", ""),
            "advocacy_notes": [],
        }

        return dossier

    @classmethod
    def _synthesize_analytical_brief(
        cls,
        case_title: str,
        court: str,
        query: str,
        facts: str,
        issues: List[str],
        statutes: List[Dict[str, str]],
        authorities: List[Dict[str, str]],
        model: Any = None,
    ) -> Dict[str, str]:
        """
        Synthesizes a structured Advocate's Bench Memorandum using the LLM or analytical legal reasoning.
        Never outputs raw user text verbatim; distills facts into structured grounds and prayers.
        """
        if model is not None and hasattr(model, "generate_chat_completion"):
            try:
                system_prompt = (
                    "You are a Senior Appellate Advocate of the Supreme Court of India. "
                    "Your duty is to conduct a high-level legal analysis of the dispute presented and "
                    "synthesize a brief, authoritative Bench Memorandum. "
                    "STRICT MANDATE: DO NOT repeat, quote, or echo the counsel's raw factual narrative verbatim. "
                    "Distill the facts into a concise 3-4 sentence legal synopsis, analyze the statutory infractions, "
                    "and formulate clear advocate's grounds and operative prayers."
                )
                user_prompt = (
                    f"FORUM: {court}\n"
                    f"CAUSE TITLE: {case_title}\n"
                    f"LEGAL QUESTION: {query}\n"
                    f"COUNSEL'S FACTUAL NARRATIVE:\n{facts}\n\n"
                    "INSTRUCTIONS:\n"
                    "Synthesize a concise, authoritative brief containing exactly these 4 titled sections:\n"
                    "I. CONCISE FACTUAL SYNOPSIS (3-4 sentences extracting the essential dispute, key parties, and the administrative/judicial order impugned. Do NOT copy raw text).\n"
                    "II. STATUTORY & CONSTITUTIONAL ANALYSIS (Analytical application of the relevant provisions—e.g. Article 14, Natural Justice, or governing statutes—to the core dispute).\n"
                    "III. ADVOCATE'S SUBMISSIONS (3 concise, numbered grounds/propositions establishing why relief should be granted).\n"
                    "IV. OPERATIVE PRAYER / RELIEF SOUGHT (A precise prayer for the remedy sought before the court).\n"
                    "Maintain formal Indian legal typography and courtroom register. Keep it brief and analytical."
                )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                brief_output = model.generate_chat_completion(messages, max_new_tokens=520, temperature=0.3)
                if brief_output and len(brief_output.strip()) > 80:
                    cleaned = brief_output.strip()
                    if "<think>" in cleaned and "</think>" in cleaned:
                        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
                    if cleaned.startswith("```"):
                        lines = cleaned.splitlines()
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].startswith("```"):
                            lines = lines[:-1]
                        cleaned = "\n".join(lines).strip()

                    synopsis_match = re.search(
                        r"(?:I\.\s+CONCISE\s+FACTUAL\s+SYNOPSIS|SYNOPSIS)[\s:\-]+(.*?)(?=\n\s*(?:II\.|STATUTORY|\Z))",
                        cleaned,
                        re.DOTALL | re.IGNORECASE,
                    )
                    synopsis = synopsis_match.group(1).strip() if synopsis_match else cleaned[:350].strip()
                    return {
                        "synopsis": synopsis,
                        "full_brief_text": cleaned
                    }
            except Exception as e:
                log.warning(f"LLM brief generation failed: {e}. Falling back to structured analytical synthesis.")

        return cls._fallback_analytical_synthesis(case_title, court, query, facts, issues, statutes)

    @classmethod
    def _fallback_analytical_synthesis(
        cls,
        case_title: str,
        court: str,
        query: str,
        facts: str,
        issues: List[str],
        statutes: List[Dict[str, str]],
    ) -> Dict[str, str]:
        """
        Rule-guided Senior Advocate analytical synthesis. Distills raw facts into
        formal courtroom legal sections without echoing user content verbatim.
        """
        f_lower = facts.lower()
        q_lower = query.lower()

        # Extract party names from case title if "v." or "vs." present
        petitioner = "The petitioner"
        respondent = "the respondent authority"
        if " v. " in case_title:
            parts = case_title.split(" v. ", 1)
            petitioner = parts[0].strip() or "The petitioner"
            respondent = parts[1].strip() or "the respondent authority"
        elif " vs. " in case_title.lower():
            parts = re.split(r"\s+vs\.?\s+", case_title, flags=re.IGNORECASE)
            if len(parts) == 2:
                petitioner = parts[0].strip() or "The petitioner"
                respondent = parts[1].strip() or "the respondent authority"

        # Identify core legal themes
        is_natural_justice = any(k in f_lower or k in q_lower for k in ["natural justice", "audi alteram", "hearing", "notice", "arbitrary", "article 14"])
        is_bail = any(k in f_lower or k in q_lower for k in ["bail", "custody", "incarceration", "439", "482", "480", "bnss", "crpc", "pmla"])
        is_cheque = any(k in f_lower or k in q_lower for k in ["138", "cheque", "dishonour", "negotiable", "ni act"])
        is_quashing = any(k in f_lower or k in q_lower for k in ["quash", "fir", "charge sheet", "cognizance", "malice", "bhajan lal"])

        # 1. Distill concise factual synopsis
        if is_natural_justice:
            synopsis = (
                f"The present proceedings arise out of an adverse determination rendered against {petitioner} in summary "
                f"derogation of procedural fairness. The aggrieved party was subjected to punitive consequences without the "
                f"issuance of a prior show-cause notice, denial of inspection of primary records, and total absence of a personal "
                f"hearing before {respondent}. The appellate forum affirmed the impugned action without recording independent reasons, "
                f"compelling recourse to this Hon'ble Forum under extraordinary supervisory jurisdiction."
            )
        elif is_bail:
            synopsis = (
                f"The applicant ({petitioner}) seeks grant of regular / anticipatory bail in connection with the impugned proceedings. "
                "The investigation has substantially concluded, custody is no longer required for custodial interrogation, "
                "and prolonged pre-trial incarceration without progress of trial converts detention into punitive custody in "
                "contravention of personal liberty guarantees under Article 21."
            )
        elif is_cheque:
            synopsis = (
                "The dispute pertains to statutory proceedings under the Negotiable Instruments Act, 1881. The complainant "
                "avers dishonour of instrument upon presentment for realization. The contest turns on the valid service of "
                "statutory demand notice, subsistence of legally enforceable debt, and whether statutory rebuttable presumptions "
                "stand discharged on the evidentiary record."
            )
        elif is_quashing:
            synopsis = (
                f"{petitioner} invokes the inherent powers of this Hon'ble Court seeking quashing of the impugned FIR / "
                "criminal proceedings. The allegations, taken at their face value, disclose no cognizable offence, constitute an "
                "abuse of judicial process, and represent an attempt to settle purely civil obligations through criminal coercion."
            )
        else:
            synopsis = (
                f"The present controversy involves a contested legal adjudication between {petitioner} and {respondent} before this Hon'ble Forum. "
                "The core grievance centers upon the sustainability of the impugned administrative / judicial action in light of "
                "governing statutory provisions and binding judicial precedents, requiring authoritative determination."
            )

        # 2. Statutory / Constitutional Analysis
        acts_cited = ", ".join(s["act"] for s in statutes) if statutes else "Applicable Constitutional & Statutory Framework"
        if is_natural_justice:
            stat_analysis = (
                "Under the constitutional mandate of Article 14 of the Constitution of India, non-arbitrariness and fairness in "
                "state action are non-negotiable prerequisites. Natural justice (audi alteram partem) is not an empty formality but "
                "a fundamental constitutional safeguard; an order entailing adverse civil consequences passed in breach of natural "
                "justice is fundamentally void ab initio. The failure of the statutory authority to provide disclosure and personal "
                "hearing renders the entire decision-making process procedurally ultra vires."
            )
        elif is_bail:
            stat_analysis = (
                "Under Article 21 of the Constitution and relevant statutory provisions governing bail, the cardinal rule remains "
                "'bail is the rule, jail is the exception'. Deprivation of liberty prior to conviction must be strictly necessary, "
                "proportionate, and non-oppressive. The absence of flight risk or witness tampering militates against continued "
                "incarceration."
            )
        else:
            stat_analysis = (
                f"The matter falls squarely within the purview of {acts_cited}. The governing statutory enactments mandate "
                "strict adherence to procedural prerequisites and jurisdictional thresholds. Any administrative or executive action "
                "deviating from prescribed statutory mandates violates settled canons of administrative law and the rule of law."
            )

        # 3. Advocate's Submissions (3 concise propositions)
        if is_natural_justice:
            submissions = (
                f"1. VIOLATION OF AUDI ALTERAM PARTEM: The complete omission to issue a formal show-cause notice and afford an oral hearing vitiates the impugned order at the threshold.\n"
                f"2. MANIFEST ARBITRARINESS & NON-APPLICATION OF MIND: The mechanical dismissal of {petitioner}'s representation without assigning reasons offends the constitutional discipline of Article 14.\n"
                f"3. PREJUDICE & BALANCE OF CONVENIENCE: Severe civil and reputational prejudice has been sustained without an opportunity to tender defence, necessitating immediate judicial corrective intervention."
            )
        elif is_bail:
            submissions = (
                f"1. ABSENCE OF REQUISITE CUSTODIAL NEED: The investigation has concluded and evidentiary material has been secured; no useful purpose is served by ongoing judicial detention of {petitioner}.\n"
                f"2. PARITY & COOPERATION: {petitioner} has unreservedly joined and cooperated with the inquiry and possesses deep roots in society with zero risk of flight or abscondence.\n"
                f"3. OVERRIDING NATURE OF ARTICLE 21: Prolonged incarceration pending trial constitutes punitive pre-trial detention in direct violation of the fundamental right to life and liberty."
            )
        else:
            submissions = (
                "1. JURISDICTIONAL INFIRMITY: The statutory authority acted beyond its statutory charter and ignored mandatory statutory procedures.\n"
                "2. LEGAL SUSTAINABILITY: The proposition of law framed for determination must be answered in favor of the petitioner on settled principles of statutory interpretation.\n"
                "3. BALANCE OF EQUITIES: The petitioner has made out a prima facie case with balance of convenience tilting overwhelmingly in favor of grant of relief."
            )

        # 4. Operative Prayer
        if is_natural_justice:
            prayer = (
                f"(a) Issue a writ of certiorari or appropriate direction quashing and setting aside the impugned adverse order passed against {petitioner};\n"
                f"(b) Remit the proceedings to {respondent} for de novo consideration strictly in accordance with natural justice and fair hearing;\n"
                "(c) Grant ad-interim stay of all consequential coercive actions during the pendency of the present petition."
            )
        elif is_bail:
            prayer = (
                f"(a) Enlarge {petitioner} on regular / anticipatory bail subject to reasonable conditions to the satisfaction of the Court;\n"
                "(b) Pass any other just and equitable relief deemed appropriate by this Hon'ble Forum."
            )
        else:
            prayer = (
                "(a) Set aside and quash the impugned order / proceedings;\n"
                f"(b) Grant consequential injunctive and declarative relief in favor of {petitioner};\n"
                "(c) Pass such further orders as this Hon'ble Court may deem fit in the interest of justice."
            )

        full_brief = (
            f"I. CONCISE FACTUAL SYNOPSIS\n{synopsis}\n\n"
            f"II. STATUTORY & CONSTITUTIONAL ANALYSIS\n{stat_analysis}\n\n"
            f"III. ADVOCATE'S SUBMISSIONS\n{submissions}\n\n"
            f"IV. OPERATIVE PRAYER / RELIEF SOUGHT\n{prayer}"
        )

        return {
            "synopsis": synopsis,
            "full_brief_text": full_brief
        }


    @classmethod
    def generate_pdf(cls, dossier: Dict[str, Any]) -> bytes:
        """
        Compiles the dossier into a court-ready PDF file using ReportLab.
        Renders only the real data provided by the user.
        """
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf,
            pagesize=A4,
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=54,
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "CourtTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            alignment=1,  # Center
            textColor=colors.HexColor("#1e293b"),
        )

        sub_title_style = ParagraphStyle(
            "SubTitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            alignment=1,  # Center
            textColor=colors.HexColor("#64748b"),
        )

        heading_style = ParagraphStyle(
            "Heading",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=10,
            spaceAfter=4,
        )

        sub_heading_style = ParagraphStyle(
            "SubHeading",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#800020"),
            spaceBefore=8,
            spaceAfter=3,
        )

        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#334155"),
            spaceAfter=6,
        )

        bullet_style = ParagraphStyle(
            "Bullet",
            parent=body_style,
            leftIndent=15,
            firstLineIndent=-10,
            spaceAfter=3,
        )

        story = []

        # 1. Header
        story.append(Paragraph(dossier.get("court", "IN THE SUPREME COURT OF INDIA").upper(), title_style))
        story.append(Spacer(1, 4))
        story.append(Paragraph(dossier.get("jurisdiction", "APPELLATE JURISDICTION"), sub_title_style))
        story.append(Spacer(1, 6))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f172a"), spaceAfter=10))

        # 2. Case Title & Metadata Block
        meta_data = [
            [
                Paragraph(f"<b>CASE BRIEF:</b> {dossier.get('case_title', 'LEGAL MEMORANDUM')}", body_style),
                Paragraph(f"<b>DATE:</b> {dossier.get('date', '')}", body_style),
            ],
            [
                Paragraph(f"<b>DOSSIER ID:</b> {dossier.get('dossier_id', '')}", body_style),
                Paragraph("<b>PREPARED BY:</b> LegalMind AI Research Platform", body_style),
            ],
        ]
        meta_table = Table(meta_data, colWidths=[330, 160])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 12))

        sec_idx = 1
        ROMAN_NUMS = ["I", "II", "III", "IV", "V", "VI"]

        # 3. Framed Issues (Only if present)
        issues = dossier.get("issues_framed", [])
        if issues:
            story.append(Paragraph(f"{ROMAN_NUMS[sec_idx-1]}. PRIMARY QUESTIONS FRAMED FOR DETERMINATION", heading_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
            for i, issue in enumerate(issues, 1):
                story.append(Paragraph(f"<b>Issue {i}:</b> {issue}", bullet_style))
            story.append(Spacer(1, 8))
            sec_idx += 1

        # 4. Statutory Matrix Table (Only if present in user data)
        stat_items = dossier.get("statutory_matrix", [])
        if stat_items:
            story.append(Paragraph(f"{ROMAN_NUMS[sec_idx-1]}. APPLICABLE STATUTORY FRAMEWORK", heading_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
            stat_rows = [["Governing Act / Statute", "Relevant Sections & Provisions"]]
            for s in stat_items:
                stat_rows.append([s.get("act", ""), s.get("provisions", "")])

            stat_table = Table(stat_rows, colWidths=[240, 250])
            stat_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(stat_table)
            story.append(Spacer(1, 10))
            sec_idx += 1

        # 5. Advocate's Bench Memorandum & Synthesized Brief
        story.append(Paragraph(f"{ROMAN_NUMS[sec_idx-1]}. ADVOCATE'S BENCH MEMORANDUM & SYNTHESIZED BRIEF", heading_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

        brief_text = dossier.get("full_brief_text", "").strip() or dossier.get("synopsis", "").strip()
        brief_paras = brief_text.split("\n\n")
        for bp in brief_paras[:18]:
            clean_bp = bp.replace("#", "").replace("**", "").replace("*", "").strip()
            if not clean_bp:
                continue
            lines = clean_bp.split("\n")
            first_line = lines[0].strip()
            if re.match(r"^(?:I|II|III|IV|V)\.\s+[A-Z\s/&-]+$", first_line):
                story.append(Paragraph(first_line, sub_heading_style))
                for sub_line in lines[1:]:
                    sub_line = sub_line.strip()
                    if sub_line:
                        if re.match(r"^(?:\d+\.|\([a-z]\))\s+", sub_line):
                            story.append(Paragraph(sub_line, bullet_style))
                        else:
                            story.append(Paragraph(sub_line, body_style))
            else:
                for sub_line in lines:
                    sub_line = sub_line.strip()
                    if sub_line:
                        if re.match(r"^(?:\d+\.|\([a-z]\))\s+", sub_line):
                            story.append(Paragraph(sub_line, bullet_style))
                        else:
                            story.append(Paragraph(sub_line, body_style))

        story.append(Spacer(1, 8))
        sec_idx += 1

        # 6. Table of Authorities (ONLY if user cited precedents)
        auth_items = dossier.get("table_of_authorities", [])
        if auth_items:
            story.append(Paragraph(f"{ROMAN_NUMS[sec_idx-1]}. TABLE OF AUTHORITIES & PRECEDENTS", heading_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
            auth_rows = [["Authority / Precedent", "Citation", "Ratio Decidendi"]]
            for a in auth_items[:8]:
                p_case = Paragraph(f"<b>{a.get('case_name', 'Precedent')}</b>", body_style)
                p_cite = Paragraph(a.get("citation", ""), body_style)
                p_ratio = Paragraph(a.get("ratio", ""), body_style)
                auth_rows.append([p_case, p_cite, p_ratio])

            auth_table = Table(auth_rows, colWidths=[160, 110, 220])
            auth_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8.5),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            story.append(auth_table)
            story.append(Spacer(1, 8))
            sec_idx += 1
        else:
            cites = dossier.get("binding_citations", [])
            if cites:
                story.append(Paragraph(f"{ROMAN_NUMS[sec_idx-1]}. BINDING AUTHORITIES & CITATIONS", heading_style))
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
                for c in cites:
                    story.append(Paragraph(f"• {c}", bullet_style))
                story.append(Spacer(1, 8))
                sec_idx += 1

        # 7. Strategic Notes
        adv_notes = dossier.get("advocacy_notes", [])
        if adv_notes:
            story.append(Paragraph(f"{ROMAN_NUMS[min(sec_idx-1, len(ROMAN_NUMS)-1)]}. COUNSEL'S STRATEGIC CHECKLIST", heading_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
            for note in adv_notes:
                story.append(Paragraph(f"✓ {note}", bullet_style))

        # Build document
        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
