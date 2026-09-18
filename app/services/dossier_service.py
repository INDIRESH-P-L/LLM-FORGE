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
    ) -> Dict[str, Any]:
        """
        Structures research into a formal advocate's bench memorandum.
        """
        today = date.today().strftime("%d %B %Y")
        clean_cites = citations or []

        # Extract or frame issues from query
        q_clean = query.strip().rstrip("?").strip()
        issues = [
            f"Whether on the facts of the case, the legal position asserted in: '{q_clean[:120]}' is sustainable under the governing statutory framework?",
            "Whether the impugned provisions or executive action satisfy the constitutional tests of reasonableness, proportionality, and non-arbitrariness under Part III of the Constitution of India?",
        ]

        # Statutory references
        statutes = [
            {"act": "Constitution of India", "provisions": "Articles 14, 19, 21, 32"},
            {"act": "Bharatiya Nyaya Sanhita, 2023 / IPC", "provisions": "Sections 103, 316, 318 BNS (S. 302, 406, 420 IPC)"},
            {"act": "Bharatiya Nagarik Suraksha Sanhita, 2023", "provisions": "Sections 480, 482 BNSS (S. 437, 439 CrPC)"},
        ]

        # Precedent knowledge repository for Table of Authorities
        KNOWN_AUTHORITIES = [
            {
                "keys": ["595", "sisodia", "manish sisodia"],
                "case_name": "Manish Sisodia v. Directorate of Enforcement",
                "citation": "(2024) INSC 595",
                "ratio": "Right to speedy trial under Art. 21 is sacred and overrides statutory bail bars when prolonged pre-trial incarceration occurs without trial progression."
            },
            {
                "keys": ["antil", "satender", "satender kumar antil"],
                "case_name": "Satender Kumar Antil v. CBI",
                "citation": "(2022) 10 SCC 51",
                "ratio": "Bail is the rule, jail is the exception; procedure must be just and non-oppressive; strict compliance with Section 41/41A CrPC (S. 35 BNSS)."
            },
            {
                "keys": ["arnesh", "arnesh kumar"],
                "case_name": "Arnesh Kumar v. State of Bihar",
                "citation": "(2014) 8 SCC 273",
                "ratio": "Mandatory notice before arrest for offences punishable up to 7 years; arrest must not be routine or automatic."
            },
            {
                "keys": ["puttaswamy", "privacy"],
                "case_name": "K.S. Puttaswamy v. Union of India",
                "citation": "(2017) 10 SCC 1",
                "ratio": "Right to privacy is an intrinsic fundamental right under Article 21; State action must satisfy four-prong proportionality test."
            },
            {
                "keys": ["bhajan lal", "bhajanlal", "482"],
                "case_name": "State of Haryana v. Bhajan Lal",
                "citation": "1992 Supp (1) SCC 335",
                "ratio": "Seven landmark categories for quashing criminal proceedings under Section 482 CrPC / Section 528 BNSS to prevent abuse of court process."
            },
            {
                "keys": ["kesavananda", "basic structure"],
                "case_name": "Kesavananda Bharati v. State of Kerala",
                "citation": "(1973) 4 SCC 225",
                "ratio": "Parliament's amending power under Article 368 cannot damage or destroy the Basic Structure of the Constitution."
            },
            {
                "keys": ["maneka", "maneka gandhi"],
                "case_name": "Maneka Gandhi v. Union of India",
                "citation": "(1978) 1 SCC 248",
                "ratio": "Procedure established by law under Article 21 must be just, fair, and reasonable, not arbitrary, fanciful or oppressive."
            }
        ]

        # Resolve Table of Authorities
        search_text = f"{query} {answer} {' '.join(clean_cites)}".lower()
        table_of_authorities = []
        seen_citations = set()

        # Match from known database
        for auth in KNOWN_AUTHORITIES:
            if any(k in search_text for k in auth["keys"]):
                if auth["citation"] not in seen_citations:
                    table_of_authorities.append({
                        "case_name": auth["case_name"],
                        "citation": auth["citation"],
                        "ratio": auth["ratio"]
                    })
                    seen_citations.add(auth["citation"])

        # Include any explicit user citations not already resolved
        for c in clean_cites:
            c_str = str(c).strip()
            if c_str and c_str not in seen_citations:
                table_of_authorities.append({
                    "case_name": "Authoritative Precedent",
                    "citation": c_str,
                    "ratio": "Binding legal authority cited in furtherance of Counsel's propositions."
                })
                seen_citations.add(c_str)

        # Fallback default authorities if none detected
        if not table_of_authorities:
            table_of_authorities = [
                {
                    "case_name": "Manish Sisodia v. Directorate of Enforcement",
                    "citation": "(2024) INSC 595",
                    "ratio": "Right to speedy trial under Art. 21 is sacred and overrides statutory bail bars."
                },
                {
                    "case_name": "Satender Kumar Antil v. CBI",
                    "citation": "(2022) 10 SCC 51",
                    "ratio": "Bail is the rule, jail is the exception; procedure must be just and non-oppressive."
                }
            ]

        dossier = {
            "dossier_id": f"DOSSIER-{date.today().strftime('%Y%m%d')}-01",
            "date": today,
            "court": court,
            "court_heading": court,
            "case_title": case_title or "LEGAL MEMORANDUM FOR COUNSEL",
            "jurisdiction": "CONSTITUTIONAL & APPELLATE JURISDICTION",
            "query": query,
            "questions_framed": issues,
            "issues_framed": issues,
            "statutory_matrix": statutes,
            "table_of_authorities": table_of_authorities,
            "binding_citations": [a["citation"] for a in table_of_authorities],
            "synopsis": answer,
            "executive_summary": answer[:600] + ("..." if len(answer) > 600 else ""),
            "full_brief_text": answer,
            "advocacy_notes": [
                "Ensure compliance with the relevant High Court / Supreme Court Practice Directions.",
                "Verify whether any cited High Court rulings have been stayed or appealed in the Supreme Court.",
                "Check notification dates for transitional clauses under the new Criminal Laws (BNS / BNSS / BSA).",
            ],
        }

        return dossier

    @classmethod
    def generate_pdf(cls, dossier: Dict[str, Any]) -> bytes:
        """
        Compiles the dossier into a court-ready PDF file using ReportLab.
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
            spaceBefore=8,
            spaceAfter=4,
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

        # 3. Framed Issues
        story.append(Paragraph("I. PRIMARY QUESTIONS FRAMED FOR DETERMINATION", heading_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
        for i, issue in enumerate(dossier.get("issues_framed", []), 1):
            story.append(Paragraph(f"<b>Issue {i}:</b> {issue}", bullet_style))
        story.append(Spacer(1, 8))

        # 4. Statutory Matrix Table
        story.append(Paragraph("II. APPLICABLE STATUTORY FRAMEWORK & TRANSITIONAL MATRIX", heading_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
        stat_rows = [["Governing Act / Statute", "Relevant Sections & Provisions"]]
        for s in dossier.get("statutory_matrix", []):
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

        # 5. Executive Legal Analysis
        story.append(Paragraph("III. EXECUTIVE LEGAL ANALYSIS & RATIO OF AUTHORITIES", heading_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))

        # Split full brief into paragraphs
        brief_paras = dossier.get("full_brief_text", "").split("\n\n")
        for bp in brief_paras[:12]:
            clean_bp = bp.replace("#", "").replace("**", "").replace("*", "").strip()
            if clean_bp:
                story.append(Paragraph(clean_bp, body_style))

        story.append(Spacer(1, 8))

        # 6. Table of Authorities
        auth_items = dossier.get("table_of_authorities", [])
        if auth_items:
            story.append(Paragraph("IV. TABLE OF AUTHORITIES & BINDING PRECEDENTS", heading_style))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
            auth_rows = [["Authority / Precedent", "Citation", "Ratio Decidendi"]]
            for a in auth_items[:6]:
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
        else:
            cites = dossier.get("binding_citations", [])
            if cites:
                story.append(Paragraph("IV. BINDING AUTHORITIES & CASE CITATIONS", heading_style))
                story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
                for c in cites:
                    story.append(Paragraph(f"• {c}", bullet_style))
                story.append(Spacer(1, 8))

        # 7. Strategic Notes
        story.append(Paragraph("V. COUNSEL'S STRATEGIC CHECKLIST", heading_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
        for note in dossier.get("advocacy_notes", []):
            story.append(Paragraph(f"✓ {note}", bullet_style))

        # Build document
        doc.build(story)
        buf.seek(0)
        return buf.getvalue()
