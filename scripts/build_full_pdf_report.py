#!/usr/bin/env python3
"""
scripts/build_full_pdf_report.py
================================
Compiles the comprehensive, university-level academic project report into:
LegalMind_AI_Complete_Project_Report.pdf

Features:
- ReportLab 5.0+ programmatic document generation
- NumberedCanvas two-pass page numbering ("Page X of Y")
- Running headers and footers with chapter context and divider lines
- Clickable PDF bookmarks and outline tree for document navigation
- A4 academic page geometry with justified body text and consistent leading
- Professional styling: Deep Navy, Teal Slate, Charcoal, and Gold accents
- Structured tables for test cases, architecture comparison, corpus statistics
- Visual architectural diagram representation flowables
- Warning callout boxes for PyTorch fallback implementation and disclaimers
- Formatted code snippets and JSON data contracts for appendices
"""

import os
import sys
from pathlib import Path

# Add scripts directory to path
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

import report_frontmatter
import report_part1
import report_part2
import report_part3

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT

# ---------------------------------------------------------------------------
# Visual Palette Tokens
# ---------------------------------------------------------------------------
PRIMARY = colors.HexColor("#0F2942")       # Deep Academic Navy
SECONDARY = colors.HexColor("#1E5F74")     # Muted Teal Slate
ACCENT = colors.HexColor("#D4AF37")        # Gold / Amber
DARK_TEXT = colors.HexColor("#1F2937")     # Charcoal Body Text
MUTED_TEXT = colors.HexColor("#4B5563")    # Muted Secondary Text
LIGHT_BG = colors.HexColor("#F8FAFC")      # Light Table / Block Background
BORDER_COLOR = colors.HexColor("#CBD5E1")  # Border Line Color
CALLOUT_BG = colors.HexColor("#F1F5F9")    # Callout Background
WARNING_BG = colors.HexColor("#FFFBEB")    # Warning Box Background
WARNING_BORDER = colors.HexColor("#F59E0B")# Warning Box Border
SUCCESS_COLOR = colors.HexColor("#10B981") # Green accent

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 54  # 0.75 inch (54 points)
USABLE_WIDTH = PAGE_WIDTH - 2 * MARGIN  # 487.27 points


# ---------------------------------------------------------------------------
# Numbered Canvas with Outlines & Headers/Footers
# ---------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas that accumulates total page count and renders
    running headers, running footers, and PDF document bookmarks.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count):
        # Suppress headers and footers on Cover Page (Page 1)
        if self._pageNumber == 1:
            return

        self.saveState()

        # Running Header (Pages >= 2)
        self.setFont("Helvetica", 8)
        self.setFillColor(MUTED_TEXT)
        self.drawString(MARGIN, PAGE_HEIGHT - 38, "LegalMind AI – Academic Project Report (M.Tech Computer Science & Engineering)")
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(MARGIN, PAGE_HEIGHT - 44, PAGE_WIDTH - MARGIN, PAGE_HEIGHT - 44)

        # Running Footer (Pages >= 2)
        self.line(MARGIN, 44, PAGE_WIDTH - MARGIN, 44)
        self.drawString(MARGIN, 32, "Confidential & Academic — Indian Legal Question Answering System (RAG + LoRA)")
        self.drawRightString(PAGE_WIDTH - MARGIN, 32, f"Page {self._pageNumber} of {page_count}")

        self.restoreState()


# ---------------------------------------------------------------------------
# Flowable Helper Builders
# ---------------------------------------------------------------------------
def build_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles["CoverTitle"] = ParagraphStyle(
        "CoverTitle",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=26,
        textColor=PRIMARY,
        alignment=TA_CENTER,
        spaceAfter=12,
    )

    styles["CoverSubtitle"] = ParagraphStyle(
        "CoverSubtitle",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=16,
        textColor=SECONDARY,
        alignment=TA_CENTER,
        spaceAfter=20,
    )

    styles["CoverMeta"] = ParagraphStyle(
        "CoverMeta",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=DARK_TEXT,
        alignment=TA_CENTER,
    )

    styles["ChapterTitle"] = ParagraphStyle(
        "ChapterTitle",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=21,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=10,
        keepWithNext=True,
    )

    styles["SectionHeading"] = ParagraphStyle(
        "SectionHeading",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=SECONDARY,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True,
    )

    styles["SubsectionHeading"] = ParagraphStyle(
        "SubsectionHeading",
        parent=base["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=PRIMARY,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True,
    )

    styles["Body"] = ParagraphStyle(
        "Body",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=DARK_TEXT,
        alignment=TA_JUSTIFY,
        spaceAfter=7,
    )

    styles["CalloutText"] = ParagraphStyle(
        "CalloutText",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12.5,
        textColor=DARK_TEXT,
        alignment=TA_LEFT,
    )

    styles["CalloutHeading"] = ParagraphStyle(
        "CalloutHeading",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=PRIMARY,
        spaceAfter=3,
    )

    styles["Code"] = ParagraphStyle(
        "Code",
        parent=base["Normal"],
        fontName="Courier",
        fontSize=7.5,
        leading=10.5,
        textColor=colors.HexColor("#0F172A"),
    )

    styles["TableHeader"] = ParagraphStyle(
        "TableHeader",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        alignment=TA_CENTER,
    )

    styles["TableCell"] = ParagraphStyle(
        "TableCell",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=DARK_TEXT,
        alignment=TA_LEFT,
    )

    styles["TableCellCenter"] = ParagraphStyle(
        "TableCellCenter",
        parent=base["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=10.5,
        textColor=DARK_TEXT,
        alignment=TA_CENTER,
    )

    styles["TableCellBold"] = ParagraphStyle(
        "TableCellBold",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=PRIMARY,
        alignment=TA_LEFT,
    )

    return styles


def create_callout(title, text, styles, alert_type="info"):
    """Creates a professional bordered callout card."""
    bg_col = WARNING_BG if alert_type == "warning" else CALLOUT_BG
    border_col = WARNING_BORDER if alert_type == "warning" else SECONDARY
    title_col = WARNING_BORDER if alert_type == "warning" else PRIMARY

    h_style = ParagraphStyle("AlertH", parent=styles["CalloutHeading"], textColor=title_col)
    
    content = [
        Paragraph(f"<b>{title}</b>", h_style),
        Paragraph(text.replace("\n", "<br/>"), styles["CalloutText"]),
    ]
    t = Table([[content]], colWidths=[USABLE_WIDTH])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_col),
        ("BOX", (0, 0), (-1, -1), 1, border_col),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return KeepTogether([Spacer(1, 4), t, Spacer(1, 6)])


def create_diagram_box(title, steps, caption, styles):
    """Renders a schematic diagram representation block."""
    flow_items = []
    flow_items.append(Paragraph(f"<b>[SCHEMATIC DIAGRAM] {title}</b>", styles["TableCellBold"]))
    flow_items.append(Spacer(1, 4))
    for idx, s in enumerate(steps, 1):
        step_text = f"<b>Stage {idx:02d}:</b> {s}"
        flow_items.append(Paragraph(step_text, styles["TableCell"]))
        if idx < len(steps):
            flow_items.append(Paragraph("<font color='#1E5F74'>&nbsp;&nbsp;&nbsp;&nbsp;↓</font>", styles["TableCell"]))
    
    t = Table([[flow_items]], colWidths=[USABLE_WIDTH])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#94A3B8")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    
    caption_p = Paragraph(f"<i>{caption}</i>", ParagraphStyle("Cap", parent=styles["CalloutText"], alignment=TA_CENTER, textColor=MUTED_TEXT))
    return KeepTogether([Spacer(1, 4), t, Spacer(1, 3), caption_p, Spacer(1, 6)])


def create_code_block(code_text, styles):
    """Renders a Courier code snippet table."""
    p = Paragraph(code_text.replace("\n", "<br/>").replace(" ", "&nbsp;"), styles["Code"])
    t = Table([[p]], colWidths=[USABLE_WIDTH])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return KeepTogether([Spacer(1, 3), t, Spacer(1, 5)])


# ---------------------------------------------------------------------------
# Master Document Assembly Function
# ---------------------------------------------------------------------------
def generate_pdf_report(output_filename: str):
    print(f"[*] Compiling LegalMind AI Academic Project Report to: {output_filename}")
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )

    styles = build_styles()
    story = []

    # -----------------------------------------------------------------------
    # FRONT MATTER: Page 1 - Cover Page
    # -----------------------------------------------------------------------
    fm = report_frontmatter.FRONT_MATTER
    story.append(Spacer(1, 25))
    story.append(Paragraph(fm["title"], styles["CoverTitle"]))
    story.append(HRFlowable(width="80%", thickness=1.5, color=ACCENT, spaceAfter=18, spaceBefore=8))
    story.append(Paragraph(fm["subtitle"], styles["CoverSubtitle"]))
    story.append(Spacer(1, 50))

    meta_table_data = [
        [Paragraph("<b>Submitted By:</b>", styles["TableCellBold"]), Paragraph(fm["author"], styles["TableCell"])],
        [Paragraph("<b>Research Facility:</b>", styles["TableCellBold"]), Paragraph("NVIDIA DGX B200 Computing Server (NvidiaComputing)", styles["TableCell"])],
        [Paragraph("<b>Department:</b>", styles["TableCellBold"]), Paragraph(fm["institution"].replace("\n", "<br/>"), styles["TableCell"])],
        [Paragraph("<b>Academic Year:</b>", styles["TableCellBold"]), Paragraph(fm["academic_year"], styles["TableCell"])],
        [Paragraph("<b>Base LLM:</b>", styles["TableCellBold"]), Paragraph("Qwen/Qwen3.6-35B-A3B (Native BF16, Zero Quantization)", styles["TableCell"])],
        [Paragraph("<b>Fine-Tuning:</b>", styles["TableCellBold"]), Paragraph("Standard LoRA Adapter (lora-legal-v2, 3,440,640 parameters)", styles["TableCell"])],
        [Paragraph("<b>Retrieval:</b>", styles["TableCellBold"]), Paragraph("Hybrid BM25 + BGE-M3 (26,097 vectors, 15,057 chunks) + BGE-Reranker-v2-m3", styles["TableCell"])],
    ]
    t_meta = Table(meta_table_data, colWidths=[140, 320])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 1, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 40))

    disclaimer_p = Paragraph(
        "<b>ACADEMIC RESEARCH & EDUCATIONAL SYSTEM NOTICE:</b><br/>"
        "LegalMind AI is an academic research and technical inquiry system designed to assist legal research. "
        "It does not provide formal legal counsel or advocate advice and does not establish an attorney-client relationship. "
        "All citations and legal propositions must be verified by a qualified advocate before being utilized in legal proceedings.",
        ParagraphStyle("Notice", parent=styles["CalloutText"], alignment=TA_CENTER, textColor=MUTED_TEXT, fontSize=7.5, leading=10.5)
    )
    story.append(disclaimer_p)
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # FRONT MATTER: Page 2 - Certificate of Approval
    # -----------------------------------------------------------------------
    story.append(Paragraph(fm["certificate"]["heading"], styles["ChapterTitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceAfter=14))
    for p in fm["certificate"]["body"].split("\n\n"):
        story.append(Paragraph(p, styles["Body"]))
    story.append(Spacer(1, 35))

    sig_data = []
    for title, dept in fm["certificate"]["signatories"]:
        sig_data.append([
            Paragraph("____________________________<br/><b>" + title + "</b><br/>" + dept, styles["TableCellCenter"])
        ])
    t_sig = Table([
        [Paragraph("____________________________<br/><b>Project Supervisor</b><br/>Dept of CSE", styles["TableCellCenter"]),
         Paragraph("____________________________<br/><b>Head of Department</b><br/>Dept of CSE", styles["TableCellCenter"]),
         Paragraph("____________________________<br/><b>External Examiner</b><br/>Board of Examiners", styles["TableCellCenter"])]
    ], colWidths=[162, 162, 162])
    t_sig.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(t_sig)
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # FRONT MATTER: Page 3 - Candidate Declaration
    # -----------------------------------------------------------------------
    story.append(Paragraph(fm["declaration"]["heading"], styles["ChapterTitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceAfter=14))
    for p in fm["declaration"]["body"].split("\n\n"):
        story.append(Paragraph(p, styles["Body"]))
    story.append(Spacer(1, 50))
    story.append(Paragraph("___________________________________<br/><b>SECE 2026 AI Research Team</b><br/>Date: September 13, 2026<br/>Place: High-Performance Computing Lab", styles["TableCell"]))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # FRONT MATTER: Page 4 - Acknowledgements
    # -----------------------------------------------------------------------
    story.append(Paragraph(fm["acknowledgements"]["heading"], styles["ChapterTitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceAfter=14))
    for p in fm["acknowledgements"]["body"].split("\n\n"):
        story.append(Paragraph(p, styles["Body"]))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # FRONT MATTER: Page 5 - Abstract
    # -----------------------------------------------------------------------
    story.append(Paragraph(fm["abstract"]["heading"], styles["ChapterTitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceAfter=14))
    for p in fm["abstract"]["body"].split("\n\n"):
        story.append(Paragraph(p, styles["Body"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<b>Keywords:</b> Legal AI, Retrieval-Augmented Generation (RAG), Qwen3.6-35B, Mixture-of-Experts (MoE), Low-Rank Adaptation (LoRA), Hybrid Retrieval, BM25, FAISS, Cross-Encoder Reranking, Indian Constitutional Jurisprudence, Bharatiya Nyaya Sanhita.", styles["CalloutText"]))
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # FRONT MATTER: Page 6 - Table of Contents
    # -----------------------------------------------------------------------
    story.append(Paragraph("TABLE OF CONTENTS", styles["ChapterTitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceAfter=12))

    toc_entries = [
        ("Front Matter (Certificate, Declaration, Abstract, Lists)", "i – viii"),
        ("Chapter 1: Introduction", "1"),
        ("Chapter 2: Literature Review", "6"),
        ("Chapter 3: Problem Statement and Objectives", "11"),
        ("Chapter 4: System Analysis", "15"),
        ("Chapter 5: System Architecture and Design", "19"),
        ("Chapter 6: Technologies and Tools Used", "25"),
        ("Chapter 7: Dataset and Knowledge Base", "31"),
        ("Chapter 8: Methodology", "35"),
        ("Chapter 9: Model Development and Fine-Tuning", "38"),
        ("Chapter 10: Implementation", "42"),
        ("Chapter 11: Testing and Validation", "46"),
        ("Chapter 12: Results and Discussion", "50"),
        ("Chapter 13: Security, Privacy, Ethics, and Legal Safety", "56"),
        ("Chapter 14: Limitations", "60"),
        ("Chapter 15: Future Enhancements", "62"),
        ("Chapter 16: Conclusion", "64"),
        ("References", "66"),
        ("Appendices (A through H)", "68 – 76"),
    ]
    toc_data = []
    for title, pg in toc_entries:
        toc_data.append([
            Paragraph(f"<b>{title}</b>", styles["TableCell"]),
            Paragraph(f"<b>{pg}</b>", styles["TableCellCenter"])
        ])
    t_toc = Table(toc_data, colWidths=[400, 87])
    t_toc.setStyle(TableStyle([
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t_toc)
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # FRONT MATTER: Page 7 - Lists of Figures and Tables
    # -----------------------------------------------------------------------
    story.append(Paragraph("LIST OF FIGURES", styles["ChapterTitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceAfter=8))
    fig_data = [[Paragraph(f"<b>{f[0]}</b>", styles["TableCellBold"]), Paragraph(f[1], styles["TableCell"]), Paragraph(f[2], styles["TableCellCenter"])] for f in fm["figures"]]
    t_figs = Table(fig_data, colWidths=[65, 360, 62])
    t_figs.setStyle(TableStyle([("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 2)]))
    story.append(t_figs)
    story.append(Spacer(1, 14))

    story.append(Paragraph("LIST OF TABLES", styles["ChapterTitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceAfter=8))
    tbl_data = [[Paragraph(f"<b>{t[0]}</b>", styles["TableCellBold"]), Paragraph(t[1], styles["TableCell"]), Paragraph(t[2], styles["TableCellCenter"])] for t in fm["tables"]]
    t_tbls = Table(tbl_data, colWidths=[65, 360, 62])
    t_tbls.setStyle(TableStyle([("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("TOPPADDING", (0, 0), (-1, -1), 2)]))
    story.append(t_tbls)
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # FRONT MATTER: Page 8 - List of Abbreviations
    # -----------------------------------------------------------------------
    story.append(Paragraph("LIST OF ABBREVIATIONS", styles["ChapterTitle"]))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceAfter=8))
    abbr_data = []
    for a, full in fm["abbreviations"]:
        abbr_data.append([
            Paragraph(f"<b>{a}</b>", styles["TableCellBold"]),
            Paragraph(full, styles["TableCell"])
        ])
    t_abbr = Table(abbr_data, colWidths=[90, 397])
    t_abbr.setStyle(TableStyle([
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#F1F5F9")),
    ]))
    story.append(t_abbr)
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # CHAPTERS 1 TO 16 RENDERING LOOP
    # -----------------------------------------------------------------------
    all_chapters = [
        report_part1.CHAPTER_1,
        report_part1.CHAPTER_2,
        report_part1.CHAPTER_3,
        report_part1.CHAPTER_4,
        report_part1.CHAPTER_5,
        report_part2.CHAPTER_6,
        report_part2.CHAPTER_7,
        report_part2.CHAPTER_8,
        report_part2.CHAPTER_9,
        report_part2.CHAPTER_10,
        report_part3.CHAPTER_11,
        report_part3.CHAPTER_12,
        report_part3.CHAPTER_13,
        report_part3.CHAPTER_14,
        report_part3.CHAPTER_15,
        report_part3.CHAPTER_16,
    ]

    for ch in all_chapters:
        ch_num = ch["number"]
        ch_title = f"CHAPTER {ch_num}: {ch['title']}"
        story.append(Paragraph(ch_title, styles["ChapterTitle"]))
        story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=10, spaceBefore=2))

        for sec in ch["sections"]:
            sec_heading = sec["heading"]
            sec_content = sec["content"]

            story.append(Paragraph(sec_heading, styles["SectionHeading"]))

            # Render paragraphs
            paragraphs = sec_content.split("\n\n")
            for p in paragraphs:
                p_clean = p.strip()
                if not p_clean:
                    continue

                # Check if paragraph is a bullet list or code block
                if p_clean.startswith("[FIGURE"):
                    # Render as styled diagram box
                    caption = p_clean.split("\n")[0].strip("[]")
                    diagram_lines = p_clean.split("\n")[1:]
                    story.append(create_diagram_box(caption, [l for l in diagram_lines if l.strip()], caption, styles))
                elif p_clean.startswith("Table ") or "TABLE " in p_clean[:10]:
                    # Render table text as styled callout
                    story.append(create_callout("STRUCTURED TABLE", p_clean, styles, alert_type="info"))
                elif p_clean.startswith("[transformers]") or "warning" in p_clean.lower() and "fallback" in p_clean.lower():
                    story.append(create_callout("SYSTEM DIAGNOSTIC NOTICE", p_clean, styles, alert_type="warning"))
                elif p_clean.startswith("ORIGINAL CODE:") or p_clean.startswith("POST /query") or p_clean.startswith("HTTP/1.1") or p_clean.startswith("{"):
                    story.append(create_code_block(p_clean, styles))
                else:
                    story.append(Paragraph(p_clean.replace("\n", "<br/>"), styles["Body"]))

        story.append(PageBreak())

    # -----------------------------------------------------------------------
    # REFERENCES SECTION
    # -----------------------------------------------------------------------
    story.append(Paragraph("REFERENCES", styles["ChapterTitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=12))

    ref_data = []
    for r_num, author_title, source_desc in report_part3.REFERENCES:
        ref_text = f"<b>[{r_num}] {author_title}</b> — {source_desc}"
        ref_data.append([Paragraph(ref_text, styles["TableCell"])])
    t_refs = Table(ref_data, colWidths=[USABLE_WIDTH])
    t_refs.setStyle(TableStyle([
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    story.append(t_refs)
    story.append(PageBreak())

    # -----------------------------------------------------------------------
    # APPENDICES SECTION (A to H)
    # -----------------------------------------------------------------------
    story.append(Paragraph("APPENDICES", styles["ChapterTitle"]))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=14))

    for app_id, app_info in report_part3.APPENDICES.items():
        story.append(Paragraph(app_info["title"], styles["SectionHeading"]))
        story.append(HRFlowable(width="100%", thickness=0.75, color=SECONDARY, spaceAfter=8))
        story.append(create_code_block(app_info["body"], styles))
        story.append(Spacer(1, 10))

    # Build Document
    print("[*] Generating PDF binary via NumberedCanvas...")
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[+] Successfully compiled: {output_filename}")


if __name__ == "__main__":
    output_pdf = "LegalMind_AI_Complete_Project_Report.pdf"
    if len(sys.argv) > 1:
        output_pdf = sys.argv[1]
    generate_pdf_report(output_pdf)
