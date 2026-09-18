#!/usr/bin/env python3
"""
scripts/build_full_markdown_report.py
====================================
Compiles all report modules (report_frontmatter, report_part1, report_part2, report_part3)
into a single, beautifully formatted Markdown file:
LegalMind_AI_Complete_Project_Report.md
"""

import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

import report_frontmatter
import report_part1
import report_part2
import report_part3

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "LegalMind_AI_Complete_Project_Report.md"

def build_markdown():
    lines = []
    
    # Title & Frontmatter
    fm = report_frontmatter.FRONT_MATTER
    lines.append(f"# {fm['title']}\n")
    lines.append(f"*{fm['subtitle']}*\n")
    lines.append(f"**Author**: {fm['author']}  ")
    lines.append(f"**Institution**: {fm['institution'].replace(chr(10), ', ')}  ")
    lines.append(f"**Academic Year**: {fm['academic_year']}\n")
    lines.append("---\n")
    
    # Certificate
    cert = fm["certificate"]
    lines.append(f"## {cert['heading']}\n")
    lines.append(f"{cert['body']}\n")
    lines.append("---\n")
    
    # Declaration
    decl = fm["declaration"]
    lines.append(f"## {decl['heading']}\n")
    lines.append(f"{decl['body']}\n")
    lines.append("---\n")
    
    # Acknowledgements
    ack = fm["acknowledgements"]
    lines.append(f"## {ack['heading']}\n")
    lines.append(f"{ack['body']}\n")
    lines.append("---\n")
    
    # Abstract
    abs_data = fm["abstract"]
    lines.append(f"## {abs_data['heading']}\n")
    lines.append(f"{abs_data['body']}\n")
    lines.append("---\n")
    
    # Abbreviations
    lines.append("## LIST OF ABBREVIATIONS\n")
    lines.append("| Abbreviation | Full Term |")
    lines.append("|---|---|")
    for abbr, full in fm["abbreviations"]:
        lines.append(f"| **{abbr}** | {full} |")
    lines.append("\n---\n")
    
    # List of Figures & Tables
    lines.append("## LIST OF FIGURES\n")
    for fig_num, fig_title, fig_page in fm["figures"]:
        lines.append(f"- **{fig_num}**: {fig_title}")
    lines.append("\n## LIST OF TABLES\n")
    for tbl_num, tbl_title, tbl_page in fm["tables"]:
        lines.append(f"- **{tbl_num}**: {tbl_title}")
    lines.append("\n---\n")
    
    # Chapters 1..5
    p1_chapters = [report_part1.CHAPTER_1, report_part1.CHAPTER_2, report_part1.CHAPTER_3, report_part1.CHAPTER_4, report_part1.CHAPTER_5]
    for ch in p1_chapters:
        lines.append(f"# CHAPTER {ch['number']}: {ch['title']}\n")
        for sec in ch["sections"]:
            lines.append(f"### {sec['heading']}\n")
            lines.append(f"{sec['content']}\n")
        lines.append("---\n")
        
    # Chapters 6..10
    p2_chapters = [report_part2.CHAPTER_6, report_part2.CHAPTER_7, report_part2.CHAPTER_8, report_part2.CHAPTER_9, report_part2.CHAPTER_10]
    for ch in p2_chapters:
        lines.append(f"# CHAPTER {ch['number']}: {ch['title']}\n")
        for sec in ch["sections"]:
            lines.append(f"### {sec['heading']}\n")
            lines.append(f"{sec['content']}\n")
        lines.append("---\n")

    # Chapters 11..16
    p3_chapters = [
        report_part3.CHAPTER_11, report_part3.CHAPTER_12, report_part3.CHAPTER_13,
        report_part3.CHAPTER_14, report_part3.CHAPTER_15, report_part3.CHAPTER_16
    ]
    for ch in p3_chapters:
        lines.append(f"# CHAPTER {ch['number']}: {ch['title']}\n")
        for sec in ch["sections"]:
            lines.append(f"### {sec['heading']}\n")
            lines.append(f"{sec['content']}\n")
        lines.append("---\n")
        
    # References
    if hasattr(report_part3, "REFERENCES"):
        lines.append("# REFERENCES\n")
        for ref in report_part3.REFERENCES:
            lines.append(f"- {ref}")
        lines.append("\n---\n")
        
    # Appendices
    if hasattr(report_part3, "APPENDICES"):
        lines.append("# APPENDICES\n")
        for app_key, app_val in report_part3.APPENDICES.items():
            title = app_val.get("title", f"APPENDIX {app_key}")
            body = app_val.get("body", "")
            lines.append(f"## {title}\n")
            lines.append(f"```\n{body}\n```\n" if "\n" in body else f"{body}\n")
            
    content = "\n".join(lines)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"[+] Successfully generated Markdown report: {OUTPUT_PATH}")

if __name__ == "__main__":
    build_markdown()
