import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import report_frontmatter
import report_part1
import report_part2
import report_part3

total_words = 0

for k, v in report_frontmatter.FRONT_MATTER.items():
    if isinstance(v, str):
        total_words += len(v.split())
    elif isinstance(v, dict):
        for sub_k, sub_v in v.items():
            if isinstance(sub_v, str):
                total_words += len(sub_v.split())
            elif isinstance(sub_v, list):
                for item in sub_v:
                    total_words += len(str(item).split())
    elif isinstance(v, list):
        for item in v:
            total_words += len(str(item).split())

for ch in [report_part1.CHAPTER_1, report_part1.CHAPTER_2, report_part1.CHAPTER_3, report_part1.CHAPTER_4, report_part1.CHAPTER_5]:
    total_words += len(ch['title'].split())
    for s in ch['sections']:
        total_words += len(s['heading'].split())
        total_words += len(s['content'].split())

for ch in [report_part2.CHAPTER_6, report_part2.CHAPTER_7, report_part2.CHAPTER_8, report_part2.CHAPTER_9, report_part2.CHAPTER_10]:
    total_words += len(ch['title'].split())
    for s in ch['sections']:
        total_words += len(s['heading'].split())
        total_words += len(s['content'].split())

for ch in [report_part3.CHAPTER_11, report_part3.CHAPTER_12, report_part3.CHAPTER_13, report_part3.CHAPTER_14, report_part3.CHAPTER_15, report_part3.CHAPTER_16]:
    total_words += len(ch['title'].split())
    for s in ch['sections']:
        total_words += len(s['heading'].split())
        total_words += len(s['content'].split())

for r in report_part3.REFERENCES:
    total_words += len(str(r).split())

for k, v in report_part3.APPENDICES.items():
    total_words += len(v['title'].split())
    total_words += len(v['body'].split())

print(f"TOTAL WORD COUNT ACROSS REPORT MODULES: {total_words:,} words")
