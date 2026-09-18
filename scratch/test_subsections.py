import re

text = """## Answer
The Constitution of India is supreme.

### Preamble
The retrieved evidence does not cover this topic.

### Fundamental Rights
Guaranteed under Part III.

## Relevant Legal Provisions
- Article 21

## Relevant Judgments
- Kesavananda Bharati

## Confidence Level
MEDIUM

## Limitations
None
"""

pattern_old = r"##\s+Answer\s+(.*?)(?=##|$)"
m_old = re.search(pattern_old, text, re.DOTALL | re.IGNORECASE)
print("OLD:", repr(m_old.group(1).strip() if m_old else None))

pattern_new = r"##\s+Answer\s+(.*?)(?=\n##\s+[A-Z]|$)"
m_new = re.search(pattern_new, text, re.DOTALL | re.IGNORECASE)
print("NEW:", repr(m_new.group(1).strip() if m_new else None))
