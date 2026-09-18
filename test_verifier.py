import sqlite3
from app.services.citation_verifier import CitationVerifier
v = CitationVerifier()
report = v.verify("Section 45 of PMLA")
for f in report.findings:
    print(f.to_dict())
