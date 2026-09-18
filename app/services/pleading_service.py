"""
app/services/pleading_service.py
================================
LegalMind AI — Voice Dictation to Court Pleading Formatter
"""
from typing import Dict, Any

class PleadingFormatterService:
    @staticmethod
    def format_pleading(raw_text: str, pleading_type: str) -> Dict[str, Any]:
        """
        Takes unstructured transcribed text and wraps it into a formal
        court pleading structure. (Mocked LLM templating for this implementation).
        """
        
        paragraphs = [p.strip() for p in raw_text.split('.') if p.strip()]
        
        formatted_paragraphs = []
        for i, p in enumerate(paragraphs):
            formatted_paragraphs.append(f"{i+1}. That {p[0].lower() + p[1:]}.")
            
        body = "\n\n".join(formatted_paragraphs)
        
        prayer = "It is, therefore, most respectfully prayed that this Hon'ble Court may be pleased to:\n"
        prayer += "a) Allow the present application in the interest of justice;\n"
        prayer += "b) Pass any other order which this Hon'ble Court may deem fit and proper."
        
        heading = "IN THE COURT OF ________________________"
        if pleading_type == "bail":
            heading = "IN THE COURT OF SESSIONS JUDGE, ________________________\nBAIL APPLICATION NO. _____ OF 2024\nUNDER SECTION 483 OF BNSS, 2023"
        elif pleading_type == "writ":
            heading = "IN THE HIGH COURT OF JUDICATURE AT ________________________\nORIGINAL JURISDICTION\nWRIT PETITION (CIVIL) NO. _____ OF 2024\nUNDER ARTICLE 226 OF THE CONSTITUTION OF INDIA"
            
        full_draft = f"{heading}\n\nIN THE MATTER OF:\n[Applicant Name] ... Applicant\n\nVERSUS\n\n[State/Respondent] ... Respondent\n\nMOST RESPECTFULLY SHOWETH:\n\n{body}\n\nPRAYER:\n{prayer}\n\n\nTHROUGH COUNSEL\n\nPLACE: \nDATE:"
        
        return {
            "pleading_type": pleading_type,
            "raw_transcription_length": len(raw_text),
            "formatted_draft": full_draft
        }
