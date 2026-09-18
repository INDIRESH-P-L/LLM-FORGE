import re
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.services.citation_verifier import RE_CASE_NAME, RE_CITATION

log = logging.getLogger("legalmind.api.citations")

router = APIRouter(prefix="/api/citations", tags=["Citation Validator"])

class CitationValidationRequest(BaseModel):
    text: str = Field(..., description="Raw text containing citations or a direct list of citations")

def parse_citation(citation: str):
    """Attempt to extract case name, year, and reporter from a citation string."""
    year_match = re.search(r'\b(19|20)\d{2}\b', citation)
    year = year_match.group(0) if year_match else "Unknown"
    
    # Try to find a reporter
    known_reporters = ['SCC', 'AIR', 'SCR', 'INSC', 'CriLJ', 'Supp']
    reporter = "Unknown"
    for rep in known_reporters:
        if re.search(r'\b' + rep + r'\b', citation, re.I):
            reporter = rep
            break
            
    # Guess if it's a statute or case
    is_statute = bool(re.search(r'\b(section|article|act)\b', citation, re.I))
    
    court = "Supreme Court of India" if (reporter in ['SCC', 'SCR', 'INSC'] or "sc" in citation.lower() or "supreme court" in citation.lower()) else "Unknown / High Court"
    
    return {
        "year": year,
        "reporter": reporter,
        "is_statute": is_statute,
        "court": court
    }

@router.post("/validate")
def validate_citations(req: CitationValidationRequest, request: Request):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    retriever = getattr(request.app.state, "retriever", None)
    if not retriever:
        log.warning("Retriever unavailable for citation check.")
        raise HTTPException(status_code=503, detail="Search engine unavailable.")

    log.info(f"[Citation Check] User input: {req.text!r}")

    extracted = []
    for m in RE_CITATION.finditer(req.text):
        extracted.append(m.group(0).strip())
    for m in RE_CASE_NAME.finditer(req.text):
        extracted.append(m.group(0).strip())

    loose_regex = re.compile(r'\b(?:[A-Za-z\.]+\s+)?(?:19|20)\d{2}\s+(?:\(\d+\)\s*)?\d{0,3}\s*[A-Za-z\.]+\s+\d+\b')
    for m in loose_regex.finditer(req.text):
        cand = m.group(0).strip()
        if not RE_CITATION.search(cand):
            extracted.append(cand)
            
    # also try to extract statutes
    statute_regex = re.compile(r'\b(?:Section|Article)\s+\d+(?:[a-zA-Z])?\b.*?(?=\.|\n|$)', re.I)
    for m in statute_regex.finditer(req.text):
        extracted.append(m.group(0).strip())

    if not extracted:
        for line in req.text.split('\n'):
            line = line.strip()
            if not line: continue
            if len(line.split()) < 15 and (" v. " in line.lower() or " vs " in line.lower() or re.search(r'\d{4}', line)):
                extracted.append(line)
                
    if not extracted:
        extracted = [req.text.strip()]
        
    extracted = list(dict.fromkeys(extracted))
    log.info(f"[Citation Check] Extracted citations: {extracted}")

    results = []
    
    for citation in extracted:
        citation_lower = citation.lower()
        
        # Check if it's purely generic text or invalid
        looks_valid = (" v. " in citation_lower or " vs " in citation_lower or " vs. " in citation_lower) or \
                      re.search(r'\d{4}', citation_lower) or \
                      re.search(r'\b(section|article)\b', citation_lower)
                      
        if not looks_valid and len(citation.split()) > 10:
             # Long unstructured text without citation markers
             status = "INVALID"
             explanation = "NO RECOGNIZABLE LEGAL CITATION"
             log.info(f"[Citation Check] '{citation}' -> INVALID")
             results.append({
                "exact_citation": citation,
                "status": status,
                "case_or_statute": "",
                "court": "",
                "year": "",
                "reporter_reference": "",
                "evidence": "",
                "explanation": explanation,
                "source_id": ""
             })
             continue
             
        # Fetch chunks
        chunks = retriever.retrieve(citation, top_k=3)
        log.info(f"[Citation Check] Retrieved {len(chunks)} documents for {citation!r}")
        
        parsed = parse_citation(citation)
        
        if not chunks:
            status = "NOT FOUND"
            explanation = "No matching records found in the verified legal database."
            log.info(f"[Citation Check] '{citation}' -> NOT FOUND")
            results.append({
                "exact_citation": citation,
                "status": status,
                "case_or_statute": "",
                "court": parsed["court"],
                "year": parsed["year"],
                "reporter_reference": parsed["reporter"],
                "evidence": "Insufficient verified source material to validate this citation. Please provide the exact case name, citation, statute, or official source.",
                "explanation": explanation,
                "source_id": ""
            })
            continue

        best_chunk = chunks[0]
        chunk_text = best_chunk.get("text", "")
        doc_title = best_chunk.get("document_title", best_chunk.get("title", ""))
        source_id = best_chunk.get("chunk_id", best_chunk.get("document_id", "Unknown ID"))
        score = best_chunk.get("rerank_score", best_chunk.get("rrf_score", 0.0))
        
        log.info(f"[Citation Check] Match score for '{citation}': {score}")

        # Basic text matching
        # Does the chunk text contain the reporter or year or case name?
        # Extract main words from citation
        cit_words = [w for w in re.findall(r'\b\w+\b', citation_lower) if w not in ['v', 'vs', 'in', 'the', 'of', 'and']]
        matches = sum(1 for w in cit_words if w in chunk_text.lower())
        match_ratio = matches / len(cit_words) if cit_words else 0
        
        status = "UNVERIFIED"
        explanation = "The source material does not strongly corroborate the citation."
        
        if match_ratio > 0.8 or (parsed["year"] != "Unknown" and parsed["year"] in chunk_text):
            # Check for fabricated reporters
            if parsed["reporter"] == "Unknown" and re.search(r'\d{4}', citation_lower):
                # We couldn't find SCC, AIR, etc. but it has a year. And it wasn't extracted by strict RE_CITATION.
                if not RE_CITATION.search(citation):
                    status = "INVALID"
                    explanation = "Reporter abbreviation not recognized. Likely an AI hallucination."
            
            if status != "INVALID":
                if match_ratio > 0.9:
                    status = "VERIFIED"
                    explanation = "The citation closely matches the verified text."
                elif match_ratio > 0.5:
                    status = "PARTIALLY VERIFIED"
                    explanation = "Some details match, but there are inconsistencies or missing elements in the retrieved source."
        
        log.info(f"[Citation Check] '{citation}' -> {status}")
        
        evidence = chunk_text if status in ["VERIFIED", "PARTIALLY VERIFIED"] else "Insufficient verified source material to validate this citation. Please provide the exact case name, citation, statute, or official source."
        if status == "INVALID":
            evidence = ""
            
        results.append({
            "exact_citation": citation,
            "status": status,
            "case_or_statute": doc_title or citation,
            "court": parsed["court"],
            "year": parsed["year"],
            "reporter_reference": parsed["reporter"],
            "evidence": evidence,
            "explanation": explanation,
            "source_id": source_id
        })

    return {
        "status": "success",
        "input_text": req.text,
        "extracted_citations": extracted,
        "validations": results
    }
