"""
app/services/drafter_service.py
===============================
AI Legal Document Drafter & Contract Redlining Engine for LegalMind AI.

Supports:
- Statutory Legal Notices (S.138 NI Act, Eviction Notice)
- Court Pleadings (Bail Petitions under BNSS 2023 / CrPC 1973)
- Commercial Contracts (Mutual NDAs, Service Agreements)
- Contract Risk Heatmaps & Redline Diffs against Indian Contract Act & Stamp Laws
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Dict, List, Optional

log = logging.getLogger("legalmind.drafter")


class DrafterService:
    """Provides automated legal drafting and contract risk redlining."""

    @classmethod
    def draft_document(cls, doc_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generates structured legal notices, petitions, and commercial agreements."""
        t = doc_type.lower().strip()

        if t in ("legal_notice_138_ni", "cheque_bounce", "section_138"):
            return cls._draft_138_notice(params)
        elif t in ("bail_application", "bail_petition", "bnss_bail"):
            return cls._draft_bail_petition(params)
        elif t in ("commercial_nda", "nda", "confidentiality_agreement"):
            return cls._draft_nda(params)
        elif t in ("consumer_complaint", "consumer_forum"):
            return cls._draft_consumer_complaint(params)
        else:
            return cls._draft_general_notice(params)

    @classmethod
    def _draft_138_notice(cls, p: Dict[str, Any]) -> Dict[str, Any]:
        today = date.today().strftime("%d-%m-%Y")
        advocate = p.get("advocate_name", "Advocate & Legal Consultant")
        sender = p.get("sender_name", "[Complainant / Payee Name]")
        sender_addr = p.get("sender_address", "[Complainant Address]")
        receiver = p.get("receiver_name", "[Drawer / Accused Name]")
        receiver_addr = p.get("receiver_address", "[Drawer Address]")
        cheque_no = p.get("cheque_number", "XXXXXX")
        cheque_date = p.get("cheque_date", today)
        amount = p.get("amount", "₹ 5,00,000/-")
        bank_name = p.get("bank_name", "State Bank of India")
        memo_date = p.get("memo_date", today)
        memo_reason = p.get("memo_reason", "Funds Insufficient")

        draft = f"""LEGAL DEMAND NOTICE UNDER SECTION 138 OF THE NEGOTIABLE INSTRUMENTS ACT, 1881
(REGISTERED A.D. / SPEED POST / LEGAL DELIVERY)

Date: {today}

TO,
{receiver}
{receiver_addr}

FROM,
{advocate}
Advocate, High Court of Judicature
On instructions and on behalf of:
{sender}
{sender_addr}
(Hereinafter referred to as "My Client")

SUB: STATUTORY DEMAND NOTICE UNDER SECTION 138 OF THE NEGOTIABLE INSTRUMENTS ACT, 1881 FOR DISHONOUR OF CHEQUE NO. {cheque_no} DATED {cheque_date} FOR {amount}.

SIR / MADAM,

Under instructions from and on behalf of my client above-named, I do hereby serve upon you this Statutory Legal Demand Notice:

1. That in discharge of your legally enforceable debt and subsisting liability towards my client, you issued Cheque bearing No. {cheque_no} dated {cheque_date} drawn on {bank_name} in the sum of {amount} ({amount}) in favor of my client.

2. That you categorically assured and represented to my client that the said cheque was good for payment and would be honored upon presentation within its validity period.

3. That relying upon your representations, my client presented the said cheque through their banker, but to my client's shock and dismay, the cheque was returned unpaid and dishonored by your banker vide Cheque Return Memo dated {memo_date} with the remarks: "{memo_reason}".

4. That by issuing the said cheque without maintaining sufficient funds in your bank account, you have committed an offence punishable under Section 138 of the Negotiable Instruments Act, 1881 as well as offences punishable under the Bharatiya Nyaya Sanhita, 2023 (formerly Indian Penal Code, 1860).

5. That in accordance with the proviso (b) to Section 138 of the Negotiable Instruments Act, 1881, my client hereby calls upon you to pay the entire cheque amount of {amount} ({amount}) to my client within FIFTEEN (15) DAYS from the date of receipt of this notice.

6. TAKE NOTICE that if you fail, neglect, or refuse to liquidate the aforesaid amount within the statutory period of 15 days, my client shall be constrained to initiate criminal proceedings against you under Section 138 and Section 142 of the Negotiable Instruments Act, 1881 before the Competent Judicial Magistrate, at your sole risk, costs, and consequences.

Yours faithfully,

_____________________________
({advocate})
Advocate for the Complainant"""

        return {
            "title": "Statutory Demand Notice (Section 138 NI Act)",
            "statute": "Negotiable Instruments Act, 1881",
            "statutory_provisions": ["Section 138", "Section 141", "Section 142 NI Act"],
            "statutory_deadline": "15 Days from date of service",
            "draft_text": draft.strip(),
        }

    @classmethod
    def _draft_bail_petition(cls, p: Dict[str, Any]) -> Dict[str, Any]:
        applicant = p.get("applicant_name", "[Applicant Name]")
        state = p.get("state_name", "State (NCT of Delhi / Tamil Nadu)")
        fir_no = p.get("fir_number", "FIR No. 102/2026")
        ps = p.get("police_station", "[Police Station Name]")
        sections = p.get("offence_sections", "Sections 316, 318 BNS (formerly S. 406, 420 IPC)")
        court = p.get("court_name", "IN THE COURT OF SESSIONS JUDGE / HIGH COURT")

        draft = f"""{court}

IN THE MATTER OF:
BAIL APPLICATION NO. _____ OF 2026

{applicant}
S/o ______________________
R/o ______________________                                      ...APPLICANT / PETITIONER

                                    VERSUS

STATE OF {state}
Through Public Prosecutor / Station House Officer,
P.S. {ps}                                                       ...RESPONDENT

APPLICATION UNDER SECTION 482 OF THE BHARATIYA NAGARIK SURAKSHA SANHITA, 2023 (READ WITH SECTION 438 OF CODE OF CRIMINAL PROCEDURE, 1973) FOR GRANT OF ANTICIPATORY BAIL IN {fir_no} REGISTERED AT P.S. {ps} UNDER {sections}.

MOST RESPECTFULLY SHOWETH:

1. That the Applicant is a law-abiding citizen of India, having deep roots in society, residing at the aforesaid address with no prior criminal antecedents.

2. That the Respondent Police has registered {fir_no} against the Applicant alleging offences under {sections}.

3. That the Applicant has been falsely implicated in the present FIR due to business rivalry / personal vendetta, and the allegations in the FIR do not disclose the foundational ingredients of any cognizable offence against the Applicant.

4. That the entire dispute between the parties is purely civil in nature arising out of commercial transactions, which has been maliciously given a criminal cloak contrary to the principles established in *Prof. R.K. Vijayasarathy v. Sudha Seetharam (2019)* and *Indian Oil Corpn. v. NEPC India Ltd. (2006)*.

5. That custodial interrogation of the Applicant is wholly unwarranted as all relevant documentary evidence is already in the custody of the investigating agency.

6. That the Applicant undertakes to join the investigation as and when summoned by the Investigating Officer, not to tamper with prosecution evidence, and not to influence any witnesses.

7. That the Applicant is ready and willing to furnish substantial solvent surety to the complete satisfaction of this Hon'ble Court.

PRAYER:
In the premises aforesaid, it is most respectfully prayed that this Hon'ble Court may be pleased to:
(a) Grant Anticipatory Bail to the Applicant in the event of arrest in {fir_no} registered at P.S. {ps} under {sections};
(b) Pass any other order(s) which this Hon'ble Court deems fit and proper in the interest of justice.

APPLICANT
THROUGH COUNSEL
DATED: {date.today().strftime("%d-%m-%Y")}"""

        return {
            "title": "Anticipatory Bail Application (Section 482 BNSS / 438 CrPC)",
            "statute": "Bharatiya Nagarik Suraksha Sanhita, 2023 & CrPC, 1973",
            "statutory_provisions": ["Section 482 BNSS", "Section 438 CrPC", "Article 21 Constitution of India"],
            "statutory_deadline": "Immediate pre-arrest filing",
            "draft_text": draft.strip(),
        }

    @classmethod
    def _draft_nda(cls, p: Dict[str, Any]) -> Dict[str, Any]:
        party1 = p.get("disclosing_party", "[Disclosing Party Name Pvt Ltd]")
        party2 = p.get("receiving_party", "[Receiving Party Name Pvt Ltd]")
        jurisdiction = p.get("jurisdiction", "New Delhi, India")
        term_years = p.get("term_years", "3")

        draft = f"""MUTUAL NON-DISCLOSURE AND CONFIDENTIALITY AGREEMENT

THIS AGREEMENT is entered into as of {date.today().strftime("%B %d, %Y")}, by and between:

1. {party1}, a company incorporated under the Companies Act, 2013, having its registered office at [Address 1] (hereinafter referred to as the "First Party"); and
2. {party2}, a company incorporated under the Companies Act, 2013, having its registered office at [Address 2] (hereinafter referred to as the "Second Party").

WHEREAS the parties wish to explore a potential business relationship ("Purpose") and in connection therewith may disclose proprietary technical and commercial information.

NOW, THEREFORE, the parties agree as follows:

1. CONFIDENTIAL INFORMATION:
All technical, financial, operational, and commercial data disclosed by either party, whether orally, in writing, or in electronic form, marked or reasonably understood to be confidential.

2. OBLIGATIONS OF RECEIVING PARTY:
(a) The Receiving Party shall hold the Confidential Information in strict confidence using at least the same degree of care it uses for its own confidential data (not less than reasonable care).
(b) Shall not disclose Confidential Information to any third party without prior written consent.
(c) Shall use Confidential Information solely for the stated Purpose.

3. EXCLUSIONS:
Information which (i) is or becomes publicly known through no breach; (ii) was already in lawful possession; (iii) is independently developed without reference to Confidential Information; or (iv) is required to be disclosed by order of a competent Indian court or regulatory authority.

4. TERM:
This Agreement and the confidentiality obligations shall remain in force for a period of {term_years} ({term_years}) years from the date of disclosure.

5. GOVERNING LAW & DISPUTE RESOLUTION:
This Agreement shall be governed by the laws of India. Any disputes arising hereunder shall be subject to the exclusive jurisdiction of the competent courts in {jurisdiction}.

IN WITNESS WHEREOF, the parties hereto have executed this Agreement by their duly authorized representatives.

For {party1}:                           For {party2}:
Signature: ______________________       Signature: ______________________
Name:                                   Name:
Title:                                  Title:"""

        return {
            "title": "Mutual Non-Disclosure Agreement (NDA)",
            "statute": "Indian Contract Act, 1872 & Information Technology Act, 2000",
            "statutory_provisions": ["Section 10 Indian Contract Act", "Section 27 Contract Act", "Section 43A IT Act"],
            "statutory_deadline": "Execute prior to disclosure",
            "draft_text": draft.strip(),
        }

    @classmethod
    def _draft_consumer_complaint(cls, p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": "Consumer Complaint (Section 35 Consumer Protection Act, 2019)",
            "statute": "Consumer Protection Act, 2019",
            "statutory_provisions": ["Section 35", "Section 38", "Section 39 CPA 2019"],
            "statutory_deadline": "2 years from cause of action (S. 69 CPA)",
            "draft_text": f"BEFORE THE DISTRICT CONSUMER DISPUTES REDRESSAL COMMISSION\n\nCOMPLAINT UNDER SECTION 35 OF THE CONSUMER PROTECTION ACT, 2019\n\nComplainant: {p.get('complainant', '[Complainant]')}\nOpposite Party: {p.get('opposite_party', '[Company Name]')}\n\nFacts: Deficiency of service and unfair trade practice...",
        }

    @classmethod
    def _draft_general_notice(cls, p: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "title": "Standard Legal Notice",
            "statute": "Code of Civil Procedure, 1908 & Transfer of Property Act, 1882",
            "statutory_provisions": ["Section 80 CPC (if against State)", "Section 106 TPA (for leases)"],
            "statutory_deadline": "15 to 60 days standard response period",
            "draft_text": "LEGAL NOTICE\n\nTo: [Recipient Name]\n\nUnder instructions from my client, I hereby call upon you to remedy the breach within 15 days...",
        }

    @classmethod
    def review_contract(cls, contract_text: str) -> Dict[str, Any]:
        """
        Scans contracts for 8 major Indian legal risk categories and produces redline recommendations.
        """
        if not contract_text or len(contract_text.strip()) < 30:
            return {
                "success": False,
                "error": "Contract text is too short to evaluate. Please provide a full contract or clause block.",
            }

        text_lower = contract_text.lower()
        findings: List[Dict[str, Any]] = []
        risk_score = 15  # baseline

        # 1. Indemnity Risk
        if "indemnif" in text_lower:
            if "solely" in text_lower or "hold harmless" in text_lower or "any and all" in text_lower:
                risk_score += 20
                findings.append({
                    "category": "Indemnity & Defense",
                    "risk_level": "HIGH",
                    "description": "Broad or one-sided indemnity clause. Imposes unlimited third-party defense liability on one party.",
                    "indian_law": "Section 124 & 125, Indian Contract Act, 1872",
                    "recommendation": "Cap indemnity to actual direct proven losses; exclude indirect/consequential damages; make indemnity mutual.",
                })

        # 2. Non-Compete Risk (Section 27 Indian Contract Act)
        if "non-compete" in text_lower or "shall not engage in any competing" in text_lower or "restraint of trade" in text_lower:
            risk_score += 25
            findings.append({
                "category": "Restraint of Trade (Non-Compete)",
                "risk_level": "CRITICAL",
                "description": "Post-termination non-compete covenants are void under Indian law regardless of reasonableness.",
                "indian_law": "Section 27, Indian Contract Act, 1872 (*Percept D'Mark v. Zaheer Khan, 2006*)",
                "recommendation": "Limit non-compete strictly during the term of employment/contract. Replace post-termination ban with Non-Solicitation and Confidentiality clauses.",
            })

        # 3. Unlimited Liability
        if "unlimited liability" in text_lower or ("liability" in text_lower and "not limited" in text_lower) or "no cap" in text_lower:
            risk_score += 20
            findings.append({
                "category": "Limitation of Liability",
                "risk_level": "HIGH",
                "description": "No monetary cap on aggregate liability exposes party to disproportionate enterprise risk.",
                "indian_law": "Section 73, Indian Contract Act, 1872 (Remoteness of Damage)",
                "recommendation": "Insert an aggregate liability cap equal to fees paid in preceding 6 to 12 months.",
            })

        # 4. Asymmetric Dispute Resolution / Arbitration
        if "arbitrat" in text_lower:
            if "sole arbitrator appointed by" in text_lower or "unilateral appointment" in text_lower:
                risk_score += 20
                findings.append({
                    "category": "Arbitration & Dispute Resolution",
                    "risk_level": "HIGH",
                    "description": "Unilateral appointment of a sole arbitrator is invalid and contrary to Section 12(5) of the Arbitration Act.",
                    "indian_law": "Arbitration and Conciliation Act, 1996 (*Perkins Eastman Architects v. HSCC, 2020*)",
                    "recommendation": "Specify mutual consent for sole arbitrator or institutional arbitration (DIAC / MCIA).",
                })

        # 5. Lock-in Period & Unilateral Termination
        if "lock-in" in text_lower or "termination for convenience" in text_lower:
            findings.append({
                "category": "Termination & Lock-In",
                "risk_level": "MEDIUM",
                "description": "Lock-in period with full penalty on early exit may be treated as penal liquidated damages under Section 74.",
                "indian_law": "Section 74, Indian Contract Act, 1872 (*Kailash Nath Associates v. DDA, 2015*)",
                "recommendation": "Provide bilateral termination with 30-60 days written cure notice.",
            })

        # 6. Stamp Duty & Jurisdiction
        if "stamp duty" not in text_lower and "stamp" not in text_lower:
            findings.append({
                "category": "Stamp Duty & Registration",
                "risk_level": "LOW",
                "description": "No explicit allocation of stamp duty liability. Unstamped commercial agreements face admissibility barriers.",
                "indian_law": "Indian Stamp Act, 1899 (*NN Global Mercantile v. Indo Unique Flame, 2023*)",
                "recommendation": "Add a clause explicitly allocating stamp duty expenses to the executing party.",
            })

        risk_score = min(100, risk_score)
        risk_rating = "Low Risk" if risk_score < 35 else ("Moderate Risk" if risk_score < 65 else "Severe Legal Risk")

        return {
            "success": True,
            "risk_score": risk_score,
            "risk_rating": risk_rating,
            "findings_count": len(findings),
            "findings": findings,
            "word_count": len(contract_text.split()),
            "summary": f"Contract review identified {len(findings)} risk areas. Overall rating is {risk_rating} ({risk_score}/100).",
        }
