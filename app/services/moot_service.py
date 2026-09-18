"""
app/services/moot_service.py
============================
Moot Court & Judicial Adversary Simulator ("Judge Mode") for LegalMind AI.

Provides high-fidelity judicial cross-examination simulating an Indian Supreme Court /
High Court Appellate Bench:
- Probes vulnerabilities, jurisdictional hurdles, and procedural bars in Counsel's submissions.
- Challenges propositions with contradictory binding Constitution Bench precedents and statutes.
- Dynamically adapts to 5 specialized Judicial Corams and 3 Judicial Temperaments.
- Supports representation for both Petitioner / Appellant and Respondent / State sides.
- Curates specialized Moot Problems with detailed factual matrixes and issue statements.
- Powered by Qwen3.6-35B-A3B LLM chat completion with thread-safe VRAM cache management.
- Features resilient legal heuristic fallback if the LLM is busy or unavailable.
- Tracks multi-turn arguments across Rounds 1 through 5+ with progressive judicial pressure.
- Produces an objective 4-metric Advocacy Scorecard (0-100) with actionable rebuttal strategies.
- Compiles formal certified Supreme Court Order Sheets & Minutes of Proceeding for export.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

log = logging.getLogger("legalmind.moot")


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


class MootCourtService:
    """Simulates an active judicial bench cross-examining advocate submissions."""

    # ───────────────────────────────────────────────────────────────────────────
    # Module 1: 5 Specialized Indian Appellate Coram Profiles
    # ───────────────────────────────────────────────────────────────────────────
    JUDGE_PERSONAS: Dict[str, Dict[str, Any]] = {
        "constitutional": {
            "id": "constitutional",
            "title": "Constitution Bench of the Supreme Court of India",
            "coram": "5-Judge Constitution Bench (Courtroom No. 1)",
            "presiding": "Hon'ble Presiding Judge (Senior Constitutional Jurist)",
            "emblem": "🏛️",
            "jurisdiction": "Part III Fundamental Rights, Basic Structure, Federalism & Proportionality",
            "style": "Doctrinal, rigorously applies Maneka Gandhi substantive due process, Puttaswamy 4-prong proportionality standard, and Article 14 manifest arbitrariness.",
            "greeting": "Counsel, this Court is convened to hear your submissions on constitutional validity and fundamental rights. State your primary proposition clearly and be prepared to address the four-prong proportionality standard.",
            "statutory_compendium": [
                {"name": "Article 14", "label": "Equality before Law & Non-Arbitrariness", "snippet": "State shall not deny to any person equality before the law or equal protection of laws."},
                {"name": "Article 19(1)(a)", "label": "Freedom of Speech and Expression", "snippet": "All citizens shall have right to freedom of speech and expression subject to reasonable restrictions under 19(2)."},
                {"name": "Article 21", "label": "Protection of Life & Personal Liberty", "snippet": "No person shall be deprived of life or personal liberty except according to procedure established by law."},
                {"name": "Article 32", "label": "Remedies for Enforcement of Part III", "snippet": "The right to move the Supreme Court by appropriate proceedings for enforcement of Part III rights is guaranteed."},
                {"name": "Article 131", "label": "Original Jurisdiction of Supreme Court", "snippet": "Exclusive jurisdiction of the Supreme Court in any dispute between Government of India and one or more States."},
            ],
            "landmark_authorities": [
                {"case": "Maneka Gandhi v. Union of India (1978) 1 SCC 248", "coram": "7-Judge Bench", "ratio": "Procedure established by law under Article 21 must be just, fair and reasonable; not arbitrary, fanciful or oppressive."},
                {"case": "Justice K.S. Puttaswamy (Retd.) v. Union of India (2017) 10 SCC 1", "coram": "9-Judge Bench", "ratio": "Privacy is an intrinsic part of personal liberty; state measures must satisfy the four-prong proportionality standard: Legitimate Goal, Rational Connection, Necessity, and Proportionality stricto sensu."},
                {"case": "Shayara Bano v. Union of India (2017) 9 SCC 1", "coram": "5-Judge Bench", "ratio": "Manifest arbitrariness is an independent substantive ground under Article 14 to strike down legislative and executive actions."},
                {"case": "Kesavananda Bharati v. State of Kerala (1973) 4 SCC 225", "coram": "13-Judge Bench", "ratio": "Parliament's amending power under Article 368 cannot alter or damage the basic structure of the Constitution."},
                {"case": "Association for Democratic Reforms v. Union of India (2024) INSC 113", "coram": "5-Judge Bench", "ratio": "Voter's right to information under Article 19(1)(a) is fundamental; disproportionate statutory opacity is unconstitutional."},
            ],
        },
        "criminal": {
            "id": "criminal",
            "title": "Criminal Appellate Division of the Supreme Court of India",
            "coram": "3-Judge Criminal Appellate Bench (Courtroom No. 3)",
            "presiding": "Hon'ble Justice (Criminal Jurisprudence Specialist)",
            "emblem": "⚖️",
            "jurisdiction": "Appellate Bail, BNSS 2023 / CrPC 1973, PMLA S.45 Twin Conditions, UAPA S.43D(5) & Evidence",
            "style": "Demands evidentiary threshold satisfaction, scrutinizes prima facie guilt, mens rea, statutory non-obstante clauses, and pre-trial custody justification.",
            "greeting": "Counsel, we are dealing with serious penal allegations. State how your client crosses the statutory threshold and why custodial interrogation is not necessary on the material on record.",
            "statutory_compendium": [
                {"name": "Section 45, PMLA 2002", "label": "Twin Conditions for Bail", "snippet": "Accused shall not be released on bail unless Public Prosecutor is given opportunity to oppose and Court is satisfied of reasonable grounds."},
                {"name": "BNSS 2023 Sec 480 / CrPC 437", "label": "Bail in Non-Bailable Offences", "snippet": "Discretion of court to grant bail with statutory limitations regarding offences punishable with death or imprisonment for life."},
                {"name": "BNSS 2023 Sec 482 / CrPC 438", "label": "Direction for Grant of Bail (Anticipatory)", "snippet": "High Court or Court of Session may direct release on bail in the event of arrest upon consideration of statutory factors."},
                {"name": "Section 43D(5), UAPA 1967", "label": "Modified Bail Pre-Conditions", "snippet": "No bail if court on perusal of case diary or report is of opinion that there are reasonable grounds for believing accusation is prima facie true."},
                {"name": "BSA 2023 Sec 63 / IEA Sec 65B", "label": "Admissibility of Electronic Records", "snippet": "Information contained in electronic records deemed document subject to mandatory certification conditions."},
            ],
            "landmark_authorities": [
                {"case": "Manish Sisodia v. Directorate of Enforcement (2024) INSC 595", "coram": "Division Bench", "ratio": "Prolonged incarceration without trial violates Article 21 and overrides statutory bail bars under Section 45 PMLA."},
                {"case": "Vijay Madanlal Choudhary v. Union of India (2022) SCC OnLine SC 929", "coram": "3-Judge Bench", "ratio": "Upheld constitutional validity of Section 45 PMLA twin conditions having reasonable nexus with serious money-laundering offences."},
                {"case": "Satender Kumar Antil v. CBI (2022) 10 SCC 51", "coram": "Division Bench", "ratio": "Bail is the rule and jail is the exception; arrest cannot be made mechanically merely because it is lawful to do so."},
                {"case": "State of Haryana v. Bhajan Lal (1992) Supp (1) SCC 335", "coram": "Division Bench", "ratio": "Seven categories where inherent powers under CrPC 482 / BNSS 528 can be exercised to quash an FIR to prevent abuse of process."},
                {"case": "Vernon v. State of Maharashtra (2023) 8 SCC 485", "coram": "Division Bench", "ratio": "Mere possession of literature or association without specific terrorist overt acts does not satisfy the prima facie bar under UAPA Section 43D(5)."},
            ],
        },
        "commercial": {
            "id": "commercial",
            "title": "Commercial & Arbitration Appellate Division",
            "coram": "Division Bench (Commercial Division, Courtroom No. 7)",
            "presiding": "Hon'ble Justice (Commercial Law Specialist)",
            "emblem": "💼",
            "jurisdiction": "Contractual Breach (S.73/74 ICA), Section 34/37 Arbitration Act, S.9 Injunctions & Specific Relief",
            "style": "Strict construction of party bargains, examines liquidated vs unliquidated damages, minimal arbitral interference, and statutory limitation.",
            "greeting": "Counsel, turn your attention to the text of the commercial agreement and governing arbitration clause. How do you establish legal injury under Section 73/74 of the Contract Act?",
            "statutory_compendium": [
                {"name": "Section 73, Contract Act 1872", "label": "Compensation for Loss or Damage", "snippet": "Compensation for loss or damage caused by breach of contract which naturally arose in the usual course of things."},
                {"name": "Section 74, Contract Act 1872", "label": "Compensation for Breach where Penalty Stipulated", "snippet": "Entitled, whether or not actual damage or loss is proved, to receive reasonable compensation not exceeding penalty stipulated."},
                {"name": "Section 34, Arbitration Act 1996", "label": "Application for Setting Aside Arbitral Award", "snippet": "Recourse against arbitral award may be made only on patent illegality or conflict with basic notion of morality or justice."},
                {"name": "Section 37, Arbitration Act 1996", "label": "Appeals against Orders", "snippet": "Appellate jurisdiction against orders under Section 8, 9, or 34 is strictly supervisory and does not permit re-appreciation of evidence."},
                {"name": "Section 9, Arbitration Act 1996", "label": "Interim Measures by Court", "snippet": "Party may apply to Court for preservation, interim custody, or sale of goods before or during arbitral proceedings."},
            ],
            "landmark_authorities": [
                {"case": "Kailash Nath Associates v. Delhi Development Authority (2015) 4 SCC 136", "coram": "Division Bench", "ratio": "Under Section 74, liquidated damages can only be awarded when it is a genuine pre-estimate of loss, and actual loss must be proved unless impossible to assess."},
                {"case": "ONGC Ltd. v. Saw Pipes Ltd. (2003) 5 SCC 705", "coram": "Division Bench", "ratio": "Where terms are clear and loss is difficult to quantify, genuine pre-estimate stipulated by commercial parties must be awarded."},
                {"case": "Associate Builders v. Delhi Development Authority (2015) 3 SCC 49", "coram": "Division Bench", "ratio": "Court sitting under Section 34 does not act as an appellate court and cannot re-appreciate evidence unless the award is perverse or patently illegal."},
                {"case": "Ssangyong Engineering & Construction v. NHAI (2019) 15 SCC 131", "coram": "Division Bench", "ratio": "Patent illegality ground under Section 34(2A) cannot be invoked to set aside an award merely on an erroneous application of law or reappreciation of evidence."},
                {"case": "Perkins Eastman Architects DPC v. HSCC (India) Ltd (2020) 20 SCC 760", "coram": "Single Judge", "ratio": "A person who is ineligible to act as an arbitrator under Section 12(5) is disqualified from unilaterally appointing another arbitrator."},
            ],
        },
        "regulatory": {
            "id": "regulatory",
            "title": "Public Interest, Environmental & Regulatory Bench",
            "coram": "Special Green & Regulatory Bench (Courtroom No. 5)",
            "presiding": "Hon'ble Presiding Justice (Environmental & Public Law)",
            "emblem": "🌿",
            "jurisdiction": "Environmental Jurisprudence, NGT Act, Precautionary Principle, Polluter Pays & Public Trust Doctrine",
            "style": "Enforces precautionary principle, polluter pays doctrine, intergenerational equity, and strict procedural compliance by administrative bodies.",
            "greeting": "Counsel, this Court exercises constitutional jurisdiction under Article 32/226 to protect public trust and environmental sustainability. Address how the impugned project complies with the Precautionary Principle.",
            "statutory_compendium": [
                {"name": "Article 48A & 51A(g)", "label": "Protection of Environment & Forests", "snippet": "The State shall endeavour to protect and improve the environment; duty of every citizen to protect and improve the natural environment."},
                {"name": "Section 20, NGT Act 2010", "label": "Application of Environmental Principles", "snippet": "The Tribunal shall, while passing any order or award, apply the principles of sustainable development, the precautionary principle and polluter pays principle."},
                {"name": "Section 3, Environment Protection Act 1986", "label": "Powers of Central Government", "snippet": "Central Government shall have power to take all such measures as it deems necessary or expedient for protecting environment quality."},
                {"name": "Forest Conservation Act 1980 Sec 2", "label": "Restriction on De-reservation of Forests", "snippet": "No State Government or authority shall make order directing forest land to be used for non-forest purpose without prior Central approval."},
            ],
            "landmark_authorities": [
                {"case": "Vellore Citizens' Welfare Forum v. Union of India (1996) 5 SCC 647", "coram": "3-Judge Bench", "ratio": "The Precautionary Principle and the Polluter Pays Principle are essential features of Sustainable Development and Part III of the Constitution."},
                {"case": "M.C. Mehta v. Union of India (Oleum Gas Leak) (1987) 1 SCC 395", "coram": "5-Judge Bench", "ratio": "An enterprise engaged in hazardous industry owes an absolute and non-delegable duty to community; no exceptions of Rylands v. Fletcher apply."},
                {"case": "Hanuman Laxman Aroskar v. Union of India (2019) 15 SCC 401", "coram": "Division Bench", "ratio": "Environmental Rule of Law demands complete disclosure of material ecological facts; suppression of sensitive flora/fauna voids environmental clearance."},
                {"case": "Goa Foundation v. Union of India (2014) 6 SCC 590", "coram": "3-Judge Bench", "ratio": "Natural resources are held in public trust; mining without sustainable limits and intergenerational equity violates Article 21."},
            ],
        },
        "tax_insolvency": {
            "id": "tax_insolvency",
            "title": "Special Tax, Corporate & Insolvency Appellate Bench",
            "coram": "Special Tax & Insolvency Bench (Courtroom No. 9)",
            "presiding": "Hon'ble Justice (Corporate, Tax & Insolvency Jurisprudence)",
            "emblem": "📊",
            "jurisdiction": "Insolvency and Bankruptcy Code (IBC 2016), S.14 Moratorium, S.53 Waterfall, Reassessment & CGST Appeals",
            "style": "Strict statutory timeline enforcement, commercial wisdom of Committee of Creditors, non-obstante priority over Crown debts, and jurisdictional validity of tax re-assessments.",
            "greeting": "Counsel, this Bench is convened to hear matters under corporate insolvency and revenue jurisprudence. How does your proposition reconcile with the non-obstante clause in Section 238 IBC and the commercial wisdom doctrine?",
            "statutory_compendium": [
                {"name": "Section 14, IBC 2016", "label": "Moratorium upon Insolvency Commencement", "snippet": "Prohibits institution or continuation of suits or execution of any judgment, decree or order against corporate debtor."},
                {"name": "Section 31, IBC 2016", "label": "Approval of Resolution Plan", "snippet": "Approved plan is binding on corporate debtor, employees, members, creditors, and Central or State Government tax authorities."},
                {"name": "Section 53, IBC 2016", "label": "Distribution of Assets (Waterfall Mechanism)", "snippet": "Order of priority: CIRP costs, workmen dues & secured creditors, financial debts of unsecured creditors, followed by government dues."},
                {"name": "Section 238, IBC 2016", "label": "Overriding Effect of the Code", "snippet": "Provisions of IBC have effect notwithstanding anything inconsistent therewith contained in any other law."},
                {"name": "Section 148, Income Tax Act 1961", "label": "Notice where Income has Escaped Assessment", "snippet": "Notice can be issued only upon satisfaction of mandatory pre-conditions and sanction of specified authority under Section 151."},
            ],
            "landmark_authorities": [
                {"case": "CoC of Essar Steel India Ltd. v. Satish Kumar Gupta (2020) 8 SCC 531", "coram": "3-Judge Bench", "ratio": "Commercial wisdom of Committee of Creditors is non-justiciable; adjudicating authority cannot interfere with distribution of funds between financial and operational creditors."},
                {"case": "Swiss Ribbons Pvt. Ltd. v. Union of India (2019) 4 SCC 17", "coram": "Division Bench", "ratio": "Upheld constitutional validity of IBC; primary objective is economic rehabilitation and resolution of corporate debtor rather than recovery of debt."},
                {"case": "Innoventive Industries Ltd. v. ICICI Bank (2018) 1 SCC 407", "coram": "Division Bench", "ratio": "Once an insolvency default is established under Section 7, the adjudicating authority has no discretion to refuse admission of petition."},
                {"case": "State Tax Officer v. Rainbow Papers Ltd. (2022) SCC OnLine SC 1162", "coram": "Division Bench", "ratio": "Statutory charge created under State VAT legislation treats the tax authority as a secured creditor under IBC."},
                {"case": "Paschimanchal Vidyut Vitran Nigam Ltd. v. Raman Ispat (2023) INSC 628", "coram": "Division Bench", "ratio": "Section 53 waterfall explicitly subordinates government dues to secured financial creditors; clarifies and limits Rainbow Papers."},
            ],
        },
    }

    # ───────────────────────────────────────────────────────────────────────────
    # Curated Catalog of Realistic Moot Problems
    # ───────────────────────────────────────────────────────────────────────────
    MOOT_PROBLEMS: List[Dict[str, Any]] = [
        {
            "id": "constitutional_surveillance",
            "title": "People's Union for Digital Rights v. Union of India",
            "coram_id": "constitutional",
            "bench_title": "Constitution Bench of the Supreme Court of India",
            "factual_matrix": (
                "The Union Government promulgated the National Digital Security & Facial Recognition Rules, 2025 under Section 69 of the IT Act. "
                "The Rules mandate real-time automated biometric surveillance across all railway transit hubs and airport check-ins, backed by predictive AI crime-risk profiling. "
                "The algorithm operates without prior judicial warrant, relying on executive authorization by an Additional Secretary (Home). "
                "The Petitioner challenges the Rules as violative of Articles 14, 19(1)(a), and 21, asserting lack of legislative backing and failure of the Puttaswamy proportionality test."
            ),
            "core_issues": [
                "Whether executive authorization without prior judicial warrant satisfies the third (necessity/least intrusive) prong of Puttaswamy proportionality?",
                "Whether algorithmic black-box risk scoring infringes Article 14 by introducing unguided discretion and manifest arbitrariness?",
                "Whether national security and transit crime prevention qualify as compelling state interests overriding individual digital privacy under Article 21?"
            ],
            "petitioner_focus": "Emphasize absence of parliamentary statute, lack of independent judicial warrant, algorithmic opacity, and violation of Puttaswamy & Maneka Gandhi.",
            "respondent_focus": "Argue sovereign security prerogative, statutory anchoring in Section 69 IT Act, procedural checks via Review Committee under PUCL (1997), and rational nexus to public order.",
            "suggested_opening": "May it please your Lordships, under Article 21 and the nine-judge Constitution Bench ruling in Puttaswamy, privacy is an inviolable fundamental right. The impugned executive surveillance scheme fails the four-prong proportionality standard because automated algorithmic tracking without prior judicial warrant is intrinsically disproportionate.",
        },
        {
            "id": "criminal_pmla_bail",
            "title": "Vikramaditya Sharma v. Directorate of Enforcement",
            "coram_id": "criminal",
            "bench_title": "Criminal Appellate Division of the Supreme Court of India",
            "factual_matrix": (
                "The Appellant, a former non-executive director of an infrastructure consortium, is arraigned under Sections 3 and 4 of PMLA 2002 in connection with a Rs. 1,400 Crore credit facility. "
                "The Appellant has been incarcerated for 28 months in pre-trial custody. Investigation has resulted in 4 supplementary prosecution complaints comprising 84,000 pages of digital records and 112 listed witnesses. "
                "Charges have not yet been framed by the Special Judge. The High Court rejected bail holding that Section 45 PMLA twin conditions were not satisfied and the offences constitute a grave economic threat."
            ),
            "core_issues": [
                "Whether prolonged pre-trial incarceration where trial cannot conclude in reasonable time dilutes or overrides Section 45 PMLA twin conditions under Article 21?",
                "Whether a non-executive director without financial signing authority can be imputed with mens rea under Section 3 PMLA?",
                "How to harmonize the strict statutory threshold in Vijay Madanlal Choudhary (2022) with the liberty jurisprudence in Manish Sisodia (2024)?"
            ],
            "petitioner_focus": "Rely on Manish Sisodia (2024), Satender Kumar Antil (2022), and Prem Prakash (2024) establishing that statutory bail bars cannot extinguish Article 21 constitutional right to speedy trial.",
            "respondent_focus": "Rely on Vijay Madanlal Choudhary (2022) and Tarun Kumar (2023) arguing economic offences constitute a distinct class, delay is attributable to voluminous defence summons, and risk of witness tampering.",
            "suggested_opening": "May it please this Hon'ble Court, under Article 21 and the binding declaration in Manish Sisodia (2024), statutory bail bars under Section 45 PMLA cannot supersede the constitutional right to speedy trial when the petitioner has suffered 28 months of pre-trial incarceration without even framing of charges.",
        },
        {
            "id": "commercial_liquidated_damages",
            "title": "AeroTech Infra Pvt Ltd v. National Logistics Corridor Authority",
            "coram_id": "commercial",
            "bench_title": "Commercial & Arbitration Appellate Division",
            "factual_matrix": (
                "AeroTech was awarded an EPC contract for constructing automated cargo transfer terminals. "
                "Owing to global semiconductor supply disruptions, commissioning was delayed by 14 months. "
                "The employer NLCA terminated the contract and invoked Clause 42.1, forfeiting the entire 10% Performance Bank Guarantee (Rs. 82 Crores) as liquidated damages. "
                "The Arbitral Tribunal held the forfeiture illegal under Section 74 Contract Act for lack of proof of actual financial loss. "
                "The High Court Commercial Division under Section 34 set aside the arbitral award, holding that under ONGC v. Saw Pipes, delay damages in public infrastructure are presumed without proof."
            ),
            "core_issues": [
                "Whether under Section 74 of the Contract Act, liquidated damages can be forfeited without proving actual damage when the facility was eventually completed by alternate contractors at lower costs?",
                "Whether the High Court under Section 34 impermissibly sat as a court of appeal and re-appreciated contractual terms contrary to Associate Builders and Ssangyong?",
                "Does the doctrine in Kailash Nath Associates (2015) require actual loss proof in public infrastructural agreements?"
            ],
            "petitioner_focus": "Argue strict limits of Section 34 review under Ssangyong & Associate Builders; establish that under Kailash Nath (2015), Section 74 requires proof of actual damage unless impossible to quantify.",
            "respondent_focus": "Rely on ONGC v. Saw Pipes (2003) and Construction & Design Services (2015) that public infrastructure delays cause intangible societal loss that cannot be calculated in rupee terms.",
            "suggested_opening": "May it please your Lordships, the Commercial Division gravely erred in setting aside the arbitral award under Section 34. Under Ssangyong and Associate Builders, an arbitral tribunal is the ultimate master of contractual interpretation. Furthermore, under Kailash Nath (2015), liquidated damages under Section 74 cannot be forfeited as penalty without proof of actual legal injury.",
        },
        {
            "id": "regulatory_green_clearance",
            "title": "Himalayan Ecological Foundation v. State Infrastructure Corp & MoEFCC",
            "coram_id": "regulatory",
            "bench_title": "Public Interest, Environmental & Regulatory Bench",
            "factual_matrix": (
                "The MoEFCC granted Environmental Clearance (EC) for widening a 125 km Himalayan highway passing through a fragile eco-sensitive buffer zone prone to catastrophic glacial flash floods. "
                "The Environmental Impact Assessment (EIA) report omitted baseline geological seismic fault-line data and did not conduct public hearings in affected downstream villages, claiming statutory exemption for strategic connectivity. "
                "The National Green Tribunal dismissed the challenge relying on executive reports of slope-stabilization geotextile engineering. "
                "The Appellants appeal under Section 22 NGT Act, invoking the Precautionary Principle and Environmental Rule of Law."
            ),
            "core_issues": [
                "Whether suppression of seismic fault-line vulnerability voids the Environmental Clearance ab initio under Hanuman Laxman Aroskar (2019)?",
                "How does the Precautionary Principle apply when geological experts caution that slope excavation creates irreversible landslide risks?",
                "Can strategic national connectivity completely displace statutory requirements of public consultation under the Environment Protection Act 1986?"
            ],
            "petitioner_focus": "Rely on Hanuman Laxman Aroskar (2019), Vellore Citizens (1996), and Article 21; establish that Environmental Rule of Law requires true disclosure and precautionary preventive measures.",
            "respondent_focus": "Emphasize sovereign strategic necessity, engineering mitigations approved by expert bodies, and judicial deference to specialized regulatory clearance under NGT Act.",
            "suggested_opening": "May it please this Hon'ble Bench, under Article 21 and the Precautionary Principle laid down in Vellore Citizens and Hanuman Laxman Aroskar, complete disclosure of seismic vulnerability is a non-negotiable condition precedent for environmental clearance. Omission of vital fault-line data vitiates the clearance ab initio.",
        },
        {
            "id": "tax_ibc_priority",
            "title": "State Tax Department v. Resolution Professional of Apex Steel Ltd & CoC",
            "coram_id": "tax_insolvency",
            "bench_title": "Special Tax, Corporate & Insolvency Appellate Bench",
            "factual_matrix": (
                "Apex Steel Ltd was admitted into Corporate Insolvency Resolution Process (CIRP). "
                "The State Tax Department filed a claim of Rs. 340 Crores for unpaid Value Added Tax and penalty, asserting statutory first charge on the assets under Section 48 of the State VAT Act. "
                "The Committee of Creditors approved a Resolution Plan allocating 1% to operational government dues and 65% to secured financial creditors under the Section 53 IBC waterfall. "
                "The State Tax Department challenges the plan, relying on the Supreme Court ruling in State Tax Officer v. Rainbow Papers (2022) that statutory first charges elevate tax dues to secured creditor status."
            ),
            "core_issues": [
                "Whether statutory tax dues under State revenue enactments constitute 'secured creditors' ranking pari passu with commercial banks under Section 53(1)(b) IBC?",
                "How does Section 238 IBC overriding clause interact with State legislation creating statutory first charges?",
                "Does the subsequent Supreme Court ruling in Paschimanchal Vidyut Vitran Nigam Ltd (2023) confine Rainbow Papers to its peculiar statutory wording?"
            ],
            "petitioner_focus": "If representing State Tax: Rely on Rainbow Papers (2022) that statutory first charge makes the State a secured creditor under Section 3(30) IBC. If representing Resolution Professional/CoC: Rely on Essar Steel (2020), Paschimanchal Vidyut (2023), and Section 238.",
            "respondent_focus": "Emphasize the sanctity of the Section 53 waterfall, non-obstante effect of Section 238, and that treating all tax claims as secured creditors would collapse corporate insolvency resolution in India.",
            "suggested_opening": "May it please your Lordships, under Section 238 of the Insolvency and Bankruptcy Code and the three-judge Bench ruling in Essar Steel, the Section 53 waterfall mechanism explicitly prioritizes financial creditors over Crown debts. The State Tax Department's reliance on Rainbow Papers stands clarified and confined by Paschimanchal Vidyut (2023).",
        },
    ]

    # ───────────────────────────────────────────────────────────────────────────
    # Module 2: LLM-Powered Dynamic Judicial Cross-Examination & Scoring
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def interject(
        cls,
        argument: str,
        bench_type: str = "constitutional",
        temperament: str = "inquisitive",
        round_num: int = 1,
        counsel_side: Optional[str] = "petitioner",
        case_topic: Optional[str] = None,
        factual_matrix: Optional[str] = None,
        prior_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates counsel's oral submission and returns high-fidelity judicial
        cross-examination, counter-precedent citation, and an analytical scorecard.
        Attempts LLM generation first via Qwen3.6-35B; seamlessly falls back to
        enhanced rule-based legal heuristics if model is offline or busy.
        """
        arg_clean = argument.strip()
        if not arg_clean:
            return {
                "success": False,
                "error": "Counsel must submit a legal proposition or argument for the Bench to consider.",
            }

        b_key = bench_type.lower() if bench_type else "constitutional"
        if b_key not in cls.JUDGE_PERSONAS:
            b_key = "constitutional"
        bench_info = cls.JUDGE_PERSONAS[b_key]

        temp_key = temperament.lower() if temperament else "inquisitive"
        if temp_key not in ("inquisitive", "textualist", "adversarial"):
            temp_key = "inquisitive"

        side_clean = (counsel_side or "petitioner").lower()
        if side_clean not in ("petitioner", "appellant", "respondent", "state"):
            side_clean = "petitioner"

        # Attempt LLM inference first
        llm = _get_llm_instance()
        if llm is not None and hasattr(llm, "generate_chat_completion"):
            try:
                llm_analysis = cls._generate_llm_cross_examination(
                    llm=llm,
                    argument=arg_clean,
                    bench_info=bench_info,
                    temperament=temp_key,
                    round_num=round_num,
                    counsel_side=side_clean,
                    case_topic=case_topic,
                    factual_matrix=factual_matrix,
                    history=prior_history,
                )
                if llm_analysis and llm_analysis.get("judicial_interjection"):
                    return cls._format_response(
                        bench_info=bench_info,
                        analysis=llm_analysis,
                        round_num=round_num,
                        bench_type=b_key,
                        temperament=temp_key,
                    )
            except Exception as ex:
                log.warning(f"Moot LLM generation encountered issue: {ex}. Falling back to rule engine.")

        # Fallback to enhanced rule-based heuristics
        fallback_analysis = cls._analyze_argument_heuristics(
            argument=arg_clean,
            bench_type=b_key,
            temperament=temp_key,
            round_num=round_num,
            counsel_side=side_clean,
            case_topic=case_topic,
            history=prior_history,
        )
        return cls._format_response(
            bench_info=bench_info,
            analysis=fallback_analysis,
            round_num=round_num,
            bench_type=b_key,
            temperament=temp_key,
        )

    @classmethod
    def _format_response(
        cls,
        bench_info: Dict[str, Any],
        analysis: Dict[str, Any],
        round_num: int,
        bench_type: str,
        temperament: str,
    ) -> Dict[str, Any]:
        """Ensures a consistent response contract with valid scorecard bounds."""
        scorecard = analysis.get("scorecard", {})
        overall = float(scorecard.get("overall_score", 75.0))
        overall = round(max(30.0, min(99.0, overall)), 1)
        grounding = round(max(30.0, min(99.0, float(scorecard.get("constitutional_grounding", 75.0)))))
        statutory = round(max(30.0, min(99.0, float(scorecard.get("statutory_precision", 75.0)))))
        precedent = round(max(30.0, min(99.0, float(scorecard.get("precedent_authority", 75.0)))))
        persuasion = round(max(30.0, min(99.0, float(scorecard.get("judicial_persuasion", 75.0)))))

        if overall >= 85:
            rating = "Formidable Advocacy (Bench Inclined to Issue Notice / Rule Nisi)"
        elif overall >= 74:
            rating = "Competent Submission (Survives Preliminary Cross-Examination)"
        elif overall >= 60:
            rating = "Requires Doctrinal Reinforcement (Struggling Against Counter-Authorities)"
        else:
            rating = "Vulnerable to Dismissal in Limine (Lacks Statutory Anchor)"

        normalized_scorecard = {
            "overall_score": overall,
            "constitutional_grounding": grounding,
            "statutory_precision": statutory,
            "precedent_authority": precedent,
            "judicial_persuasion": persuasion,
            "readiness_rating": scorecard.get("readiness_rating") or rating,
        }

        return {
            "success": True,
            "round": round_num,
            "bench_type": bench_type,
            "temperament": temperament,
            "bench_title": bench_info["title"],
            "bench_coram": bench_info["coram"],
            "presiding_judge": bench_info["presiding"],
            "judicial_interjection": analysis.get("judicial_interjection", "Counsel, state your binding statutory basis."),
            "counter_precedent": analysis.get("counter_precedent", "Binding Appellate Authority"),
            "counter_precedent_ratio": analysis.get("counter_precedent_ratio", "Statutory provisions must be strictly construed."),
            "rebuttal_tip": analysis.get("rebuttal_tip", "Anchor your proposition in express statutory wording and larger bench ratios."),
            "strengths": analysis.get("strengths") or ["Submitted with professional courtroom posture."],
            "vulnerabilities": analysis.get("vulnerabilities") or ["Needs reconciliation with conflicting larger bench authorities."],
            "statutory_compendium": bench_info.get("statutory_compendium", []),
            "landmark_authorities": bench_info.get("landmark_authorities", []),
            "scorecard": normalized_scorecard,
        }

    # ───────────────────────────────────────────────────────────────────────────
    # LLM Cross-Examination Prompting & Parsing
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _generate_llm_cross_examination(
        cls,
        llm: Any,
        argument: str,
        bench_info: Dict[str, Any],
        temperament: str,
        round_num: int,
        counsel_side: str,
        case_topic: Optional[str],
        factual_matrix: Optional[str],
        history: Optional[List[Dict[str, str]]],
    ) -> Optional[Dict[str, Any]]:
        """Invokes LegalMindModel with specialized appellate judicial instructions."""
        temp_instructions = {
            "inquisitive": "Adopt an inquisitive, intellectually probing tone. Test the logical boundaries, doctrinal consistency, and the four-prong proportionality of Counsel's propositions.",
            "textualist": "Adopt a strict textualist approach. Demand exact statutory wording, section numbers, non-obstante clauses, punctuation, and question broad equitable expansions beyond the bare act.",
            "adversarial": "Adopt an aggressive, adversarial cross-examination posture. Relentlessly challenge factual premises, highlight contradictions with larger bench precedents, and warn of floodgate consequences.",
        }.get(temperament, "Adopt a probing judicial tone.")

        stage_escalation = {
            1: "Round 1 (Opening Proposition & Jurisdiction): Scrutinize foundational standing, jurisdictional bar, or prima facie threshold.",
            2: "Round 2 (Statutory Text & Bare Act Interpretation): Confront Counsel with statutory exceptions, provisos, and literal wording.",
            3: "Round 3 (Binding Precedent & Distinguishment): Press Counsel with contradictory Constitution Bench or larger bench rulings.",
            4: "Round 4 (Consequences, Slippery Slope & Public Interest): Challenge Counsel on policy fallout, floodgate litigation, or administrative disruption.",
            5: "Round 5 (Final Formulation & Prayer): Demand Counsel's exact formulation of relief, limiting principle, and summary prayer.",
        }.get(min(round_num, 5), f"Round {round_num}: Deep scrutiny of Counsel's sustained arguments and final relief.")

        system_prompt = f"""You are the Presiding Judge of the {bench_info['title']} ({bench_info['coram']}).
Emblem: {bench_info['emblem']}
Jurisdiction: {bench_info['jurisdiction']}
Judicial Style: {bench_info['style']}

You are presiding over a high-stakes Supreme Court appellate hearing.
The Counsel appearing before you represents the {counsel_side.upper()}.
Hearing Stage: {stage_escalation}
Temperament Instruction: {temp_instructions}

{f"Moot Problem / Case Topic: {case_topic}" if case_topic else ""}
{f"Factual Matrix: {factual_matrix}" if factual_matrix else ""}

Your Task:
Examine Counsel's oral submission critically. Do NOT act as an AI assistant. Act strictly as the Hon'ble Supreme Court Justice on the Bench.
1. Interject with a sharp, realistic judicial question or challenge (2-4 sentences). Address Counsel formally ('Counsel, ...').
2. Cite a specific conflicting landmark Supreme Court precedent or statutory provision that creates a hurdle for Counsel's proposition.
3. Provide the binding legal ratio of that counter-precedent.
4. Provide a tactical rebuttal suggestion ('rebuttal_tip') advising Counsel how to overcome your judicial hurdle.
5. Identify 1-3 specific analytical strengths and 1-3 specific vulnerabilities in Counsel's submission.
6. Score the argument across 4 metrics (0-100): constitutional_grounding, statutory_precision, precedent_authority, judicial_persuasion, and compute overall_score.

You MUST respond ONLY with a valid, parseable JSON object matching this exact schema:
{{
  "judicial_interjection": "Counsel, ...",
  "counter_precedent": "Case Name (Year) Citation",
  "counter_precedent_ratio": "Ratio decidendi of the cited case...",
  "rebuttal_tip": "How counsel can rebut or distinguish...",
  "strengths": ["Strength 1", "Strength 2"],
  "vulnerabilities": ["Vulnerability 1", "Vulnerability 2"],
  "scorecard": {{
    "overall_score": 82.0,
    "constitutional_grounding": 85,
    "statutory_precision": 80,
    "precedent_authority": 82,
    "judicial_persuasion": 80,
    "readiness_rating": "Competent Submission"
  }}
}}"""

        user_content_parts = []
        if history:
            user_content_parts.append("PRIOR HEARING TRANSCRIPT:")
            for idx, h in enumerate(history[-3:], 1):
                user_content_parts.append(f"Round {idx} Counsel: {h.get('counsel', '')}")
                user_content_parts.append(f"Round {idx} Bench: {h.get('bench', '')}")
            user_content_parts.append("\nCURRENT ROUND SUBMISSION:")

        user_content_parts.append(f"COUNSEL'S ORAL ARGUMENT:\n\"{argument}\"")
        user_content = "\n".join(user_content_parts)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        raw_output = llm.generate_chat_completion(
            messages=messages,
            max_new_tokens=450,
            temperature=0.5,
            top_p=0.9,
        )

        return cls._parse_llm_json(raw_output)

    @classmethod
    def _parse_llm_json(cls, raw: str) -> Optional[Dict[str, Any]]:
        """Robust parser for LLM JSON output handling markdown code blocks, trailing text, or truncated closing brackets."""
        if not raw:
            return None

        # Clean markdown codeblocks
        cleaned = raw.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1]
            if "```" in cleaned:
                cleaned = cleaned.split("```", 1)[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1]
            if "```" in cleaned:
                cleaned = cleaned.split("```", 1)[0]

        cleaned = cleaned.strip()

        # 1. Try direct json.loads
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "judicial_interjection" in parsed:
                return parsed
        except Exception:
            pass

        # 2. Try searching for outermost JSON brackets { ... }
        match = re.search(r"(\{.*\})", cleaned, re.DOTALL)
        if match:
            try:
                candidate = match.group(1)
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and "judicial_interjection" in parsed:
                    return parsed
            except Exception:
                pass

        # 3. Resilient field-level extraction if JSON was truncated or has syntax flaws
        interjection_match = re.search(r'"judicial_interjection"\s*:\s*"((?:[^"\\]|\\.)*)"?', cleaned, re.DOTALL)
        if not interjection_match:
            interjection_match = re.search(r'"judicial_interjection"\s*:\s*"([^"]+)', cleaned)

        if interjection_match:
            interjection_text = interjection_match.group(1).replace(r'\"', '"').replace(r'\n', '\n').strip()
            if interjection_text:
                counter_match = re.search(r'"counter_precedent"\s*:\s*"((?:[^"\\]|\\.)*)"?', cleaned)
                ratio_match = re.search(r'"counter_precedent_ratio"\s*:\s*"((?:[^"\\]|\\.)*)"?', cleaned)
                rebuttal_match = re.search(r'"rebuttal_tip"\s*:\s*"((?:[^"\\]|\\.)*)"?', cleaned)
                score_match = re.search(r'"overall_score"\s*:\s*(\d+(?:\.\d+)?)', cleaned)

                overall_val = float(score_match.group(1)) if score_match else 80.0

                return {
                    "judicial_interjection": interjection_text,
                    "counter_precedent": counter_match.group(1) if counter_match else "Binding Appellate Precedent",
                    "counter_precedent_ratio": ratio_match.group(1) if ratio_match else "Procedure established by law must be strictly adhered to.",
                    "rebuttal_tip": rebuttal_match.group(1) if rebuttal_match else "Distinguish on factual matrix and lack of statutory anchor.",
                    "strengths": ["Clear articulation of fundamental rights.", "Addressed primary constitutional issue directly."],
                    "vulnerabilities": ["Needs reconciliation with statutory exceptions."],
                    "scorecard": {
                        "overall_score": overall_val,
                        "constitutional_grounding": overall_val + 2,
                        "statutory_precision": overall_val - 3,
                        "precedent_authority": overall_val,
                        "judicial_persuasion": overall_val - 1,
                        "readiness_rating": "Competent Submission" if overall_val >= 75 else "Requires Reinforcement"
                    }
                }

        log.warning(f"Could not parse structured JSON from LLM output: {raw[:200]}...")
        return None

    # ───────────────────────────────────────────────────────────────────────────
    # Enhanced Legal Heuristics & Rule-Based Fallback Engine
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _analyze_argument_heuristics(
        cls,
        argument: str,
        bench_type: str,
        temperament: str,
        round_num: int,
        counsel_side: str = "petitioner",
        case_topic: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Evaluates advocacy quality using rich Indian appellate legal rules."""
        text = argument.lower()
        word_count = len(argument.split())

        grounding = 70.0
        statutory = 70.0
        precedent = 65.0
        persuasion = 70.0

        strengths: List[str] = []
        vulnerabilities: List[str] = []

        # Length / Substantiveness check
        if word_count < 12:
            vulnerabilities.append("Counsel's submission is too laconic; lacks foundational statutory citation and factual grounding.")
            persuasion -= 15
            statutory -= 10
        elif word_count > 60:
            strengths.append("Thorough factual exposition with detailed legal framing.")
            persuasion += 8

        # Citation of binding authorities check
        has_citations = bool(re.search(r"\b(scc|air|insc|scale|scr|manupatra|\d{4}\b|\bv\b|\bvs\b)\b", text))
        if has_citations:
            strengths.append("Cites formal judicial precedent and case reporter citations.")
            precedent += 18
            grounding += 10
        else:
            vulnerabilities.append("Did not cite any binding Supreme Court or High Court judgment on point.")
            precedent -= 12

        # Statutory sections check
        has_statutes = bool(re.search(r"\b(article|section|sec\.|art\.|clause|sub-section|proviso|schedule)\b", text))
        if has_statutes:
            strengths.append("Anchored on specific statutory provisions and bare act wording.")
            statutory += 16
        else:
            vulnerabilities.append("Lacks explicit reference to the governing section or constitutional article.")
            statutory -= 14

        # Side awareness nuance
        is_respondent = counsel_side in ("respondent", "state")

        # Coram & Topic Specific Evaluation
        if bench_type == "criminal" or any(k in text for k in ["bail", "pmla", "twin conditions", "section 45", "custody", "bnss", "crpc", "arrest", "uapa"]):
            counter_precedent = "Vijay Madanlal Choudhary v. Union of India (2022) SCC OnLine SC 929"
            counter_precedent_ratio = "Section 45 PMLA twin conditions are constitutional and mandatory; the court must record satisfaction of prima facie innocence before granting bail in money-laundering offences."

            if not is_respondent:
                if round_num == 1:
                    interjection = (
                        "Counsel, you are appearing in a matter involving serious scheduled offences under PMLA. "
                        "The 3-Judge Bench in *Vijay Madanlal Choudhary (2022)* upheld the constitutional validity of Section 45 twin conditions. "
                        "How do you surmount the statutory bar requiring this Court to record a positive finding of innocence, when bank trail evidence is prima facie on record?"
                    )
                    rebuttal_tip = (
                        "Cite *Manish Sisodia v. Directorate of Enforcement (2024) INSC 595* and *Prem Prakash v. ED (2024)*: statutory bail bars cannot extinguish the higher constitutional guarantee "
                        "under Article 21. Prolonged pre-trial incarceration without trial commencement dilutes Section 45 twin conditions."
                    )
                else:
                    interjection = (
                        f"Counsel, in Round {round_num}, you reiterate the delay argument. But under *Tarun Kumar v. ED (2023)*, economic offences constitute a distinct class. "
                        "Where voluminous digital evidence requires meticulous appreciation, can the accused benefit from delays occasioned by multiple interlocutory filings?"
                    )
                    rebuttal_tip = (
                        "Rely on *Satender Kumar Antil (2022)* and *Vernon (2023)*: gravity of offence cannot justify indefinite incarceration. "
                        "Point out that the agency has already seized all digital forensic evidence, extinguishing tampering risks."
                    )
            else:
                interjection = (
                    "Learned Counsel for the State/ED, the accused has already spent over two years in custody without charges being framed. "
                    "How does the State justify continuing pre-trial custody in light of *Manish Sisodia (2024)* and Article 21's mandate on speedy justice?"
                )
                rebuttal_tip = (
                    "Highlight flight risk, influence over key prosecution witnesses, and that the trial delay was caused by repeated discharge and inspection applications filed by the defence."
                )
            statutory += 12
            precedent += 10

        elif bench_type == "commercial" or any(k in text for k in ["contract", "section 74", "section 73", "liquidated damages", "arbitration", "section 34", "patent illegality"]):
            counter_precedent = "Kailash Nath Associates v. Delhi Development Authority (2015) 4 SCC 136"
            counter_precedent_ratio = "Under Section 74 Contract Act, liquidated damages can only be awarded when actual loss is proved, unless the nature of the contract makes it impossible to assess loss."

            if round_num == 1:
                interjection = (
                    "Counsel, please open the landmark ruling in *Kailash Nath Associates (2015)*. "
                    "Section 74 of the Contract Act does not dispense with proof of loss; it merely stipulates the maximum ceiling of reasonable compensation. "
                    "Since your client has not adduced evidence of actual financial injury, how can the entire liquidated amount be forfeited?"
                )
                rebuttal_tip = (
                    "Distinguish *Kailash Nath* by relying on *ONGC v. Saw Pipes (2003) 5 SCC 705* and *Construction & Design Services (2015)*: in complex infrastructure projects, "
                    "delay causes intangible societal loss that is impossible to calculate in monetary terms, making the pre-estimated sum fully enforceable."
                )
            else:
                interjection = (
                    f"Counsel, moving to Round {round_num}: you are challenging an arbitral award under Section 34. "
                    "Under *Associate Builders (2015)* and *Ssangyong (2019)*, an arbitrator's contractual interpretation is final unless patently illegal. "
                    "Are you not inviting this Court to impermissibly sit as a court of appeal and substitute our own view for the arbitrator's findings?"
                )
                rebuttal_tip = (
                    "Demonstrate that the arbitrator completely ignored express negative covenants or rewrote the bargain, which constitutes patent illegality on the face of the award under *PSA SICAL (2021)*."
                )
            statutory += 14
            precedent += 12

        elif bench_type == "regulatory" or any(k in text for k in ["environment", "precautionary", "polluter pays", "ngt", "forest", "sustainable", "public trust"]):
            counter_precedent = "Vellore Citizens' Welfare Forum v. Union of India (1996) 5 SCC 647"
            counter_precedent_ratio = "The Precautionary Principle and Polluter Pays Principle are essential components of Sustainable Development and are read into Article 21 of the Constitution."

            interjection = (
                "Counsel, this Court is bound by the Precautionary Principle laid down in *Vellore Citizens* and the Environmental Rule of Law in *Hanuman Laxman Aroskar (2019)*. "
                "Where scientific certainty is lacking regarding ecological fragility, the burden shifts entirely to your client to demonstrate that the developmental activity is benign. "
                "Where in your impact assessment is that burden discharged?"
            )
            rebuttal_tip = (
                "Submit that comprehensive mitigation safeguards, compensatory afforestation, and independent expert oversight ensure sustainable development without irreversible ecological harm."
            )
            grounding += 15
            statutory += 10

        elif bench_type == "tax_insolvency" or any(k in text for k in ["ibc", "insolvency", "cirp", "moratorium", "section 14", "section 53", "waterfall", "rainbow papers", "coc"]):
            counter_precedent = "CoC of Essar Steel India Ltd. v. Satish Kumar Gupta (2020) 8 SCC 531"
            counter_precedent_ratio = "The commercial wisdom of the Committee of Creditors is non-justiciable; Section 53 waterfall strictly subordinates operational and government tax dues to secured financial lenders."

            interjection = (
                "Counsel, the 3-Judge Bench in *Essar Steel (2020)* and *Paschimanchal Vidyut (2023)* settled that Section 53's waterfall explicitly subordinates Crown and tax debts. "
                "How do you argue against the non-obstante clause in Section 238 of the IBC, which overrides any statutory first charges claimed under state revenue enactments?"
            )
            rebuttal_tip = (
                "Rely on *State Tax Officer v. Rainbow Papers (2022)* where statutory creation of a first charge was held to qualify the State as a secured creditor under Section 3(30) IBC."
            )
            statutory += 16
            precedent += 14

        else:
            # Constitutional / Article 14 / Article 21 / Default
            counter_precedent = "Justice K.S. Puttaswamy (Retd.) v. Union of India (2017) 10 SCC 1"
            counter_precedent_ratio = "State action restricting personal liberty must satisfy the four-prong proportionality test: Legitimate Goal, Rational Connection, Necessity (least intrusive means), and Proportionality stricto sensu."

            if round_num == 1:
                interjection = (
                    "Counsel, while Article 21 guarantees personal liberty, the State asserts that national security and administrative efficiency constitute compelling interests. "
                    "Where in your submissions have you established that the impugned measure fails the third prong of the *Puttaswamy* test—namely, that a less restrictive alternative was available?"
                )
                rebuttal_tip = (
                    "Emphasize that unguided executive discretion without prior independent judicial authorization is intrinsically disproportionate under *Modern Dental College (2016)*."
                )
            else:
                interjection = (
                    f"Counsel, in Round {round_num}: how do you distinguish the Constitution Bench ruling in *Ram Lubhaya Bagga (1998)* requiring judicial deference to executive policy, "
                    "from your submission claiming manifest arbitrariness under *Shayara Bano (2017)*?"
                )
                rebuttal_tip = (
                    "Rely on *Association for Democratic Reforms (2024)*: policy decisions that infringe core fundamental rights enjoy no presumption of judicial deference."
                )
            grounding += 15
            precedent += 12

        # Apply Temperament nuances
        if temperament == "textualist":
            interjection = f"[Strict Textual Construction] {interjection} Please direct this Bench to the exact sub-section and words in the bare act supporting your proposition."
            statutory += 6
        elif temperament == "adversarial":
            interjection = f"[Adversarial Interlocution] {interjection} If we endorse your submission today, wouldn't that set a disastrous precedent opening the floodgates to endless frivolous litigation?"
            persuasion -= 5

        # Normalize metrics (35 - 98)
        grounding = max(35.0, min(98.0, grounding))
        statutory = max(35.0, min(98.0, statutory))
        precedent = max(35.0, min(98.0, precedent))
        persuasion = max(35.0, min(98.0, persuasion))

        overall_score = round(
            (grounding * 0.30) + (statutory * 0.30) + (precedent * 0.25) + (persuasion * 0.15),
            1,
        )

        if not strengths:
            strengths.append("Presented with professional courtroom decorum.")
        if not vulnerabilities:
            vulnerabilities.append("Requires closer reconciliation with conflicting larger bench authorities.")

        return {
            "judicial_interjection": interjection,
            "counter_precedent": counter_precedent,
            "counter_precedent_ratio": counter_precedent_ratio,
            "rebuttal_tip": rebuttal_tip,
            "strengths": strengths,
            "vulnerabilities": vulnerabilities,
            "scorecard": {
                "overall_score": overall_score,
                "constitutional_grounding": grounding,
                "statutory_precision": statutory,
                "precedent_authority": precedent,
                "judicial_persuasion": persuasion,
            },
        }

    # ───────────────────────────────────────────────────────────────────────────
    # Module 5: Supreme Court Order Sheet & Minutes Export Generator
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def generate_session_record(
        cls,
        bench_type: str,
        temperament: str,
        rounds: List[Dict[str, Any]],
        scorecard: Dict[str, Any],
        case_title: Optional[str] = None,
        counsel_side: Optional[str] = "petitioner",
    ) -> str:
        """
        Compiles the moot court argument session into an authentic, certified
        Supreme Court Order Sheet / Minutes of Proceeding.
        """
        b_key = bench_type.lower() if bench_type else "constitutional"
        bench_info = cls.JUDGE_PERSONAS.get(b_key, cls.JUDGE_PERSONAS["constitutional"])
        now_str = datetime.now().strftime("%d %B %Y, %H:%M IST")

        title = case_title or "IN RE: SPECIAL APPELLATE REFERENCE"
        side_label = (counsel_side or "Petitioner").capitalize()

        lines = [
            "=" * 78,
            "                 IN THE SUPREME COURT OF INDIA",
            "                     APPELLATE JURISDICTION",
            "=" * 78,
            f"RECORD OF PROCEEDINGS: {title.upper()}",
            "-" * 78,
            f"CORAM:        {bench_info['coram'].upper()}",
            f"PRESIDING:    {bench_info['presiding'].upper()}",
            f"JURISDICTION: {bench_info['jurisdiction']}",
            f"TEMPERAMENT:  {temperament.upper()} MODE",
            f"APPEARANCE:   LEARNED COUNSEL FOR THE {side_label.upper()}",
            f"DATE & TIME:  {now_str}",
            "=" * 78,
            "\n[ RECORD OF ORAL SUBMISSIONS & JUDICIAL INTERLOCUTION ]\n",
        ]

        for i, r in enumerate(rounds, 1):
            counsel_sub = r.get("counsel", "No oral submission recorded.")
            bench_interject = r.get("bench", "Bench observed without question.")
            lines.append(f"─── ROUND {i} ─────────────────────────────────────────────────────────────")
            lines.append(f"ORAL SUBMISSION BY COUNSEL FOR {side_label.upper()}:")
            lines.append(f"  \"{counsel_sub}\"\n")
            lines.append(f"JUDICIAL INTERJECTION & CROSS-EXAMINATION BY THE BENCH:")
            lines.append(f"  \"{bench_interject}\"\n")

        lines.append("-" * 78)
        lines.append("[ BENCH EVALUATION & ADVOCACY SCORECARD ]")
        lines.append(f"Overall Advocacy Rating: {scorecard.get('overall_score', 80)}/100 ({scorecard.get('readiness_rating', 'Competent')})")
        lines.append(f"• Constitutional Grounding: {scorecard.get('constitutional_grounding', 80)}%")
        lines.append(f"• Statutory Precision:      {scorecard.get('statutory_precision', 75)}%")
        lines.append(f"• Precedent Authority:      {scorecard.get('precedent_authority', 78)}%")
        lines.append(f"• Judicial Persuasion:      {scorecard.get('judicial_persuasion', 72)}%")
        lines.append("\nRECOMMENDED TACTICAL REBUTTAL STRATEGY:")
        lines.append(f"  {scorecard.get('rebuttal_tip', 'Frame arguments within strict statutory provisions.')}")
        lines.append("-" * 78)
        lines.append("[ BENCH DISPOSITION & DIRECTIONS ]")
        overall_num = float(scorecard.get("overall_score", 75))
        if overall_num >= 85:
            lines.append("ORDER: Submissions of Counsel heard at length. Issue Notice returnable in four weeks.")
            lines.append("       Interim relief granted in terms of prayer clause (b) until the next date of hearing.")
        elif overall_num >= 70:
            lines.append("ORDER: Heard learned Counsel. List the matter on next Tuesday for continuation of oral arguments.")
            lines.append("       Counsel is directed to file a concise written compendium of authorities not exceeding five pages.")
        else:
            lines.append("ORDER: Learned Counsel failed to satisfy the Court on the preliminary jurisdictional hurdle.")
            lines.append("       Counsel prays for leave to withdraw the petition with liberty to approach the High Court. Dismissed as withdrawn.")
        lines.append("=" * 78)
        lines.append("         [ OFFICIALLY CERTIFIED COPY OF SIMULATED APPELLATE RECORD ]")
        lines.append("=" * 78)

        return "\n".join(lines)
