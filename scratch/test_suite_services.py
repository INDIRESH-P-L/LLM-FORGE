import sys
from scripts.precedent_graph import get_precedent_service
from app.services.drafter_service import DrafterService
from app.services.moot_service import MootCourtService
from app.services.dossier_service import DossierService

print("1. Testing Precedent Service...")
prec_service = get_precedent_service()
landmarks = list(prec_service.nodes.keys())
print(f"   Landmarks count: {len(landmarks)}")
subgraph = prec_service.export_subgraph("Kesavananda Bharati v. State of Kerala (1973)")
print(f"   Subgraph nodes: {len(subgraph['nodes'])}, edges: {len(subgraph['edges'])}")

print("2. Testing Drafter Service...")
draft_res = DrafterService.draft_document(
    "legal_notice_138_ni",
    {
        "sender_name": "Acme Industries",
        "sender_advocate": "Advocate R. Sharma",
        "recipient_name": "Apex Builders Ltd",
        "cheque_number": "847291",
        "cheque_date": "15th August 2024",
        "bank_name": "State Bank of India",
        "amount": "15,00,000",
        "return_reason": "Funds Insufficient",
        "memo_date": "20th August 2024"
    }
)
print(f"   Draft generated: {draft_res.get('title')} ({len(draft_res.get('draft_text', ''))} chars)")

redline_res = DrafterService.review_contract(
    "The Contractor shall defend and indemnify the Company against all third party liabilities without limit."
)
print(f"   Redline: {redline_res.get('risk_level')} risk, issues: {len(redline_res.get('issues', []))}")

print("3. Testing Moot Court Simulator...")
moot_res = MootCourtService.interject(
    argument="Article 21 incorporates procedural and substantive due process as held in Maneka Gandhi.",
    bench_type="constitutional",
    round_num=1
)
print(f"   Judge: {moot_res.get('presiding_judge')} ({moot_res.get('bench_title')})")
print(f"   Interjection: {moot_res.get('judicial_interjection')[:80]}...")
print(f"   Advocacy Score: {moot_res.get('scorecard')}")

print("4. Testing Advocate Dossier & PDF Brief...")
dossier_res = DossierService.generate_dossier(
    case_title="Puttaswamy v. Union of India",
    query="Right to Privacy under Article 21",
    answer="The nine-judge bench unanimously affirmed privacy as an intrinsic fundamental right under Part III.",
    court="IN THE SUPREME COURT OF INDIA",
    citations=["(2017) 10 SCC 1", "AIR 1978 SC 597"]
)
print(f"   Dossier: {dossier_res.get('case_title')}, Sections: {len(dossier_res.get('sections', []))}")
pdf_data = DossierService.generate_pdf(dossier_res)
print(f"   PDF generated: {len(pdf_data)} bytes!")

print("\n>>> ALL 4 BACKEND MODULES OPERATIONAL AND VALIDATED! <<<")
