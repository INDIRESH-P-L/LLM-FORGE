#!/usr/bin/env python3
"""
fine_tuning/datasets/build_sft_dataset.py
==========================================
LegalMind AI — SFT Dataset Builder

Creates the initial LoRA fine-tuning dataset from two sources:

1. SEED EXAMPLES (hardcoded, high-quality) — cover all task types:
   - Legal question answering
   - Case summarization
   - Issue extraction
   - Case comparison
   - Citation formatting
   - Evidence-grounded answering
   - Uncertainty / abstention
   - Legal drafting

2. RAG-DERIVED EXAMPLES (when corpus is available) — generate structured
   training examples from ingested legal documents.

Usage:
    python fine_tuning/datasets/build_sft_dataset.py --output fine_tuning/datasets/

Output:
    legal_sft_train.jsonl   (80% split)
    legal_sft_eval.jsonl    (20% split)
"""

import json
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Seed training examples — task coverage
# ---------------------------------------------------------------------------
SEED_EXAMPLES = [
    # ── Legal question answering ──────────────────────────────────────────
    {
        "instruction": "Answer the following Indian legal question using the retrieved evidence.",
        "input": "What does Article 21 of the Constitution of India guarantee?",
        "output": (
            "## Answer\n"
            "Article 21 of the Constitution of India guarantees the fundamental right to life and personal liberty. "
            "It states: 'No person shall be deprived of his life or personal liberty except according to procedure established by law.' [1]\n\n"
            "## Relevant Legal Provisions\n"
            "- Constitution of India, Article 21 — Right to Life and Personal Liberty\n\n"
            "## Relevant Judgments\n"
            "- Maneka Gandhi v. Union of India (1978) 1 SCC 248 — Expanded Article 21 to require a just, fair, and reasonable procedure\n"
            "- K.S. Puttaswamy v. Union of India (2017) 10 SCC 1 — Right to privacy is a fundamental right under Article 21\n\n"
            "## Legal Reasoning\n"
            "The Supreme Court in Maneka Gandhi held that 'procedure established by law' must not be arbitrary, fanciful, or oppressive. "
            "This transformed Article 21 from a narrow procedural guarantee into a substantive right covering dignity, privacy, livelihood, and health.\n\n"
            "## Evidence / Citations\n"
            "[1] Constitution of India — Article 21\n"
            "[2] Maneka Gandhi v. Union of India — Supreme Court — 1978 — (1978) 1 SCC 248\n\n"
            "## Confidence Level\nHIGH — Core constitutional provision, extensively interpreted by Supreme Court.\n\n"
            "## Limitations\nThis is a foundational answer. Specific applications of Article 21 depend on the factual context of the legal matter."
        ),
    },
    {
        "instruction": "Explain the conditions for anticipatory bail in India based on the retrieved legal documents.",
        "input": "What are the conditions for anticipatory bail under Section 438 CrPC?",
        "output": (
            "## Answer\n"
            "Anticipatory bail under Section 438 of the Code of Criminal Procedure (CrPC) is a direction by a Sessions Court or High Court "
            "to release a person on bail in the event of arrest. [1]\n\n"
            "## Relevant Legal Provisions\n"
            "- CrPC Section 438 — Direction for grant of bail to person apprehending arrest\n\n"
            "## Relevant Judgments\n"
            "- Sushila Aggarwal v. State (NCT of Delhi) (2020) 5 SCC 1 — Anticipatory bail need not be time-limited; can continue until trial\n"
            "- Gurbaksh Singh Sibbia v. State of Punjab (1980) 2 SCC 565 — Broad discretion of court; no exhaustive conditions\n\n"
            "## Legal Reasoning\n"
            "Courts consider: (i) nature and gravity of accusation, (ii) antecedents of the applicant, "
            "(iii) possibility of fleeing justice, (iv) whether accusation is made with intent to humiliate. "
            "The court may impose conditions such as availability for interrogation and surrender of passport. [2]\n\n"
            "## Evidence / Citations\n"
            "[1] Code of Criminal Procedure, 1973 — Section 438\n"
            "[2] Gurbaksh Singh Sibbia v. State of Punjab — Supreme Court — 1980 — (1980) 2 SCC 565\n\n"
            "## Confidence Level\nHIGH — Well-established statutory provision with clear Supreme Court guidance.\n\n"
            "## Limitations\nSection 438 has been amended by the Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023, which replaces CrPC. "
            "Please verify the applicable statute for your jurisdiction and date of offence."
        ),
    },
    {
        "instruction": "Answer the following Indian legal question using retrieved evidence. If evidence is insufficient, say so.",
        "input": "What is mens rea in Indian criminal law?",
        "output": (
            "## Answer\n"
            "Mens rea (Latin: 'guilty mind') is the mental element required for a crime — the intention, knowledge, or recklessness "
            "accompanying a prohibited act. In Indian criminal law, most offences under the Indian Penal Code (IPC) require proof of mens rea. [1]\n\n"
            "## Relevant Legal Provisions\n"
            "- Indian Penal Code, 1860 — General Exceptions (Sections 76–106) — Define absence of mens rea as a defence\n"
            "- IPC Section 300 — Murder requires specific intent (dolus specialis)\n\n"
            "## Relevant Judgments\n"
            "- State of Maharashtra v. M.H. George AIR 1965 SC 722 — Strict liability offences do not require mens rea\n\n"
            "## Legal Reasoning\n"
            "The principle actus non facit reum nisi mens sit rea ('the act does not make a person guilty unless the mind is also guilty') "
            "underlies Indian criminal law. However, regulatory and strict liability statutes (e.g., prevention of food adulteration) may dispense with mens rea.\n\n"
            "## Evidence / Citations\n"
            "[1] Indian Penal Code, 1860 — General Principles\n\n"
            "## Confidence Level\nMEDIUM — General principle is clear; specific applications vary by offence.\n\n"
            "## Limitations\nNote that the IPC has been replaced by the Bharatiya Nyaya Sanhita (BNS) 2023 for offences after its commencement. Check applicable version."
        ),
    },

    # ── Uncertainty / abstention ──────────────────────────────────────────
    {
        "instruction": "Answer the following Indian legal question using retrieved evidence. If evidence is insufficient, explicitly state so.",
        "input": "What was the exact salary of the petitioner in Sharma v. XYZ Corp (2019) Delhi HC?",
        "output": (
            "## Answer\n"
            "The retrieved evidence does not contain the case 'Sharma v. XYZ Corp (2019) Delhi HC' or any details regarding the petitioner's salary in that case. "
            "I cannot provide this information without verified evidence.\n\n"
            "## Evidence / Citations\nNo relevant sources retrieved.\n\n"
            "## Confidence Level\nLOW — Specific case details not found in the legal corpus.\n\n"
            "## Limitations\n"
            "This question requires access to the specific judgment, which is not in the current knowledge base. "
            "Please search Indian Kanoon (indiankanoon.org) or SCC Online for the full judgment text."
        ),
    },

    # ── Case summarization ─────────────────────────────────────────────────
    {
        "instruction": "Summarize the following Supreme Court judgment in clear legal language.",
        "input": (
            "Summarize: Kesavananda Bharati v. State of Kerala (1973) 4 SCC 225 — "
            "This case challenged the Kerala Land Reforms Act. The Supreme Court, in a 13-judge bench, "
            "held by 7:6 that Parliament can amend the Constitution under Article 368 but cannot "
            "destroy its 'basic structure'. The basic structure doctrine prevents constitutional amendments "
            "that would alter fundamental features such as supremacy of the Constitution, republican and "
            "democratic form of government, secularism, separation of powers, and judicial review."
        ),
        "output": (
            "## Answer\n"
            "**Kesavananda Bharati v. State of Kerala (1973) 4 SCC 225** is a landmark 13-judge Supreme Court ruling "
            "that established the **Basic Structure Doctrine**.\n\n"
            "**Key Holding**: Parliament has wide power to amend the Constitution under Article 368, but cannot amend it "
            "so as to destroy or damage its basic structure or essential features.\n\n"
            "**Basic Structure includes**: Supremacy of Constitution, Republic and Democratic polity, Secularism, "
            "Separation of powers, Judicial review, Federalism, Rule of law, Fundamental rights.\n\n"
            "**Significance**: This ruling is the cornerstone of Indian constitutional law. It limits Parliament's amending power "
            "and empowers the Supreme Court to strike down constitutional amendments that violate basic structure.\n\n"
            "## Relevant Legal Provisions\n"
            "- Constitution of India, Article 368 — Power of Parliament to amend the Constitution\n\n"
            "## Confidence Level\nHIGH — Landmark judgment, extensively documented.\n\n"
            "## Limitations\nThis is a summary. The full judgment spans over 700 pages across multiple opinions."
        ),
    },

    # ── Issue extraction ───────────────────────────────────────────────────
    {
        "instruction": "Extract the key legal issues from the following case description.",
        "input": (
            "A factory worker was dismissed without notice after being accused of theft. "
            "The employer did not conduct a domestic enquiry. The worker challenges the dismissal "
            "claiming violation of natural justice and seeks reinstatement."
        ),
        "output": (
            "## Answer\n"
            "The key legal issues in this matter are:\n\n"
            "1. **Violation of Natural Justice** — Was the worker dismissed without being given an opportunity to be heard? "
            "The principle audi alteram partem requires that no person be condemned unheard.\n\n"
            "2. **Requirement of Domestic Enquiry** — Under the Industrial Disputes Act, 1947 and standing orders, "
            "dismissal for misconduct typically requires a fair domestic enquiry.\n\n"
            "3. **Entitlement to Reinstatement** — Whether the worker is entitled to reinstatement with back wages "
            "under Section 11A of the Industrial Disputes Act.\n\n"
            "4. **Proportionality of Punishment** — Whether dismissal is proportionate to the alleged misconduct.\n\n"
            "## Relevant Legal Provisions\n"
            "- Industrial Disputes Act, 1947 — Section 11A (reinstatement), Section 25F\n"
            "- Natural Justice — Audi alteram partem, Nemo judex in causa sua\n\n"
            "## Confidence Level\nHIGH — Issues clearly identifiable from the facts.\n\n"
            "## Limitations\nThe specific applicable standing orders and service rules depend on the industry and establishment."
        ),
    },

    # ── Evidence-grounded answering ────────────────────────────────────────
    {
        "instruction": "Answer the question based strictly on the retrieved evidence below. Do not fabricate information.",
        "input": (
            "Question: What is the right to equality?\n\n"
            "Retrieved Evidence:\n"
            "[Source 1] Constitution of India — Article 14\n"
            "The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.\n\n"
            "[Source 2] Constitution of India — Article 15\n"
            "The State shall not discriminate against any citizen on grounds only of religion, race, caste, sex, place of birth or any of them."
        ),
        "output": (
            "## Answer\n"
            "The right to equality under the Constitution of India has two primary aspects:\n\n"
            "1. **Equality before law** (Article 14) — The State shall not deny to any person equality before the law "
            "or the equal protection of the laws within the territory of India. [1]\n\n"
            "2. **Prohibition of discrimination** (Article 15) — The State shall not discriminate against any citizen "
            "on grounds only of religion, race, caste, sex, or place of birth. [2]\n\n"
            "## Relevant Legal Provisions\n"
            "- Constitution of India, Article 14 — Equality before law\n"
            "- Constitution of India, Article 15 — Prohibition of discrimination\n\n"
            "## Evidence / Citations\n"
            "[1] Constitution of India — Article 14\n"
            "[2] Constitution of India — Article 15\n\n"
            "## Confidence Level\nHIGH — Answer is directly sourced from constitutional text.\n\n"
            "## Limitations\nThis answer is based on the retrieved constitutional provisions. "
            "Detailed application requires examination of relevant Supreme Court judgments."
        ),
    },

    # ── Legal drafting assistance ──────────────────────────────────────────
    {
        "instruction": "Draft a brief legal notice for the following situation.",
        "input": (
            "Situation: A landlord has not returned the security deposit of ₹50,000 to a tenant "
            "who vacated the flat 60 days ago despite multiple requests."
        ),
        "output": (
            "## Answer\n"
            "**LEGAL NOTICE**\n\n"
            "To,\n[Name of Landlord]\n[Address]\n\nDate: [Date]\n\n"
            "Subject: Demand for Return of Security Deposit — ₹50,000\n\n"
            "Sir/Madam,\n\n"
            "I, [Tenant Name], resident of [Address], hereby issue this legal notice to you.\n\n"
            "I vacated your premises at [Property Address] on [Vacation Date] after fulfilling all obligations "
            "under our tenancy agreement. As per our agreement, you were required to return the security deposit "
            "of ₹50,000 (Rupees Fifty Thousand) within a reasonable time of vacating.\n\n"
            "Despite repeated oral and written requests, you have failed and neglected to return the said amount. "
            "Your continued withholding of my money constitutes unlawful enrichment and breach of contract.\n\n"
            "You are hereby called upon to return the security deposit of ₹50,000 within 15 (fifteen) days "
            "of receipt of this notice, failing which I shall be constrained to initiate legal proceedings "
            "before the appropriate Consumer Forum/Civil Court for recovery of the amount along with interest "
            "and costs, without further notice.\n\n"
            "Issued through:\n[Advocate Name]\n[Bar Registration No.]\n\n"
            "## Confidence Level\nMEDIUM — Standard format; actual notice should be reviewed by a qualified advocate.\n\n"
            "## Limitations\nThis is a template. The applicable law (rent control, contract law) depends on the state and specific facts."
        ),
    },

    # ── Case comparison ───────────────────────────────────────────────────
    {
        "instruction": "Compare the two Supreme Court judgments described below on the issue of right to privacy.",
        "input": (
            "Case 1: Kharak Singh v. State of UP (1962) — SC held that domiciliary visits by police violated Article 19(1)(d) "
            "but did NOT recognise a general right to privacy under the Constitution.\n\n"
            "Case 2: K.S. Puttaswamy v. Union of India (2017) — 9-judge bench unanimously held that right to privacy "
            "is a fundamental right under Article 21 of the Constitution."
        ),
        "output": (
            "## Answer\n"
            "**Comparison: Kharak Singh (1962) vs. K.S. Puttaswamy (2017) on Right to Privacy**\n\n"
            "| Aspect | Kharak Singh (1962) | K.S. Puttaswamy (2017) |\n"
            "|--------|--------------------|-----------------------|\n"
            "| Year | 1962 | 2017 |\n"
            "| Bench | 6 judges | 9 judges (unanimous) |\n"
            "| Privacy right recognised? | No — majority held no general right to privacy | Yes — fundamental right under Article 21 |\n"
            "| Constitutional basis | Article 19(1)(d) — freedom of movement | Article 21 — right to life and personal liberty |\n"
            "| Scope | Narrow (physical surveillance) | Broad (informational, bodily, decisional privacy) |\n\n"
            "**Key Shift**: The Puttaswamy judgment overruled the privacy holding in Kharak Singh, establishing "
            "that privacy is intrinsic to liberty, dignity, and autonomy under Article 21.\n\n"
            "## Legal Reasoning\n"
            "The Puttaswamy court noted that Kharak Singh was decided in an era before the substantive transformation "
            "of Article 21 in Maneka Gandhi (1978). The 2017 judgment brought Indian constitutional law in line with "
            "international human rights standards.\n\n"
            "## Confidence Level\nHIGH — Both judgments are well-documented landmark cases.\n\n"
            "## Limitations\nThis comparison is limited to the privacy aspect. Both judgments address multiple other issues."
        ),
    },
]


# ---------------------------------------------------------------------------
# Build and split dataset
# ---------------------------------------------------------------------------

def build_dataset(output_dir: str, seed: int = 42) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    examples = list(SEED_EXAMPLES)  # Add seed examples
    
    # Load from master_sft_final.jsonl if available
    sft_source = Path("/home/sece2026-student12/LegalMindAI/data/fine_tuning/Indian-Legal-CPT-SFT/master_sft_final.jsonl")
    if sft_source.exists():
        print(f"Loading additional examples from {sft_source}...")
        try:
            with open(sft_source, "r", encoding="utf-8") as f:
                sft_data = [json.loads(line) for line in f if line.strip()]
            # Sample 2000 examples to keep training time reasonable
            random.seed(seed)
            sampled_sft = random.sample(sft_data, min(2000, len(sft_data)))
            # The master_sft uses ChatML format: {"messages": [{"role": "user", "content": "..."}, ...]}
            for item in sampled_sft:
                messages = item.get("messages", [])
                user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
                asst_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
                if user_msg and asst_msg:
                    examples.append({
                        "instruction": "Answer the following Indian legal question.",
                        "input": user_msg,
                        "output": asst_msg
                    })
            print(f"Added {len(sampled_sft)} RAG-derived examples.")
        except Exception as e:
            print(f"Error loading {sft_source}: {e}")

    random.seed(seed)
    random.shuffle(examples)

    split_idx = int(len(examples) * 0.8)
    train_set = examples[:split_idx] if split_idx > 0 else examples
    eval_set  = examples[split_idx:] if split_idx < len(examples) else []

    # Write train
    train_file = out / "legal_sft_train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for ex in train_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Write eval
    eval_file = out / "legal_sft_eval.jsonl"
    with open(eval_file, "w", encoding="utf-8") as f:
        for ex in eval_set:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"SFT dataset built:")
    print(f"  Total examples : {len(examples)}")
    print(f"  Train          : {len(train_set)} → {train_file}")
    print(f"  Eval           : {len(eval_set)} → {eval_file}")
    print(f"\nTask coverage:")
    task_types = {
        "Legal Q&A": 3,
        "Abstention": 1,
        "Summarization": 1,
        "Issue extraction": 1,
        "Evidence-grounded": 1,
        "Legal drafting": 1,
        "Case comparison": 1,
    }
    for task, count in task_types.items():
        print(f"  {task:<25}: {count} example(s)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LegalMind AI — SFT Dataset Builder")
    parser.add_argument("--output", "-o", default="fine_tuning/datasets/",
                        help="Output directory (default: fine_tuning/datasets/)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(args.output, args.seed)
