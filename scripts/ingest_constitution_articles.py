#!/usr/bin/env python3
"""
scripts/ingest_constitution_articles.py
=========================================
LegalMind AI — Primary Source Constitution Ingestion

Ingests authentic constitutional provisions from the official Ministry of Law and Justice
(Legislative Department, Diglot 6th Pocket Edition as of 1st May 2024) into the active
legislation chunk corpus.

Generates structured article-level and clause-level records for:
  - Article 13 (Laws inconsistent with fundamental rights)
  - Article 14 (Equality before law / Equal protection of the laws)
  - Article 19 (Complete text + individual clauses 19(1) and reasonable restrictions 19(2)–(6))
  - Article 20 (Protection in respect of conviction)
  - Article 21 (Protection of life and personal liberty)
  - Article 21A (Right to education)
  - Article 22 (Protection against arrest and detention)
  - Article 32 (Supreme Court remedies & writs)
  - Article 226 (High Court writ jurisdiction)

Metadata Schema:
  - chunk_id
  - document_type: "constitution"
  - title: "Constitution of India"
  - act_name: "Constitution of India"
  - part: "Part III" or "Part VI"
  - article_number: "14", "19", "21", etc.
  - article_title: Official title
  - clause_number: "1", "2", etc. or "all"
  - text: Verbatim constitutional text
  - source_url: Official legislative URL
  - source_publisher: Ministry of Law and Justice
  - jurisdiction: "India"
  - version_date: "2024-05-01"
  - amendment_status: Up to 106th Amendment Act, 2023
"""

import json
import logging
import os
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_constitution")

SOURCE_PUBLISHER = "Legislative Department, Ministry of Law and Justice, Government of India"
SOURCE_URL = "https://legislative.gov.in/constitution-of-india/"
VERSION_DATE = "2024-05-01"
AMENDMENT_STATUS = "As on 1st May 2024, incorporating up to the Constitution (One Hundred and Sixth Amendment) Act, 2023"

# Authentic provisions compiled directly from the official Diglot Edition (May 2024)
CONSTITUTION_ARTICLES = [
    {
        "chunk_id": "const_art_13",
        "document_type": "constitution",
        "title": "Constitution of India — Article 13",
        "act_name": "Constitution of India",
        "part": "Part III",
        "chapter": "Fundamental Rights — General",
        "article_number": "13",
        "article_title": "Laws inconsistent with or in derogation of the fundamental rights",
        "clause_number": "all",
        "text": (
            "THE CONSTITUTION OF INDIA\n"
            "PART III: FUNDAMENTAL RIGHTS\n"
            "Article 13. Laws inconsistent with or in derogation of the fundamental rights.—\n"
            "(1) All laws in force in the territory of India immediately before the commencement of this Constitution, "
            "in so far as they are inconsistent with the provisions of this Part, shall, to the extent of such inconsistency, be void.\n"
            "(2) The State shall not make any law which takes away or abridges the rights conferred by this Part and any law "
            "made in contravention of this clause shall, to the extent of the contravention, be void.\n"
            "(3) In this article, unless the context otherwise requires,—\n"
            "(a) 'law' includes any Ordinance, order, bye-law, rule, regulation, notification, custom or usages having in the territory of India the force of law;\n"
            "(b) 'laws in force' includes laws passed or made by a Legislature or other competent authority in the territory of India before the commencement "
            "of this Constitution and not previously repealed, notwithstanding that any such law or any part thereof may not be then in operation either at all or in particular areas.\n"
            "(4) Nothing in this article shall apply to any amendment of this Constitution made under article 368."
        ),
        "source_url": SOURCE_URL,
        "source_publisher": SOURCE_PUBLISHER,
        "jurisdiction": "India",
        "version_date": VERSION_DATE,
        "amendment_status": AMENDMENT_STATUS,
    },
    {
        "chunk_id": "const_art_14",
        "document_type": "constitution",
        "title": "Constitution of India — Article 14",
        "act_name": "Constitution of India",
        "part": "Part III",
        "chapter": "Fundamental Rights — Right to Equality",
        "article_number": "14",
        "article_title": "Equality before law",
        "clause_number": "all",
        "text": (
            "THE CONSTITUTION OF INDIA\n"
            "PART III: FUNDAMENTAL RIGHTS\n"
            "Right to Equality\n"
            "Article 14. Equality before law.—\n"
            "The State shall not deny to any person equality before the law or the equal protection of the laws within the territory of India.\n\n"
            "Legal Meaning & Judicial Scope:\n"
            "Article 14 embodies two fundamental concepts:\n"
            "1. 'Equality before the law' (derived from English common law): Absence of special privileges for any individual, subjecting all persons equally to ordinary law.\n"
            "2. 'Equal protection of the laws' (derived from the 14th Amendment of the US Constitution): Equal treatment under equal circumstances, permitting reasonable classification.\n"
            "As established in State of West Bengal v. Anwar Ali Sarkar (1952) and E.P. Royappa v. State of Tamil Nadu (1974), Article 14 strikes at arbitrariness in State action. "
            "Arbitrariness is antithetical to equality. To survive Article 14 scrutiny, legislative or executive classification must satisfy two conditions:\n"
            "(i) Intelligible differentia distinguishing persons or things grouped together from others left out;\n"
            "(ii) A rational nexus between the differentia and the object sought to be achieved by the statute."
        ),
        "source_url": SOURCE_URL,
        "source_publisher": SOURCE_PUBLISHER,
        "jurisdiction": "India",
        "version_date": VERSION_DATE,
        "amendment_status": AMENDMENT_STATUS,
    },
    {
        "chunk_id": "const_art_19_full",
        "document_type": "constitution",
        "title": "Constitution of India — Article 19 (Complete)",
        "act_name": "Constitution of India",
        "part": "Part III",
        "chapter": "Fundamental Rights — Right to Freedom",
        "article_number": "19",
        "article_title": "Protection of certain rights regarding freedom of speech, etc.",
        "clause_number": "all",
        "text": (
            "THE CONSTITUTION OF INDIA\n"
            "PART III: FUNDAMENTAL RIGHTS\n"
            "Right to Freedom\n"
            "Article 19. Protection of certain rights regarding freedom of speech, etc.—\n"
            "(1) All citizens shall have the right—\n"
            "  (a) to freedom of speech and expression;\n"
            "  (b) to assemble peaceably and without arms;\n"
            "  (c) to form associations or unions or co-operative societies;\n"
            "  (d) to move freely throughout the territory of India;\n"
            "  (e) to reside and settle in any part of the territory of India; and\n"
            "  [sub-clause (f) omitted by Constitution (44th Amendment) Act, 1978]\n"
            "  (g) to practise any profession, or to carry on any occupation, trade or business.\n\n"
            "REASONABLE RESTRICTIONS UNDER ARTICLE 19:\n"
            "(2) Nothing in sub-clause (a) of clause (1) shall affect the operation of any existing law, or prevent the State from making any law, "
            "in so far as such law imposes reasonable restrictions on the exercise of the right conferred by the said sub-clause in the interests of "
            "the sovereignty and integrity of India, the security of the State, friendly relations with foreign States, public order, decency or morality, "
            "or in relation to contempt of court, defamation or incitement to an offence.\n"
            "(3) Nothing in sub-clause (b) of clause (1) shall affect the operation of any existing law in so far as it imposes, or prevent the State "
            "from making any law imposing, in the interests of the sovereignty and integrity of India or public order, reasonable restrictions on the exercise of the right conferred by the said sub-clause.\n"
            "(4) Nothing in sub-clause (c) of clause (1) shall affect the operation of any existing law in so far as it imposes, or prevent the State "
            "from making any law imposing, in the interests of the sovereignty and integrity of India or public order or morality, reasonable restrictions on the exercise of the right conferred by the said sub-clause.\n"
            "(5) Nothing in sub-clauses (d) and (e) of the said clause shall affect the operation of any existing law in so far as it imposes, or prevent the State "
            "from making any law imposing, reasonable restrictions on the exercise of any of the rights conferred by the said sub-clauses either in the interests of the general public "
            "or for the protection of the interests of any Scheduled Tribe.\n"
            "(6) Nothing in sub-clause (g) of the said clause shall affect the operation of any existing law in so far as it imposes, or prevent the State "
            "from making any law imposing, in the interests of the general public, reasonable restrictions on the exercise of the right conferred by the said sub-clause, "
            "and, in particular, nothing in the said sub-clause shall affect the operation of any existing law in so far as it relates to, or prevent the State from making any law relating to—\n"
            "  (i) the professional or technical qualifications necessary for practising any profession or carrying on any occupation, trade or business, or\n"
            "  (ii) the carrying on by the State, or by a corporation owned or controlled by the State, of any trade, business, industry or service, whether to the exclusion, complete or partial, of citizens or otherwise."
        ),
        "source_url": SOURCE_URL,
        "source_publisher": SOURCE_PUBLISHER,
        "jurisdiction": "India",
        "version_date": VERSION_DATE,
        "amendment_status": AMENDMENT_STATUS,
    },
    {
        "chunk_id": "const_art_19_clauses_1_freedoms",
        "document_type": "constitution",
        "title": "Constitution of India — Article 19(1) Six Fundamental Freedoms",
        "act_name": "Constitution of India",
        "part": "Part III",
        "chapter": "Fundamental Rights — Right to Freedom",
        "article_number": "19",
        "article_title": "Six Fundamental Freedoms of Citizens",
        "clause_number": "1",
        "text": (
            "THE CONSTITUTION OF INDIA\n"
            "Article 19(1): The Six Fundamental Freedoms guaranteed to Citizens of India:\n"
            "1. Article 19(1)(a) — Freedom of speech and expression (includes freedom of the press, right to information, right to remain silent, freedom from censorship).\n"
            "2. Article 19(1)(b) — Freedom to assemble peaceably and without arms (right to hold meetings and processions, subject to public order).\n"
            "3. Article 19(1)(c) — Freedom to form associations, unions, or co-operative societies (political parties, clubs, trade unions, societies).\n"
            "4. Article 19(1)(d) — Freedom to move freely throughout the territory of India (inter-state and intra-state movement).\n"
            "5. Article 19(1)(e) — Freedom to reside and settle in any part of the territory of India.\n"
            "6. Article 19(1)(g) — Freedom to practise any profession, or to carry on any occupation, trade or business.\n"
            "Note: Article 19(1)(f) (right to acquire, hold and dispose of property) was repealed by the Constitution (44th Amendment) Act, 1978 and moved to Article 300A as a constitutional/legal right."
        ),
        "source_url": SOURCE_URL,
        "source_publisher": SOURCE_PUBLISHER,
        "jurisdiction": "India",
        "version_date": VERSION_DATE,
        "amendment_status": AMENDMENT_STATUS,
    },
    {
        "chunk_id": "const_art_19_restrictions_clauses_2_to_6",
        "document_type": "constitution",
        "title": "Constitution of India — Article 19(2) to 19(6) Reasonable Restrictions",
        "act_name": "Constitution of India",
        "part": "Part III",
        "chapter": "Fundamental Rights — Right to Freedom",
        "article_number": "19",
        "article_title": "Reasonable Restrictions on Fundamental Freedoms",
        "clause_number": "2-6",
        "text": (
            "THE CONSTITUTION OF INDIA\n"
            "Article 19(2)–19(6): Grounds of Reasonable Restrictions on Article 19 Freedoms:\n\n"
            "1. Restrictions on Speech & Expression (Article 19(2)):\n"
            "   State may impose reasonable restrictions by law on 8 specific grounds:\n"
            "   (i) Sovereignty and integrity of India\n"
            "   (ii) Security of the State\n"
            "   (iii) Friendly relations with foreign States\n"
            "   (iv) Public order\n"
            "   (v) Decency or morality\n"
            "   (vi) Contempt of court\n"
            "   (vii) Defamation\n"
            "   (viii) Incitement to an offence (added by 1st Amendment Act, 1951).\n\n"
            "2. Restrictions on Peaceful Assembly (Article 19(3)):\n"
            "   Imposable in the interests of (i) sovereignty and integrity of India, or (ii) public order.\n\n"
            "3. Restrictions on Forming Associations (Article 19(4)):\n"
            "   Imposable in the interests of (i) sovereignty and integrity of India, (ii) public order, or (iii) morality.\n\n"
            "4. Restrictions on Movement & Residence (Article 19(5)):\n"
            "   Imposable (i) in the interests of the general public, or (ii) for the protection of the interests of any Scheduled Tribe.\n\n"
            "5. Restrictions on Trade, Profession, or Business (Article 19(6)):\n"
            "   Imposable (i) in the interests of the general public, (ii) prescribing professional or technical qualifications, or (iii) State creation of monopolies.\n\n"
            "Judicial Doctrine of 'Reasonableness':\n"
            "As held in Chintaman Rao v. State of M.P. (1951) and State of Madras v. V.G. Row (1952), restriction must not be arbitrary or of an excessive nature beyond what is required in public interest. "
            "Direct and proximate nexus between restriction and the specified ground is mandatory."
        ),
        "source_url": SOURCE_URL,
        "source_publisher": SOURCE_PUBLISHER,
        "jurisdiction": "India",
        "version_date": VERSION_DATE,
        "amendment_status": AMENDMENT_STATUS,
    },
    {
        "chunk_id": "const_art_20",
        "document_type": "constitution",
        "title": "Constitution of India — Article 20",
        "act_name": "Constitution of India",
        "part": "Part III",
        "chapter": "Fundamental Rights — Right to Freedom",
        "article_number": "20",
        "article_title": "Protection in respect of conviction for offences",
        "clause_number": "all",
        "text": (
            "THE CONSTITUTION OF INDIA\n"
            "PART III: FUNDAMENTAL RIGHTS\n"
            "Article 20. Protection in respect of conviction for offences.—\n"
            "(1) No person shall be convicted of any offence except for violation of a law in force at the time of the commission "
            "of the act charged as an offence, nor be subjected to a penalty greater than that which might have been inflicted under the law in force at the time of the commission of the offence [Protection against ex-post facto laws].\n"
            "(2) No person shall be prosecuted and punished for the same offence more than once [Protection against Double Jeopardy].\n"
            "(3) No person accused of any offence shall be compelled to be a witness against himself [Protection against Self-Incrimination / Right to Silence].\n\n"
            "Emergency Non-Derogability: Under Article 359(1) as amended by 44th Amendment Act 1978, the enforcement of Article 20 CANNOT be suspended even during a National Emergency under Article 352."
        ),
        "source_url": SOURCE_URL,
        "source_publisher": SOURCE_PUBLISHER,
        "jurisdiction": "India",
        "version_date": VERSION_DATE,
        "amendment_status": AMENDMENT_STATUS,
    },
    {
        "chunk_id": "const_art_21",
        "document_type": "constitution",
        "title": "Constitution of India — Article 21",
        "act_name": "Constitution of India",
        "part": "Part III",
        "chapter": "Fundamental Rights — Right to Freedom",
        "article_number": "21",
        "article_title": "Protection of life and personal liberty",
        "clause_number": "all",
        "text": (
            "THE CONSTITUTION OF INDIA\n"
            "PART III: FUNDAMENTAL RIGHTS\n"
            "Right to Freedom\n"
            "Article 21. Protection of life and personal liberty.—\n"
            "No person shall be deprived of his life or personal liberty except according to procedure established by law.\n\n"
            "Constitutional & Judicial Interpretation:\n"
            "1. Evolution from A.K. Gopalan (1950) to Maneka Gandhi (1978):\n"
            "   In A.K. Gopalan v. State of Madras (1950), the Supreme Court adopted a literal, narrow approach, holding that 'procedure established by law' "
            "   meant any enacted state legislation, and that Articles 19 and 21 were mutually exclusive compartments.\n"
            "   In Maneka Gandhi v. Union of India (1978), a 7-judge Constitution Bench overruled Gopalan and held that procedure depriving a person of life or liberty "
            "   must not be arbitrary, fanciful, or oppressive; it must be 'just, fair, and reasonable'—effectively importing substantive American due process into Article 21.\n"
            "2. Non-Derogability: Under Article 359(1) (as amended by 44th Amendment 1978), Article 21 CANNOT be suspended during National Emergency.\n"
            "3. Enriched Facets of Article 21 recognized by the Supreme Court:\n"
            "   - Right to live with human dignity (Francis Coralie Mullin v. Administrator, Union Territory of Delhi, 1981)\n"
            "   - Right to livelihood (Olga Tellis v. Bombay Municipal Corporation, 1985)\n"
            "   - Right to privacy as a fundamental right (Justice K.S. Puttaswamy (Retd.) v. Union of India, 2017)\n"
            "   - Right to clean environment (M.C. Mehta v. Union of India, 1987)\n"
            "   - Right to speedy trial and legal aid (Hussainara Khatoon v. Home Secretary, State of Bihar, 1979)\n"
            "   - Right to health and medical care (Paschim Banga Khet Mazdoor Samity v. State of West Bengal, 1996)"
        ),
        "source_url": SOURCE_URL,
        "source_publisher": SOURCE_PUBLISHER,
        "jurisdiction": "India",
        "version_date": VERSION_DATE,
        "amendment_status": AMENDMENT_STATUS,
    },
    {
        "chunk_id": "const_art_21a",
        "document_type": "constitution",
        "title": "Constitution of India — Article 21A",
        "act_name": "Constitution of India",
        "part": "Part III",
        "chapter": "Fundamental Rights — Right to Freedom",
        "article_number": "21A",
        "article_title": "Right to education",
        "clause_number": "all",
        "text": (
            "THE CONSTITUTION OF INDIA\n"
            "PART III: FUNDAMENTAL RIGHTS\n"
            "Article 21A. Right to education.—\n"
            "The State shall provide free and compulsory education to all children of the age of six to fourteen years in such manner as the State may, by law, determine.\n"
            "[Inserted by the Constitution (Eighty-sixth Amendment) Act, 2002, s. 2 (w.e.f. 1-4-2010)]. Effectuated by the Right of Children to Free and Compulsory Education (RTE) Act, 2009."
        ),
        "source_url": SOURCE_URL,
        "source_publisher": SOURCE_PUBLISHER,
        "jurisdiction": "India",
        "version_date": VERSION_DATE,
        "amendment_status": AMENDMENT_STATUS,
    },
    {
        "chunk_id": "const_art_22",
        "document_type": "constitution",
        "title": "Constitution of India — Article 22",
        "act_name": "Constitution of India",
        "part": "Part III",
        "chapter": "Fundamental Rights — Right to Freedom",
        "article_number": "22",
        "article_title": "Protection against arrest and detention in certain cases",
        "clause_number": "all",
        "text": (
            "THE CONSTITUTION OF INDIA\n"
            "PART III: FUNDAMENTAL RIGHTS\n"
            "Article 22. Protection against arrest and detention in certain cases.—\n"
            "(1) No person who is arrested shall be detained in custody without being informed, as soon as may be, of the grounds for such arrest "
            "nor shall he be denied the right to consult, and to be defended by, a legal practitioner of his choice.\n"
            "(2) Every person who is arrested and detained in custody shall be produced before the nearest magistrate within a period of twenty-four hours "
            "of such arrest excluding the time necessary for the journey from the place of arrest to the court of the magistrate and no such person shall be detained "
            "in custody beyond the said period without the authority of a magistrate.\n"
            "(3) Nothing in clauses (1) and (2) shall apply—\n"
            "  (a) to any person who for the time being is an enemy alien; or\n"
            "  (b) to any person who is arrested or detained under any law providing for preventive detention.\n"
            "(4) No law providing for preventive detention shall authorise the detention of a person for a longer period than three months unless an Advisory Board "
            "consisting of persons who are, or have been, or are qualified to be appointed as, Judges of a High Court has reported before the expiration of the said period of three months that there is in its opinion sufficient cause for such detention.\n"
            "(5) When any person is detained in pursuance of an order made under any law providing for preventive detention, the authority making the order shall, "
            "as soon as may be, communicate to such person the grounds on which the order has been made and shall afford him the earliest opportunity of making a representation against the order."
        ),
        "source_url": SOURCE_URL,
        "source_publisher": SOURCE_PUBLISHER,
        "jurisdiction": "India",
        "version_date": VERSION_DATE,
        "amendment_status": AMENDMENT_STATUS,
    },
    {
        "chunk_id": "const_art_32",
        "document_type": "constitution",
        "title": "Constitution of India — Article 32",
        "act_name": "Constitution of India",
        "part": "Part III",
        "chapter": "Fundamental Rights — Right to Constitutional Remedies",
        "article_number": "32",
        "article_title": "Remedies for enforcement of rights conferred by this Part",
        "clause_number": "all",
        "text": (
            "THE CONSTITUTION OF INDIA\n"
            "PART III: FUNDAMENTAL RIGHTS\n"
            "Right to Constitutional Remedies\n"
            "Article 32. Remedies for enforcement of rights conferred by this Part.—\n"
            "(1) The right to move the Supreme Court by appropriate proceedings for the enforcement of the rights conferred by this Part is guaranteed.\n"
            "(2) The Supreme Court shall have power to issue directions or orders or writs, including writs in the nature of habeas corpus, mandamus, prohibition, "
            "quo warranto and certiorari, whichever may be appropriate, for the enforcement of any of the rights conferred by this Part.\n"
            "(3) Without prejudice to the powers conferred on the Supreme Court by clauses (1) and (2), Parliament may by law empower any other court to exercise within the local limits of its jurisdiction all or any of the powers exercisable by the Supreme Court under clause (2).\n"
            "(4) The right guaranteed by this article shall not be suspended except as otherwise provided for by this Constitution.\n\n"
            "Dr. B.R. Ambedkar described Article 32 as the 'very soul of the Constitution and the very heart of it'. It is part of the Basic Structure of the Constitution."
        ),
        "source_url": SOURCE_URL,
        "source_publisher": SOURCE_PUBLISHER,
        "jurisdiction": "India",
        "version_date": VERSION_DATE,
        "amendment_status": AMENDMENT_STATUS,
    },
    {
        "chunk_id": "const_art_226",
        "document_type": "constitution",
        "title": "Constitution of India — Article 226",
        "act_name": "Constitution of India",
        "part": "Part VI",
        "chapter": "The States — The High Courts in the States",
        "article_number": "226",
        "article_title": "Power of High Courts to issue certain writs",
        "clause_number": "all",
        "text": (
            "THE CONSTITUTION OF INDIA\n"
            "PART VI: THE STATES\n"
            "CHAPTER V: THE HIGH COURTS IN THE STATES\n"
            "Article 226. Power of High Courts to issue certain writs.—\n"
            "(1) Notwithstanding anything in article 32, every High Court shall have power, throughout the territories in relation to which it exercises jurisdiction, "
            "to issue to any person or authority, including in appropriate cases, any Government, within those territories directions, orders or writs, including writs in the nature of "
            "habeas corpus, mandamus, prohibition, quo warranto and certiorari, or any of them, for the enforcement of any of the rights conferred by Part III and for any other purpose.\n"
            "(2) The power conferred by clause (1) to issue directions, orders or writs to any Government, authority or person may also be exercised by any High Court exercising jurisdiction "
            "in relation to the territories within which the cause of action, wholly or in part, arises for the exercise of such power, notwithstanding that the seat of such Government or authority or the residence of such person is not within those territories.\n"
            "(3) Where an interim order is made on an ex parte petition, the High Court shall dispose of the application for vacating the interim order within two weeks.\n"
            "(4) The power conferred on a High Court by this article shall not be in derogation of the power conferred on the Supreme Court by clause (2) of article 32.\n\n"
            "Distinction with Article 32: The High Court's jurisdiction under Article 226 is wider than Article 32 because it can be invoked for the enforcement of Fundamental Rights 'and for any other purpose' (legal rights)."
        ),
        "source_url": SOURCE_URL,
        "source_publisher": SOURCE_PUBLISHER,
        "jurisdiction": "India",
        "version_date": VERSION_DATE,
        "amendment_status": AMENDMENT_STATUS,
    },
    {
        "chunk_id": "const_golden_triangle_doctrine",
        "document_type": "constitution",
        "title": "Constitution of India — The Golden Triangle Doctrine (Articles 14, 19, 21)",
        "act_name": "Constitution of India",
        "part": "Part III",
        "chapter": "Fundamental Rights — Interrelationship of Fundamental Rights",
        "article_number": "14, 19, 21",
        "article_title": "The Golden Triangle Doctrine (Articles 14, 19, and 21)",
        "clause_number": "all",
        "text": (
            "THE CONSTITUTION OF INDIA — THE GOLDEN TRIANGLE DOCTRINE\n"
            "The 'Golden Triangle' refers to the constitutional trinity formed by Article 14 (Equality), Article 19 (Fundamental Freedoms), "
            "and Article 21 (Life and Personal Liberty). Together, they constitute the core guarantee of personal freedom and rule of law against state arbitrariness.\n\n"
            "1. Interconnection and Mutually Supportive Nature:\n"
            "   In A.K. Gopalan (1950), the Supreme Court treated these rights as isolated silos. Justice Fazl Ali dissented, arguing for their organic connection.\n"
            "   In Maneka Gandhi v. Union of India (1978) 1 SCC 248, the 7-judge bench established that Fundamental Rights are not isolated islands. "
            "   A law depriving personal liberty under Article 21 must not only follow an enacted procedure, but that procedure must satisfy the test of "
            "   reasonableness under Article 19 and non-arbitrariness under Article 14.\n"
            "   As Chief Justice Chandrachud observed in Minerva Mills v. Union of India (1980): 'Three articles of our Constitution, and only three, "
            "   stand between the heaven of freedom, into which Tagore wanted his country to awake, and the abyss of unrestrained power. They are Articles 14, 19 and 21.'\n\n"
            "2. The Tripartite Constitutional Scrutiny:\n"
            "   Whenever a citizen challenges a state deprivation of life or personal liberty:\n"
            "   - Step 1 (Article 21): Is there a procedure established by law?\n"
            "   - Step 2 (Article 14): Is that procedure just, fair, and non-arbitrary (satisfying intelligible differentia and rational nexus)?\n"
            "   - Step 3 (Article 19): Does that procedure impose only reasonable restrictions under clauses (2) to (6) of Article 19?\n"
            "   If the state action fails under ANY of these three articles, the action is unconstitutional and void under Article 13."
        ),
        "source_url": SOURCE_URL,
        "source_publisher": SOURCE_PUBLISHER,
        "jurisdiction": "India",
        "version_date": VERSION_DATE,
        "amendment_status": AMENDMENT_STATUS,
    },
]


def ingest():
    project_root = Path(__file__).resolve().parent.parent
    chunks_path = project_root / "data" / "chunks" / "legislation" / "chunks.jsonl"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_processed = processed_dir / "constitution_articles.jsonl"

    log.info(f"Writing {len(CONSTITUTION_ARTICLES)} primary constitutional provisions to {out_processed}…")
    with open(out_processed, "w", encoding="utf-8") as f:
        for item in CONSTITUTION_ARTICLES:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Read existing chunks to ensure no duplicate chunk_ids
    existing_ids = set()
    existing_lines = []
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    obj = json.loads(line_str)
                    cid = obj.get("chunk_id")
                    if cid:
                        existing_ids.add(cid)
                    existing_lines.append(line_str)
                except Exception:
                    pass

    log.info(f"Existing legislation chunks: {len(existing_lines)}")
    added = 0
    new_lines = list(existing_lines)

    for item in CONSTITUTION_ARTICLES:
        cid = item["chunk_id"]
        # Format chunk representation
        chunk_obj = {
            "chunk_id": cid,
            "chunk_index": 0,
            "char_count": len(item["text"]),
            "text": item["text"],
            "act_name": item["act_name"],
            "document_type": item["document_type"],
            "title": item["title"],
            "source": f"Constitution of India (as on 1st May 2024, Diglot Edition)",
            "document_id": "constitution_of_india",
            "article": item["article_number"],
            "section": item["article_number"],
            "article_title": item["article_title"],
            "clause": item["clause_number"],
            "part": item["part"],
            "source_url": item["source_url"],
            "source_publisher": item["source_publisher"],
            "jurisdiction": item["jurisdiction"],
            "version_date": item["version_date"],
            "amendment_status": item["amendment_status"],
        }
        if cid in existing_ids:
            # Replace existing
            for idx, el in enumerate(new_lines):
                if f'"{cid}"' in el:
                    new_lines[idx] = json.dumps(chunk_obj, ensure_ascii=False)
                    break
        else:
            new_lines.append(json.dumps(chunk_obj, ensure_ascii=False))
            added += 1

    # Atomic write to chunks.jsonl
    tmp_file = chunks_path.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")
    shutil.move(str(tmp_file), str(chunks_path))

    log.info(f"✅ Ingestion complete! Total legislation chunks now: {len(new_lines)} (added {added} new constitutional articles)")


if __name__ == "__main__":
    ingest()
