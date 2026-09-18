#!/usr/bin/env python3
"""
scripts/precedent_graph.py
==========================
LegalMind AI — Precedent Relationship Graph Backend

Extracts, indexes, and queries precedent relationship structures:
- Case name, court, year, official citation, bench/judge
- Legal issue addressed
- Statutes cited
- Cases cited
- Treatment relationships:
    * overruled_cases (expressly or impliedly overruled)
    * followed_cases (approved / affirmed)
    * distinguished_cases (distinguished on facts or law)
    * relied_upon_cases (principles adopted)

Includes a curated knowledge graph of landmark Supreme Court of India precedents
and an extraction engine for case law texts.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class PrecedentNode:
    case_name: str
    court: str
    year: int
    citation: str
    judge: Optional[str] = None
    bench_size: Optional[int] = None
    legal_issue: Optional[str] = None
    statutes_cited: List[str] = field(default_factory=list)
    cases_cited: List[str] = field(default_factory=list)
    overruled_cases: List[str] = field(default_factory=list)
    followed_cases: List[str] = field(default_factory=list)
    distinguished_cases: List[str] = field(default_factory=list)
    relied_upon_cases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Curated Landmark Precedent Knowledge Graph
# ---------------------------------------------------------------------------

LANDMARK_PRECEdENTS: List[PrecedentNode] = [
    PrecedentNode(
        case_name="Kesavananda Bharati v. State of Kerala",
        court="Supreme Court of India",
        year=1973,
        citation="AIR 1973 SC 1461 : (1973) 4 SCC 225",
        judge="S.M. Sikri, C.J. (13-Judge Constitution Bench)",
        bench_size=13,
        legal_issue="Constitutional validity of 24th, 25th, 29th Amendments; Basic Structure Doctrine and amending power under Article 368.",
        statutes_cited=["Constitution of India, Articles 13, 31, 31C, 368"],
        cases_cited=[
            "I.C. Golaknath v. State of Punjab (1967)",
            "Sajjan Singh v. State of Rajasthan (1965)",
            "Shankari Prasad v. Union of India (1951)",
        ],
        overruled_cases=["I.C. Golaknath v. State of Punjab (1967) (prospectively overruled)"],
        followed_cases=[],
        distinguished_cases=[],
        relied_upon_cases=["Shankari Prasad v. Union of India (1951)", "Sajjan Singh v. State of Rajasthan (1965)"],
    ),
    PrecedentNode(
        case_name="Maneka Gandhi v. Union of India",
        court="Supreme Court of India",
        year=1978,
        citation="AIR 1978 SC 597 : (1978) 1 SCC 248",
        judge="M.H. Beg, C.J., P.N. Bhagwati, V.R. Krishna Iyer, JJ. (7-Judge Bench)",
        bench_size=7,
        legal_issue="Right to travel abroad under Article 21; Interconnection of Articles 14, 19, and 21 (Golden Triangle); 'Procedure established by law' requires fairness, justice and non-arbitrariness (Due Process).",
        statutes_cited=["Constitution of India, Articles 14, 19, 21, 32", "Passports Act, 1967, Section 10(3)(c)"],
        cases_cited=["A.K. Gopalan v. State of Madras (1950)", "Satwant Singh Sawhney v. D. Ramarathnam (1967)"],
        overruled_cases=["A.K. Gopalan v. State of Madras (1950) (on isolation of Fundamental Rights)"],
        followed_cases=["Satwant Singh Sawhney v. D. Ramarathnam (1967)"],
        distinguished_cases=[],
        relied_upon_cases=["Rustom Cavasjee Cooper v. Union of India (Bank Nationalisation Case) (1970)"],
    ),
    PrecedentNode(
        case_name="Justice K.S. Puttaswamy (Retd.) v. Union of India",
        court="Supreme Court of India",
        year=2017,
        citation="AIR 2017 SC 4161 : (2017) 10 SCC 1",
        judge="J.S. Khehar, C.J., J. Chelameswar, S.A. Bobde, D.Y. Chandrachud, JJ. (9-Judge Bench)",
        bench_size=9,
        legal_issue="Whether the Right to Privacy is a Fundamental Right under the Constitution of India.",
        statutes_cited=["Constitution of India, Articles 14, 19, 21"],
        cases_cited=[
            "M.P. Sharma v. Satish Chandra (1954)",
            "Kharak Singh v. State of U.P. (1963)",
            "Govind v. State of M.P. (1975)",
            "ADM Jabalpur v. Shivkant Shukla (1976)",
        ],
        overruled_cases=[
            "M.P. Sharma v. Satish Chandra (1954) (to the extent it held privacy is not protected)",
            "Kharak Singh v. State of U.P. (1963) (majority view on privacy)",
            "ADM Jabalpur v. Shivkant Shukla (1976)",
        ],
        followed_cases=["Gobind v. State of M.P. (1975)", "R. Rajagopal v. State of T.N. (1994)"],
        distinguished_cases=[],
        relied_upon_cases=["Maneka Gandhi v. Union of India (1978)"],
    ),
    PrecedentNode(
        case_name="Navtej Singh Johar v. Union of India",
        court="Supreme Court of India",
        year=2018,
        citation="AIR 2018 SC 4321 : (2018) 10 SCC 1",
        judge="Dipak Misra, C.J., R.F. Nariman, A.M. Khanwilkar, D.Y. Chandrachud, Indu Malhotra, JJ. (5-Judge Bench)",
        bench_size=5,
        legal_issue="Constitutionality of Section 377 Indian Penal Code criminalizing consensual homosexual acts between adults.",
        statutes_cited=["Constitution of India, Articles 14, 15, 19, 21", "Indian Penal Code, 1860, Section 377"],
        cases_cited=["Suresh Kumar Koushal v. Naz Foundation (2014)", "NALSA v. Union of India (2014)"],
        overruled_cases=["Suresh Kumar Koushal v. Naz Foundation (2014)"],
        followed_cases=["NALSA v. Union of India (2014)", "K.S. Puttaswamy v. Union of India (2017)"],
        distinguished_cases=[],
        relied_upon_cases=["Maneka Gandhi v. Union of India (1978)"],
    ),
    PrecedentNode(
        case_name="Joseph Shine v. Union of India",
        court="Supreme Court of India",
        year=2018,
        citation="AIR 2018 SC 4898 : (2019) 3 SCC 39",
        judge="Dipak Misra, C.J., R.F. Nariman, A.M. Khanwilkar, D.Y. Chandrachud, Indu Malhotra, JJ. (5-Judge Bench)",
        bench_size=5,
        legal_issue="Constitutionality of Section 497 IPC (Adultery) and Section 198(2) CrPC.",
        statutes_cited=["Constitution of India, Articles 14, 15, 21", "Indian Penal Code, 1860, Section 497", "Code of Criminal Procedure, 1973, Section 198(2)"],
        cases_cited=["Yusuf Abdul Aziz v. State of Bombay (1954)", "Sowmithri Vishnu v. Union of India (1985)", "V. Revathi v. Union of India (1988)"],
        overruled_cases=[
            "Yusuf Abdul Aziz v. State of Bombay (1954)",
            "Sowmithri Vishnu v. Union of India (1985)",
            "V. Revathi v. Union of India (1988)",
        ],
        followed_cases=["K.S. Puttaswamy v. Union of India (2017)"],
        distinguished_cases=[],
        relied_upon_cases=[],
    ),
    PrecedentNode(
        case_name="Shreya Singhal v. Union of India",
        court="Supreme Court of India",
        year=2015,
        citation="AIR 2015 SC 1523 : (2015) 5 SCC 1",
        judge="J. Chelameswar, R.F. Nariman, JJ. (2-Judge Bench)",
        bench_size=2,
        legal_issue="Constitutionality of Section 66A of the Information Technology Act, 2000 on grounds of vague restrictions on freedom of speech.",
        statutes_cited=["Constitution of India, Article 19(1)(a), Article 19(2)", "Information Technology Act, 2000, Section 66A"],
        cases_cited=["Romesh Thappar v. State of Madras (1950)", "Kameshwar Singh v. State of Bihar (1962)"],
        overruled_cases=[],
        followed_cases=["Romesh Thappar v. State of Madras (1950)"],
        distinguished_cases=[],
        relied_upon_cases=[],
    ),
    PrecedentNode(
        case_name="Indra Sawhney v. Union of India",
        court="Supreme Court of India",
        year=1992,
        citation="AIR 1993 SC 477 : 1992 Supp (3) SCC 217",
        judge="M.H. Kania, C.J. (9-Judge Bench)",
        bench_size=9,
        legal_issue="Scope of reservation under Article 16(4); 50% cap on total reservations; exclusion of 'creamy layer' from OBC quota.",
        statutes_cited=["Constitution of India, Articles 14, 15, 16, 335, 340"],
        cases_cited=["M.R. Balaji v. State of Mysore (1963)", "T. Devadasan v. Union of India (1964)", "State of Kerala v. N.M. Thomas (1976)"],
        overruled_cases=[],
        followed_cases=["M.R. Balaji v. State of Mysore (1963) (affirmed on 50% rule)"],
        distinguished_cases=[],
        relied_upon_cases=[],
    ),
    PrecedentNode(
        case_name="S.R. Bommai v. Union of India",
        court="Supreme Court of India",
        year=1994,
        citation="AIR 1994 SC 1918 : (1994) 3 SCC 1",
        judge="A.M. Ahmadi, J.S. Verma, P.B. Sawant, K. Ramaswamy, S.C. Agrawal, B.P. Jeevan Reddy, JJ. (9-Judge Bench)",
        bench_size=9,
        legal_issue="Judicial review of presidential proclamation imposing President's Rule under Article 356; Secularism and Federalism as basic structure.",
        statutes_cited=["Constitution of India, Articles 355, 356, 74(2)"],
        cases_cited=["State of Rajasthan v. Union of India (1977)"],
        overruled_cases=["State of Rajasthan v. Union of India (1977) (narrowed on immunity from judicial review)"],
        followed_cases=["Kesavananda Bharati v. State of Kerala (1973)"],
        distinguished_cases=[],
        relied_upon_cases=[],
    ),
    PrecedentNode(
        case_name="Vishaka v. State of Rajasthan",
        court="Supreme Court of India",
        year=1997,
        citation="AIR 1997 SC 3011 : (1997) 6 SCC 241",
        judge="J.S. Verma, C.J., Sujata V. Manohar, B.N. Kirpal, JJ. (3-Judge Bench)",
        bench_size=3,
        legal_issue="Formulation of binding guidelines preventing sexual harassment of women at workplace in the absence of legislation.",
        statutes_cited=["Constitution of India, Articles 14, 19(1)(g), 21, 32", "CEDAW Convention"],
        cases_cited=[],
        overruled_cases=[],
        followed_cases=[],
        distinguished_cases=[],
        relied_upon_cases=[],
    ),
    PrecedentNode(
        case_name="A.K. Gopalan v. State of Madras",
        court="Supreme Court of India",
        year=1950,
        citation="AIR 1950 SC 27",
        judge="H.J. Kania, C.J. (6-Judge Bench)",
        bench_size=6,
        legal_issue="Preventive detention and interpretation of Article 21; Separation of Articles 19 and 21.",
        statutes_cited=["Constitution of India, Articles 19, 21, 22", "Preventive Detention Act, 1950"],
        cases_cited=[],
        overruled_cases=[],
        followed_cases=[],
        distinguished_cases=[],
        relied_upon_cases=[],
    ),
]


class PrecedentGraphService:
    """Service for indexing and retrieving precedent relationships."""

    def __init__(self, precedents: Optional[List[PrecedentNode]] = None):
        self._nodes: Dict[str, PrecedentNode] = {}
        self._index_by_keyword: Dict[str, Set[str]] = {}
        items = precedents or LANDMARK_PRECEdENTS
        for p in items:
            self.add_node(p)

    @property
    def nodes(self) -> Dict[str, PrecedentNode]:
        return self._nodes

    def fuzzy_match_case(self, case_name: str) -> Optional[str]:
        p = self.get_precedent(case_name)
        return p.case_name if p else None

    def add_node(self, node: PrecedentNode) -> None:
        key = self._normalize_name(node.case_name)
        self._nodes[key] = node
        # Index keywords
        words = re.findall(r"\b\w{3,}\b", node.case_name.lower())
        for w in words:
            self._index_by_keyword.setdefault(w, set()).add(key)

    def _normalize_name(self, name: str) -> str:
        s = name.lower()
        s = re.sub(r"\bv\b|\bvs\b|\bversus\b", "v", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def get_precedent(self, case_name: str) -> Optional[PrecedentNode]:
        norm = self._normalize_name(case_name)
        if norm in self._nodes:
            return self._nodes[norm]

        # Substring search
        for k, v in self._nodes.items():
            if norm in k or k in norm:
                return v

        # Keyword matching
        words = [w for w in re.findall(r"\b\w{3,}\b", norm) if w not in ("state", "union", "india")]
        best_match = None
        best_count = 0
        for k, v in self._nodes.items():
            count = sum(1 for w in words if w in k)
            if count > best_count:
                best_count = count
                best_match = v

        return best_match if best_count >= 1 else None

    def search_precedents(self, query: str, limit: int = 5) -> List[PrecedentNode]:
        results = []
        q_norm = self._normalize_name(query)
        words = re.findall(r"\b\w{3,}\b", q_norm)

        scored: List[Tuple[int, PrecedentNode]] = []
        for key, node in self._nodes.items():
            score = 0
            if q_norm in key or key in q_norm:
                score += 10
            for w in words:
                if w in key:
                    score += 2
                if node.legal_issue and w in node.legal_issue.lower():
                    score += 1
                for sc in node.statutes_cited:
                    if w in sc.lower():
                        score += 1
            if score > 0:
                scored.append((score, node))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [node for _, node in scored[:limit]]

    def get_relationships(self, case_name: str) -> Dict[str, Any]:
        """
        Returns inward and outward relationship edges for a given case.
        """
        node = self.get_precedent(case_name)
        if not node:
            return {
                "case_found": False,
                "case_name": case_name,
                "relationships": {},
            }

        # Find any other cases in the graph that cite or overrule this node
        inward_citations: List[str] = []
        overruled_by: List[str] = []
        followed_by: List[str] = []

        norm_target = self._normalize_name(node.case_name)
        for k, other in self._nodes.items():
            if k == norm_target:
                continue
            for ov in other.overruled_cases:
                if any(part in ov.lower() for part in norm_target.split(" v ")[0].split()):
                    overruled_by.append(other.case_name)
            for fol in other.followed_cases:
                if any(part in fol.lower() for part in norm_target.split(" v ")[0].split()):
                    followed_by.append(other.case_name)
            for cc in other.cases_cited:
                if any(part in cc.lower() for part in norm_target.split(" v ")[0].split()):
                    inward_citations.append(other.case_name)

        return {
            "case_found": True,
            "case": node.to_dict(),
            "outward": {
                "overruled_cases": node.overruled_cases,
                "followed_cases": node.followed_cases,
                "distinguished_cases": node.distinguished_cases,
                "relied_upon_cases": node.relied_upon_cases,
                "statutes_cited": node.statutes_cited,
            },
            "inward": {
                "overruled_by": list(set(overruled_by)),
                "followed_by": list(set(followed_by)),
                "cited_by": list(set(inward_citations)),
            },
        }

    def find_cases_citing_provision(self, provision: str) -> List[PrecedentNode]:
        """Finds all cases that cite a specific constitutional article or statutory provision."""
        p_clean = provision.lower().strip()
        matched = []
        for node in self._nodes.values():
            for sc in node.statutes_cited:
                if p_clean in sc.lower():
                    matched.append(node)
                    break
        return matched

    def find_overruled_cases(self, case_name: str) -> List[str]:
        """Returns cases explicitly or prospectively overruled by case_name."""
        node = self.get_precedent(case_name)
        return node.overruled_cases if node else []

    def find_following_cases(self, case_name: str) -> List[str]:
        """Returns cases followed/affirmed by case_name."""
        node = self.get_precedent(case_name)
        return node.followed_cases if node else []

    def find_distinguished_cases(self, case_name: str) -> List[str]:
        """Returns cases distinguished by case_name."""
        node = self.get_precedent(case_name)
        return node.distinguished_cases if node else []

    def export_subgraph(self, case_name: str, depth: int = 1) -> Dict[str, Any]:
        """Exports nodes and directed edges as a graph structure for visual exploration."""
        node = self.get_precedent(case_name)
        if not node:
            return {"nodes": [], "edges": []}

        nodes_map = {
            node.case_name: {
                "id": node.case_name,
                "court": node.court,
                "year": node.year,
                "citation": node.citation,
            }
        }
        edges = []

        for ov in node.overruled_cases:
            target = ov.split("(")[0].strip()
            nodes_map[target] = {"id": target, "type": "case"}
            edges.append({"from": node.case_name, "to": target, "type": "overruled"})

        for fol in node.followed_cases:
            target = fol.split("(")[0].strip()
            nodes_map[target] = {"id": target, "type": "case"}
            edges.append({"from": node.case_name, "to": target, "type": "followed"})

        for dist in node.distinguished_cases:
            target = dist.split("(")[0].strip()
            nodes_map[target] = {"id": target, "type": "case"}
            edges.append({"from": node.case_name, "to": target, "type": "distinguished"})

        for rel in node.relied_upon_cases:
            target = rel.split("(")[0].strip()
            nodes_map[target] = {"id": target, "type": "case"}
            edges.append({"from": node.case_name, "to": target, "type": "relied_upon"})

        return {
            "root": node.case_name,
            "nodes": list(nodes_map.values()),
            "edges": edges,
        }


# Global singleton instance
_default_graph: Optional[PrecedentGraphService] = None


def get_precedent_service() -> PrecedentGraphService:
    global _default_graph
    if _default_graph is None:
        _default_graph = PrecedentGraphService()
    return _default_graph
