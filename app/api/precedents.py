"""
app/api/precedents.py
=====================
FastAPI router for Precedent Relationship Graph & Case Lineage Visualizer.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from scripts.precedent_graph import get_precedent_service

log = logging.getLogger("legalmind.api.precedents")

router = APIRouter(tags=["precedents"])


class PrecedentNodeModel(BaseModel):
    case_name: str
    court: str
    year: int
    citation: str
    judge: Optional[str] = None
    bench_size: Optional[int] = None
    legal_issue: Optional[str] = None
    statutes_cited: List[str] = []
    cases_cited: List[str] = []
    overruled_cases: List[str] = []
    followed_cases: List[str] = []
    distinguished_cases: List[str] = []
    relied_upon_cases: List[str] = []


class GraphEdgeModel(BaseModel):
    source: str
    target: str
    relation: str


class PrecedentGraphResponse(BaseModel):
    root: str
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


@router.get("/precedents/landmarks")
@router.get("/api/precedents/landmarks")
async def get_landmark_cases():
    """Returns curated landmark Supreme Court of India precedents for graph selection."""
    service = get_precedent_service()
    landmarks = []
    for node in service.nodes.values():
        landmarks.append({
            "case_name": node.case_name,
            "court": node.court,
            "year": node.year,
            "citation": node.citation,
            "bench_size": node.bench_size,
            "legal_issue": node.legal_issue,
            "overruled_count": len(node.overruled_cases),
            "followed_count": len(node.followed_cases),
            "cited_count": len(node.cases_cited),
        })
    return {"success": True, "count": len(landmarks), "landmarks": landmarks}


@router.get("/precedents/graph")
@router.get("/api/precedents/graph")
async def get_precedent_subgraph(
    case: str = Query("Maneka Gandhi v. Union of India", description="Target landmark case name"),
    depth: int = Query(1, ge=1, le=3),
):
    """
    Returns nodes and directed relationship edges (overruled, followed, distinguished, relied_upon)
    formatted for interactive Canvas/SVG visual rendering.
    """
    service = get_precedent_service()
    matched_case = service.fuzzy_match_case(case)
    if not matched_case:
        matched_case = case

    subgraph = service.export_subgraph(matched_case, depth=depth)
    if not subgraph["nodes"]:
        # Fallback to root node
        subgraph["nodes"] = [{
            "id": matched_case,
            "label": matched_case,
            "court": "Supreme Court of India",
            "year": 1978,
            "citation": "AIR / SCC",
            "type": "root"
        }]

    # Format nodes with display properties for visualizer
    formatted_nodes = []
    for n in subgraph["nodes"]:
        is_root = (n.get("id") == subgraph.get("root"))
        node_info = service.get_precedent(n.get("id"))
        formatted_nodes.append({
            "id": n.get("id"),
            "label": n.get("id").split("v.")[0].strip() if "v." in n.get("id", "") else n.get("id"),
            "full_name": n.get("id"),
            "court": node_info.court if node_info else n.get("court", "Supreme Court of India"),
            "year": node_info.year if node_info else n.get("year", 2000),
            "citation": node_info.citation if node_info else n.get("citation", ""),
            "bench_size": node_info.bench_size if node_info else 5,
            "is_root": is_root,
            "issue": node_info.legal_issue if node_info else "Referenced in landmark constitutional jurisprudence.",
        })

    formatted_edges = []
    for e in subgraph.get("edges", []):
        formatted_edges.append({
            "source": e.get("from"),
            "target": e.get("to"),
            "relation": e.get("type"),
        })

    return {
        "success": True,
        "root": subgraph.get("root", matched_case),
        "nodes": formatted_nodes,
        "edges": formatted_edges,
    }


@router.get("/precedents/details")
@router.get("/api/precedents/details")
async def get_precedent_details(case: str = Query(..., description="Case name")):
    """Returns granular ratio decidendi and statutory context for a case."""
    service = get_precedent_service()
    node = service.get_precedent(case)
    if not node:
        raise HTTPException(status_code=404, detail=f"Case '{case}' not found in precedent graph.")
    return {"success": True, "case": node.to_dict()}
