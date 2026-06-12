from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from agents.copilot_agent import run_copilot, generate_summary
from services.search_service import search_papers

router = APIRouter()


class CopilotRequest(BaseModel):
    query:      str
    discipline: str = "all"
    limit:      int = 10


@router.post("/analyse")
async def copilot_analyse(req: CopilotRequest):
    """
    Full Research Copilot: search papers then generate intelligence report.
    Returns: key papers, trends, gaps, future directions, suggested experiments.
    """
    try:
        papers, meta = await search_papers(
            query=req.query, discipline=req.discipline,
            year_from=None, year_to=None, limit=req.limit,
        )
        paper_dicts = [p.model_dump() for p in papers]
        report      = await run_copilot(req.query, paper_dicts)
        report["search_meta"] = {
            "interpreted_query": meta.get("expanded_query", req.query),
            "intent":            meta.get("intent", "general"),
            "papers_analysed":   len(papers),
        }
        return report
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/summary")
async def copilot_summary(req: CopilotRequest):
    """Lightweight research landscape summary (faster than full analyse)."""
    try:
        papers, _ = await search_papers(
            query=req.query, discipline=req.discipline,
            year_from=None, year_to=None, limit=6,
        )
        summary = await generate_summary(req.query, [p.model_dump() for p in papers])
        return {"query": req.query, "summary": summary, "paper_count": len(papers)}
    except Exception as e:
        raise HTTPException(500, str(e))
