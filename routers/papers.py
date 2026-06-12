"""
Saved papers collection — bookmark papers from search results.
"""
from fastapi import APIRouter, HTTPException, Query
from core.config import get_settings

router = APIRouter()


@router.post("/")
async def save_paper(paper: dict):
    """Save a paper to the collection."""
    if not get_settings().database_url:
        raise HTTPException(503, "Database not configured")
    if not paper.get("paper_id") or not paper.get("title"):
        raise HTTPException(400, "paper_id and title are required")
    from db.repositories import save_paper as db_save, insert_event
    paper_db_id = await db_save(paper)
    await insert_event(
        event_type="save_paper",
        discipline=paper.get("discipline_tag"),
        meta={"paper_id": paper.get("paper_id"), "title": paper.get("title")},
    )
    return {"saved": True, "id": paper_db_id, "paper_id": paper.get("paper_id")}


@router.get("/")
async def get_saved_papers(
    discipline: str = Query(None, description="Filter: aerospace | materials | textile | all"),
    limit:      int = Query(50, le=200, ge=1),
):
    """List all saved papers, newest first."""
    if not get_settings().database_url:
        raise HTTPException(503, "Database not configured")
    from db.repositories import get_saved_papers
    rows = await get_saved_papers(discipline=discipline, limit=limit)
    for r in rows:
        if r.get("saved_at"):
            r["saved_at"] = r["saved_at"].isoformat()
    return {"total": len(rows), "papers": rows}


@router.delete("/{paper_id}")
async def delete_saved_paper(paper_id: str):
    """Remove a paper from the saved collection."""
    if not get_settings().database_url:
        raise HTTPException(503, "Database not configured")
    from db.repositories import delete_saved_paper
    deleted = await delete_saved_paper(paper_id)
    if not deleted:
        raise HTTPException(404, f"Paper '{paper_id}' not found in saved collection")
    return {"deleted": True, "paper_id": paper_id}
