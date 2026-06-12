import time
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from rag.pipeline import ingest_pdf, chat_with_pdf, extract_properties
from analytics.tracker import get_tracker
from models.schemas import ChatRequest, ChatResponse, PropertyExtractionResponse
from core.config import get_settings

router = APIRouter()


@router.post("/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    content_length = request.headers.get("content-length")
    max_bytes = get_settings().max_pdf_size_mb * 1024 * 1024
    if content_length and int(content_length) > max_bytes:
        raise HTTPException(
            400,
            f"File too large ({int(content_length)/(1024*1024):.1f}MB). Max is {get_settings().max_pdf_size_mb}MB."
        )

    contents  = await file.read()
    size_mb   = len(contents) / (1024 * 1024)
    if size_mb > get_settings().max_pdf_size_mb:
        raise HTTPException(400, f"File too large ({size_mb:.1f}MB). Max is {get_settings().max_pdf_size_mb}MB.")

    t0 = time.time()
    try:
        session_id = await ingest_pdf(contents, file.filename, file_size_mb=round(size_mb, 2))
        latency    = round((time.time() - t0) * 1000, 1)

        # Update latency in DB now that we know it
        if get_settings().database_url:
            try:
                from db.repositories import insert_pdf_session
                await insert_pdf_session(
                    session_id=session_id, filename=file.filename,
                    file_size_mb=round(size_mb, 2), chunk_count=0, latency_ms=latency,
                )
            except Exception:
                pass

        get_tracker().record_pdf(file.filename, session_id, 0, latency)
        return {
            "session_id": session_id,
            "filename":   file.filename,
            "size_mb":    round(size_mb, 2),
            "message":    "PDF indexed via Qdrant vector search. Use session_id to chat.",
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = await chat_with_pdf(req.session_id, req.question)
        return ChatResponse(
            session_id = req.session_id,
            answer     = result["answer"],
            sources    = result["sources"],
            history    = result["history"],
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/sessions")
async def list_sessions(limit: int = 20):
    """List recent PDF sessions from the database."""
    if not get_settings().database_url:
        raise HTTPException(503, "Database not configured")
    from db.repositories import get_pdf_sessions
    rows = await get_pdf_sessions(limit=limit)
    for r in rows:
        for k in ("created_at", "last_accessed"):
            if r.get(k):
                r[k] = r[k].isoformat()
    return {"total": len(rows), "sessions": rows}


@router.get("/extract-properties/{session_id}", response_model=PropertyExtractionResponse)
async def extract(session_id: str):
    try:
        props = await extract_properties(session_id)
        return PropertyExtractionResponse(session_id=session_id, properties=props)
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))
