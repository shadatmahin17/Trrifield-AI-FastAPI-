"""
Library router — shared community PDF library with on-demand column extraction.
"""
import time, logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Query, BackgroundTasks
from core.config import get_settings
from core.storage import r2_enabled, upload_pdf_to_r2, delete_pdf_from_r2
from services.paper_extractor import extract_column, extract_metadata_from_text, COLUMN_DEFINITIONS

router = APIRouter()
logger = logging.getLogger(__name__)


async def _extract_text_from_bytes(file_bytes: bytes) -> str:
    """Extract plain text from PDF bytes using pymupdf."""
    import pymupdf
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


@router.post("/upload")
async def upload_to_library(
    request:          Request,
    background_tasks: BackgroundTasks,
    file:             UploadFile = File(...),
    discipline:       str = Query("general"),
    uploaded_by:      str = Query("anonymous"),
):
    """
    Upload a PDF to the shared library.
    1. Saves to Cloudflare R2 (permanent storage)
    2. Ingests into Qdrant (for PDF chat)
    3. Extracts metadata (title, authors, abstract, DOI) via LLM
    4. Stores in library_papers table
    5. Background: extracts TL;DR column
    """
    if not get_settings().database_url:
        raise HTTPException(503, "Database not configured")
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    max_bytes = get_settings().max_pdf_size_mb * 1024 * 1024
    cl = request.headers.get("content-length")
    if cl and int(cl) > max_bytes:
        raise HTTPException(400, f"File too large. Max {get_settings().max_pdf_size_mb}MB")

    contents = await file.read()
    size_mb  = round(len(contents) / (1024 * 1024), 2)
    if size_mb * 1024 * 1024 > max_bytes:
        raise HTTPException(400, f"File too large ({size_mb}MB). Max {get_settings().max_pdf_size_mb}MB")

    t0 = time.time()

    # 1. Ingest into Qdrant for PDF chat
    from rag.pipeline import ingest_pdf
    session_id = await ingest_pdf(contents, file.filename, file_size_mb=size_mb)

    # 2. Extract text + metadata
    paper_text = await _extract_text_from_bytes(contents)
    metadata   = await extract_metadata_from_text(paper_text)

    # 3. Detect discipline from text if not provided
    if discipline == "general" and paper_text:
        from services.search_service import _tag_discipline
        discipline = _tag_discipline(
            metadata.get("title") or file.filename,
            metadata.get("abstract"),
            [],
        )

    # 4. Upload to R2 if configured
    r2_key = None; r2_url = None
    if r2_enabled():
        try:
            r2_key = f"papers/{session_id}/{file.filename}"
            r2_url = await upload_pdf_to_r2(contents, r2_key, file.filename)
        except Exception as e:
            logger.warning(f"R2 upload failed (continuing without): {e}")

    # 5. Save to database
    from db.repositories import insert_library_paper
    paper_id = await insert_library_paper(
        session_id   = session_id,
        filename     = file.filename,
        title        = metadata.get("title") or file.filename.replace(".pdf",""),
        authors      = metadata.get("authors") or [],
        abstract     = metadata.get("abstract"),
        doi          = metadata.get("doi"),
        journal      = metadata.get("journal"),
        year         = metadata.get("year"),
        discipline   = discipline,
        file_size_mb = size_mb,
        r2_key       = r2_key,
        r2_url       = r2_url,
        chunk_count  = 0,
        uploaded_by  = uploaded_by,
    )

    latency = round((time.time() - t0) * 1000, 1)
    logger.info(f"Library upload: {file.filename} → paper_id={paper_id}, session={session_id}, {latency}ms")

    # 6. Background: pre-extract TL;DR so it's ready immediately
    async def _pre_extract_tldr():
        try:
            from db.repositories import get_column, set_column
            if not await get_column(paper_id, "tldr"):
                content = await extract_column(paper_text, "tldr")
                await set_column(paper_id, "tldr", content)
        except Exception as e:
            logger.warning(f"TL;DR pre-extraction failed: {e}")

    background_tasks.add_task(_pre_extract_tldr)

    return {
        "paper_id":   paper_id,
        "session_id": session_id,
        "title":      metadata.get("title") or file.filename.replace(".pdf",""),
        "authors":    metadata.get("authors") or [],
        "doi":        metadata.get("doi"),
        "journal":    metadata.get("journal"),
        "year":       metadata.get("year"),
        "discipline": discipline,
        "size_mb":    size_mb,
        "r2_url":     r2_url,
        "latency_ms": latency,
    }


@router.get("/")
async def list_library(
    discipline: str  = Query(None),
    search:     str  = Query(None),
    limit:      int  = Query(50, le=200),
    offset:     int  = Query(0),
):
    """List all public library papers with optional discipline filter and search."""
    if not get_settings().database_url:
        raise HTTPException(503, "Database not configured")
    from db.repositories import get_library_papers
    papers = await get_library_papers(discipline=discipline, search=search, limit=limit, offset=offset)
    return {"total": len(papers), "papers": papers, "offset": offset}


@router.get("/stats")
async def library_stats():
    """Community library statistics."""
    if not get_settings().database_url:
        raise HTTPException(503, "Database not configured")
    from db.repositories import get_library_stats
    return await get_library_stats()


@router.get("/columns")
async def list_column_types():
    """All supported extraction column types."""
    return {
        "columns": [
            {"key": k, "label": v["label"]}
            for k, v in COLUMN_DEFINITIONS.items()
        ]
    }


@router.get("/{paper_id}")
async def get_paper(paper_id: int):
    """Get a single library paper with all cached column extractions."""
    if not get_settings().database_url:
        raise HTTPException(503, "Database not configured")
    from db.repositories import get_library_paper, get_all_columns
    paper = await get_library_paper(paper_id)
    if not paper:
        raise HTTPException(404, f"Paper {paper_id} not found")
    columns = await get_all_columns(paper_id)
    return {**paper, "extracted_columns": columns}


@router.post("/{paper_id}/extract/{column_key}")
async def extract_paper_column(paper_id: int, column_key: str):
    """
    Extract a specific column for a paper on-demand.
    Returns cached value if already extracted, otherwise runs LLM extraction.
    """
    if not get_settings().database_url:
        raise HTTPException(503, "Database not configured")
    if column_key not in COLUMN_DEFINITIONS:
        raise HTTPException(400, f"Unknown column. Valid: {list(COLUMN_DEFINITIONS.keys())}")

    from db.repositories import get_library_paper, get_column, set_column

    paper = await get_library_paper(paper_id)
    if not paper:
        raise HTTPException(404, f"Paper {paper_id} not found")

    # Return cached value if available
    cached = await get_column(paper_id, column_key)
    if cached:
        return {
            "paper_id":   paper_id,
            "column_key": column_key,
            "label":      COLUMN_DEFINITIONS[column_key]["label"],
            "content":    cached,
            "cached":     True,
        }

    # Extract from PDF session
    paper_text = ""
    try:
        from vectorstore.qdrant_store import get_store
        store   = get_store()
        results = store.search(paper["session_id"], "methods results findings conclusion", top_k=8)
        paper_text = "\n\n".join(r["text"] for r in results)
    except Exception as e:
        logger.warning(f"Could not retrieve paper text for extraction: {e}")
        # Try R2 if available
        if paper.get("r2_url") and r2_enabled():
            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(paper["r2_url"])
                if resp.status_code == 200:
                    paper_text = await _extract_text_from_bytes(resp.content)

    if not paper_text:
        raise HTTPException(422, "Could not retrieve paper text for extraction")

    content = await extract_column(paper_text, column_key)
    await set_column(paper_id, column_key, content)

    return {
        "paper_id":   paper_id,
        "column_key": column_key,
        "label":      COLUMN_DEFINITIONS[column_key]["label"],
        "content":    content,
        "cached":     False,
    }


@router.post("/{paper_id}/extract-batch")
async def extract_batch_columns(
    paper_id:    int,
    column_keys: list[str],
    background_tasks: BackgroundTasks,
):
    """
    Request extraction of multiple columns. Returns cached immediately,
    queues uncached ones in background. Frontend polls for results.
    """
    if not get_settings().database_url:
        raise HTTPException(503, "Database not configured")

    from db.repositories import get_library_paper, get_column, set_column, get_all_columns

    paper = await get_library_paper(paper_id)
    if not paper:
        raise HTTPException(404, f"Paper {paper_id} not found")

    # Validate keys
    invalid = [k for k in column_keys if k not in COLUMN_DEFINITIONS]
    if invalid:
        raise HTTPException(400, f"Unknown column keys: {invalid}")

    # Split into cached vs needed
    all_cached  = await get_all_columns(paper_id)
    results     = {k: all_cached[k] for k in column_keys if k in all_cached}
    to_extract  = [k for k in column_keys if k not in all_cached]

    async def _extract_all():
        try:
            from vectorstore.qdrant_store import get_store
            store      = get_store()
            chunks     = store.search(paper["session_id"], "methods results findings conclusion abstract", top_k=10)
            paper_text = "\n\n".join(r["text"] for r in chunks)
            for key in to_extract:
                try:
                    content = await extract_column(paper_text, key)
                    await set_column(paper_id, key, content)
                except Exception as e:
                    logger.warning(f"Column {key} extraction failed: {e}")
        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")

    if to_extract:
        background_tasks.add_task(_extract_all)

    return {
        "paper_id":       paper_id,
        "cached":         results,
        "queued":         to_extract,
        "queued_count":   len(to_extract),
        "message":        f"{len(results)} ready, {len(to_extract)} extracting in background. Poll GET /{paper_id} to get results.",
    }


@router.delete("/{paper_id}")
async def delete_paper(paper_id: int):
    """Remove a paper from the library (also deletes from R2)."""
    if not get_settings().database_url:
        raise HTTPException(503, "Database not configured")
    from db.repositories import delete_library_paper
    deleted = await delete_library_paper(paper_id)
    if not deleted:
        raise HTTPException(404, f"Paper {paper_id} not found")
    if deleted.get("r2_key") and r2_enabled():
        await delete_pdf_from_r2(deleted["r2_key"])
    return {"deleted": True, "paper_id": paper_id}
