"""
RAG pipeline — position-aware ingestion + clean citation answers.

pymupdf block coordinates are TOP-LEFT origin (y increases downward),
same as HTML canvas. No coordinate flip needed on the frontend.
We also store page_height so the frontend can scale bbox correctly.
"""
import uuid, logging
from vectorstore.qdrant_store import get_store
from core.llm import llm_call
from core.config import get_settings
from prompts.templates import PDF_CHAT_SYSTEM, PROPERTY_EXTRACT
from models.schemas import ChatMessage

logger = logging.getLogger(__name__)

_chat_history_mem: dict[str, list[ChatMessage]] = {}


# ── Position-aware chunking ────────────────────────────────────────────────

def _extract_blocks(doc) -> list[dict]:
    """
    Extract text blocks from pymupdf preserving page, bbox, and page_height.

    pymupdf coordinate system: TOP-LEFT origin, y increases downward.
    bbox = [x0, y0, x1, y1] in PDF points (1 point = 1/72 inch).
    page_height stored so frontend can scale to canvas pixels.
    """
    blocks = []
    for page_num, page in enumerate(doc, start=1):
        page_rect   = page.rect
        page_height = round(page_rect.height, 1)
        page_width  = round(page_rect.width,  1)

        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, *_ = block
            text = text.strip()
            if text and len(text.split()) > 5:
                blocks.append({
                    "text":        text,
                    "page":        page_num,
                    "page_height": page_height,
                    "page_width":  page_width,
                    "bbox":        [round(x0,1), round(y0,1), round(x1,1), round(y1,1)],
                })
    return blocks


def _merge_blocks_into_chunks(
    blocks:        list[dict],
    chunk_words:   int = 350,
    overlap_words: int = 60,
) -> list[dict]:
    """
    Merge blocks into ~chunk_words chunks.

    BUG FIX: previously used union_bbox across all blocks in a chunk,
    which produced a bbox spanning the full page — making the highlight
    cover the entire page yellow.

    Now we store the bbox of the FIRST block only as the anchor.
    This gives a tight, accurate highlight at the start of the passage.
    """
    chunks   = []
    buf_text = []
    buf_meta = {}

    def flush():
        if not buf_text or not buf_meta:
            return
        text = " ".join(buf_text)
        if len(text.split()) > 15:
            chunks.append({**buf_meta, "text": text})

    for blk in blocks:
        is_new_chunk = not buf_meta

        if is_new_chunk:
            # Anchor bbox = first block of this chunk only
            buf_meta = {
                "page":        blk["page"],
                "page_height": blk["page_height"],
                "page_width":  blk["page_width"],
                "bbox":        blk["bbox"],   # first block — tight highlight
            }

        buf_text.extend(blk["text"].split())

        if len(buf_text) >= chunk_words:
            flush()
            overlap  = buf_text[-overlap_words:]
            buf_text = list(overlap)
            # New chunk starts at current block
            buf_meta = {
                "page":        blk["page"],
                "page_height": blk["page_height"],
                "page_width":  blk["page_width"],
                "bbox":        blk["bbox"],
            }

    flush()
    return chunks


# ── DB helpers ─────────────────────────────────────────────────────────────

async def _load_history_from_db(session_id: str) -> list[ChatMessage]:
    try:
        from db.repositories import get_chat_history
        rows = await get_chat_history(session_id)
        return [ChatMessage(role=r["role"], content=r["content"]) for r in rows]
    except Exception as e:
        logger.warning(f"Could not load chat history from DB: {e}")
        return []


async def _session_exists(session_id: str) -> bool:
    if session_id in _chat_history_mem:
        return True
    if get_settings().database_url:
        try:
            from db.repositories import session_exists_in_db
            if await session_exists_in_db(session_id):
                _chat_history_mem[session_id] = await _load_history_from_db(session_id)
                return True
        except Exception as e:
            logger.warning(f"DB session check failed: {e}")
    return get_store().session_exists(session_id)


# ── Ingest ─────────────────────────────────────────────────────────────────

async def ingest_pdf(file_bytes: bytes, filename: str, file_size_mb: float = 0.0) -> str:
    import pymupdf

    session_id  = str(uuid.uuid4())
    doc         = pymupdf.open(stream=file_bytes, filetype="pdf")
    total_pages = len(doc)

    if total_pages == 0:
        raise ValueError("PDF has no pages.")

    blocks = _extract_blocks(doc)
    doc.close()

    if not blocks:
        raise ValueError("Could not extract text from PDF.")

    chunks = _merge_blocks_into_chunks(blocks)
    if not chunks:
        raise ValueError("PDF text too short to process.")

    store = get_store()
    n     = store.ingest_with_positions(session_id, chunks, filename)

    if get_settings().database_url:
        try:
            from db.repositories import insert_pdf_session, insert_event
            await insert_pdf_session(
                session_id=session_id, filename=filename,
                file_size_mb=file_size_mb, chunk_count=n, latency_ms=0,
            )
            await insert_event(
                event_type="pdf_upload",
                meta={"filename": filename, "chunks": n,
                      "pages": total_pages, "session_id": session_id},
            )
        except Exception as e:
            logger.warning(f"DB session persist failed (non-fatal): {e}")

    _chat_history_mem[session_id] = []
    logger.info(f"PDF ingested: {filename} → {n} chunks, {total_pages} pages, session={session_id}")
    return session_id


# ── Context builder ────────────────────────────────────────────────────────

def _build_context_and_sources(results: list[dict]) -> tuple[str, list[dict]]:
    """
    Build numbered context for prompt + structured sources for frontend.

    Context sent to LLM:
        [1] (page 3)
        "chunk text..."

    Sources for frontend (ALL retrieved — frontend filters to cited only):
        { ref, page, page_height, page_width, bbox, snippet }
    """
    context_parts = []
    sources       = []

    for i, r in enumerate(results, start=1):
        page        = r.get("page", 1)
        page_height = r.get("page_height")
        page_width  = r.get("page_width")
        text        = r.get("text", "")
        bbox        = r.get("bbox")
        snippet     = text[:200].strip()
        if len(text) > 200:
            snippet += "…"

        context_parts.append(f'[{i}] (page {page})\n"{text}"')
        sources.append({
            "ref":         i,
            "page":        page,
            "page_height": page_height,
            "page_width":  page_width,
            "bbox":        bbox,
            "snippet":     snippet,
        })

    return "\n\n".join(context_parts), sources


# ── Chat ───────────────────────────────────────────────────────────────────

async def chat_with_pdf(session_id: str, question: str) -> dict:
    if not await _session_exists(session_id):
        raise ValueError(f"Session '{session_id}' not found. Upload a PDF first.")

    if session_id not in _chat_history_mem:
        _chat_history_mem[session_id] = await _load_history_from_db(session_id)

    store   = get_store()
    results = store.search(session_id, question, top_k=6)

    if not results:
        return {
            "answer":  "This information is not found in the uploaded paper.",
            "sources": [],
            "history": [m.model_dump() for m in _chat_history_mem[session_id]],
        }

    results.sort(key=lambda r: r["score"], reverse=True)
    context, sources = _build_context_and_sources(results)
    system   = PDF_CHAT_SYSTEM.format(context=context)
    history  = _chat_history_mem[session_id]
    messages = [m.model_dump() for m in history] + [{"role": "user", "content": question}]

    answer = await llm_call(
        system=system, messages=messages, max_tokens=1024, task="pdf_chat",
    )

    _chat_history_mem[session_id].append(ChatMessage(role="user",      content=question))
    _chat_history_mem[session_id].append(ChatMessage(role="assistant", content=answer))

    if get_settings().database_url:
        try:
            from db.repositories import append_chat_message
            await append_chat_message(session_id, "user",      question)
            await append_chat_message(session_id, "assistant", answer)
        except Exception as e:
            logger.warning(f"Chat history DB write failed (non-fatal): {e}")

    return {
        "answer":  answer,
        "sources": sources,
        "history": [m.model_dump() for m in _chat_history_mem[session_id]],
    }


# ── Property extraction ────────────────────────────────────────────────────

async def extract_properties(session_id: str) -> list[dict]:
    if not await _session_exists(session_id):
        raise ValueError(f"Session '{session_id}' not found.")

    store = get_store()
    queries = [
        "tensile strength flexural strength Young's modulus mechanical properties",
        "fibre volume fraction void content density weight",
        "impact strength fracture toughness interlaminar shear",
        "test standard ASTM ISO specimen dimensions",
    ]
    all_chunks, seen = [], set()
    for q in queries:
        for r in store.search(session_id, q, top_k=4):
            if r["chunk_index"] not in seen:
                seen.add(r["chunk_index"])
                all_chunks.append(r)

    if not all_chunks:
        return []

    context = "\n\n---\n\n".join(r["text"] for r in all_chunks[:10])

    import json
    raw = await llm_call(
        system="You are a materials science data extraction specialist.",
        messages=[{"role": "user", "content": f"{PROPERTY_EXTRACT}\n\nTEXT:\n{context}"}],
        max_tokens=2048, prefer_json=True, task="property_extract",
    )
    raw = raw.strip().replace("```json","").replace("```","").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []
