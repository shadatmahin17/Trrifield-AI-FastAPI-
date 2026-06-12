"""
RAG pipeline — position-aware ingestion + clean citation answers.

Ingest:  pymupdf extracts text block-by-block, preserving page_number + bbox.
         Each chunk stored in Qdrant with {text, page, bbox, chunk_index}.

Chat:    Retrieved chunks numbered [1]..[N].
         Prompt instructs Claude to cite inline as [1], [2] etc.
         Response includes structured sources with page + snippet for frontend.
"""
import uuid, re, logging
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
    Extract text blocks from pymupdf with page number and bounding box.
    Returns list of {text, page, bbox: [x0,y0,x1,y1]}.
    """
    blocks = []
    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("blocks"):
            # block = (x0, y0, x1, y1, text, block_no, block_type)
            x0, y0, x1, y1, text, *_ = block
            text = text.strip()
            if text and len(text.split()) > 5:
                blocks.append({
                    "text": text,
                    "page": page_num,
                    "bbox": [round(x0,1), round(y0,1), round(x1,1), round(y1,1)],
                })
    return blocks


def _merge_blocks_into_chunks(
    blocks: list[dict],
    chunk_words: int = 350,
    overlap_words: int = 60,
) -> list[dict]:
    """
    Merge small blocks into chunks of ~chunk_words words.
    Each chunk carries: text, page (of first block), bbox (union of blocks).
    Overlapping chunks share overlap_words from the previous chunk for context.
    """
    chunks   = []
    buf_text = []
    buf_page = None
    buf_bbox = None

    def flush():
        nonlocal buf_text, buf_page, buf_bbox
        if not buf_text:
            return
        text = " ".join(buf_text)
        if len(text.split()) > 15:
            chunks.append({"text": text, "page": buf_page, "bbox": buf_bbox})
        buf_text = []
        buf_page = None
        buf_bbox = None

    def union_bbox(a, b):
        if a is None: return b
        if b is None: return a
        return [min(a[0],b[0]), min(a[1],b[1]), max(a[2],b[2]), max(a[3],b[3])]

    for blk in blocks:
        words = blk["text"].split()
        if buf_page is None:
            buf_page = blk["page"]
            buf_bbox = blk["bbox"]

        buf_text.extend(words)
        buf_bbox = union_bbox(buf_bbox, blk["bbox"])

        if len(buf_text) >= chunk_words:
            flush()
            # Overlap: carry last overlap_words into next chunk
            prev_words = " ".join(buf_text[-overlap_words:]) if buf_text else ""
            buf_text   = prev_words.split() if prev_words else []
            buf_page   = blk["page"]
            buf_bbox   = blk["bbox"]

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

    session_id = str(uuid.uuid4())
    doc        = pymupdf.open(stream=file_bytes, filetype="pdf")
    total_pages = len(doc)

    if total_pages == 0:
        raise ValueError("PDF has no pages.")

    # Position-aware extraction
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
    logger.info(f"PDF ingested: {filename} → {n} chunks across {total_pages} pages")
    return session_id


# ── Chat ───────────────────────────────────────────────────────────────────

def _build_context_and_sources(results: list[dict]) -> tuple[str, list[dict]]:
    """
    Build numbered context block for the prompt + structured sources for the frontend.

    Context sent to LLM:
        [1] (page 3)
        "...chunk text..."

        [2] (page 5)
        "...chunk text..."

    Sources returned to frontend:
        [
          { "ref": 1, "page": 3, "bbox": [...], "snippet": "first 180 chars…" },
          ...
        ]
    """
    context_parts = []
    sources       = []

    for i, r in enumerate(results, start=1):
        page    = r.get("page", 1)
        text    = r.get("text", "")
        bbox    = r.get("bbox")
        snippet = text[:180].strip()
        if len(text) > 180:
            snippet += "…"

        context_parts.append(f"[{i}] (page {page})\n\"{text}\"")
        sources.append({
            "ref":     i,
            "page":    page,
            "bbox":    bbox,
            "snippet": snippet,
        })

    return "\n\n".join(context_parts), sources


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
        "sources": sources,   # structured list, not raw strings
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
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []
