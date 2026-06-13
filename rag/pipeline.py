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

def _extract_lines(doc) -> list[dict]:
    """
    Extract every text line from the PDF with its own bbox.

    Working at LINE level (not block level) means:
    - Each line has its own tight bbox
    - When we chunk lines together, the first line of each chunk
      has the exact bbox where that chunk's text actually starts
    - No more "first line of block" vs "first line of chunk" mismatch

    pymupdf coordinate system: TOP-LEFT origin, y increases downward.
    bbox = [x0, y0, x1, y1] in PDF points (1/72 inch).
    """
    lines = []
    for page_num, page in enumerate(doc, start=1):
        page_rect   = page.rect
        page_height = round(page_rect.height, 1)
        page_width  = round(page_rect.width,  1)

        page_dict = page.get_text("dict")
        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:   # skip image blocks
                continue
            for line in block.get("lines", []):
                # Join all spans in this line into one string
                text = " ".join(
                    span.get("text", "") for span in line.get("spans", [])
                ).strip()
                if not text or len(text) < 10:
                    continue

                lb = line.get("bbox", block["bbox"])
                lines.append({
                    "text":        text,
                    "page":        page_num,
                    "page_height": page_height,
                    "page_width":  page_width,
                    "bbox":        [round(lb[0],1), round(lb[1],1),
                                    round(lb[2],1), round(lb[3],1)],
                })
    return lines


def _chunk_lines(
    lines:         list[dict],
    chunk_words:   int = 350,
    overlap_lines: int = 4,
) -> list[dict]:
    """
    Merge lines into ~chunk_words chunks.

    HIGHLIGHT ANCHOR FIX:
    When overlap carries lines from the previous chunk forward, those
    overlap lines are NOT the semantic start of the new chunk — they're
    carried-over context. The bbox must point to the FIRST NEW line
    (i.e. the line after the overlap), not the first overlap line.

    We track this with `anchor_idx` — the index in buf_lines where
    new (non-overlap) content starts.
    """
    chunks   = []
    buf_lines: list[dict] = []
    anchor_idx = 0   # index of first NEW line in buf_lines

    def flush():
        if not buf_lines:
            return
        text = " ".join(l["text"] for l in buf_lines)
        if len(text.split()) < 15:
            return
        # Use the first NEW line (after overlap) as the highlight anchor
        anchor = buf_lines[anchor_idx]
        pw     = anchor.get("page_width") or 595.0
        b      = anchor["bbox"]
        # Extend x1 to cover full text width (two-column fix)
        wide_bbox = [b[0], b[1], round(pw - 50, 1), b[3]]
        chunks.append({
            "text":        text,
            "page":        anchor["page"],
            "page_height": anchor["page_height"],
            "page_width":  anchor["page_width"],
            "bbox":        wide_bbox,
        })

    for line in lines:
        buf_lines.append(line)
        word_count = sum(len(l["text"].split()) for l in buf_lines)

        if word_count >= chunk_words:
            flush()
            # Keep last overlap_lines as context for next chunk
            overlap    = buf_lines[-overlap_lines:]
            buf_lines  = list(overlap)
            anchor_idx = 0   # all lines in new buffer are overlap initially
            # The next NEW line added will update anchor_idx
            # We set anchor_idx to len(overlap) so first new line becomes anchor
            anchor_idx = len(overlap)

    # Flush remaining
    if buf_lines:
        anchor_idx = min(anchor_idx, len(buf_lines) - 1)
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

    lines = _extract_lines(doc)
    doc.close()

    if not lines:
        raise ValueError("Could not extract text from PDF.")

    chunks = _chunk_lines(lines)
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
