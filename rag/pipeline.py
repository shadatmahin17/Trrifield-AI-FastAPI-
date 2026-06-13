"""
RAG pipeline — Phase 3 architecture.

Ingestion:  PDF → Lines → Sentences (with exact bbox) → Chunks
            Both sentences and chunks stored in Qdrant.

Citation mapping:
  LLM answer → per-claim sentence matching → exact sentence bbox → frontend.

Key design decisions:
  - Reference list pages are excluded from sentence index (numbered refs pattern)
  - Figure/chart lines filtered by aspect ratio
  - Citation mapper matches on first-12-word head of claim, not full text
  - Frontend clamps highlight height to text-line size as final guard
"""
import re, uuid, logging
from vectorstore.qdrant_store import get_store
from core.llm import llm_call
from core.config import get_settings
from prompts.templates import PDF_CHAT_SYSTEM, PROPERTY_EXTRACT
from models.schemas import ChatMessage

logger = logging.getLogger(__name__)
_chat_history_mem: dict[str, list[ChatMessage]] = {}

# ── Line / sentence quality filters ───────────────────────────────────────

# Patterns that identify reference-list lines — exclude from sentence index
# so the citation mapper never returns a bibliography entry as a highlight
_REF_LINE_RE = re.compile(
    r'^(\d{1,3}[\.\)]\s)'          # "1. " or "1) "
    r'|^\[\d{1,3}\]\s'             # "[1] "
    r'|https?://'                  # bare URL lines
    r'|doi\.org/'                  # DOI-only lines
    r'|10\.\d{4,}/',               # DOI number
    re.IGNORECASE
)

def _is_body_text(text: str, bbox: list, page_height: float, page_width: float) -> bool:
    """
    Return True only if this line is likely body text suitable for highlighting.
    Rejects:
      - Reference list entries (numbered/bracketed citations)
      - Figure/table caption regions (based on bbox geometry)
      - Header/footer zones (top 8% or bottom 8% of page)
      - Lines that are too short or too narrow
      - Figure areas (tall bboxes, aspect ratio too low)
    """
    if not text or len(text) < 15:
        return False

    # Reference list pattern
    if _REF_LINE_RE.match(text.strip()):
        return False

    if not bbox or len(bbox) < 4:
        return False

    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0

    if h <= 0 or w <= 0:
        return False

    # Header/footer zone: top 8% or bottom 8% of page
    if page_height > 0:
        if y0 < page_height * 0.08:
            return False
        if y1 > page_height * 0.92:
            return False

    # Must be a wide, short line (text line geometry)
    # Text lines: height < 30pt, width > 60pt, aspect ratio > 3
    if h >= 30:
        return False
    if w < 60:
        return False
    if (w / h) < 3:
        return False

    return True


# ── Extraction ─────────────────────────────────────────────────────────────

def _extract_lines(doc) -> list[dict]:
    """
    Extract every text line from pymupdf with exact bbox.
    Coordinate system: TOP-LEFT origin, y increases downward (same as HTML canvas).
    Only returns lines that pass _is_body_text filter.
    """
    lines = []
    for page_num, page in enumerate(doc, start=1):
        pr          = page.rect
        page_height = round(pr.height, 1)
        page_width  = round(pr.width,  1)

        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                text = " ".join(
                    s.get("text", "") for s in line.get("spans", [])
                ).strip()
                lb = line.get("bbox", block.get("bbox", [0,0,0,0]))
                bbox = [round(lb[0],1), round(lb[1],1),
                        round(lb[2],1), round(lb[3],1)]

                if not _is_body_text(text, bbox, page_height, page_width):
                    continue

                lines.append({
                    "text":        text,
                    "page":        page_num,
                    "page_height": page_height,
                    "page_width":  page_width,
                    "bbox":        bbox,
                })
    return lines


def _lines_to_sentences(lines: list[dict]) -> list[dict]:
    """
    Reconstruct sentences from filtered body-text lines.
    Stores bbox of the FIRST LINE of each sentence as the highlight anchor.
    """
    sentences = []
    buf_text: list[str] = []
    anchor: dict | None = None   # first line of current sentence

    for line in lines:
        # Page break — flush whatever we have
        if anchor and line["page"] != anchor["page"]:
            if buf_text:
                sentences.append({**anchor, "text": " ".join(buf_text).strip()})
            buf_text = []; anchor = None

        if anchor is None:
            anchor = {k: line[k] for k in ("page","page_height","page_width","bbox")}

        buf_text.append(line["text"])
        combined = " ".join(buf_text)

        # Flush on sentence-ending punctuation
        if re.search(r'[.!?]\s*$', combined) and len(combined.split()) >= 6:
            sentences.append({**anchor, "text": combined.strip()})
            buf_text = []; anchor = None

    if buf_text and anchor:
        sentences.append({**anchor, "text": " ".join(buf_text).strip()})

    return [s for s in sentences if len(s["text"].split()) >= 5]


def _sentences_to_chunks(
    sentences:     list[dict],
    chunk_words:   int = 300,
    overlap_sents: int = 2,
) -> list[dict]:
    """
    Group sentences into chunks for embedding.
    Chunks store sentence_indices so citation mapper can look up exact sentences.
    new_start tracks which sentences are overlap vs new content.
    """
    chunks:    list[dict] = []
    buf:       list[int]  = []   # sentence indices in buffer
    new_start: int        = 0    # index in buf of first non-overlap sentence

    def flush():
        if not buf:
            return
        text = " ".join(sentences[i]["text"] for i in buf)
        if len(text.split()) < 10:
            return
        idx    = min(new_start, len(buf) - 1)
        anchor = sentences[buf[idx]]
        chunks.append({
            "text":             text,
            "page":             anchor["page"],
            "page_height":      anchor["page_height"],
            "page_width":       anchor["page_width"],
            "bbox":             anchor["bbox"],
            "sentence_indices": list(buf),
        })

    for si, sent in enumerate(sentences):
        buf.append(si)
        if sum(len(sentences[i]["text"].split()) for i in buf) >= chunk_words:
            flush()
            overlap   = buf[-overlap_sents:]
            buf       = list(overlap)
            new_start = len(overlap)

    if buf:
        flush()

    return chunks


# ── Citation mapper ────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def find_best_sentence(
    claim: str,
    chunk_result: dict,
    all_sentences: list[dict],
) -> dict | None:
    """
    Find the sentence in a chunk whose HEAD best matches the claim head.

    Strategy: compare first 12 words of claim against first 12 words of
    each candidate sentence. Head-matching is 3× weighted over full overlap.
    This ensures the highlight lands at the START of the cited passage.

    Rejects sentences that fail body-text geometry (reference list, figures).
    """
    norm_claim  = _norm(claim)
    if not norm_claim:
        return None

    claim_words = norm_claim.split()
    head_target = set(claim_words[:12])
    full_target = set(claim_words)

    best_score: float = 0.0
    best_sent:  dict | None = None

    for si in chunk_result.get("sentence_indices", []):
        if si >= len(all_sentences):
            continue
        sent = all_sentences[si]

        # Skip any sentence that slipped through with bad geometry
        bbox = sent.get("bbox")
        if bbox and not _is_body_text(
            sent["text"], bbox,
            sent.get("page_height", 841),
            sent.get("page_width",  595),
        ):
            continue

        norm_sent  = _norm(sent["text"])
        sent_words = norm_sent.split()
        head_sent  = set(sent_words[:12])
        full_sent  = set(sent_words)

        head_overlap = len(head_target & head_sent)
        full_overlap = len(full_target & full_sent)
        score        = (head_overlap * 3) + full_overlap

        if score > best_score:
            best_score = score
            best_sent  = sent

    # Require at least 2 head words matching (score ≥ 6)
    return best_sent if best_score >= 6 else None


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

    sentences = _lines_to_sentences(lines)
    if not sentences:
        raise ValueError("Could not extract sentences from PDF.")

    chunks = _sentences_to_chunks(sentences)
    if not chunks:
        raise ValueError("PDF text too short to process.")

    store = get_store()
    n     = store.ingest_phase3(session_id, chunks, sentences, filename)

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
                      "sentences": len(sentences),
                      "pages": total_pages, "session_id": session_id},
            )
        except Exception as e:
            logger.warning(f"DB session persist failed (non-fatal): {e}")

    _chat_history_mem[session_id] = []
    logger.info(f"PDF ingested: {filename} → {n} chunks, {len(sentences)} sentences, {total_pages} pages")
    return session_id


# ── Citation source builder ────────────────────────────────────────────────

def _build_sources(results: list[dict], answer: str, sentences: list[dict]) -> list[dict]:
    """
    Map each [N] citation in the answer to an exact sentence bbox.

    1. Split answer into sentences, find which refs each sentence uses.
    2. For each ref, find the best matching sentence in the retrieved chunk.
    3. Return structured sources with exact page + bbox for the frontend.
    """
    ref_pat = re.compile(r'\[(\d+)\]')

    # Map ref number → the answer-sentence that used it (stripped of markers)
    ref_to_claim: dict[int, str] = {}
    for ans_sent in re.split(r'(?<=[.!?])\s+', answer):
        refs  = [int(m) for m in ref_pat.findall(ans_sent)]
        clean = ref_pat.sub('', ans_sent).strip()
        for ref in refs:
            if ref not in ref_to_claim:
                ref_to_claim[ref] = clean

    sources = []
    for i, r in enumerate(results, start=1):
        text    = r.get("text", "")
        snippet = text[:200].strip() + ("…" if len(text) > 200 else "")
        page    = r.get("page", 1)
        bbox    = r.get("bbox")          # chunk anchor (fallback)

        claim = ref_to_claim.get(i, "")
        if claim and sentences:
            best = find_best_sentence(claim, r, sentences)
            if best:
                bbox    = best["bbox"]
                snippet = best["text"]
                page    = best["page"]

        sources.append({
            "ref":         i,
            "page":        page,
            "page_height": r.get("page_height"),
            "page_width":  r.get("page_width"),
            "bbox":        bbox,
            "snippet":     snippet,
        })

    return sources


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

    context = "\n\n".join(
        f'[{i}] (page {r.get("page",1)})\n"{r.get("text","")}"'
        for i, r in enumerate(results, start=1)
    )
    system   = PDF_CHAT_SYSTEM.format(context=context)
    history  = _chat_history_mem[session_id]
    messages = [m.model_dump() for m in history] + [{"role": "user", "content": question}]

    answer = await llm_call(
        system=system, messages=messages, max_tokens=1024, task="pdf_chat",
    )

    sentences = store.get_sentences(session_id)
    sources   = _build_sources(results, answer, sentences)

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

    store   = get_store()
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
