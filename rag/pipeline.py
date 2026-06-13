"""
RAG pipeline — Phase 4+5 architecture.

Phase 4: Multi-column reading order correction
  - Detects two-column layouts using block x-position analysis
  - Sorts blocks: left column top→bottom, then right column top→bottom
  - Prevents sentence/chunk corruption in IEEE, Elsevier, Springer papers

Phase 5: Multi-bounding-box sentence highlighting
  - Each sentence stores ALL line bboxes (not just the first line)
  - Frontend renders multiple highlight rectangles per citation
  - SciSpace-level highlight precision for multi-line sentences

Ingestion:
  PDF → Column-ordered Lines → Sentences (multi-bbox) → Chunks
  Both sentences and chunks stored in Qdrant (Phase 4+5 schema).

Citation mapping:
  LLM answer → per-claim sentence matching → ALL sentence bboxes → frontend.
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

_REF_LINE_RE = re.compile(
    r'^(\d{1,3}[\.\ )]\s)'
    r'|^\[\d{1,3}\]\s'
    r'|https?://'
    r'|doi\.org/'
    r'|10\.\d{4,}/',
    re.IGNORECASE
)


def _is_body_text(text: str, bbox: list, page_height: float, page_width: float) -> bool:
    """
    Return True only if this line is likely body text suitable for highlighting.
    Rejects reference list entries, figure areas, headers/footers, short/narrow lines.
    """
    if not text or len(text) < 15:
        return False
    if _REF_LINE_RE.match(text.strip()):
        return False
    if not bbox or len(bbox) < 4:
        return False

    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0

    if h <= 0 or w <= 0:
        return False
    if page_height > 0:
        if y0 < page_height * 0.08:
            return False
        if y1 > page_height * 0.92:
            return False
    if h >= 30:
        return False
    if w < 60:
        return False
    if (w / h) < 3:
        return False

    return True


# ── Phase 4: Multi-column reading order ───────────────────────────────────

def _detect_column_boundary(blocks: list[dict], page_width: float) -> float | None:
    """
    Detect the x-coordinate gap that separates two columns.

    Strategy: collect all block x0 positions. If there is a clear bimodal
    distribution (left cluster and right cluster), the midpoint between the
    two clusters is returned as the column boundary. Returns None for
    single-column pages.

    IEEE/Elsevier papers typically have a 10-15pt gutter between columns.
    We detect this by finding the largest gap in x0 positions that is
    between 20% and 70% of the page width (ruling out margins).
    """
    if not blocks or page_width <= 0:
        return None

    x0_positions = sorted({
        round(b["bbox"][0], 1)
        for b in blocks
        if b.get("type") == 0 and b.get("bbox")
    })

    if len(x0_positions) < 2:
        return None

    min_x = page_width * 0.20
    max_x = page_width * 0.70

    best_gap   = 0.0
    best_mid   = None

    for i in range(len(x0_positions) - 1):
        gap = x0_positions[i + 1] - x0_positions[i]
        mid = (x0_positions[i] + x0_positions[i + 1]) / 2
        if min_x <= mid <= max_x and gap > best_gap:
            best_gap = gap
            best_mid = mid

    # Require a meaningful gap (at least 10pt) to call it two-column
    return best_mid if best_gap >= 10 else None


def _sort_blocks_reading_order(
    blocks: list[dict],
    col_boundary: float | None,
    page_height: float,
) -> list[dict]:
    """
    Sort blocks into correct reading order.

    Single-column: top-to-bottom by y0.
    Two-column:    left column top→bottom, then right column top→bottom.

    Within each column, sort by y0. Ties broken by x0.
    """
    text_blocks = [b for b in blocks if b.get("type") == 0 and b.get("bbox")]

    if col_boundary is None:
        return sorted(text_blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))

    left  = [b for b in text_blocks if b["bbox"][0] < col_boundary]
    right = [b for b in text_blocks if b["bbox"][0] >= col_boundary]

    left_sorted  = sorted(left,  key=lambda b: (b["bbox"][1], b["bbox"][0]))
    right_sorted = sorted(right, key=lambda b: (b["bbox"][1], b["bbox"][0]))

    return left_sorted + right_sorted


# ── Phase 4+5: Line extraction with column-correct order ──────────────────

def _extract_lines(doc) -> list[dict]:
    """
    Extract body-text lines from PDF in correct reading order.

    Phase 4: Detects and corrects multi-column layout per page so that
             lines from the left column always precede the right column.
    Phase 5: Each line carries its own bbox — used later to build multi-bbox
             sentence objects.

    Coordinate system: TOP-LEFT origin, y increases downward.
    """
    lines = []

    for page_num, page in enumerate(doc, start=1):
        pr          = page.rect
        page_height = round(pr.height, 1)
        page_width  = round(pr.width,  1)

        raw_blocks     = page.get_text("dict").get("blocks", [])
        col_boundary   = _detect_column_boundary(raw_blocks, page_width)
        ordered_blocks = _sort_blocks_reading_order(raw_blocks, col_boundary, page_height)

        for block in ordered_blocks:
            for line in block.get("lines", []):
                text = " ".join(
                    s.get("text", "") for s in line.get("spans", [])
                ).strip()
                lb   = line.get("bbox", block.get("bbox", [0, 0, 0, 0]))
                bbox = [round(lb[0], 1), round(lb[1], 1),
                        round(lb[2], 1), round(lb[3], 1)]

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


# ── Phase 5: Multi-bbox sentence reconstruction ────────────────────────────

def _lines_to_sentences(lines: list[dict]) -> list[dict]:
    """
    Reconstruct sentences from body-text lines.

    Phase 5 upgrade: Instead of storing only the FIRST line's bbox,
    every sentence now stores a 'bboxes' list containing ALL line bboxes
    that make up that sentence. The frontend renders one highlight
    rectangle per bbox, giving SciSpace-level multi-line highlighting.

    Schema per sentence:
    {
        "text":        str,
        "page":        int,
        "page_height": float,
        "page_width":  float,
        "bbox":        list[float],   # first-line bbox (backward-compat fallback)
        "bboxes":      list[list[float]]  # ALL line bboxes ← Phase 5
    }
    """
    sentences: list[dict] = []
    buf_text:  list[str]  = []
    buf_bboxes: list[list] = []
    anchor: dict | None   = None

    for line in lines:
        # Page break — flush whatever we have
        if anchor and line["page"] != anchor["page"]:
            if buf_text:
                sentences.append({
                    **anchor,
                    "text":   " ".join(buf_text).strip(),
                    "bboxes": buf_bboxes,
                })
            buf_text = []; buf_bboxes = []; anchor = None

        if anchor is None:
            anchor = {k: line[k] for k in ("page", "page_height", "page_width", "bbox")}

        buf_text.append(line["text"])
        buf_bboxes.append(line["bbox"])
        combined = " ".join(buf_text)

        # Flush on sentence-ending punctuation (min 6 words)
        if re.search(r'[.!?]\s*$', combined) and len(combined.split()) >= 6:
            sentences.append({
                **anchor,
                "text":   combined.strip(),
                "bboxes": list(buf_bboxes),
            })
            buf_text = []; buf_bboxes = []; anchor = None

    if buf_text and anchor:
        sentences.append({
            **anchor,
            "text":   " ".join(buf_text).strip(),
            "bboxes": list(buf_bboxes),
        })

    return [s for s in sentences if len(s["text"].split()) >= 5]


def _sentences_to_chunks(
    sentences:     list[dict],
    chunk_words:   int = 300,
    overlap_sents: int = 2,
) -> list[dict]:
    """
    Group sentences into chunks for embedding.
    Chunks store sentence_indices so citation mapper can look up exact sentences.
    """
    chunks:    list[dict] = []
    buf:       list[int]  = []
    new_start: int        = 0

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
            "bboxes":           anchor.get("bboxes", [anchor["bbox"]]),
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
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def find_best_sentence(
    claim:         str,
    chunk_result:  dict,
    all_sentences: list[dict],
) -> dict | None:
    """
    Find the sentence whose head best matches the claim head.

    Returns the full sentence dict (including 'bboxes' list for Phase 5
    multi-highlight). Falls back to None if no match exceeds threshold.
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
    n     = store.ingest_phase45(session_id, chunks, sentences, filename)

    if get_settings().database_url:
        try:
            from db.repositories import insert_pdf_session, insert_event
            await insert_pdf_session(
                session_id=session_id, filename=filename,
                file_size_mb=file_size_mb, chunk_count=n, latency_ms=0,
            )
            await insert_event(
                event_type="pdf_upload",
                meta={
                    "filename":  filename,
                    "chunks":    n,
                    "sentences": len(sentences),
                    "pages":     total_pages,
                    "session_id": session_id,
                    "phase":     "4+5",
                },
            )
        except Exception as e:
            logger.warning(f"DB session persist failed (non-fatal): {e}")

    _chat_history_mem[session_id] = []
    logger.info(
        f"PDF ingested (Phase 4+5): {filename} → "
        f"{n} chunks, {len(sentences)} sentences, {total_pages} pages"
    )
    return session_id


# ── Citation source builder ────────────────────────────────────────────────

def _build_sources(results: list[dict], answer: str, sentences: list[dict]) -> list[dict]:
    """
    Map each [N] citation in the LLM answer to an exact sentence.

    Phase 5: Returns 'bboxes' list (all line bboxes) so the frontend can
    render multiple highlight rectangles — one per physical line of text.
    Also returns 'bbox' (first line) as a backward-compatible fallback.
    """
    ref_pat = re.compile(r'\[(\d+)\]')

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
        bbox    = r.get("bbox")
        bboxes  = r.get("bboxes") or ([bbox] if bbox else [])

        claim = ref_to_claim.get(i, "")
        if claim and sentences:
            best = find_best_sentence(claim, r, sentences)
            if best:
                bbox    = best["bbox"]
                bboxes  = best.get("bboxes") or [bbox]
                snippet = best["text"]
                page    = best["page"]

        sources.append({
            "ref":         i,
            "page":        page,
            "page_height": r.get("page_height"),
            "page_width":  r.get("page_width"),
            "bbox":        bbox,           # first-line fallback
            "bboxes":      bboxes,         # ← Phase 5: all line bboxes
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
        f'[{i}] (page {r.get("page", 1)})\n"{r.get("text", "")}"'
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
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []
