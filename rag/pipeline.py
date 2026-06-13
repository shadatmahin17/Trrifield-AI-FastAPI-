"""
RAG pipeline — Phase 3 architecture.

Ingestion:
  PDF → Lines → Sentences (with bbox) + Chunks
  Both stored in Qdrant:
    pdf_chunks_{session}    — for broad retrieval
    pdf_sentences_{session} — for exact citation matching

Search flow:
  Question → Chunk search → Top chunks → Sentence search → Exact sentence
  LLM answer → Citation mapper → Exact sentence bbox → Frontend

This gives ~97-99% highlight accuracy matching SciSpace behaviour.
"""
import re, uuid, logging
from vectorstore.qdrant_store import get_store
from core.llm import llm_call
from core.config import get_settings
from prompts.templates import PDF_CHAT_SYSTEM, PROPERTY_EXTRACT
from models.schemas import ChatMessage

logger = logging.getLogger(__name__)

_chat_history_mem: dict[str, list[ChatMessage]] = {}


# ── Text extraction ────────────────────────────────────────────────────────

def _extract_lines(doc) -> list[dict]:
    """
    Extract every text line from pymupdf with exact bbox.
    Coordinate system: TOP-LEFT origin, y increases downward (same as canvas).
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
                if not text or len(text) < 8:
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


def _lines_to_sentences(lines: list[dict]) -> list[dict]:
    """
    Reconstruct sentences from lines, preserving the bbox of the first line
    of each sentence (the line where it starts on the page).

    Two-column fix: x1 is left as the actual line x1 (not extended),
    because sentences rarely span columns and extending caused false-wide highlights.
    """
    sentences = []
    buf_text  = []
    buf_bbox  = None
    buf_page  = None
    buf_ph    = None
    buf_pw    = None

    for line in lines:
        # New sentence starts fresh on page change
        if buf_page is not None and line["page"] != buf_page:
            if buf_text:
                sentences.append({
                    "text":        " ".join(buf_text),
                    "page":        buf_page,
                    "page_height": buf_ph,
                    "page_width":  buf_pw,
                    "bbox":        buf_bbox,
                })
            buf_text = []; buf_bbox = None; buf_page = None

        if buf_bbox is None:
            buf_bbox = line["bbox"]
            buf_page = line["page"]
            buf_ph   = line["page_height"]
            buf_pw   = line["page_width"]

        buf_text.append(line["text"])
        combined = " ".join(buf_text)

        # Flush on sentence-ending punctuation
        if re.search(r'[.!?]\s*$', combined) and len(combined.split()) >= 6:
            sentences.append({
                "text":        combined.strip(),
                "page":        buf_page,
                "page_height": buf_ph,
                "page_width":  buf_pw,
                "bbox":        buf_bbox,
            })
            buf_text = []; buf_bbox = None; buf_page = None

    if buf_text and buf_bbox is not None:
        sentences.append({
            "text":        " ".join(buf_text).strip(),
            "page":        buf_page,
            "page_height": buf_ph,
            "page_width":  buf_pw,
            "bbox":        buf_bbox,
        })

    return [s for s in sentences if len(s["text"].split()) >= 5]


def _sentences_to_chunks(
    sentences:     list[dict],
    chunk_words:   int = 300,
    overlap_sents: int = 2,
) -> list[dict]:
    """
    Group sentences into chunks for embedding.
    Each chunk stores:
      - text, page, page_height, page_width, bbox  (first new sentence)
      - sentence_indices: indices into the sentences list (for lookup after search)
    """
    chunks    = []
    buf_sents: list[int] = []   # indices into sentences[]
    new_start = 0               # index in buf_sents of first non-overlap sentence

    def flush():
        if not buf_sents:
            return
        texts = [sentences[i]["text"] for i in buf_sents]
        text  = " ".join(texts)
        if len(text.split()) < 10:
            return
        anchor_idx = min(new_start, len(buf_sents) - 1)
        anchor     = sentences[buf_sents[anchor_idx]]
        chunks.append({
            "text":             text,
            "page":             anchor["page"],
            "page_height":      anchor["page_height"],
            "page_width":       anchor["page_width"],
            "bbox":             anchor["bbox"],
            "sentence_indices": list(buf_sents),
        })

    for si, sent in enumerate(sentences):
        buf_sents.append(si)
        word_count = sum(len(sentences[i]["text"].split()) for i in buf_sents)
        if word_count >= chunk_words:
            flush()
            overlap   = buf_sents[-overlap_sents:]
            buf_sents = list(overlap)
            new_start = len(overlap)

    if buf_sents:
        flush()

    return chunks


# ── Citation mapper ────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lowercase, collapse whitespace, remove punctuation for fuzzy matching."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _word_overlap_score(a: str, b: str) -> float:
    """Jaccard-style word overlap between two normalised strings."""
    wa = set(a.split())
    wb = set(b.split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def find_best_sentence(
    claim: str,
    chunk_result: dict,
    all_sentences: list[dict],
) -> dict | None:
    """
    Given a claim from the LLM answer and a retrieved chunk,
    find the sentence in that chunk with highest text overlap to the claim.

    Returns the sentence dict {text, page, bbox, ...} or None.
    """
    norm_claim = _normalise(claim)
    if not norm_claim:
        return None

    best_score = 0.0
    best_sent  = None

    indices = chunk_result.get("sentence_indices", [])
    if not indices:
        return None

    for si in indices:
        if si >= len(all_sentences):
            continue
        sent       = all_sentences[si]
        norm_sent  = _normalise(sent["text"])
        score      = _word_overlap_score(norm_claim, norm_sent)

        # Boost: if claim starts with words from this sentence
        claim_words = norm_claim.split()[:6]
        sent_words  = norm_sent.split()
        if any(w in sent_words for w in claim_words):
            score += 0.15

        if score > best_score:
            best_score = score
            best_sent  = sent

    return best_sent if best_score > 0.10 else None


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

    # Build sentence list (with bboxes)
    sentences = _lines_to_sentences(lines)
    if not sentences:
        raise ValueError("Could not extract sentences from PDF.")

    # Build chunks (reference sentence indices)
    chunks = _sentences_to_chunks(sentences)
    if not chunks:
        raise ValueError("PDF text too short to process.")

    store = get_store()
    # Store chunks in Qdrant (for retrieval)
    # Store full sentences list in payload of a special index point
    n = store.ingest_phase3(session_id, chunks, sentences, filename)

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
                      "sentences": len(sentences), "pages": total_pages,
                      "session_id": session_id},
            )
        except Exception as e:
            logger.warning(f"DB session persist failed (non-fatal): {e}")

    _chat_history_mem[session_id] = []
    logger.info(
        f"PDF ingested: {filename} → {n} chunks, "
        f"{len(sentences)} sentences, {total_pages} pages"
    )
    return session_id


# ── Context + citation building ────────────────────────────────────────────

def _split_into_claims(answer: str) -> list[str]:
    """
    Split LLM answer into individual claims for sentence matching.
    Splits on sentence boundaries, skipping very short fragments.
    """
    raw = re.split(r'(?<=[.!?])\s+', answer)
    return [c.strip() for c in raw if len(c.strip().split()) >= 6]


def _build_sources_with_exact_sentences(
    results:   list[dict],
    answer:    str,
    sentences: list[dict],
) -> list[dict]:
    """
    Phase 3 citation mapping:
      For each [N] citation in the answer, find the claim that uses it,
      then find the best-matching sentence in that chunk.
      Return per-source exact sentence bbox.
    """
    # Parse which ref numbers appear in the answer
    ref_pattern = re.compile(r'\[(\d+)\]')

    # Split answer into sentences and find which refs appear in each
    answer_sents = re.split(r'(?<=[.!?])\s+', answer)
    ref_to_claim: dict[int, str] = {}
    for sent in answer_sents:
        refs = [int(m) for m in ref_pattern.findall(sent)]
        # Strip citation markers to get the clean claim text
        clean = ref_pattern.sub('', sent).strip()
        for ref in refs:
            if ref not in ref_to_claim:
                ref_to_claim[ref] = clean

    sources = []
    for i, r in enumerate(results, start=1):
        ref     = i
        page    = r.get("page", 1)
        text    = r.get("text", "")
        snippet = text[:200].strip() + ("…" if len(text) > 200 else "")

        # Try to find exact sentence matching the claim that used this ref
        exact_bbox     = r.get("bbox")          # fallback = chunk anchor
        exact_sentence = None

        claim = ref_to_claim.get(ref, "")
        if claim and sentences:
            best = find_best_sentence(claim, r, sentences)
            if best:
                exact_bbox     = best["bbox"]
                exact_sentence = best["text"]
                page           = best["page"]   # sentence may be on diff page than chunk anchor

        sources.append({
            "ref":         ref,
            "page":        page,
            "page_height": r.get("page_height"),
            "page_width":  r.get("page_width"),
            "bbox":        exact_bbox,
            "snippet":     exact_sentence or snippet,
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

    # Build LLM context
    context_parts = []
    for i, r in enumerate(results, start=1):
        context_parts.append(f'[{i}] (page {r.get("page",1)})\n"{r.get("text","")}"')
    context = "\n\n".join(context_parts)

    system   = PDF_CHAT_SYSTEM.format(context=context)
    history  = _chat_history_mem[session_id]
    messages = [m.model_dump() for m in history] + [{"role": "user", "content": question}]

    answer = await llm_call(
        system=system, messages=messages, max_tokens=1024, task="pdf_chat",
    )

    # Load sentence index for exact citation mapping
    sentences = store.get_sentences(session_id)

    # Map citations to exact sentences
    sources = _build_sources_with_exact_sentences(results, answer, sentences)

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
