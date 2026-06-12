import uuid
import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
from core.config import get_settings
from core.llm import llm_call

_settings     = get_settings()
_chat_history: dict[str, list[dict]] = {}
_chroma_client = None
_embed_fn      = None


def _get_chroma_client():
    global _chroma_client, _embed_fn
    if _chroma_client is None:
        _embed_fn      = ONNXMiniLM_L6_V2()
        _chroma_client = chromadb.PersistentClient(path=_settings.chroma_path)
    return _chroma_client


def _get_or_create_collection(session_id: str):
    return _get_chroma_client().get_or_create_collection(
        name=f"pdf_{session_id}",
        embedding_function=_embed_fn,
    )


def _chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    words  = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += chunk_size - overlap
    return chunks


async def ingest_pdf(file_bytes: bytes, filename: str) -> str:
    import pymupdf
    session_id = str(uuid.uuid4())

    doc       = pymupdf.open(stream=file_bytes, filetype="pdf")
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    chunks = _chunk_text(full_text)
    if not chunks:
        raise ValueError("Could not extract text from PDF.")

    col = _get_or_create_collection(session_id)
    col.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"source": filename, "chunk_index": i} for i in range(len(chunks))],
    )

    _chat_history[session_id] = []
    return session_id


async def chat_with_pdf(session_id: str, question: str) -> dict:
    if session_id not in _chat_history:
        raise ValueError(f"Session '{session_id}' not found. Please upload a PDF first.")

    col       = _get_or_create_collection(session_id)
    results   = col.query(query_texts=[question], n_results=5)
    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    context   = "\n\n---\n\n".join(docs)
    sources   = [f"chunk {m['chunk_index']}" for m in metadatas]
    history   = _chat_history[session_id]

    system = (
        "You are TriField AI, an expert research assistant specialising in "
        "aerospace structures, advanced materials, and textile engineering. "
        "Answer using ONLY the context from the PDF. Cite the relevant section. "
        "If the answer is not in the context, say so clearly.\n\n"
        f"CONTEXT FROM PDF:\n{context}"
    )

    # Uses Anthropic first, falls back to Groq automatically
    answer = await llm_call(
        system=system,
        messages=history + [{"role": "user", "content": question}],
        max_tokens=1024,
    )

    _chat_history[session_id].append({"role": "user",      "content": question})
    _chat_history[session_id].append({"role": "assistant", "content": answer})

    return {"answer": answer, "sources": sources, "history": _chat_history[session_id]}


async def extract_properties(session_id: str) -> list[dict]:
    if session_id not in _chat_history:
        raise ValueError(f"Session '{session_id}' not found.")

    col     = _get_or_create_collection(session_id)
    results = col.query(
        query_texts=["mechanical properties tensile strength modulus fibre volume fraction"],
        n_results=8,
    )
    context = "\n\n---\n\n".join(results["documents"][0])

    prompt = (
        "Extract ALL material and mechanical properties from the text below. "
        "Return a JSON array where each item has: "
        "property_name, value, unit, test_standard (if mentioned), page_ref (if mentioned). "
        "Examples: tensile strength, Young's modulus, flexural strength, "
        "fibre volume fraction, void content, density, impact strength.\n\n"
        f"TEXT:\n{context}"
    )

    import json
    raw = await llm_call(
        system="You are a materials science data extractor.",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        prefer_json=True,
    )
    raw = raw.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return []
