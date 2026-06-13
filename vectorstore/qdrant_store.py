"""
Qdrant vector store — Phase 4+5 architecture.

Phase 4: Column-order-corrected ingestion (handled upstream in pipeline.py).
Phase 5: Multi-bbox sentence storage.
  - Each sentence now stores 'bboxes': list[list[float]] — all line bboxes.
  - 'bbox' (single) is retained as a backward-compatible fallback.
  - Sentences stored as individual Qdrant points (not a single JSON blob)
    for scalability with large papers (3000+ sentences).

Per session collections:
  pdf_chunks_{session}    — chunk embeddings for broad semantic retrieval
  pdf_sentences_{session} — individual sentence points with multi-bbox payload
"""
import json, logging
from typing import Optional
from core.config import get_settings

logger = logging.getLogger(__name__)

CHUNK_PREFIX    = "pdf_chunks_"
SENTENCE_PREFIX = "pdf_sentences_"
VECTOR_SIZE     = 384


def _get_embedding_model():
    try:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        return ONNXMiniLM_L6_V2()
    except Exception as e:
        logger.error(f"Embedding model load failed: {e}")
        raise


def _get_qdrant_client():
    from qdrant_client import QdrantClient
    s = get_settings()
    if s.qdrant_url and s.qdrant_api_key:
        return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
    return QdrantClient(path=s.qdrant_local_path)


class QdrantPDFStore:

    def __init__(self):
        self._client   = None
        self._embed_fn = None

    def _get_client(self):
        if self._client is None or self._embed_fn is None:
            self._client   = _get_qdrant_client()
            self._embed_fn = _get_embedding_model()
        return self._client

    _client_ = _get_client

    def _chunk_col(self, session_id: str) -> str:
        return f"{CHUNK_PREFIX}{session_id}"

    def _sentence_col(self, session_id: str) -> str:
        return f"{SENTENCE_PREFIX}{session_id}"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        result = self._embed_fn(texts)
        return result if isinstance(result[0], list) else [list(r) for r in result]

    # ── Legacy ingest (no position) ───────────────────────────────────────
    def ingest(self, session_id: str, chunks: list[str], filename: str) -> int:
        positioned = [{
            "text": c, "page": 1, "page_height": None,
            "page_width": None, "bbox": None, "bboxes": [],
            "sentence_indices": [],
        } for c in chunks]
        return self.ingest_phase45(session_id, positioned, [], filename)

    # ── Phase 2 ingest (position-aware, no sentence index) ────────────────
    def ingest_with_positions(self, session_id: str, chunks: list[dict], filename: str) -> int:
        return self.ingest_phase45(session_id, chunks, [], filename)

    # ── Phase 3 ingest (backward compat — routes to Phase 4+5) ───────────
    def ingest_phase3(
        self,
        session_id: str,
        chunks:     list[dict],
        sentences:  list[dict],
        filename:   str,
    ) -> int:
        return self.ingest_phase45(session_id, chunks, sentences, filename)

    # ── Phase 4+5 ingest ──────────────────────────────────────────────────
    def ingest_phase45(
        self,
        session_id: str,
        chunks:     list[dict],
        sentences:  list[dict],
        filename:   str,
    ) -> int:
        """
        Ingest chunks and sentences into Qdrant.

        Phase 5 change: Sentences are stored as individual points
        (one point per sentence) instead of a single JSON blob.
        Each sentence point carries:
          - bbox:   list[float]        — first-line bbox (fallback)
          - bboxes: list[list[float]]  — ALL line bboxes (Phase 5)
          - sentence_index: int        — for fast lookup from chunks

        This replaces the single-payload approach and scales correctly
        to papers with 5000+ sentences without memory overhead.
        """
        from qdrant_client.models import (
            VectorParams, Distance, PointStruct,
        )

        client    = self._get_client()
        chunk_col = self._chunk_col(session_id)
        sent_col  = self._sentence_col(session_id)

        # ── Chunk collection ──────────────────────────────────────────────
        client.recreate_collection(
            collection_name=chunk_col,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

        texts = [c["text"] for c in chunks]
        all_embeddings = []
        for i in range(0, len(texts), 32):
            all_embeddings.extend(self._embed(texts[i:i + 32]))

        chunk_points = [
            PointStruct(
                id=i,
                vector=all_embeddings[i],
                payload={
                    "text":             chunks[i]["text"],
                    "chunk_index":      i,
                    "page":             chunks[i].get("page", 1),
                    "page_height":      chunks[i].get("page_height"),
                    "page_width":       chunks[i].get("page_width"),
                    "bbox":             chunks[i].get("bbox"),
                    "bboxes":           chunks[i].get("bboxes") or [],
                    "sentence_indices": chunks[i].get("sentence_indices", []),
                    "source":           filename,
                },
            )
            for i in range(len(chunks))
        ]
        client.upsert(collection_name=chunk_col, points=chunk_points)

        # ── Sentence collection (Phase 5: individual points) ───────────────
        if sentences:
            client.recreate_collection(
                collection_name=sent_col,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

            # Use a tiny deterministic dummy vector per sentence.
            # We never do ANN search on sentences — we look them up by
            # sentence_index from chunk payload. The vector is required by
            # Qdrant but is never used for retrieval.
            zero_vec = [0.0] * VECTOR_SIZE

            sent_points = []
            for si, sent in enumerate(sentences):
                bbox   = sent.get("bbox")
                bboxes = sent.get("bboxes") or ([bbox] if bbox else [])
                sent_points.append(PointStruct(
                    id=si,
                    vector=zero_vec,
                    payload={
                        "sentence_index": si,
                        "text":           sent["text"],
                        "page":           sent.get("page", 1),
                        "page_height":    sent.get("page_height"),
                        "page_width":     sent.get("page_width"),
                        "bbox":           bbox,
                        "bboxes":         bboxes,   # ← Phase 5
                    },
                ))

            # Upsert in batches of 256 to stay within Qdrant payload limits
            for i in range(0, len(sent_points), 256):
                client.upsert(
                    collection_name=sent_col,
                    points=sent_points[i:i + 256],
                )

        logger.info(
            f"Phase4+5 ingest: {len(chunks)} chunks, "
            f"{len(sentences)} sentences → session {session_id}"
        )
        return len(chunks)

    # ── Search chunks ─────────────────────────────────────────────────────
    def search(self, session_id: str, query: str, top_k: int = 5) -> list[dict]:
        client    = self._get_client()
        query_vec = self._embed([query])[0]
        results   = client.search(
            collection_name=self._chunk_col(session_id),
            query_vector=query_vec,
            limit=top_k,
        )
        return [
            {
                "text":             r.payload.get("text", ""),
                "chunk_index":      r.payload.get("chunk_index", 0),
                "page":             r.payload.get("page", 1),
                "page_height":      r.payload.get("page_height"),
                "page_width":       r.payload.get("page_width"),
                "bbox":             r.payload.get("bbox"),
                "bboxes":           r.payload.get("bboxes") or [],
                "sentence_indices": r.payload.get("sentence_indices", []),
                "source":           r.payload.get("source", ""),
                "score":            r.score,
            }
            for r in results
        ]

    # ── Retrieve sentences ─────────────────────────────────────────────────
    def get_sentences(self, session_id: str) -> list[dict]:
        """
        Return all sentences for a session, each with 'bboxes' (Phase 5).

        Phase 5: Sentences are stored as individual Qdrant points.
        Retrieval uses scroll (no vector needed) and sorts by sentence_index
        to guarantee original document order.
        """
        try:
            client   = self._get_client()
            sent_col = self._sentence_col(session_id)

            all_sentences = []
            offset = None

            while True:
                result, next_offset = client.scroll(
                    collection_name=sent_col,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for point in result:
                    p = point.payload
                    all_sentences.append({
                        "sentence_index": p.get("sentence_index", 0),
                        "text":           p.get("text", ""),
                        "page":           p.get("page", 1),
                        "page_height":    p.get("page_height"),
                        "page_width":     p.get("page_width"),
                        "bbox":           p.get("bbox"),
                        "bboxes":         p.get("bboxes") or [],
                    })
                if next_offset is None:
                    break
                offset = next_offset

            # Sort by original document order
            all_sentences.sort(key=lambda s: s["sentence_index"])
            return all_sentences

        except Exception as e:
            logger.warning(f"Could not load sentences for {session_id}: {e}")
            return []

    # ── Retrieve specific sentences by index ───────────────────────────────
    def get_sentences_by_indices(
        self, session_id: str, indices: list[int]
    ) -> list[dict]:
        """
        Fast lookup of specific sentences by their sentence_index.
        Used by citation mapper to avoid loading all sentences.
        """
        if not indices:
            return []
        try:
            client  = self._get_client()
            results = client.retrieve(
                collection_name=self._sentence_col(session_id),
                ids=indices,
                with_payload=True,
            )
            sentences = []
            for r in results:
                p = r.payload
                sentences.append({
                    "sentence_index": p.get("sentence_index", 0),
                    "text":           p.get("text", ""),
                    "page":           p.get("page", 1),
                    "page_height":    p.get("page_height"),
                    "page_width":     p.get("page_width"),
                    "bbox":           p.get("bbox"),
                    "bboxes":         p.get("bboxes") or [],
                })
            sentences.sort(key=lambda s: s["sentence_index"])
            return sentences
        except Exception as e:
            logger.warning(f"Sentence index lookup failed for {session_id}: {e}")
            return []

    def delete_session(self, session_id: str):
        client = self._get_client()
        for col in [self._chunk_col(session_id), self._sentence_col(session_id)]:
            try:
                client.delete_collection(col)
            except Exception as e:
                logger.warning(f"Could not delete {col}: {e}")

    def session_exists(self, session_id: str) -> bool:
        try:
            self._get_client().get_collection(self._chunk_col(session_id))
            return True
        except Exception:
            return False


_store: Optional[QdrantPDFStore] = None

def get_store() -> QdrantPDFStore:
    global _store
    if _store is None:
        _store = QdrantPDFStore()
    return _store
