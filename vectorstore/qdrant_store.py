"""
Qdrant vector store — Phase 3 dual collection architecture.

Per session:
  pdf_chunks_{session}    — chunk embeddings for broad retrieval
  pdf_sentences_{session} — full sentence list stored as a single metadata point
                            (no embedding needed — looked up by session_id)
"""
import json, logging
from typing import Optional
from core.config import get_settings

logger = logging.getLogger(__name__)

CHUNK_PREFIX    = "pdf_chunks_"
SENTENCE_PREFIX = "pdf_sentences_"
VECTOR_SIZE     = 384
SENTENCE_POINT_ID = 0   # single point storing all sentences as payload


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

    _client_ = _get_client   # alias for health probe

    def _chunk_col(self, session_id: str) -> str:
        return f"{CHUNK_PREFIX}{session_id}"

    def _sentence_col(self, session_id: str) -> str:
        return f"{SENTENCE_PREFIX}{session_id}"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        result = self._embed_fn(texts)
        return result if isinstance(result[0], list) else [list(r) for r in result]

    # ── Legacy ingest (no position) ───────────────────────────────────────
    def ingest(self, session_id: str, chunks: list[str], filename: str) -> int:
        positioned = [{"text": c, "page": 1, "page_height": None,
                       "page_width": None, "bbox": None,
                       "sentence_indices": []} for c in chunks]
        return self.ingest_with_positions(session_id, positioned, filename)

    # ── Phase 2 ingest (position-aware, no sentence index) ────────────────
    def ingest_with_positions(self, session_id: str, chunks: list[dict], filename: str) -> int:
        return self.ingest_phase3(session_id, chunks, [], filename)

    # ── Phase 3 ingest (chunks + sentence index) ──────────────────────────
    def ingest_phase3(
        self,
        session_id: str,
        chunks:     list[dict],   # [{text, page, page_height, page_width, bbox, sentence_indices}]
        sentences:  list[dict],   # [{text, page, page_height, page_width, bbox}]
        filename:   str,
    ) -> int:
        from qdrant_client.models import (
            VectorParams, Distance, PointStruct, CollectionInfo
        )

        client      = self._get_client()
        chunk_col   = self._chunk_col(session_id)
        sent_col    = self._sentence_col(session_id)

        # ── Create / recreate chunk collection ────────────────────────────
        client.recreate_collection(
            collection_name=chunk_col,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

        texts = [c["text"] for c in chunks]
        all_embeddings = []
        for i in range(0, len(texts), 32):
            all_embeddings.extend(self._embed(texts[i:i+32]))

        points = [
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
                    "sentence_indices": chunks[i].get("sentence_indices", []),
                    "source":           filename,
                },
            )
            for i in range(len(chunks))
        ]
        client.upsert(collection_name=chunk_col, points=points)

        # ── Store sentence index in a dedicated collection ─────────────────
        # We store the entire sentence list as a JSON payload on a single
        # dummy point (id=0). No embedding needed — looked up by session.
        if sentences:
            client.recreate_collection(
                collection_name=sent_col,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            # Dummy zero vector — we only use this point for its payload
            zero_vec = [0.0] * VECTOR_SIZE
            client.upsert(
                collection_name=sent_col,
                points=[PointStruct(
                    id=SENTENCE_POINT_ID,
                    vector=zero_vec,
                    payload={"sentences": json.dumps(sentences)},
                )],
            )

        logger.info(
            f"Phase3 ingest: {len(chunks)} chunks, "
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
                "sentence_indices": r.payload.get("sentence_indices", []),
                "source":           r.payload.get("source", ""),
                "score":            r.score,
            }
            for r in results
        ]

    # ── Retrieve sentence index ───────────────────────────────────────────
    def get_sentences(self, session_id: str) -> list[dict]:
        """Return all sentences for a session. Empty list if not stored."""
        try:
            client  = self._get_client()
            results = client.retrieve(
                collection_name=self._sentence_col(session_id),
                ids=[SENTENCE_POINT_ID],
                with_payload=True,
            )
            if results:
                raw = results[0].payload.get("sentences", "[]")
                return json.loads(raw)
        except Exception as e:
            logger.warning(f"Could not load sentences for {session_id}: {e}")
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
