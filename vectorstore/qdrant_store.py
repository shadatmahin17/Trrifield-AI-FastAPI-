"""
Qdrant vector store — position-aware ingestion.
Each point payload: { text, page, bbox, chunk_index, source }
"""
import logging
from typing import Optional
from core.config import get_settings

logger = logging.getLogger(__name__)

COLLECTION_PREFIX = "pdf_"
VECTOR_SIZE       = 384


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
        logger.info("Using Qdrant Cloud")
        return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
    logger.info(f"Using local Qdrant at {s.qdrant_local_path}")
    return QdrantClient(path=s.qdrant_local_path)


class QdrantPDFStore:

    def __init__(self):
        self._client   = None
        self._embed_fn = None
        self._sessions: dict[str, dict] = {}

    def _get_client(self):
        if self._client is None or self._embed_fn is None:
            self._client   = _get_qdrant_client()
            self._embed_fn = _get_embedding_model()
        return self._client

    _client_ = _get_client  # alias for health probe

    def _col(self, session_id: str) -> str:
        return f"{COLLECTION_PREFIX}{session_id}"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        result = self._embed_fn(texts)
        return result if isinstance(result[0], list) else [list(r) for r in result]

    # ── Legacy ingest (plain text chunks, no position) ─────────────────────
    def ingest(self, session_id: str, chunks: list[str], filename: str) -> int:
        positioned = [
            {"text": c, "page": 1, "bbox": None}
            for c in chunks
        ]
        return self.ingest_with_positions(session_id, positioned, filename)

    # ── Position-aware ingest ──────────────────────────────────────────────
    def ingest_with_positions(
        self,
        session_id: str,
        chunks: list[dict],   # [{text, page, bbox}, ...]
        filename:   str,
    ) -> int:
        from qdrant_client.models import VectorParams, Distance, PointStruct

        client = self._get_client()
        col    = self._col(session_id)

        client.recreate_collection(
            collection_name=col,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

        texts = [c["text"] for c in chunks]

        # Embed in batches of 32
        all_embeddings = []
        for i in range(0, len(texts), 32):
            all_embeddings.extend(self._embed(texts[i:i+32]))

        points = [
            PointStruct(
                id=i,
                vector=all_embeddings[i],
                payload={
                    "text":        chunks[i]["text"],
                    "chunk_index": i,
                    "page":        chunks[i].get("page", 1),
                    "bbox":        chunks[i].get("bbox"),   # [x0,y0,x1,y1] or None
                    "source":      filename,
                },
            )
            for i in range(len(chunks))
        ]
        client.upsert(collection_name=col, points=points)
        self._sessions[session_id] = {"filename": filename, "chunk_count": len(chunks)}
        logger.info(f"Qdrant: stored {len(chunks)} positioned chunks for {session_id}")
        return len(chunks)

    def search(self, session_id: str, query: str, top_k: int = 5) -> list[dict]:
        client    = self._get_client()
        query_vec = self._embed([query])[0]
        results   = client.search(
            collection_name=self._col(session_id),
            query_vector=query_vec,
            limit=top_k,
        )
        return [
            {
                "text":        r.payload.get("text", ""),
                "chunk_index": r.payload.get("chunk_index", 0),
                "page":        r.payload.get("page", 1),
                "bbox":        r.payload.get("bbox"),
                "source":      r.payload.get("source", ""),
                "score":       r.score,
            }
            for r in results
        ]

    def delete_session(self, session_id: str):
        try:
            self._get_client().delete_collection(self._col(session_id))
            self._sessions.pop(session_id, None)
        except Exception as e:
            logger.warning(f"Could not delete session {session_id}: {e}")

    def session_exists(self, session_id: str) -> bool:
        try:
            self._get_client().get_collection(self._col(session_id))
            return True
        except Exception:
            return False


_store: Optional[QdrantPDFStore] = None

def get_store() -> QdrantPDFStore:
    global _store
    if _store is None:
        _store = QdrantPDFStore()
    return _store
