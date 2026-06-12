"""
Qdrant vector store for PDF embeddings.
Supports both local (file-based) and cloud (Qdrant Cloud) modes.
"""
import logging
from typing import Optional
from core.config import get_settings

logger = logging.getLogger(__name__)

COLLECTION_PREFIX = "pdf_"
VECTOR_SIZE       = 384   # all-MiniLM-L6-v2 output size


def _get_embedding_model():
    """Lazy-load ONNX embedding model."""
    try:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        return ONNXMiniLM_L6_V2()
    except Exception as e:
        logger.error(f"Embedding model load failed: {e}")
        raise


def _get_qdrant_client():
    """Return Qdrant client — cloud if configured, else local."""
    from qdrant_client import QdrantClient
    s = get_settings()
    if s.qdrant_url and s.qdrant_api_key:
        logger.info("Using Qdrant Cloud")
        return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
    logger.info(f"Using local Qdrant at {s.qdrant_local_path}")
    return QdrantClient(path=s.qdrant_local_path)


class QdrantPDFStore:
    """Manages PDF chunk embeddings in Qdrant."""

    def __init__(self):
        self._client   = None
        self._embed_fn = None
        self._sessions: dict[str, dict] = {}

    def _get_client(self):
        # BUG FIX: renamed from _client_() (confusing trailing underscore).
        # Guard both _client and _embed_fn so neither is re-initialised independently.
        if self._client is None or self._embed_fn is None:
            self._client   = _get_qdrant_client()
            self._embed_fn = _get_embedding_model()
        return self._client

    # Keep the old name as an alias so health.py probe still works
    _client_ = _get_client

    def _collection_name(self, session_id: str) -> str:
        return f"{COLLECTION_PREFIX}{session_id}"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text chunks using ONNX MiniLM."""
        result = self._embed_fn(texts)
        return result if isinstance(result[0], list) else [list(r) for r in result]

    def ingest(self, session_id: str, chunks: list[str], filename: str) -> int:
        """Embed and store PDF chunks. Returns number of chunks stored."""
        from qdrant_client.models import VectorParams, Distance, PointStruct

        client = self._get_client()
        col    = self._collection_name(session_id)

        # Create collection
        client.recreate_collection(
            collection_name=col,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

        # Embed in batches of 32
        all_embeddings = []
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            all_embeddings.extend(self._embed(batch))

        # Upload points
        points = [
            PointStruct(
                id=i,
                vector=all_embeddings[i],
                payload={"text": chunks[i], "chunk_index": i, "source": filename},
            )
            for i in range(len(chunks))
        ]
        client.upsert(collection_name=col, points=points)
        self._sessions[session_id] = {"filename": filename, "chunk_count": len(chunks)}
        logger.info(f"Qdrant: stored {len(chunks)} chunks for session {session_id}")
        return len(chunks)

    def search(self, session_id: str, query: str, top_k: int = 5) -> list[dict]:
        """Semantic search over a PDF session's chunks."""
        client = self._get_client()
        col    = self._collection_name(session_id)

        query_vec = self._embed([query])[0]
        results   = client.search(collection_name=col, query_vector=query_vec, limit=top_k)

        return [
            {
                "text":        r.payload.get("text", ""),
                "chunk_index": r.payload.get("chunk_index", 0),
                "source":      r.payload.get("source", ""),
                "score":       r.score,
            }
            for r in results
        ]

    def delete_session(self, session_id: str):
        """Remove a PDF session's collection."""
        try:
            self._get_client().delete_collection(self._collection_name(session_id))
            self._sessions.pop(session_id, None)
        except Exception as e:
            logger.warning(f"Could not delete session {session_id}: {e}")

    def session_exists(self, session_id: str) -> bool:
        try:
            self._get_client().get_collection(self._collection_name(session_id))
            return True
        except Exception:
            return False


# ── Singleton ──────────────────────────────────────────────────────────────
_store: Optional[QdrantPDFStore] = None

def get_store() -> QdrantPDFStore:
    global _store
    if _store is None:
        _store = QdrantPDFStore()
    return _store
