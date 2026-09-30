import re
import math
import uuid
import logging
import threading
from typing import Any
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class FastFeatureEmbedder:
    """
    Ultra-fast, zero-dependency subword n-gram feature hashing embedder.
    Produces unit-normalized dense vectors of length `dim` (384) in milliseconds.
    Used when PyTorch/SentenceTransformers is not loaded to preserve <50MB RAM footprint.
    """
    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def encode(self, sentences: str | list[str], normalize_embeddings: bool = True, show_progress_bar: bool = False) -> Any:
        is_single = isinstance(sentences, str)
        input_list = [sentences] if is_single else sentences

        import numpy as np
        vectors = []
        for text in input_list:
            vec = np.zeros(self.dim, dtype=np.float32)
            tokens = re.findall(r"\b\w+\b", text.lower())
            for token in tokens:
                idx = abs(hash(token)) % self.dim
                vec[idx] += 1.0
                for i in range(len(token) - 2):
                    sub_idx = abs(hash(token[i:i+3])) % self.dim
                    vec[sub_idx] += 0.5
            norm = np.linalg.norm(vec)
            if norm > 0 and normalize_embeddings:
                vec = vec / norm
            vectors.append(vec)

        arr = np.array(vectors, dtype=np.float32)
        return arr[0] if is_single else arr


class FastCrossEncoder:
    """
    Fast subword Jaccard & lexical overlap cross-encoder scorer.
    Computes pairwise relevance logits mapped via sigmoid to [0, 1].
    """
    def predict(self, pairs: list[list[str]]) -> list[float]:
        logits = []
        for query, passage in pairs:
            q_tokens = set(re.findall(r"\b\w+\b", query.lower()))
            p_tokens = set(re.findall(r"\b\w+\b", passage.lower()))
            if not q_tokens or not p_tokens:
                logits.append(-2.0)
                continue
            intersection = q_tokens.intersection(p_tokens)
            union = q_tokens.union(p_tokens)
            jaccard = len(intersection) / len(union) if union else 0.0
            overlap_ratio = len(intersection) / len(q_tokens) if q_tokens else 0.0
            logit = (jaccard * 3.0 + overlap_ratio * 3.0) - 1.0
            logits.append(float(logit))
        return logits


class ModelRegistry:
    """
    Thread-safe lazy-loading singleton for local neural models
    (Dense Embedder & Cross-Encoder Reranker).
    """
    _embedding_model: Any = None
    _reranker_model: Any = None
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def get_embedding_model(cls) -> Any:
        if cls._embedding_model is None:
            with cls._lock:
                if cls._embedding_model is None:
                    if HAS_SENTENCE_TRANSFORMERS:
                        try:
                            import torch
                            torch.set_num_threads(1)
                            logger.info("Loading dense embedding model: %s", settings.EMBEDDING_MODEL_NAME)
                            cls._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
                        except Exception as exc:
                            logger.warning("Could not load SentenceTransformer (%s). Using FastFeatureEmbedder fallback.", exc)
                            cls._embedding_model = FastFeatureEmbedder(dim=settings.EMBEDDING_DIMENSION)
                    else:
                        logger.info("SentenceTransformers not installed. Using FastFeatureEmbedder fallback.")
                        cls._embedding_model = FastFeatureEmbedder(dim=settings.EMBEDDING_DIMENSION)
        return cls._embedding_model

    @classmethod
    def get_reranker_model(cls) -> Any:
        if cls._reranker_model is None:
            with cls._lock:
                if cls._reranker_model is None:
                    if HAS_SENTENCE_TRANSFORMERS:
                        try:
                            logger.info("Loading cross-encoder reranker model: %s", settings.RERANKER_MODEL_NAME)
                            cls._reranker_model = CrossEncoder(settings.RERANKER_MODEL_NAME)
                        except Exception as exc:
                            logger.warning("Could not load CrossEncoder (%s). Using FastCrossEncoder fallback.", exc)
                            cls._reranker_model = FastCrossEncoder()
                    else:
                        logger.info("SentenceTransformers not installed. Using FastCrossEncoder fallback.")
                        cls._reranker_model = FastCrossEncoder()
        return cls._reranker_model


def tokenize_text(text: str) -> list[str]:
    """Tokenize and normalize text for BM25 Okapi."""
    return re.findall(r"\b\w+\b", text.lower())


def sigmoid(x: float) -> float:
    """Logistic sigmoid transformation to map unbounded logits to [0, 1]."""
    return 1.0 / (1.0 + math.exp(-max(min(x, 20.0), -20.0)))


class BM25IndexStore:
    """
    In-memory thread-safe BM25 index synced with persisted document chunks.
    """
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.chunk_ids: list[str] = []
        self.chunk_records: dict[str, dict[str, Any]] = {}
        self.tokenized_corpus: list[list[str]] = []
        self.bm25_model: BM25Okapi | None = None

    def add_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """
        Adds new chunk records to the BM25 store and rebuilds the BM25 model.
        Each chunk is a dict with keys: id, content, document_id, chunk_index, metadata.
        """
        with self._lock:
            for chunk in chunks:
                cid = str(chunk["id"])
                if cid not in self.chunk_records:
                    self.chunk_ids.append(cid)
                    self.chunk_records[cid] = chunk
                    self.tokenized_corpus.append(tokenize_text(chunk["content"]))
                else:
                    self.chunk_records[cid] = chunk
                    idx = self.chunk_ids.index(cid)
                    self.tokenized_corpus[idx] = tokenize_text(chunk["content"])

            if self.tokenized_corpus:
                self.bm25_model = BM25Okapi(self.tokenized_corpus)
                logger.info("Rebuilt BM25 index with %d total chunks", len(self.chunk_ids))

    def search(self, query: str, top_k: int = 15) -> list[dict[str, Any]]:
        """
        Searches the BM25 index for the most relevant chunks.
        """
        with self._lock:
            if not self.bm25_model or not self.chunk_ids:
                return []

            query_tokens = tokenize_text(query)
            if not query_tokens:
                return []

            scores = self.bm25_model.get_scores(query_tokens)
            indexed_scores = [(idx, float(score)) for idx, score in enumerate(scores) if score > 0.0]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)

            results: list[dict[str, Any]] = []
            for rank, (idx, score) in enumerate(indexed_scores[:top_k], start=1):
                cid = self.chunk_ids[idx]
                record = self.chunk_records[cid]
                results.append({
                    "id": cid,
                    "document_id": record["document_id"],
                    "chunk_index": record["chunk_index"],
                    "content": record["content"],
                    "metadata": record.get("metadata", {}),
                    "sparse_score": score,
                    "sparse_rank": rank
                })
            return results

    def clear(self) -> None:
        with self._lock:
            self.chunk_ids.clear()
            self.chunk_records.clear()
            self.tokenized_corpus.clear()
            self.bm25_model = None


# Global in-memory BM25 index store
bm25_store = BM25IndexStore()


class QdrantVectorService:
    """
    Async client service for managing dense embeddings and vector storage in Qdrant.
    """
    def __init__(self) -> None:
        self.client: AsyncQdrantClient | None = None

    async def get_client(self) -> AsyncQdrantClient:
        if self.client is None:
            if settings.QDRANT_HOST.startswith(("http://", "https://", ":memory:")):
                self.client = AsyncQdrantClient(
                    url=settings.QDRANT_HOST,
                    api_key=settings.QDRANT_API_KEY,
                    timeout=10.0
                )
            else:
                self.client = AsyncQdrantClient(
                    host=settings.QDRANT_HOST,
                    port=settings.QDRANT_PORT,
                    api_key=settings.QDRANT_API_KEY,
                    timeout=10.0
                )
        return self.client

    async def ensure_collection(self) -> None:
        client = await self.get_client()
        collections_response = await client.get_collections()
        existing = [c.name for c in collections_response.collections]
        if settings.QDRANT_COLLECTION not in existing:
            logger.info("Creating Qdrant collection: %s", settings.QDRANT_COLLECTION)
            await client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=qmodels.VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=qmodels.Distance.COSINE
                )
            )

    async def upsert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        """
        Embeds chunks using SentenceTransformer and upserts into Qdrant.
        """
        if not chunks:
            return

        client = await self.get_client()
        await self.ensure_collection()

        model = ModelRegistry.get_embedding_model()
        contents = [c["content"] for c in chunks]
        embeddings = model.encode(contents, normalize_embeddings=True, show_progress_bar=False)

        points = []
        for chunk, emb in zip(chunks, embeddings):
            point_id = str(chunk.get("embedding_id") or chunk.get("id") or uuid.uuid4())
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=emb.tolist(),
                    payload={
                        "chunk_id": str(chunk["id"]),
                        "document_id": chunk["document_id"],
                        "chunk_index": chunk["chunk_index"],
                        "content": chunk["content"],
                        "metadata": chunk.get("metadata", {})
                    }
                )
            )

        await client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=points,
            wait=True
        )
        logger.info("Upserted %d points to Qdrant collection %s", len(points), settings.QDRANT_COLLECTION)

    async def search_dense(self, query: str, top_k: int = 15) -> list[dict[str, Any]]:
        """
        Dense vector semantic similarity search using cosine distance in Qdrant.
        """
        client = await self.get_client()
        await self.ensure_collection()

        model = ModelRegistry.get_embedding_model()
        query_vector = model.encode(query, normalize_embeddings=True, show_progress_bar=False).tolist()

        search_results = await client.search(
            collection_name=settings.QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )

        results: list[dict[str, Any]] = []
        for rank, hit in enumerate(search_results, start=1):
            payload = hit.payload or {}
            results.append({
                "id": payload.get("chunk_id", str(hit.id)),
                "document_id": payload.get("document_id", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "content": payload.get("content", ""),
                "metadata": payload.get("metadata", {}),
                "dense_score": float(hit.score),
                "dense_rank": rank
            })
        return results

    async def close(self) -> None:
        if self.client is not None:
            await self.client.close()
            self.client = None


qdrant_service = QdrantVectorService()


def reciprocal_rank_fusion(
    dense_results: list[dict[str, Any]],
    sparse_results: list[dict[str, Any]],
    k: int = 60,
    top_n: int = 10
) -> list[dict[str, Any]]:
    """
    Executes Reciprocal Rank Fusion (RRF) between dense vector hits and BM25 sparse hits.
    Formula: RRF(d) = sum_{m in {dense, sparse}} (1 / (k + rank_m(d)))
    """
    merged: dict[str, dict[str, Any]] = {}

    for hit in dense_results:
        cid = str(hit["id"])
        rank = hit["dense_rank"]
        score_contrib = 1.0 / (k + rank)
        if cid not in merged:
            merged[cid] = {
                "id": cid,
                "document_id": hit["document_id"],
                "chunk_index": hit["chunk_index"],
                "content": hit["content"],
                "metadata": hit.get("metadata", {}),
                "dense_score": hit.get("dense_score", 0.0),
                "dense_rank": rank,
                "sparse_score": 0.0,
                "sparse_rank": None,
                "rrf_score": score_contrib
            }
        else:
            merged[cid]["rrf_score"] += score_contrib
            merged[cid]["dense_score"] = hit.get("dense_score", 0.0)
            merged[cid]["dense_rank"] = rank

    for hit in sparse_results:
        cid = str(hit["id"])
        rank = hit["sparse_rank"]
        score_contrib = 1.0 / (k + rank)
        if cid not in merged:
            merged[cid] = {
                "id": cid,
                "document_id": hit["document_id"],
                "chunk_index": hit["chunk_index"],
                "content": hit["content"],
                "metadata": hit.get("metadata", {}),
                "dense_score": 0.0,
                "dense_rank": None,
                "sparse_score": hit.get("sparse_score", 0.0),
                "sparse_rank": rank,
                "rrf_score": score_contrib
            }
        else:
            merged[cid]["rrf_score"] += score_contrib
            merged[cid]["sparse_score"] = hit.get("sparse_score", 0.0)
            merged[cid]["sparse_rank"] = rank

    fused_candidates = list(merged.values())
    fused_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)

    for rank, item in enumerate(fused_candidates, start=1):
        item["rrf_rank"] = rank

    return fused_candidates[:top_n]


def rerank_chunks(
    query: str,
    chunks: list[dict[str, Any]],
    threshold: float = 0.20,
    top_n: int = 5
) -> list[dict[str, Any]]:
    """
    Reranks candidate chunks with CrossEncoder (ms-marco-MiniLM-L-6-v2),
    maps raw logits via sigmoid to [0, 1], and prunes chunks falling below threshold.
    """
    if not chunks:
        return []

    model = ModelRegistry.get_reranker_model()
    pairs = [[query, chunk["content"]] for chunk in chunks]
    raw_scores = model.predict(pairs)

    scored_chunks: list[dict[str, Any]] = []
    for chunk, raw_score in zip(chunks, raw_scores):
        normalized_score = float(sigmoid(float(raw_score)))
        chunk_copy = dict(chunk)
        chunk_copy["rerank_raw_logit"] = float(raw_score)
        chunk_copy["rerank_score"] = normalized_score
        scored_chunks.append(chunk_copy)

    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Filter and prune below threshold
    filtered = [c for c in scored_chunks if c["rerank_score"] >= threshold]

    for rank, item in enumerate(filtered, start=1):
        item["final_rank"] = rank

    return filtered[:top_n]
