import pytest
from app.retrieval import (
    tokenize_text,
    sigmoid,
    BM25IndexStore,
    reciprocal_rank_fusion,
    rerank_chunks
)


def test_tokenize_text():
    raw = "AgentForge: Reciprocal Rank Fusion & Dense Retrieval!"
    tokens = tokenize_text(raw)
    assert "agentforge" in tokens
    assert "reciprocal" in tokens
    assert "fusion" in tokens
    assert "retrieval" in tokens


def test_sigmoid_bounds():
    assert sigmoid(0.0) == 0.5
    assert sigmoid(100.0) == pytest.approx(1.0)
    assert sigmoid(-100.0) == pytest.approx(0.0)
    assert 0.0 <= sigmoid(-2.5) <= 1.0


def test_bm25_store_operations():
    store = BM25IndexStore()
    store.clear()
    assert len(store.chunk_ids) == 0

    chunks = [
        {
            "id": "chunk-1",
            "document_id": "doc-A",
            "chunk_index": 0,
            "content": "PostgreSQL database with asyncpg stores execution traces.",
            "metadata": {"source": "test"}
        },
        {
            "id": "chunk-2",
            "document_id": "doc-B",
            "chunk_index": 0,
            "content": "Qdrant vector database handles dense embeddings.",
            "metadata": {"source": "test"}
        }
    ]

    store.add_chunks(chunks)
    assert len(store.chunk_ids) == 2

    # Search query
    results = store.search("PostgreSQL database", top_k=5)
    assert len(results) >= 1
    assert results[0]["document_id"] == "doc-A"
    assert "sparse_score" in results[0]

    store.clear()
    assert len(store.chunk_ids) == 0


def test_reciprocal_rank_fusion():
    dense_hits = [
        {"id": "c1", "document_id": "doc1", "chunk_index": 0, "content": "text 1", "dense_score": 0.9, "dense_rank": 1},
        {"id": "c2", "document_id": "doc2", "chunk_index": 0, "content": "text 2", "dense_score": 0.8, "dense_rank": 2},
    ]
    sparse_hits = [
        {"id": "c2", "document_id": "doc2", "chunk_index": 0, "content": "text 2", "sparse_score": 5.2, "sparse_rank": 1},
        {"id": "c3", "document_id": "doc3", "chunk_index": 0, "content": "text 3", "sparse_score": 3.1, "sparse_rank": 2},
    ]

    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=60, top_n=5)
    assert len(fused) == 3

    # Document c2 appears in both dense and sparse, so its RRF score should be highest
    assert fused[0]["id"] == "c2"
    assert fused[0]["rrf_rank"] == 1
    expected_c2_rrf = (1.0 / (60 + 2)) + (1.0 / (60 + 1))
    assert fused[0]["rrf_score"] == pytest.approx(expected_c2_rrf)


def test_rerank_chunks_filtering():
    chunks = [
        {"id": "c1", "document_id": "doc1", "content": "AgentForge RRF algorithm fuses sparse and dense search."},
        {"id": "c2", "document_id": "doc2", "content": "Baking chocolate cake with vanilla ice cream."}
    ]

    query = "How does AgentForge fuse search algorithms?"
    reranked = rerank_chunks(query, chunks, threshold=0.10, top_n=5)

    assert len(reranked) >= 1
    # Highly relevant chunk c1 should rank first
    assert reranked[0]["id"] == "c1"
    assert "rerank_score" in reranked[0]
