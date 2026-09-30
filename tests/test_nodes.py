import pytest
from app.nodes import (
    QueryRewriterNode,
    RelevanceGraderNode,
    GeneratorNode
)


def test_query_rewriter_clean_query():
    raw1 = "Can you please explain how Qdrant vector database works?"
    cleaned1 = QueryRewriterNode.clean_query(raw1)
    assert "how Qdrant vector database works" in cleaned1

    raw2 = "Tell me about Reciprocal Rank Fusion"
    cleaned2 = QueryRewriterNode.clean_query(raw2)
    assert "Reciprocal Rank Fusion" in cleaned2


@pytest.mark.asyncio
async def test_query_rewriter_execution():
    state = {
        "run_id": "test-run",
        "query": "Could you please explain LangGraph node reflection?",
        "iterations": 0,
        "needs_revision": False,
        "traces": []
    }
    result = await QueryRewriterNode.execute(state)
    assert "rewritten_query" in result
    assert "LangGraph node reflection" in result["rewritten_query"]
    assert len(result["traces"]) == 1


@pytest.mark.asyncio
async def test_query_rewriter_reflection_relaxation():
    state = {
        "run_id": "test-run",
        "query": "What is x7z999 special jargon?",
        "iterations": 1,
        "needs_revision": True,
        "traces": []
    }
    result = await QueryRewriterNode.execute(state)
    assert "overview" in result["rewritten_query"] or "key facts" in result["rewritten_query"]


@pytest.mark.asyncio
async def test_relevance_grader_node_pass():
    state = {
        "filtered_documents": [
            {"id": "c1", "rerank_score": 0.85},
            {"id": "c2", "rerank_score": 0.60}
        ],
        "iterations": 0,
        "traces": []
    }
    res = await RelevanceGraderNode.execute(state)
    assert res["needs_revision"] is False


@pytest.mark.asyncio
async def test_relevance_grader_node_fail():
    state = {
        "filtered_documents": [],
        "iterations": 0,
        "traces": []
    }
    res = await RelevanceGraderNode.execute(state)
    assert res["needs_revision"] is True
    assert res["iterations"] == 1


@pytest.mark.asyncio
async def test_generator_node_citations():
    state = {
        "query": "What database persists execution traces?",
        "filtered_documents": [
            {
                "id": "chunk-101",
                "document_id": "doc-postgres",
                "content": "PostgreSQL with asyncpg persists all execution traces.",
                "rerank_score": 0.92
            }
        ],
        "traces": []
    }
    res = await GeneratorNode.execute(state)
    assert "[1]" in res["generation"]
    assert len(res["citations"]) == 1
    assert res["citations"][0]["document_id"] == "doc-postgres"
    assert res["citations"][0]["chunk_id"] == "chunk-101"
