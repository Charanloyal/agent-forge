import os
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

# Ensure SQLite fallback for local test environment
os.environ["USE_SQLITE"] = "true"
os.environ["QDRANT_HOST"] = ":memory:"


@pytest.mark.asyncio
async def test_health_check_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/healthz")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "bm25_indexed_chunks" in data


@pytest.mark.asyncio
async def test_ingest_and_query_flow():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Ingest Documents
        ingest_payload = {
            "documents": [
                {
                    "document_id": "test-doc-01",
                    "content": "AgentForge implements Reciprocal Rank Fusion (RRF) with constant k=60 to fuse dense vector rankings with sparse BM25 scores.",
                    "metadata": {"test": True}
                },
                {
                    "document_id": "test-doc-02",
                    "content": "Execution traces and evaluation metrics are persisted in the database for observability.",
                    "metadata": {"test": True}
                }
            ]
        }
        ingest_resp = await client.post("/v1/ingest", json=ingest_payload)
        assert ingest_resp.status_code == 201
        ingest_data = ingest_resp.json()
        assert ingest_data["status"] == "success"
        assert ingest_data["documents_ingested"] == 2
        assert len(ingest_data["chunk_ids"]) == 2

        # 2. Query Agent Workflow
        query_payload = {
            "query": "Can you explain how AgentForge fuses search scores and stores traces?"
        }
        query_resp = await client.post("/v1/query", json=query_payload)
        assert query_resp.status_code == 200
        query_data = query_resp.json()
        assert "run_id" in query_data
        assert "answer" in query_data
        assert "evaluation_metrics" in query_data
        assert "execution_trace" in query_data
        assert len(query_data["execution_trace"]) >= 5

        run_id = query_data["run_id"]

        # 3. Retrieve Execution Trace
        trace_resp = await client.get(f"/v1/traces/{run_id}")
        assert trace_resp.status_code == 200
        trace_data = trace_resp.json()
        assert trace_data["run_id"] == run_id
        assert trace_data["total_steps"] >= 5
        assert "steps" in trace_data
