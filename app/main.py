import os
import re
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.database import get_db, init_db, close_db
from app.models import DocumentChunk, AgentExecutionTrace, EvaluationMetric
from app.retrieval import qdrant_service, bm25_store
from app.graph import run_agent_workflow
from app.evaluator import EvaluationMetrics

# Configure logging
settings = get_settings()
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("agentforge.main")


def chunk_document_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> list[str]:
    """
    Deterministically splits document text into overlapping chunks respecting
    sentence and paragraph boundaries.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    if len(clean_text) <= chunk_size:
        return [clean_text]

    chunks: list[str] = []
    start = 0
    text_length = len(clean_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            # Look backwards for sentence delimiter or paragraph break
            match = re.search(r"([.\n!?])\s+", clean_text[start:end])
            if match:
                last_delim_idx = -1
                for m in re.finditer(r"([.\n!?])\s+", clean_text[start:end]):
                    last_delim_idx = m.end()
                if last_delim_idx > chunk_size // 3:
                    end = start + last_delim_idx
            else:
                # Fallback to nearest whitespace
                last_space = clean_text[start:end].rfind(" ")
                if last_space > chunk_size // 3:
                    end = start + last_space

        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start += max(1, (end - start) - chunk_overlap)
        if end >= text_length:
            break

    return chunks


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown orchestration.
    Initializes PostgreSQL tables, verifies Qdrant collection, and rebuilds BM25 index.
    """
    logger.info("Starting up AgentForge...")
    try:
        await init_db()
    except Exception as exc:
        logger.warning("Database initialization warning: %s", exc)

    try:
        await qdrant_service.ensure_collection()
    except Exception as exc:
        logger.warning("Qdrant collection setup warning: %s", exc)

    # Hydrate in-memory BM25 index from persistent PostgreSQL chunks
    try:
        from app.database import async_session_factory
        async with async_session_factory() as session:
            stmt = select(DocumentChunk)
            result = await session.execute(stmt)
            existing_chunks = result.scalars().all()
            if existing_chunks:
                records = [
                    {
                        "id": str(c.id),
                        "document_id": c.document_id,
                        "chunk_index": c.chunk_index,
                        "content": c.content,
                        "metadata": c.metadata_json
                    }
                    for c in existing_chunks
                ]
                bm25_store.add_chunks(records)
                logger.info("Hydrated BM25 store with %d persisted chunks from database.", len(records))
    except Exception as exc:
        logger.warning("Could not hydrate BM25 on startup (database might be empty): %s", exc)

    yield

    logger.info("Shutting down AgentForge...")
    await qdrant_service.close()
    await close_db()
    logger.info("AgentForge shutdown complete.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production Agentic RAG system with automated multi-hop reflection, hybrid retrieval, and an evaluation scoring harness.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "AgentForge API operational. Visit /docs for OpenAPI specifications."}


# ---------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------

class DocumentPayload(BaseModel):
    document_id: str = Field(..., description="Unique parent document identifier")
    content: str = Field(..., min_length=1, description="Raw textual content to ingest")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata tags, author, date, source URL")


class IngestRequest(BaseModel):
    documents: list[DocumentPayload] = Field(..., min_length=1)


class IngestResponse(BaseModel):
    status: str
    documents_ingested: int
    chunks_created: int
    chunk_ids: list[str]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Conversational or search query")


class CitationItem(BaseModel):
    citation_index: int
    chunk_id: str
    document_id: str
    relevance_score: float
    snippet: str
    metadata: dict[str, Any] = {}


class TraceStep(BaseModel):
    step_name: str
    step_index: int
    input_state: dict[str, Any]
    output_state: dict[str, Any]
    latency_ms: float


class QueryResponse(BaseModel):
    run_id: str
    query: str
    rewritten_query: str
    answer: str
    citations: list[CitationItem]
    evaluation_metrics: dict[str, Any]
    execution_trace: list[TraceStep]
    iterations: int


class HealthStatus(BaseModel):
    status: str
    version: str
    database_connected: bool
    qdrant_connected: bool
    bm25_indexed_chunks: int


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.get("/healthz", response_model=HealthStatus, tags=["Health"])
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Liveness and readiness health probe reporting service dependencies status.
    """
    db_ok = False
    qdrant_ok = False

    # Check PostgreSQL
    try:
        res = await db.execute(select(1))
        db_ok = res.scalar() == 1
    except Exception as exc:
        logger.error("Database health check failed: %s", exc)

    # Check Qdrant
    try:
        client = await qdrant_service.get_client()
        cols = await client.get_collections()
        qdrant_ok = cols is not None
    except Exception as exc:
        logger.error("Qdrant health check failed: %s", exc)

    all_healthy = db_ok and qdrant_ok
    return HealthStatus(
        status="healthy" if all_healthy else "degraded",
        version=settings.APP_VERSION,
        database_connected=db_ok,
        qdrant_connected=qdrant_ok,
        bm25_indexed_chunks=len(bm25_store.chunk_ids)
    )


@app.post("/v1/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED, tags=["Ingestion"])
async def ingest_documents(payload: IngestRequest, db: AsyncSession = Depends(get_db)):
    """
    Ingests document batches:
    1. Chunks text using recursive boundary chunker.
    2. Computes dense embeddings and stores vectors in Qdrant.
    3. Indexes chunks in in-memory BM25 Okapi sparse store.
    4. Persists chunk records in PostgreSQL.
    """
    try:
        all_chunks_to_upsert: list[dict[str, Any]] = []
        db_records_to_insert: list[DocumentChunk] = []
        created_chunk_ids: list[str] = []

        for doc in payload.documents:
            raw_chunks = chunk_document_text(
                text=doc.content,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )

            for idx, chunk_text in enumerate(raw_chunks):
                chunk_uuid = uuid.uuid4()
                embedding_uuid = str(uuid.uuid4())
                chunk_id_str = str(chunk_uuid)
                created_chunk_ids.append(chunk_id_str)

                chunk_dict = {
                    "id": chunk_id_str,
                    "embedding_id": embedding_uuid,
                    "document_id": doc.document_id,
                    "chunk_index": idx,
                    "content": chunk_text,
                    "metadata": doc.metadata
                }
                all_chunks_to_upsert.append(chunk_dict)

                db_records_to_insert.append(
                    DocumentChunk(
                        id=chunk_id_str,
                        document_id=doc.document_id,
                        chunk_index=idx,
                        content=chunk_text,
                        metadata_json=doc.metadata,
                        embedding_id=embedding_uuid
                    )
                )

        if all_chunks_to_upsert:
            # 1. Upsert into Qdrant
            try:
                await qdrant_service.upsert_chunks(all_chunks_to_upsert)
            except Exception as e:
                logger.warning("Qdrant vector upsert warning: %s", e)

            # 2. Add to BM25 sparse index
            try:
                bm25_store.add_chunks(all_chunks_to_upsert)
            except Exception as e:
                logger.warning("BM25 sparse index warning: %s", e)

            # 3. Persist to Database
            try:
                db.add_all(db_records_to_insert)
                await db.commit()
            except Exception as e:
                logger.warning("Database persistence warning: %s", e)

        return IngestResponse(
            status="success",
            documents_ingested=len(payload.documents),
            chunks_created=len(all_chunks_to_upsert),
            chunk_ids=created_chunk_ids
        )
    except Exception as exc:
        logger.exception("Document ingestion exception: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion error: {str(exc)}"
        )


def sanitize_json_obj(obj: Any) -> Any:
    """
    Recursively converts numpy types, UUIDs, and complex types to JSON-safe Python primitives.
    """
    if obj is None:
        return None
    if isinstance(obj, (int, str, bool)):
        return obj
    if isinstance(obj, float):
        return obj
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "tolist"):
        try:
            return [sanitize_json_obj(x) for x in obj.tolist()]
        except Exception:
            pass
    if isinstance(obj, dict):
        return {str(k): sanitize_json_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_json_obj(x) for x in obj]
    return str(obj)


@app.post("/v1/query", response_model=QueryResponse, tags=["Agentic RAG"])
async def query_rag_agent(payload: QueryRequest, db: AsyncSession = Depends(get_db)):
    """
    Executes the AgentForge LangGraph workflow:
    - Multi-hop reflection and query normalization
    - Parallel hybrid retrieval (Qdrant + BM25 with RRF k=60)
    - Cross-Encoder neural re-ranking
    - Relevance grading & query relaxation routing
    - Grounded generation with citation bracket indices
    - Hallucination checking & reflection loop
    - Programmatic evaluation scoring harness (Precision, Faithfulness, Relevance)
    - Full persistence of execution traces and evaluation metrics
    """
    try:
        run_id = str(uuid.uuid4())

        # Execute State Machine
        try:
            final_state = await run_agent_workflow(query=payload.query, run_id=run_id)
        except Exception as workflow_err:
            logger.exception("run_agent_workflow exception: %s", workflow_err)
            import traceback
            tb_str = traceback.format_exc()
            return JSONResponse(
                status_code=500,
                content={
                    "detail": f"Workflow error: {str(workflow_err)}",
                    "traceback": tb_str
                }
            )

        final_state = sanitize_json_obj(final_state)

        # Persist Execution Traces
        traces = final_state.get("traces", [])
        trace_entities: list[AgentExecutionTrace] = []
        for t in traces:
            trace_entities.append(
                AgentExecutionTrace(
                    id=str(uuid.uuid4()),
                    run_id=run_id,
                    step_name=str(t["step_name"]),
                    step_index=int(t["step_index"]),
                    input_state=sanitize_json_obj(t.get("input_state", {})),
                    output_state=sanitize_json_obj(t.get("output_state", {})),
                    latency_ms=float(t.get("latency_ms", 0.0))
                )
            )
        if trace_entities:
            try:
                db.add_all(trace_entities)
            except Exception as e:
                logger.warning("Trace entity persistence warning: %s", e)

        # Persist Evaluation Metrics
        eval_scores = sanitize_json_obj(final_state.get("evaluation_scores", {}))
        if eval_scores:
            try:
                metric_entity = EvaluationMetric(
                    id=str(uuid.uuid4()),
                    run_id=run_id,
                    query=payload.query,
                    context_precision=float(eval_scores.get("context_precision", 0.0)),
                    faithfulness_score=float(eval_scores.get("faithfulness_score", 0.0)),
                    answer_relevance=float(eval_scores.get("answer_relevance", 0.0)),
                    evaluation_details=eval_scores
                )
                db.add(metric_entity)
            except Exception as e:
                logger.warning("Metric entity persistence warning: %s", e)

        try:
            await db.commit()
        except Exception as e:
            logger.warning("DB commit warning: %s", e)

        # Format Citations & Traces
        raw_citations = final_state.get("citations", [])
        formatted_citations = [
            {
                "citation_index": int(c["citation_index"]),
                "chunk_id": str(c["chunk_id"]),
                "document_id": str(c["document_id"]),
                "relevance_score": float(c["relevance_score"]),
                "snippet": str(c["snippet"]),
                "metadata": sanitize_json_obj(c.get("metadata", {}))
            }
            for c in raw_citations
        ]

        formatted_traces = [
            {
                "step_name": str(t["step_name"]),
                "step_index": int(t["step_index"]),
                "input_state": sanitize_json_obj(t.get("input_state", {})),
                "output_state": sanitize_json_obj(t.get("output_state", {})),
                "latency_ms": float(t.get("latency_ms", 0.0))
            }
            for t in traces
        ]

        response_dict = {
            "run_id": run_id,
            "query": payload.query,
            "rewritten_query": str(final_state.get("rewritten_query", "")),
            "answer": str(final_state.get("generation", "")),
            "citations": formatted_citations,
            "evaluation_metrics": eval_scores,
            "execution_trace": formatted_traces,
            "iterations": int(final_state.get("iterations", 0))
        }

        return JSONResponse(content=sanitize_json_obj(response_dict))
    except Exception as exc:
        logger.exception("Error executing RAG query: %s", exc)
        import traceback
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"Query execution error: {str(exc)}",
                "traceback": traceback.format_exc()
            }
        )


@app.get("/v1/traces/{run_id}", tags=["Observability"])
async def get_execution_trace(run_id: str, db: AsyncSession = Depends(get_db)):
    """
    Retrieves full step-by-step latency, input/output transitions, and evaluation metrics
    for a completed run from PostgreSQL.
    """
    try:
        run_uuid = uuid.UUID(run_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid UUID string format: '{run_id}'"
        )

    # Fetch Traces
    stmt_traces = (
        select(AgentExecutionTrace)
        .where(AgentExecutionTrace.run_id == run_uuid)
        .order_by(AgentExecutionTrace.step_index.asc())
    )
    res_traces = await db.execute(stmt_traces)
    trace_rows = res_traces.scalars().all()

    if not trace_rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No execution traces found for run ID: '{run_id}'"
        )

    # Fetch Evaluation Metric
    stmt_metric = select(EvaluationMetric).where(EvaluationMetric.run_id == run_uuid)
    res_metric = await db.execute(stmt_metric)
    metric_row = res_metric.scalar_one_or_none()

    total_latency = sum(t.latency_ms for t in trace_rows)

    return {
        "run_id": run_id,
        "total_steps": len(trace_rows),
        "total_latency_ms": round(total_latency, 2),
        "evaluation_metric": {
            "context_precision": metric_row.context_precision if metric_row else None,
            "faithfulness_score": metric_row.faithfulness_score if metric_row else None,
            "answer_relevance": metric_row.answer_relevance if metric_row else None,
            "details": metric_row.evaluation_details if metric_row else None
        } if metric_row else None,
        "steps": [
            {
                "step_name": t.step_name,
                "step_index": t.step_index,
                "input_state": t.input_state,
                "output_state": t.output_state,
                "latency_ms": t.latency_ms,
                "timestamp": t.created_at.isoformat()
            }
            for t in trace_rows
        ]
    }
