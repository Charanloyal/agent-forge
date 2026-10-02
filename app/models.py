import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Text,
    Integer,
    Float,
    DateTime,
    Index,
    func
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.types import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models."""
    pass


class DocumentChunk(Base):
    """
    Persisted raw and chunked document data with vector mapping references.
    """
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        nullable=False,
    )
    document_id: Mapped[str] = mapped_column(
        String(255),
        index=True,
        nullable=False,
        comment="Logical parent document identifier"
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        comment="Sequential index of this chunk within the parent document"
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Extracted textual content of the chunk"
    )
    metadata_json: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=dict,
        comment="Arbitrary metadata such as title, source URI, authors, tags"
    )
    embedding_id: Mapped[str] = mapped_column(
        String(64),
        index=True,
        nullable=False,
        comment="Unique point ID reference in the Qdrant vector collection"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )

    __table_args__ = (
        Index("ix_doc_chunk_lookup", "document_id", "chunk_index"),
    )


class AgentExecutionTrace(Base):
    """
    Tracks step-by-step latency, node transitions, and execution graph states
    for every LangGraph run invocation.
    """
    __tablename__ = "agent_execution_traces"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        nullable=False
    )
    run_id: Mapped[str] = mapped_column(
        String(36),
        index=True,
        nullable=False,
        comment="Unique identifier for the entire end-to-end query run"
    )
    step_name: Mapped[str] = mapped_column(
        String(100),
        index=True,
        nullable=False,
        comment="Name of the executing agent node"
    )
    step_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Sequential step counter in the LangGraph trajectory"
    )
    input_state: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=dict,
        comment="Snapshot of relevant state fields before node execution"
    )
    output_state: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=dict,
        comment="State delta or update produced by this node"
    )
    latency_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Execution time in milliseconds for this node"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )

    __table_args__ = (
        Index("ix_trace_run_step", "run_id", "step_index"),
    )


class EvaluationMetric(Base):
    """
    Persisted evaluation benchmark scores for groundedness, context precision,
    and answer relevance calculated by the programmatic scoring harness.
    """
    __tablename__ = "evaluation_metrics"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        nullable=False
    )
    run_id: Mapped[str] = mapped_column(
        String(36),
        index=True,
        unique=True,
        nullable=False,
        comment="Run ID associated with this evaluation"
    )
    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Original user query input"
    )
    context_precision: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Context precision score (0.0 to 1.0)"
    )
    faithfulness_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Sentence-level groundedness/faithfulness score (0.0 to 1.0)"
    )
    answer_relevance: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Semantic cosine similarity between original query and answer"
    )
    evaluation_details: Mapped[dict] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=dict,
        comment="Granular per-sentence claims, retrieved chunk scores, and NLI flags"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
