import pytest
from app.evaluator import (
    split_sentences,
    calculate_cosine_similarity,
    compute_context_precision,
    compute_faithfulness,
    compute_answer_relevance,
    evaluate_response
)
import numpy as np


def test_split_sentences():
    text = "AgentForge is an enterprise platform [1]. It uses Reciprocal Rank Fusion [2]! Does it support Qdrant?"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert "[1]" not in sentences[0]
    assert "[2]" not in sentences[1]


def test_calculate_cosine_similarity():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    assert calculate_cosine_similarity(v1, v2) == pytest.approx(1.0)

    v3 = np.array([0.0, 1.0, 0.0])
    assert calculate_cosine_similarity(v1, v3) == pytest.approx(0.5)  # normalized (0+1)/2


def test_compute_context_precision():
    chunks = [
        {"content": "AgentForge fuses dense vectors and BM25 sparse scores with RRF."},
        {"content": "Irrelevant text about baking pizza crust at high temperatures."}
    ]
    query = "How does AgentForge fuse search scores?"

    avg_prec, prec_list = compute_context_precision(query, chunks, threshold=0.20)
    assert avg_prec > 0.0
    assert len(prec_list) == 2


def test_compute_faithfulness():
    chunks = [
        {"id": "c1", "content": "AgentForge persists execution traces in PostgreSQL using asyncpg."}
    ]
    answer = "AgentForge persists execution traces in PostgreSQL using asyncpg [1]."

    score, evals = compute_faithfulness(answer, chunks, entailment_threshold=0.30)
    assert score >= 0.8
    assert len(evals) == 1
    assert evals[0].is_grounded is True


def test_evaluate_response_harness():
    query = "What database is used in AgentForge?"
    answer = "AgentForge uses PostgreSQL for storing traces and metrics."
    chunks = [
        {"id": "c1", "content": "AgentForge stores all traces and metrics in PostgreSQL."}
    ]

    metrics = evaluate_response(query, answer, chunks)
    assert metrics.context_precision >= 0.0
    assert metrics.faithfulness_score >= 0.0
    assert metrics.answer_relevance >= 0.0
    assert metrics.total_claims == 1
