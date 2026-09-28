import re
import math
import numpy as np
from typing import Any
from pydantic import BaseModel, Field
from app.retrieval import ModelRegistry, sigmoid


class ClaimEvaluation(BaseModel):
    statement: str
    is_grounded: bool
    support_score: float
    supporting_chunk_id: str | None = None


class EvaluationMetrics(BaseModel):
    context_precision: float = Field(..., ge=0.0, le=1.0)
    faithfulness_score: float = Field(..., ge=0.0, le=1.0)
    answer_relevance: float = Field(..., ge=0.0, le=1.0)
    total_claims: int
    grounded_claims: int
    claim_evaluations: list[ClaimEvaluation] = []
    ranked_chunk_precisions: list[float] = []


def split_sentences(text: str) -> list[str]:
    """
    Splits generation text into atomic sentence units, stripping citation tags like [1], [2].
    """
    clean_text = re.sub(r"\[\d+\]", "", text)
    # Split on sentence terminals followed by whitespace
    raw_sentences = re.split(r"(?<=[.!?])\s+", clean_text.strip())
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]
    return sentences if sentences else [clean_text.strip()] if clean_text.strip() else []


def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Calculates normalized cosine similarity between two 1D vector representations.
    cos(theta) = (A . B) / (||A|| * ||B||)
    """
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot_prod = np.dot(vec_a, vec_b)
    similarity = float(dot_prod / (norm_a * norm_b))
    return max(0.0, min(1.0, (similarity + 1.0) / 2.0))


def compute_context_precision(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    threshold: float = 0.25
) -> tuple[float, list[float]]:
    """
    Calculates Context Precision@K.
    Precision@k = (Count of relevant documents in top-k) / k
    Overall Context Precision = Mean Average Precision (MAP) of relevant retrieved documents.
    """
    if not retrieved_chunks:
        return 0.0, []

    model = ModelRegistry.get_reranker_model()
    pairs = [[query, chunk["content"]] for chunk in retrieved_chunks]
    raw_scores = model.predict(pairs)

    relevant_flags: list[bool] = []
    precision_at_k: list[float] = []
    running_relevant = 0

    for k, raw_score in enumerate(raw_scores, start=1):
        norm_score = sigmoid(float(raw_score))
        is_rel = norm_score >= threshold
        if is_rel:
            running_relevant += 1
        relevant_flags.append(is_rel)
        precision_at_k.append(running_relevant / k)

    total_relevant = sum(relevant_flags)
    if total_relevant == 0:
        return 0.0, precision_at_k

    # Calculate MAP: sum of (Precision@k * rel(k)) / total_relevant
    avg_precision = sum(
        p * (1 if rel else 0) for p, rel in zip(precision_at_k, relevant_flags)
    ) / total_relevant

    return round(float(avg_precision), 4), [round(p, 4) for p in precision_at_k]


def compute_faithfulness(
    answer: str,
    context_chunks: list[dict[str, Any]],
    entailment_threshold: float = 0.45
) -> tuple[float, list[ClaimEvaluation]]:
    """
    Evaluates groundedness: checks whether each atomic statement in the generated answer
    is semantically supported by the cited context chunks without hallucination.
    Uses Cross-Encoder sentence-pair scoring against context passages.
    """
    statements = split_sentences(answer)
    if not statements:
        return 1.0, []

    if not context_chunks:
        # If no context exists but statements were generated, groundedness is 0
        evals = [
            ClaimEvaluation(statement=stmt, is_grounded=False, support_score=0.0)
            for stmt in statements
        ]
        return 0.0, evals

    reranker = ModelRegistry.get_reranker_model()
    claim_evaluations: list[ClaimEvaluation] = []
    grounded_count = 0

    for stmt in statements:
        best_score = -1.0
        best_chunk_id = None

        # Compare claim against each context chunk
        pairs = [[chunk["content"], stmt] for chunk in context_chunks]
        scores = reranker.predict(pairs)

        for chunk, score in zip(context_chunks, scores):
            norm_score = sigmoid(float(score))
            if norm_score > best_score:
                best_score = norm_score
                best_chunk_id = str(chunk.get("id", ""))

        is_grounded = best_score >= entailment_threshold
        if is_grounded:
            grounded_count += 1

        claim_evaluations.append(
            ClaimEvaluation(
                statement=stmt,
                is_grounded=is_grounded,
                support_score=round(best_score, 4),
                supporting_chunk_id=best_chunk_id if is_grounded else None
            )
        )

    faithfulness_ratio = round(grounded_count / len(statements), 4)
    return faithfulness_ratio, claim_evaluations


def compute_answer_relevance(query: str, answer: str) -> float:
    """
    Calculates semantic cosine similarity between the user's original query and
    the synthesized answer using dense embeddings.
    """
    clean_answer = answer.strip()
    if not clean_answer:
        return 0.0

    embedder = ModelRegistry.get_embedding_model()
    embeddings = embedder.encode([query, clean_answer], normalize_embeddings=True)
    vec_q = embeddings[0]
    vec_a = embeddings[1]

    relevance = calculate_cosine_similarity(vec_q, vec_a)
    return round(float(relevance), 4)


def evaluate_response(
    query: str,
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
    precision_threshold: float = 0.25,
    entailment_threshold: float = 0.45
) -> EvaluationMetrics:
    """
    Unified programmatic evaluation harness.
    Executes Context Precision, Faithfulness/Groundedness, and Answer Relevance.
    """
    context_precision, precision_at_k = compute_context_precision(
        query=query,
        retrieved_chunks=retrieved_chunks,
        threshold=precision_threshold
    )

    faithfulness, claim_evals = compute_faithfulness(
        answer=answer,
        context_chunks=retrieved_chunks,
        entailment_threshold=entailment_threshold
    )

    answer_relevance = compute_answer_relevance(
        query=query,
        answer=answer
    )

    grounded_count = sum(1 for c in claim_evals if c.is_grounded)

    return EvaluationMetrics(
        context_precision=context_precision,
        faithfulness_score=faithfulness,
        answer_relevance=answer_relevance,
        total_claims=len(claim_evals),
        grounded_claims=grounded_count,
        claim_evaluations=claim_evals,
        ranked_chunk_precisions=precision_at_k
    )
