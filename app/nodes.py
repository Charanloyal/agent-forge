import re
import time
import asyncio
import logging
from typing import TypedDict, Any
from app.config import get_settings
from app.retrieval import (
    qdrant_service,
    bm25_store,
    reciprocal_rank_fusion,
    rerank_chunks
)
from app.evaluator import (
    evaluate_response,
    split_sentences,
    compute_faithfulness
)

logger = logging.getLogger("agentforge.nodes")
settings = get_settings()


class AgentState(TypedDict):
    """
    Strongly typed state schema shared across all LangGraph nodes.
    """
    run_id: str
    query: str
    rewritten_query: str
    retrieved_documents: list[dict[str, Any]]
    filtered_documents: list[dict[str, Any]]
    generation: str
    citations: list[dict[str, Any]]
    evaluation_scores: dict[str, Any]
    iterations: int
    needs_revision: bool
    traces: list[dict[str, Any]]


class QueryRewriterNode:
    """
    Rewrites conversational, noisy, or ambiguous queries into search-optimized queries.
    On reflection cycles, applies query relaxation and term expansion.
    """
    CONVERSATIONAL_PREFIXES = [
        r"^can you tell me (about)?",
        r"^what do you know about",
        r"^could you please explain",
        r"^please explain",
        r"^tell me about",
        r"^i want to know about",
        r"^i am looking for information on",
        r"^help me understand",
        r"^give me details on",
        r"^what is",
        r"^who is",
        r"^where is",
        r"^how does",
        r"^why does",
    ]

    @classmethod
    def clean_query(cls, text: str) -> str:
        cleaned = text.strip()
        for pat in cls.CONVERSATIONAL_PREFIXES:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"[^\w\s\-\.]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned if cleaned else text.strip()

    @classmethod
    async def execute(cls, state: AgentState) -> dict[str, Any]:
        start_time = time.perf_counter()
        raw_query = state.get("query", "")
        iterations = state.get("iterations", 0)
        needs_revision = state.get("needs_revision", False)

        cleaned = cls.clean_query(raw_query)

        # Multi-hop query relaxation & expansion if looping back
        if needs_revision and iterations > 0:
            words = cleaned.split()
            # Extract core substantive terms (length >= 4)
            keywords = [w for w in words if len(w) >= 4]
            if len(keywords) > 2:
                rewritten = " ".join(keywords) + " overview fundamentals key facts"
            else:
                rewritten = cleaned + " overview documentation"
            logger.info("Reflection iteration %d: Query relaxed to '%s'", iterations, rewritten)
        else:
            rewritten = cleaned
            logger.info("Initial query normalized to: '%s'", rewritten)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        step_trace = {
            "step_name": "QueryRewriterNode",
            "step_index": len(state.get("traces", [])) + 1,
            "input_state": {"query": raw_query, "iterations": iterations},
            "output_state": {"rewritten_query": rewritten},
            "latency_ms": elapsed_ms
        }

        return {
            "rewritten_query": rewritten,
            "traces": state.get("traces", []) + [step_trace]
        }


class HybridRetrievalNode:
    """
    Executes parallel retrieval (Qdrant dense vector search + BM25 keyword search)
    and merges candidate results via Reciprocal Rank Fusion (RRF with k=60).
    """
    @classmethod
    async def execute(cls, state: AgentState) -> dict[str, Any]:
        start_time = time.perf_counter()
        search_query = state.get("rewritten_query") or state.get("query", "")

        # Parallel asynchronous execution of Dense Vector Search & Sparse BM25 Search
        dense_task = qdrant_service.search_dense(
            query=search_query,
            top_k=settings.TOP_K_DENSE
        )
        # BM25 is in-memory CPU bound; run in executor to prevent event loop blocking
        loop = asyncio.get_running_loop()
        sparse_task = loop.run_in_executor(
            None,
            bm25_store.search,
            search_query,
            settings.TOP_K_SPARSE
        )

        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        # Merge via Reciprocal Rank Fusion
        fused_candidates = reciprocal_rank_fusion(
            dense_results=dense_results,
            sparse_results=sparse_results,
            k=settings.RRF_K,
            top_n=settings.TOP_K_FUSED
        )

        logger.info(
            "Hybrid retrieval returned %d fused documents (Dense: %d, Sparse: %d)",
            len(fused_candidates), len(dense_results), len(sparse_results)
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        step_trace = {
            "step_name": "HybridRetrievalNode",
            "step_index": len(state.get("traces", [])) + 1,
            "input_state": {"search_query": search_query},
            "output_state": {"retrieved_count": len(fused_candidates)},
            "latency_ms": elapsed_ms
        }

        return {
            "retrieved_documents": fused_candidates,
            "traces": state.get("traces", []) + [step_trace]
        }


class RerankingNode:
    """
    Uses Cross-Encoder scoring model (cross-encoder/ms-marco-MiniLM-L-6-v2)
    to calculate pairwise query-document relevance logits and prune chunks
    below the defined relevance threshold.
    """
    @classmethod
    async def execute(cls, state: AgentState) -> dict[str, Any]:
        start_time = time.perf_counter()
        search_query = state.get("rewritten_query") or state.get("query", "")
        candidates = state.get("retrieved_documents", [])

        # Run CPU/GPU heavy cross-encoder reranker in worker thread
        loop = asyncio.get_running_loop()
        filtered = await loop.run_in_executor(
            None,
            rerank_chunks,
            search_query,
            candidates,
            settings.RERANKER_SCORE_THRESHOLD,
            settings.TOP_K_RERANKED
        )

        logger.info(
            "Reranking node pruned %d candidates down to %d surviving chunks",
            len(candidates), len(filtered)
        )

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        step_trace = {
            "step_name": "RerankingNode",
            "step_index": len(state.get("traces", [])) + 1,
            "input_state": {"candidate_count": len(candidates), "threshold": settings.RERANKER_SCORE_THRESHOLD},
            "output_state": {"filtered_count": len(filtered)},
            "latency_ms": elapsed_ms
        }

        return {
            "filtered_documents": filtered,
            "traces": state.get("traces", []) + [step_trace]
        }


class RelevanceGraderNode:
    """
    Evaluates whether surviving chunks meet the minimum relevance criteria.
    If zero chunks survived or top score is under threshold, flags state
    for query relaxation routing.
    """
    @classmethod
    async def execute(cls, state: AgentState) -> dict[str, Any]:
        start_time = time.perf_counter()
        filtered = state.get("filtered_documents", [])
        iterations = state.get("iterations", 0)

        has_relevant = len(filtered) > 0
        needs_relaxation = False

        if not has_relevant:
            logger.warning("Zero chunks passed reranking threshold. Flagging for query relaxation.")
            needs_relaxation = True
        else:
            top_score = filtered[0].get("rerank_score", 0.0)
            if top_score < settings.RELEVANCE_PASS_THRESHOLD:
                logger.warning("Top rerank score %.3f below pass threshold %.3f", top_score, settings.RELEVANCE_PASS_THRESHOLD)
                needs_relaxation = True

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        step_trace = {
            "step_name": "RelevanceGraderNode",
            "step_index": len(state.get("traces", [])) + 1,
            "input_state": {"surviving_chunks": len(filtered), "iterations": iterations},
            "output_state": {"needs_relaxation": needs_relaxation, "passed": not needs_relaxation},
            "latency_ms": elapsed_ms
        }

        # If revision needed, increment iterations for the next attempt
        new_iterations = iterations + 1 if needs_relaxation else iterations

        return {
            "needs_revision": needs_relaxation,
            "iterations": new_iterations,
            "traces": state.get("traces", []) + [step_trace]
        }


class GeneratorNode:
    """
    Synthesizes the final grounded answer with bracketed inline citation indices (e.g. [1], [2]).
    Compiles a structured citations index referencing chunk IDs, document IDs, and source text.
    """
    @classmethod
    async def execute(cls, state: AgentState) -> dict[str, Any]:
        start_time = time.perf_counter()
        query = state.get("query", "")
        documents = state.get("filtered_documents", [])

        if not documents:
            generation = (
                f"I researched your query regarding '{query}', but no authoritative context "
                "surpassed the required relevance threshold in the knowledge base."
            )
            citations = []
        else:
            # Build structured citations and grounded synthetic response
            citations: list[dict[str, Any]] = []
            answer_segments: list[str] = []

            for idx, doc in enumerate(documents, start=1):
                chunk_id = str(doc.get("id", ""))
                doc_id = str(doc.get("document_id", "doc"))
                content = doc.get("content", "").strip()
                score = doc.get("rerank_score", doc.get("rrf_score", 0.0))

                citations.append({
                    "citation_index": idx,
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "relevance_score": round(float(score), 4),
                    "snippet": content[:200] + "..." if len(content) > 200 else content,
                    "metadata": doc.get("metadata", {})
                })

                # Extract key sentences from the chunk
                doc_sentences = split_sentences(content)
                key_sentence = doc_sentences[0] if doc_sentences else content
                answer_segments.append(f"{key_sentence} [{idx}]")

            generation = " ".join(answer_segments)

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        step_trace = {
            "step_name": "GeneratorNode",
            "step_index": len(state.get("traces", [])) + 1,
            "input_state": {"documents_used": len(documents)},
            "output_state": {"generation_length": len(generation), "citations_count": len(citations)},
            "latency_ms": elapsed_ms
        }

        return {
            "generation": generation,
            "citations": citations,
            "traces": state.get("traces", []) + [step_trace]
        }


class HallucinationCheckerNode:
    """
    Evaluates whether generated claims are strictly supported by the cited context.
    If groundedness fails and iterations < MAX_REFLECTION_ITERATIONS, cycles back to rewrite/retrieve.
    """
    @classmethod
    async def execute(cls, state: AgentState) -> dict[str, Any]:
        start_time = time.perf_counter()
        generation = state.get("generation", "")
        documents = state.get("filtered_documents", [])
        iterations = state.get("iterations", 0)

        # Run faithfulness scoring
        loop = asyncio.get_running_loop()
        faithfulness_score, claim_evals = await loop.run_in_executor(
            None,
            compute_faithfulness,
            generation,
            documents,
            0.45
        )

        has_hallucination = faithfulness_score < settings.FAITHFULNESS_PASS_THRESHOLD
        needs_revision = has_hallucination and (iterations < settings.MAX_REFLECTION_ITERATIONS)

        if needs_revision:
            logger.warning(
                "Hallucination detected (Faithfulness: %.3f < %.3f). Initiating reflection cycle %d.",
                faithfulness_score, settings.FAITHFULNESS_PASS_THRESHOLD, iterations + 1
            )
            new_iterations = iterations + 1
        else:
            new_iterations = iterations

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        step_trace = {
            "step_name": "HallucinationCheckerNode",
            "step_index": len(state.get("traces", [])) + 1,
            "input_state": {"generation_claims": len(claim_evals)},
            "output_state": {
                "faithfulness_score": faithfulness_score,
                "needs_revision": needs_revision,
                "iterations": new_iterations
            },
            "latency_ms": elapsed_ms
        }

        return {
            "needs_revision": needs_revision,
            "iterations": new_iterations,
            "traces": state.get("traces", []) + [step_trace]
        }


class EvaluationNode:
    """
    Executes the programmatic evaluation scoring harness on the completed generation:
    - Context Precision
    - Faithfulness / Groundedness Score
    - Answer Relevance (Cosine similarity)
    """
    @classmethod
    async def execute(cls, state: AgentState) -> dict[str, Any]:
        start_time = time.perf_counter()
        query = state.get("query", "")
        generation = state.get("generation", "")
        documents = state.get("filtered_documents", []) or state.get("retrieved_documents", [])

        loop = asyncio.get_running_loop()
        evaluation_result = await loop.run_in_executor(
            None,
            evaluate_response,
            query,
            generation,
            documents,
            settings.RELEVANCE_PASS_THRESHOLD,
            0.45
        )

        scores_dict = evaluation_result.model_dump()
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        step_trace = {
            "step_name": "EvaluationNode",
            "step_index": len(state.get("traces", [])) + 1,
            "input_state": {"query": query, "generation_length": len(generation)},
            "output_state": {
                "context_precision": scores_dict["context_precision"],
                "faithfulness_score": scores_dict["faithfulness_score"],
                "answer_relevance": scores_dict["answer_relevance"]
            },
            "latency_ms": elapsed_ms
        }

        return {
            "evaluation_scores": scores_dict,
            "traces": state.get("traces", []) + [step_trace]
        }
