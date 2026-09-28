import uuid
import logging
from langgraph.graph import StateGraph, START, END
from app.config import get_settings
from app.nodes import (
    AgentState,
    QueryRewriterNode,
    HybridRetrievalNode,
    RerankingNode,
    RelevanceGraderNode,
    GeneratorNode,
    HallucinationCheckerNode,
    EvaluationNode
)

logger = logging.getLogger("agentforge.graph")
settings = get_settings()


def route_after_relevance_grader(state: AgentState) -> str:
    """
    Conditional edge router evaluating whether chunks survived reranking.
    If zero chunks survived or top score is inadequate, cycles back to
    QueryRewriterNode for query relaxation (up to MAX_REFLECTION_ITERATIONS).
    """
    needs_revision = state.get("needs_revision", False)
    iterations = state.get("iterations", 0)

    if needs_revision and iterations < settings.MAX_REFLECTION_ITERATIONS:
        logger.info(
            "Routing to 'query_rewriter' for query relaxation (iteration %d of %d)",
            iterations, settings.MAX_REFLECTION_ITERATIONS
        )
        return "query_rewriter"

    logger.info("Relevance gate passed or max iterations reached. Routing to 'generator'.")
    return "generator"


def route_after_hallucination_checker(state: AgentState) -> str:
    """
    Conditional edge router evaluating faithfulness / groundedness score.
    If claims are unsupported by context, cycles back to QueryRewriterNode
    for multi-hop reflection and broader retrieval (up to MAX_REFLECTION_ITERATIONS).
    """
    needs_revision = state.get("needs_revision", False)
    iterations = state.get("iterations", 0)

    if needs_revision and iterations < settings.MAX_REFLECTION_ITERATIONS:
        logger.info(
            "Routing to 'query_rewriter' due to hallucination detection (iteration %d of %d)",
            iterations, settings.MAX_REFLECTION_ITERATIONS
        )
        return "query_rewriter"

    logger.info("Hallucination check passed or max iterations reached. Routing to 'evaluator'.")
    return "evaluator"


def build_agent_graph() -> StateGraph:
    """
    Assembles the production LangGraph state machine with cyclic reflection loops.
    """
    builder = StateGraph(AgentState)

    # Register Nodes
    builder.add_node("query_rewriter", QueryRewriterNode.execute)
    builder.add_node("hybrid_retriever", HybridRetrievalNode.execute)
    builder.add_node("reranker", RerankingNode.execute)
    builder.add_node("relevance_grader", RelevanceGraderNode.execute)
    builder.add_node("generator", GeneratorNode.execute)
    builder.add_node("hallucination_checker", HallucinationCheckerNode.execute)
    builder.add_node("evaluator", EvaluationNode.execute)

    # Establish Workflow Edges
    builder.add_edge(START, "query_rewriter")
    builder.add_edge("query_rewriter", "hybrid_retriever")
    builder.add_edge("hybrid_retriever", "reranker")
    builder.add_edge("reranker", "relevance_grader")

    # Conditional Branching 1: Relevance Check -> Query Rewriter (Relaxation) OR Generator
    builder.add_conditional_edges(
        "relevance_grader",
        route_after_relevance_grader,
        {
            "query_rewriter": "query_rewriter",
            "generator": "generator"
        }
    )

    # Generator -> Hallucination Checker
    builder.add_edge("generator", "hallucination_checker")

    # Conditional Branching 2: Hallucination Check -> Query Rewriter (Multi-Hop Reflection) OR Evaluator
    builder.add_conditional_edges(
        "hallucination_checker",
        route_after_hallucination_checker,
        {
            "query_rewriter": "query_rewriter",
            "evaluator": "evaluator"
        }
    )

    # Final Evaluation -> END
    builder.add_edge("evaluator", END)

    return builder


# Compile the singleton runnable graph
compiled_agent_graph = build_agent_graph().compile()


async def run_agent_workflow(query: str, run_id: str | None = None) -> AgentState:
    """
    Executes the compiled AgentForge LangGraph state machine from an initial query.
    """
    execution_id = run_id or str(uuid.uuid4())
    initial_state: AgentState = {
        "run_id": execution_id,
        "query": query,
        "rewritten_query": "",
        "retrieved_documents": [],
        "filtered_documents": [],
        "generation": "",
        "citations": [],
        "evaluation_scores": {},
        "iterations": 0,
        "needs_revision": False,
        "traces": []
    }

    final_state = await compiled_agent_graph.ainvoke(initial_state)
    return final_state
