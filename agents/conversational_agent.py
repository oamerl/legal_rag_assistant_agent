"""
Conversational RAG agent.

Uses LangGraph checkpointer for session state and provides a tool to run
the full retrieval and generation pipeline.
"""

from __future__ import annotations

import logging
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import get_settings
from core.models import PipelineData
from core.memory import memory_provider
from generation.pipeline import create_generation_pipeline
from retrieval.pipeline import create_retrieval_pipeline
from generation.llm_client import create_llm_client

logger = logging.getLogger(__name__)


@tool(return_direct=True)
def ask_legal_rag_pipeline(query: str) -> str:
    """
    Searches the uploaded legal documents and generates a highly accurate, structured answer with citations.
    Use this tool whenever the user asks a legal question or asks for information from the documents.
    Pass the user's exact input as the query without any rephrasing, summarization, or modifications.
    Do not use this tool for conversational pleasantries.
    """
    logger.info("Agent invoked RAG pipeline tool for query: %s", query)
    
    # 1. Retrieval
    retrieval_pipeline = create_retrieval_pipeline()
    data = PipelineData(query=query)
    
    try:
        data = retrieval_pipeline.run(data)
    except Exception as exc:
        logger.exception("Retrieval failed during agent tool execution.")
        return f"Retrieval failed: {exc}"
        
    # 2. Generation
    generation_pipeline = create_generation_pipeline()
    
    try:
        data = generation_pipeline.run(data)
    except Exception as exc:
        logger.exception("Generation failed during agent tool execution.")
        return f"Generation failed: {exc}"
        
    answer = data.legal_answer
    if answer is None:
        return "No answer could be generated from the documents."
        
    # Format the structured answer into a string for the agent to read
    output_parts = [
        f"Answer (confidence: {answer.confidence:.1%}):\n{answer.answer}\n"
    ]
    
    if answer.citations:
        output_parts.append("Citations:")
        for i, cit in enumerate(answer.citations, 1):
            cit_str = f"[{i}] {cit.document_name or 'Unknown Document'}"
            if cit.page_number:
                cit_str += f", Page {cit.page_number}"
            if cit.section_title or cit.section_id:
                sec = cit.section_title or cit.section_id
                cit_str += f", Section: {sec}"
            output_parts.append(cit_str)
            if cit.quoted_text:
                output_parts.append(f"    \"{cit.quoted_text}\"")
                
    if answer.caveats:
        output_parts.append(f"\nCaveats: {', '.join(answer.caveats)}")
        
    if answer.unanswered_parts:
        output_parts.append(f"\nUnanswered: {', '.join(answer.unanswered_parts)}")
        
    # Include RAGAS metrics if available so the agent knows the quality
    if data.guard_result and data.guard_result.faithfulness_score is not None:
        f_score = data.guard_result.faithfulness_score
        r_score = data.guard_result.answer_relevancy_score or 0.0
        
        if f_score > 0.0 or r_score > 0.0:
            output_parts.append(f"\nMetrics: Faithfulness={f_score:.2f}, Relevancy={r_score:.2f}")

    return "\n".join(output_parts)


def get_conversational_agent():
    """
    Creates and returns the conversational LangGraph agent with SQLite checkpointer.
    """
    llm = create_llm_client()
    
    system_message = (
        "You are a helpful and professional Legal AI Assistant. "
        "Your primary job is to help users understand their uploaded legal documents. "
        "Whenever a user asks a question about the documents, the law, or a specific topic, "
        "you MUST use the `ask_legal_rag_pipeline` tool to find the information. "
        "IMPORTANT: When calling the tool, you must pass the exact raw message from the user without altering, summarizing, or rephrasing a single word. "
        "The retrieval pipeline handles query transformation itself. "
        "If the user says hi or asks a general conversational question, you can reply directly but never answer any legal question without using the tool."
    )
    
    # We use the centralized memory provider which manages the SQLite connection, 
    # directory creation, and LangGraph's .setup()
    checkpointer = memory_provider.get_checkpointer("default")
    
    agent_executor = create_agent(
        llm,
        tools=[ask_legal_rag_pipeline],
        system_prompt=system_message,
        checkpointer=checkpointer
    )
    
    return agent_executor

