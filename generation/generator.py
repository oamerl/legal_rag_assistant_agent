"""
LLM generator — produces grounded legal answers with citations.

Uses LangChain's structured output to enforce the LegalAnswer schema,
ensuring every response includes citations, confidence, and caveats.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from core.exceptions import GenerationError
from core.models import Citation, LegalAnswer, PipelineData
from core.pipeline import PipelineStage
from generation.llm_client import create_llm_client
from generation.prompt_templates import (
    LEGAL_QA_SYSTEM_PROMPT,
    NOT_FOUND_SYSTEM_PROMPT,
    format_not_found_prompt,
    format_user_prompt,
)

logger = logging.getLogger(__name__)


class CitationSchema(BaseModel):
    """Schema for a single citation extracted from the context."""
    document_name: str | None = Field(description="Name of the source document. Leave empty or null if not provided in the context.", default="")
    page_number: int | None = Field(description="Page number where the claim is supported. Leave empty or null if not explicitly stated in the context.", default=0)
    section_id: str | None = Field(description="Section ID of the source text. Leave empty or null if not provided.", default="")
    section_title: str | None = Field(description="Title of the section. Leave empty or null if not provided.", default="")
    quoted_text: str = Field(description="Exact quote from the source supporting the claim. Must be an exact substring match.", default="")


class LegalAnswerSchema(BaseModel):
    """Schema for the structured output of the generator."""
    answer: str = Field(description="The detailed legal answer to the user's query")
    citations: list[CitationSchema] = Field(default_factory=list, description="List of citations used in the answer")
    unanswered_parts: list[str] | None = Field(default=None, description="Parts of the query that could not be answered with the given context. Leave null if fully answered.")
    caveats: list[str] | None = Field(default=None, description="Any caveats or limitations of the answer. Leave null if none.")


class LegalGenerator:
    """
    Generates grounded legal answers with structured citations.

    Uses a two-path approach:
        - If context is available → generate answer with citations
        - If no relevant context → generate a "not found" response
    """

    def __init__(self, llm=None) -> None:
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            self._llm = create_llm_client()
        return self._llm

    def generate(
        self,
        query: str,
        context: str,
        is_relevant: bool = True,
    ) -> LegalAnswer:
        """
        Generate a legal answer from query and retrieved context.

        Parameters
        ----------
        query : str
            The user's question.
        context : str
            Assembled context from the retrieval pipeline.
        is_relevant : bool
            Whether the retrieval found relevant chunks.

        Returns
        -------
        LegalAnswer
            Structured answer with citations and confidence.
        """
        try:
            llm = self._get_llm()

            if not is_relevant or not context.strip():
                return self._generate_not_found(query, llm)

            return self._generate_answer(query, context, llm)

        except GenerationError:
            raise
        except Exception as exc:
            raise GenerationError(f"Generation failed: {exc}") from exc

    def _generate_answer(self, query: str, context: str, llm) -> LegalAnswer:
        """Generate a grounded answer with citations."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=LEGAL_QA_SYSTEM_PROMPT),
            HumanMessage(content=format_user_prompt(query, context)),
        ]

        logger.info("Generating answer for: '%s'", query[:80])
        structured_llm = llm.with_structured_output(LegalAnswerSchema)
        response = structured_llm.invoke(messages)

        citations = [
            Citation(
                document_name=c.document_name,
                page_number=c.page_number,
                section_id=c.section_id,
                section_title=c.section_title,
                quoted_text=c.quoted_text,
            ) for c in response.citations
        ]

        answer = LegalAnswer(
            answer=response.answer,
            citations=citations,
            confidence=0.0,     # Default; will be updated with reranker scores average
            unanswered_parts=response.unanswered_parts or [],
            caveats=response.caveats or [],
        )

        logger.info("Generated answer (%d chars)", len(answer.answer))
        return answer

    def _generate_not_found(self, query: str, llm) -> LegalAnswer:
        """Generate a helpful not-found response."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=NOT_FOUND_SYSTEM_PROMPT),
            HumanMessage(content=format_not_found_prompt(query)),
        ]

        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        return LegalAnswer(
            answer=content,
            citations=[],
            confidence=0.0,
            unanswered_parts=[query],
            caveats=["No relevant information found in uploaded documents."],
        )


class GenerationStage(PipelineStage):
    """Pipeline stage: generate the final legal answer."""

    def __init__(self, generator: LegalGenerator | None = None) -> None:
        self._generator = generator or LegalGenerator()

    def process(self, data: PipelineData) -> PipelineData:
        answer = self._generator.generate(
            query=data.query,
            context=data.assembled_context,
            is_relevant=data.retrieval_is_relevant,
        )

        # Enrich confidence from reranker scores
        if data.ranked_chunks and data.retrieval_is_relevant:
            avg_score = sum(rc.score for rc in data.ranked_chunks) / len(data.ranked_chunks)
            answer.confidence = round(avg_score, 3)

        data.legal_answer = answer
        return data
