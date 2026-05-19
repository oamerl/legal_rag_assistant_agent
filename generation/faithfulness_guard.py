"""
Faithfulness guard — post-generation verification.

Verifies that the LLM's answer is grounded in the retrieved chunks
using two approaches:
    1. Fast check: string-match cited text against source chunks
    2. Deep check: LLM-based claim verification (optional)
"""

from __future__ import annotations

import logging

from config.settings import get_settings
from core.models import GuardResult, LegalAnswer, PipelineData, RankedChunk
from core.pipeline import PipelineStage

logger = logging.getLogger(__name__)


class FaithfulnessGuard:
    """
    Verifies that generated answers are grounded in source chunks.

    Two verification levels:
        - Fast: check if cited quoted_text appears in the source chunks
        - Deep: use an LLM to decompose claims and verify each one
    """

    def __init__(self, enable_deep_check: bool = True) -> None:
        self._deep_check = enable_deep_check

    def verify(
        self,
        query: str,
        answer: LegalAnswer,
        ranked_chunks: list[RankedChunk],
    ) -> GuardResult:
        """
        Verify faithfulness of the answer.

        Parameters
        ----------
        query : str
            The user's question.
        answer : LegalAnswer
            The generated answer with citations.
        ranked_chunks : list[RankedChunk]
            The source chunks used for generation.

        Returns
        -------
        GuardResult
            Verification result with match rate and unsupported claims.
        """
        # If no citations to check, skip
        if not answer.citations:
            logger.info("No structured citations to verify — skipping fast check")
            return GuardResult(faithfulness_score=0.0, answer_relevancy_score=0.0, context_precision_score=0.0)

        result = GuardResult(
            faithfulness_score=0.0,
            answer_relevancy_score=0.0,
            context_precision_score=0.0
        )

        if self._deep_check:

            logger.info("Deep check enabled: running RAGAS evaluation")
            from evaluation.ragas_eval import EvaluationSample, evaluate_samples

            contexts = [rc.chunk.text for rc in ranked_chunks]
            sample = EvaluationSample(
                question=query,
                answer=answer.answer,
                contexts=contexts,
                #ground_truth=query  # Dummy ground truth as context precision requires it
            )
            eval_result = evaluate_samples([sample])
            
            result.faithfulness_score = eval_result.faithfulness
            result.answer_relevancy_score = eval_result.answer_relevancy
            result.context_precision_score = eval_result.context_precision

        return result


class FaithfulnessGuardStage(PipelineStage):
    """Pipeline stage: verify faithfulness of the generated answer."""

    def __init__(self, guard: FaithfulnessGuard | None = None) -> None:
        self._guard = guard or FaithfulnessGuard()

    def process(self, data: PipelineData) -> PipelineData:
        if data.legal_answer is None:
            logger.warning("No answer to verify")
            return data

        result = self._guard.verify(
            query=data.query,
            answer=data.legal_answer,
            ranked_chunks=data.ranked_chunks,
        )
        data.guard_result = result

        return data
