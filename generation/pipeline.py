"""
Generation pipeline — orchestrates: Generate → (optional) FaithfulnessGuard.

Each step is a PipelineStage that reads from and writes to the shared
PipelineData object. The pipeline is composed using the Pipeline class
from core.pipeline.
"""

from __future__ import annotations

import logging

from config.settings import get_settings
from core.models import PipelineData
from core.pipeline import Pipeline, PipelineStage
from generation.faithfulness_guard import FaithfulnessGuard, FaithfulnessGuardStage
from generation.generator import GenerationStage, LegalGenerator

logger = logging.getLogger(__name__)


# ── Composed Pipeline ─────────────────────────────────────────────────


def create_generation_pipeline(
    generator: LegalGenerator | None = None,
    guard: FaithfulnessGuard | None = None,
    *,
    enable_faithfulness_guard: bool | None = None,
) -> Pipeline:
    """
    Build the default generation pipeline.

    Stages:
        1. GenerationStage              — produce the LegalAnswer via LLM
        2. FaithfulnessGuardStage       — verify faithfulness using RAGAS metrics (optional)

    Parameters
    ----------
    generator : LegalGenerator | None
        Optional override for the generation stage.
    guard : FaithfulnessGuard | None
        Optional override for the faithfulness guard.
    enable_faithfulness_guard : bool | None
        Explicitly enable/disable the guard stage.
        When *None* (default), the value is read from
        ``settings.enable_faithfulness_guard``.

    Returns
    -------
    Pipeline
        Ready-to-run generation pipeline.
    """
    settings = get_settings()

    if enable_faithfulness_guard is None:
        enable_faithfulness_guard = settings.enable_faithfulness_guard

    stages: list[PipelineStage] = [
        GenerationStage(generator),
    ]

    if enable_faithfulness_guard:
        stages.append(FaithfulnessGuardStage(guard))

    return Pipeline(stages, name="GenerationPipeline")


def generate(query: str, assembled_context: str, conversation_id: str = "") -> PipelineData:
    """
    Convenience function: run the generation pipeline for a query.

    Parameters
    ----------
    query : str
        The user's natural language question.
    assembled_context : str
        Context assembled by the retrieval pipeline.
    conversation_id : str, optional
        For session tracking.

    Returns
    -------
    PipelineData
        Contains legal_answer and optional guard_result.
    """
    pipeline = create_generation_pipeline()
    data = PipelineData(
        query=query,
        assembled_context=assembled_context,
        conversation_id=conversation_id,
    )
    return pipeline.run(data)
