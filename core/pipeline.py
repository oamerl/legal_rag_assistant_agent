"""
Pipeline Pattern — base classes for composing processing stages.

Both the ingestion and retrieval flows are sequential pipelines.
Each stage reads from and writes to a shared PipelineData object.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from core.models import PipelineData

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """
    Abstract base for a single processing step.

    Subclasses implement `process()` which receives a PipelineData
    object, performs its work, mutates / enriches the data, and
    returns it for the next stage.
    """

    @property
    def stage_name(self) -> str:
        """Human-readable name used in logs and diagnostics."""
        return self.__class__.__name__

    @abstractmethod
    def process(self, data: PipelineData) -> PipelineData:
        """Execute this stage and return the (possibly mutated) data."""

    def __repr__(self) -> str:
        return f"<{self.stage_name}>"


class Pipeline:
    """
    Composes an ordered list of PipelineStage instances.

    Running a pipeline feeds the data through each stage sequentially,
    logging timing information and catching errors per-stage.
    """

    def __init__(self, stages: list[PipelineStage], name: str = "Pipeline"):
        self._stages = stages
        self._name = name

    def run(self, data: PipelineData) -> PipelineData:
        """Execute all stages in order, returning the final data."""
        logger.info(
            "%s started with %d stage(s): %s",
            self._name,
            len(self._stages),
            " → ".join(s.stage_name for s in self._stages),
        )

        for stage in self._stages:
            stage_start = time.perf_counter()
            logger.info("[%s] %s — starting", self._name, stage.stage_name)

            data = stage.process(data)

            elapsed = time.perf_counter() - stage_start
            logger.info(
                "[%s] %s — completed in %.3fs",
                self._name,
                stage.stage_name,
                elapsed,
            )
            data.diagnostics[f"{stage.stage_name}_time_s"] = round(elapsed, 3)

        logger.info("%s finished", self._name)
        return data

    def __repr__(self) -> str:
        stages_str = " → ".join(s.stage_name for s in self._stages)
        return f"<{self._name}: {stages_str}>"
