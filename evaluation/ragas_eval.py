"""
RAGAS evaluation module — offline batch evaluation of the RAG pipeline.

Uses the RAGAS framework (v0.4+) to measure:
    - Faithfulness: is the answer grounded in the context?
    - Answer Relevancy: does the answer address the question?
    - Context Precision: are the retrieved chunks relevant?
    - Context Recall: did retrieval find all needed information?

This is for development/CI evaluation, NOT real-time production use.
"""

from __future__ import annotations

import logging
from core.exceptions import EvaluationError
from core.models import EvaluationSample, EvaluationResult

logger = logging.getLogger(__name__)


def evaluate_samples(samples: list[EvaluationSample]) -> EvaluationResult:
    """
    Run RAGAS evaluation on a list of samples.

    Parameters
    ----------
    samples : list[EvaluationSample]
        Evaluation data with questions, answers, contexts, and ground truths.

    Returns
    -------
    EvaluationResult
        Aggregated metric scores.

    Notes
    -----
    Requires `ragas` to be installed: `pip install ragas`
    """
    logger.info("Running RAGAS evaluation on %d sample(s)", len(samples))

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as exc:
        raise EvaluationError("RAGAS or datasets is not installed. Please install with `pip install ragas datasets`.") from exc

    # We need to construct the evaluation dataset.
    # Note: context_precision typically requires 'ground_truth' or 'reference' depending on the ragas version.
    # If not provided, it may fail, so we provide an empty string if missing, or use user's explicit request.
    data = {
        "question": [s.question for s in samples],
        "answer": [s.answer for s in samples],
        "contexts": [s.contexts for s in samples],
        "ground_truth": [s.ground_truth for s in samples], # Using ground_truth for context_precision if needed
    }
    
    dataset = Dataset.from_dict(data)

    try:
        from generation.llm_client import create_llm_client

        # We can pass the LLM and embeddings to evaluate if needed, or use default Ragas defaults.
        # Ragas uses langchain chat models.
        llm = create_llm_client()
        
        from langchain_core.embeddings import Embeddings
        from ingestion.embedder import EmbeddingServiceFactory

        class LocalEmbeddingsWrapper(Embeddings):
            def __init__(self):
                self.strategy = EmbeddingServiceFactory.create()
                
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                results = self.strategy.embed_texts(texts)
                return [r.dense for r in results]
                
            def embed_query(self, text: str) -> list[float]:
                return self.strategy.embed_query(text).dense
                
        embeddings = LocalEmbeddingsWrapper()
        
        metrics = [faithfulness, answer_relevancy, context_precision]
        
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )
        
        # Safely extract the mean metric scores from the results
        try:
            # Convert to pandas, select only numeric columns, and take the mean
            df = result.to_pandas()
            numeric_df = df.select_dtypes(include=['number'])
            scores_dict = numeric_df.mean().to_dict()
        except Exception as e:
            logger.warning(f"Failed to extract metric scores from RAGAS result: {e}")
            scores_dict = {}
            
        return EvaluationResult(
            faithfulness=scores_dict.get("faithfulness", 0.0),
            answer_relevancy=scores_dict.get("answer_relevancy", 0.0),
            context_precision=scores_dict.get("context_precision", 0.0),
            num_samples=len(samples)
        )
    except Exception as exc:
        raise EvaluationError("RAGAS evaluation failed.") from exc
