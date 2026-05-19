"""
LLM client — wraps OpenRouter via LangChain's ChatOpenAI.

Provides a unified interface for LLM calls, using OpenRouter as
the gateway to models like GPT-4o, Claude 3.5 Sonnet, etc.
"""

from __future__ import annotations

import logging

from config.settings import get_settings
from core.exceptions import LLMClientError

logger = logging.getLogger(__name__)


def create_llm_client(
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    use_small_model: bool = False,
):
    """
    Create a LangChain ChatOpenAI instance configured for OpenRouter.

    Parameters
    ----------
    model : str, optional
        Model identifier (e.g. "openai/gpt-4o"). Defaults to config.
    temperature : float, optional
        Sampling temperature. Defaults to config (0.0 for legal).
    max_tokens : int, optional
        Max output tokens. Defaults to config.
    use_small_model : bool
        If True, use the smaller/cheaper model for auxiliary tasks
        (query routing, HyDE, decomposition).

    Returns
    -------
    ChatOpenAI
        A LangChain chat model pointed at OpenRouter.

    Raises
    ------
    LLMClientError
        If the API key is missing or LangChain is not installed.
    """
    settings = get_settings()

    if not settings.openrouter_api_key:
        raise LLMClientError(
            "OpenRouter API key not set. Add OPENROUTER_API_KEY to your .env file."
        )

    try:
        from langchain_openai import ChatOpenAI

        selected_model = model or (
            settings.llm_model_small if use_small_model else settings.llm_model
        )

        llm = ChatOpenAI(
            model=selected_model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=settings.openrouter_base_url,
            temperature=temperature if temperature is not None else settings.llm_temperature,
            max_tokens=max_tokens or settings.llm_max_tokens,
        )

        logger.info(
            "LLM client created: model=%s, temp=%.1f",
            selected_model,
            temperature if temperature is not None else settings.llm_temperature,
        )
        return llm

    except ImportError as exc:
        raise LLMClientError(
            "langchain-openai not installed. Run: pip install langchain-openai"
        ) from exc
    except Exception as exc:
        raise LLMClientError(f"Failed to create LLM client: {exc}") from exc
