"""
Legal RAG Assistant — CLI Entry Point.

Provides an interactive command-line interface for:
    - Uploading and ingesting legal documents
    - Asking questions with citation-backed answers
    - Managing conversations
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

# Ensure project root is in the path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import logging

from config.settings import get_settings
from core.models import PipelineData
from core.memory import memory_provider
from ingestion.pipeline import create_ingestion_pipeline
from agents.conversational_agent import get_conversational_agent

from utils.logging import setup_logging

logger = logging.getLogger(__name__)


def print_banner() -> None:
    """Display the application banner."""
    print(
        "\n"
        "╔══════════════════════════════════════════╗\n"
        "║     ⚖️  Legal RAG Assistant  ⚖️            ║\n"
        "║  AI-Powered Legal Document Analysis      ║\n"
        "╚══════════════════════════════════════════╝\n"
    )


def ingest_document(file_path: str) -> PipelineData:
    """Ingest a single document through the full pipeline."""
    path = Path(file_path).resolve()
    if not path.exists():
        print(f"❌ File not found: {path}")
        return PipelineData()

    print(f"📄 Ingesting: {path.name}")
    print("   Parsing → Chunking → Embedding → Indexing ...")

    ingestion_pipeline = create_ingestion_pipeline()
    data = PipelineData(file_path=str(path))

    try:
        data = ingestion_pipeline.run(data)
        chunks_count = data.diagnostics.get("chunks_indexed", 0)
        print(f"   ✅ Done! {chunks_count} chunk(s) indexed")
        print(f"   📊 Timings: {data.diagnostics}")
        return data
    except Exception as exc:
        print(f"   ❌ Ingestion failed: {exc}")
        logger.exception("Ingestion error")
        return data


def ask_agent(query: str, conversation_id: str, agent) -> str:
    """Run the user query through the conversational agent."""
    
    # LangChain isolates conversations using the thread_id inside the config.
    # The checkpointer itself ("default") manages the database file, while thread_id manages the specific chat.
    thread_id = memory_provider.make_thread_id("local_cli_user", conversation_id)
    config = {"configurable": {"thread_id": thread_id}}
    try:
        response = agent.invoke(
            {"messages": [("user", query)]},
            config=config
        )
        # response['messages'] contains the conversation. The last message is the AIMessage.
        last_msg = response["messages"][-1]
        
        output = [
            f"\n{'─'*60}",
            f"📝 Answer:",
            f"{'─'*60}",
            last_msg.content,
            f"{'─'*60}"
        ]
        return "\n".join(output)
        
    except Exception as exc:
        logger.exception("Agent execution failed.")
        return f"❌ Agent failed: {exc}"


def main() -> None:
    """Main CLI loop."""
    setup_logging()
    settings = get_settings()
    agent = get_conversational_agent()

    print_banner()
    print(f"📋 Config: device={settings.device}")
    print(f"LLM: {settings.llm_model} via OpenRouter")
    print(f"Dense Embeddings: {settings.embedding_provider}")
    print(f"Sparse Embeddings: {settings.sparse_model_name}")
    print(f"Reranker: {settings.reranker_provider}")
    print(f"Evaluation enabled: {settings.enable_faithfulness_guard}")
    print("\n")

    # Simple session choice
    session_choice = input("Enter conversation ID to resume (or press Enter for new): ").strip()
    if session_choice:
        conversation_id = session_choice
        print(f"📂 Resuming session: {conversation_id}")
    else:
        conversation_id = str(uuid.uuid4())
        print(f"📝 New conversation: {conversation_id}")

    print("\nCommands:")
    print("  /upload <path>  — Upload and ingest a document")
    print("  /quit           — Exit")
    print("  (anything else) — Ask a question\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("👋 Goodbye!")
            break

        elif user_input.startswith("/upload "):
            file_path = user_input[8:].strip().strip('"').strip("'")
            ingest_document(file_path)

        else:
            response = ask_agent(user_input, conversation_id, agent)
            print(response)


if __name__ == "__main__":
    main()
