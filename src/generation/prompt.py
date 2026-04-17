"""
Prompt Templates
----------------
Builds structured prompts that instruct the LLM to answer
using only the retrieved context (grounded generation).

Templates provided:
  - rag_prompt:      Standard RAG with cited context
  - chat_prompt:     Multi-turn conversational RAG
  - strict_prompt:   Forces "I don't know" for out-of-context queries
"""

from typing import List, Optional
from src.vectorstore.faiss_store import SearchResult


def build_rag_prompt(query: str, results: List[SearchResult]) -> str:
    """
    Build a standard RAG prompt with numbered context chunks.

    The LLM is instructed to answer using only the provided context
    and cite which chunk number(s) support its answer.
    """
    context_blocks = []
    for result in results:
        source = result.chunk.metadata.get("source", "unknown")
        page = result.chunk.metadata.get("page", "")
        page_info = f", page {page}" if page else ""
        context_blocks.append(
            f"[{result.rank}] (Source: {source}{page_info})\n{result.chunk.content}"
        )

    context_str = "\n\n".join(context_blocks)

    prompt = f"""You are a precise question-answering assistant. Use ONLY the provided context to answer the question. If the answer is not in the context, say "I don't have enough information to answer this."

CONTEXT:
{context_str}

QUESTION: {query}

INSTRUCTIONS:
- Answer based solely on the context above
- Be concise and factual
- If referencing specific context, mention which source number (e.g., "According to [1]...")
- Do not make up information

ANSWER:"""
    return prompt


def build_chat_prompt(
    query: str,
    results: List[SearchResult],
    chat_history: Optional[List[dict]] = None
) -> List[dict]:
    """
    Build a multi-turn conversation prompt for chat-based LLMs.
    Returns a list of message dicts (OpenAI format).
    """
    context_blocks = []
    for result in results:
        context_blocks.append(
            f"[{result.rank}] {result.chunk.content}"
        )
    context_str = "\n\n".join(context_blocks)

    system_message = {
        "role": "system",
        "content": (
            "You are a helpful assistant that answers questions based on the provided context. "
            "Stick to the context. If unsure, say so. Be concise."
            f"\n\nCONTEXT:\n{context_str}"
        )
    }

    messages = [system_message]

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": query})
    return messages


def build_strict_prompt(query: str, results: List[SearchResult]) -> str:
    """
    Strict RAG prompt that forces explicit 'I don't know' responses
    when the context doesn't contain the answer.

    Best for production use cases where hallucination is unacceptable.
    """
    context_blocks = [
        f"CHUNK {result.rank}:\n{result.chunk.content}"
        for result in results
    ]
    context_str = "\n---\n".join(context_blocks)

    prompt = f"""Answer the question using ONLY the chunks below. Do not use any prior knowledge.

If none of the chunks contain the answer, respond exactly with:
"I don't have enough information in the provided documents to answer this question."

{context_str}

Question: {query}
Answer:"""
    return prompt


def format_sources(results: List[SearchResult]) -> str:
    """Format source citations for display alongside the answer."""
    if not results:
        return ""

    lines = ["**Sources:**"]
    seen = set()
    for result in results:
        source = result.chunk.metadata.get("source", "unknown")
        page = result.chunk.metadata.get("page", "")
        key = f"{source}:{page}"
        if key not in seen:
            page_info = f" (page {page})" if page else ""
            lines.append(f"  [{result.rank}] {source}{page_info}")
            seen.add(key)

    return "\n".join(lines)
