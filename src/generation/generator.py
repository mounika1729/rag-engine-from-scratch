"""
src/generation/generator.py — Groq-only LLM response generation.

Uses Groq API (fast + free tier)
"""

from __future__ import annotations
from dataclasses import dataclass
from loguru import logger
from src.models import RAGResponse, RetrievedChunk
from config import settings
from dotenv import load_dotenv
import os

load_dotenv()

# ── Prompt templates ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise and helpful assistant. 
Answer the user's question ONLY using the provided context.
If the answer is not in the context, say "I don't have enough information to answer that."
Always be concise and cite the relevant parts of the context."""

RAG_PROMPT_TEMPLATE = """CONTEXT:
{context}

---
QUESTION: {question}

ANSWER:"""


def build_prompt(question: str, chunks: list[RetrievedChunk], max_context_chars: int = 3000) -> str:
    context_parts = []
    total_chars = 0

    for i, rc in enumerate(chunks, 1):
        source = rc.chunk.metadata.get("source", "unknown")
        snippet = f"[Source {i}: {source}]\n{rc.chunk.content}"
        if total_chars + len(snippet) > max_context_chars:
            break
        context_parts.append(snippet)
        total_chars += len(snippet)

    context = "\n\n".join(context_parts)
    return RAG_PROMPT_TEMPLATE.format(context=context, question=question)


# ── Groq Generator ────────────────────────────────────────────────────────────

@dataclass
class GroqGenerator:
    model: str = "llama-3.1-8b-instant"

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> RAGResponse:
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY not found in environment")

        client = Groq(api_key=api_key)
        prompt = build_prompt(question, chunks)

        logger.debug(f"[Groq] Calling {self.model} | context chunks: {len(chunks)}")

        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        answer = resp.choices[0].message.content.strip()

        return RAGResponse(
            question=question,
            answer=answer,
            retrieved_chunks=chunks,
            model=self.model,
        )


# ── Generator Selector (Groq only) ────────────────────────────────────────────

def get_generator(model: str | None = None):
    name = model or os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    return GroqGenerator(model=name)