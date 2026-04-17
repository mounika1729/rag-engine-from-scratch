"""
scripts/demo_query.py — Run several demo queries against an existing index.
Run: python scripts/demo_query.py
"""
import sys; sys.path.insert(0, ".")
from src.pipeline import RAGPipeline
from rich.console import Console
from rich.panel import Panel

console = Console()

QUESTIONS = [
    "What is Retrieval-Augmented Generation?",
    "What are the main components of a RAG system?",
    "How does vector search work in RAG?",
]

def main():
    pipeline = RAGPipeline.build(load_existing=True)
    for q in QUESTIONS:
        resp = pipeline.query(q)
        console.print(Panel(resp.answer, title=f"[yellow]{q}[/yellow]", border_style="yellow"))
        console.print(f"  [dim]Sources: {len(resp.sources)} | Tokens: {resp.completion_tokens}[/dim]\n")

if __name__ == "__main__":
    main()
