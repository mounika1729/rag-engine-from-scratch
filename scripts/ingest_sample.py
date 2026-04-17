"""
scripts/ingest_sample.py — Quick demo that ingests a sample Wikipedia article.
Run: python scripts/ingest_sample.py
"""
import sys; sys.path.insert(0, ".")
from src.pipeline import RAGPipeline
from rich.console import Console

console = Console()

URL = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"

def main():
    console.print("[bold cyan]Building RAG pipeline...[/bold cyan]")
    pipeline = RAGPipeline.build(vector_store_type="faiss")
    console.print(f"[cyan]Ingesting: {URL}[/cyan]")
    n = pipeline.ingest(URL)
    console.print(f"[green]✓ Indexed {n} chunks[/green]")
    pipeline.save()
    console.print("[green]✓ Saved index to disk[/green]")
    console.print("\n[bold]Try asking:[/bold] python main.py query 'What is RAG?' --show-sources")

if __name__ == "__main__":
    main()
