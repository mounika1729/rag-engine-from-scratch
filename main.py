"""
main.py — CLI entrypoint for the RAG system.

Commands:
  ingest  <source>   Index a file, URL, or directory
  query   <question> Ask a question
  serve              Start the FastAPI server
  eval    <file>     Run evaluation on a JSON file of samples
"""

import argparse
import json
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()
from dotenv import load_dotenv
load_dotenv()

def cmd_ingest(args):
    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline.build()
    n = pipeline.ingest(args.source)
    pipeline.save()
    console.print(f"[green]✓ Indexed {n} chunks from '{args.source}'[/green]")


def cmd_query(args):
    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline.build(load_existing=True)
    response = pipeline.query(args.question)
    console.print(Panel(Markdown(response.answer), title="[bold cyan]Answer[/bold cyan]", border_style="cyan"))
    if args.show_sources:
        console.print("\n[bold]Sources:[/bold]")
        for i, src in enumerate(response.sources, 1):
            console.print(f"  {i}. [yellow]{src['metadata'].get('source','?')}[/yellow] (score={src['score']:.3f})")
            console.print(f"     {src['preview'][:120]}...")


def cmd_serve(args):
    import uvicorn
    from config import settings
    uvicorn.run("src.api:app", host=settings.api_host, port=settings.api_port, reload=settings.api_reload)


def cmd_eval(args):
    from src.pipeline import RAGPipeline
    from src.evaluation.evaluator import EvalSample, run_evaluation
    pipeline = RAGPipeline.build(load_existing=True)
    samples_data = json.loads(Path(args.file).read_text())
    samples = [EvalSample(**s) for s in samples_data]
    run_evaluation(pipeline, samples)


def main():
    parser = argparse.ArgumentParser(description="RAG From Scratch CLI")
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Index a file, URL, or directory")
    p_ingest.add_argument("source", help="Path or URL to ingest")

    # query
    p_query = sub.add_parser("query", help="Ask a question")
    p_query.add_argument("question", help="Your question")
    p_query.add_argument("--show-sources", action="store_true", help="Show retrieved source chunks")

    # serve
    sub.add_parser("serve", help="Start the FastAPI server")

    # eval
    p_eval = sub.add_parser("eval", help="Run evaluation")
    p_eval.add_argument("file", help="Path to JSON eval samples file")

    args = parser.parse_args()

    if args.command == "ingest":       cmd_ingest(args)
    elif args.command == "query":      cmd_query(args)
    elif args.command == "serve":      cmd_serve(args)
    elif args.command == "eval":       cmd_eval(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
