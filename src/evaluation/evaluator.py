"""
src/evaluation/evaluator.py — RAG evaluation metrics (no RAGAS dependency).

Metrics computed from scratch:
  • hit_rate      — Was the correct document retrieved at all?
  • mrr           — Mean Reciprocal Rank of the first correct result
  • context_precision — Fraction of retrieved chunks that are relevant
  • answer_faithfulness — Rough lexical overlap between answer and context
  • answer_relevance — Rough lexical overlap between answer and question
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Callable
from src.models import RAGResponse


def _tokenise(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


@dataclass
class EvalSample:
    question: str
    expected_answer: str
    relevant_doc_ids: list[str] = field(default_factory=list)  # ground-truth doc ids


@dataclass
class EvalResult:
    question: str
    hit: bool
    reciprocal_rank: float
    context_precision: float
    answer_faithfulness: float
    answer_relevance: float

    @property
    def summary(self) -> dict:
        return {
            "question": self.question[:60],
            "hit": self.hit,
            "rr": round(self.reciprocal_rank, 3),
            "ctx_precision": round(self.context_precision, 3),
            "faithfulness": round(self.answer_faithfulness, 3),
            "relevance": round(self.answer_relevance, 3),
        }


def evaluate_response(response: RAGResponse, sample: EvalSample) -> EvalResult:
    """Evaluate a single RAGResponse against a ground-truth sample."""

    # Hit rate & MRR
    hit = False
    rr = 0.0
    for rank, rc in enumerate(response.retrieved_chunks, 1):
        if rc.chunk.doc_id in sample.relevant_doc_ids:
            hit = True
            rr = 1.0 / rank
            break

    # Context precision — fraction of retrieved chunks that are relevant
    relevant = sum(1 for rc in response.retrieved_chunks if rc.chunk.doc_id in sample.relevant_doc_ids)
    ctx_precision = relevant / len(response.retrieved_chunks) if response.retrieved_chunks else 0.0

    # Faithfulness — token overlap between answer and retrieved context
    context_tokens = _tokenise(" ".join(rc.chunk.content for rc in response.retrieved_chunks))
    answer_tokens = _tokenise(response.answer)
    faithfulness = len(answer_tokens & context_tokens) / (len(answer_tokens) + 1e-9)

    # Relevance — token overlap between answer and question
    question_tokens = _tokenise(response.question)
    relevance = len(answer_tokens & question_tokens) / (len(answer_tokens) + 1e-9)

    return EvalResult(
        question=response.question,
        hit=hit,
        reciprocal_rank=rr,
        context_precision=ctx_precision,
        answer_faithfulness=faithfulness,
        answer_relevance=relevance,
    )


def run_evaluation(
    pipeline,
    samples: list[EvalSample],
    verbose: bool = True,
) -> dict:
    """
    Run the full evaluation suite over a list of EvalSamples.
    Returns aggregate metrics.
    """
    from rich.console import Console
    from rich.table import Table

    results: list[EvalResult] = []
    for sample in samples:
        response = pipeline.query(sample.question)
        result = evaluate_response(response, sample)
        results.append(result)

    # Aggregate
    n = len(results)
    agg = {
        "n_samples": n,
        "hit_rate": sum(r.hit for r in results) / n,
        "mrr": sum(r.reciprocal_rank for r in results) / n,
        "context_precision": sum(r.context_precision for r in results) / n,
        "answer_faithfulness": sum(r.answer_faithfulness for r in results) / n,
        "answer_relevance": sum(r.answer_relevance for r in results) / n,
    }

    if verbose:
        console = Console()
        table = Table(title="RAG Evaluation Results")
        for col in ["question", "hit", "rr", "ctx_precision", "faithfulness", "relevance"]:
            table.add_column(col, justify="right" if col != "question" else "left")
        for r in results:
            s = r.summary
            table.add_row(s["question"], str(s["hit"]), str(s["rr"]), str(s["ctx_precision"]), str(s["faithfulness"]), str(s["relevance"]))
        console.print(table)
        console.print(f"\n[bold green]Aggregate:[/bold green] {agg}")

    return agg
