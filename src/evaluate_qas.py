#!/usr/bin/env python3
"""
QAS Evaluation Script: Run all questions from QASPER dataset through RAG pipeline.
Saves results with latency and answer generation metrics.
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from functools import wraps

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

# Configure LangSmith tracing
from langsmith.wrappers import wrap_openai
from langsmith import traceable

console = Console()

# LangSmith configuration
if os.getenv("LANGSMITH_API_KEY"):
    os.environ.setdefault("LANGSMITH_TRACING", "true")

from cli_rag import (
    get_embedding,
    get_weaviate_client,
    retrieve_documents,
    generate_answer,
    get_cohere_client,
)

console = Console()

# Configuration
DATA_FILE = "data/dummy_data/qasper-train-v0.1.json"
RESULTS_DIR = "results"
TOP_K = int(os.getenv("TOP_K", "5"))


def load_qas_questions(data_file):
    """Load all questions from the QAS dataset."""
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        questions = []
        for paper_id, paper_data in data.items():
            if "qas" not in paper_data:
                continue
            
            for qa_item in paper_data["qas"]:
                questions.append({
                    "paper_id": paper_id,
                    "title": paper_data.get("title", "Unknown"),
                    "question_id": qa_item.get("question_id", "unknown"),
                    "question": qa_item.get("question", ""),
                })
        
        console.print(f"[green]Loaded {len(questions)} questions from dataset[/green]")
        return questions
    except Exception as e:
        console.print(f"[red]Error loading QAS data: {e}[/red]")
        sys.exit(1)


def ensure_results_dir(tag):
    """Create results directory if it doesn't exist."""
    result_path = Path(RESULTS_DIR) / tag
    result_path.mkdir(parents=True, exist_ok=True)
    return result_path


def evaluate_question(question, weaviate_client, tag):
    """
    Evaluate a single question through RAG pipeline.
    Returns dict with results and metrics.
    Traced individually with tag in metadata.
    """
    question_text = question["question"]
    question_id = question["question_id"]
    
    # Create a traced wrapper for this specific question
    @traceable(
        run_type="chain",
        name=f"evaluate_question_{question_id[:8]}",
        tags=["rag-evaluation", tag],
        metadata={"tag": tag, "question_id": question_id}
    )
    def _evaluate():
        result = {
            "paper_id": question["paper_id"],
            "title": question["title"],
            "question_id": question_id,
            "question": question_text,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "embedding_latency_ms": 0,
                "retrieval_latency_ms": 0,
                "generation_latency_ms": 0,
                "total_latency_ms": 0,
                "documents_retrieved": 0,
            },
            "answer": None,
            "error": None,
        }
        
        try:
            total_start = time.time()
            
            # Step 1: Embedding
            embed_start = time.time()
            query_embedding = get_embedding(question_text)
            embed_latency = (time.time() - embed_start) * 1000
            result["metrics"]["embedding_latency_ms"] = round(embed_latency, 2)
            
            if query_embedding is None:
                result["error"] = "Failed to generate embedding"
                return result
            
            # Step 2: Retrieval
            retrieval_start = time.time()
            documents = retrieve_documents(query_embedding, weaviate_client, top_k=TOP_K)
            retrieval_latency = (time.time() - retrieval_start) * 1000
            result["metrics"]["retrieval_latency_ms"] = round(retrieval_latency, 2)
            result["metrics"]["documents_retrieved"] = len(documents)
            
            if not documents:
                result["answer"] = "No relevant documents found."
                result["metrics"]["total_latency_ms"] = round((time.time() - total_start) * 1000, 2)
                return result
            
            # Step 3: Generation
            gen_start = time.time()
            answer_text, citations = generate_answer(question_text, documents)
            gen_latency = (time.time() - gen_start) * 1000
            result["metrics"]["generation_latency_ms"] = round(gen_latency, 2)
            
            # Store answer text and convert citations to serializable format
            result["answer"] = answer_text
            result["citations"] = [
                {
                    "sources": [{"id": source.id} for source in citation.sources]
                }
                for citation in citations
            ] if citations else []
            result["metrics"]["total_latency_ms"] = round((time.time() - total_start) * 1000, 2)
            
        except Exception as e:
            result["error"] = str(e)
            result["metrics"]["total_latency_ms"] = round((time.time() - total_start) * 1000, 2)
        
        return result
    
    return _evaluate()


@traceable(run_type="chain", name="run_evaluation", tags=["rag-evaluation"], metadata={"evaluation_type": "batch"})
def run_evaluation(questions, tag):
    """Run evaluation for all questions."""
    results_path = ensure_results_dir(tag)
    results = {
        "tag": tag,
        "timestamp": datetime.now().isoformat(),
        "total_questions": len(questions),
        "questions_completed": 0,
        "questions_failed": 0,
        "average_latency_ms": 0,
        "results": []
    }
    
    weaviate_client = get_weaviate_client()
    latencies = []
    
    try:
        with Progress(
            SpinnerColumn(),
            BarColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                "[cyan]Evaluating questions...",
                total=len(questions)
            )
            
            for question in questions:
                result = evaluate_question(question, weaviate_client, tag)
                results["results"].append(result)
                
                if result["error"]:
                    results["questions_failed"] += 1
                else:
                    results["questions_completed"] += 1
                    latencies.append(result["metrics"]["total_latency_ms"])
                
                progress.advance(task)
        
        # Calculate average latency
        if latencies:
            results["average_latency_ms"] = round(sum(latencies) / len(latencies), 2)
    
    finally:
        weaviate_client.close()
    
    return results, results_path


def save_results(results, results_path):
    """Save results to JSON file."""
    results_file = results_path / "results.json"
    
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(f"[green]Results saved to: {results_file}[/green]")
        
        # Print summary
        console.print("\n[bold cyan]Evaluation Summary[/bold cyan]")
        console.print(f"  Tag: {results['tag']}")
        console.print(f"  Total Questions: {results['total_questions']}")
        console.print(f"  Completed: {results['questions_completed']}")
        console.print(f"  Failed: {results['questions_failed']}")
        console.print(f"  Average Latency: {results['average_latency_ms']}ms")
        console.print()
        
        return results_file
    except Exception as e:
        console.print(f"[red]Error saving results: {e}[/red]")
        sys.exit(1)


@click.command()
@click.option(
    '--tag',
    required=False,
    help='Tag to identify this evaluation run (default: timestamp)',
)
@click.option(
    '--limit',
    type=int,
    default=None,
    help='Limit number of questions to evaluate (for testing)',
)
def main(tag, limit):
    """Run QAS evaluation through RAG pipeline."""
    
    # Generate tag if not provided
    if not tag:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    console.print(f"[bold cyan]Starting QAS Evaluation[/bold cyan]\n")
    console.print(f"Tag: [yellow]{tag}[/yellow]")
    console.print(f"Results will be saved to: [yellow]{RESULTS_DIR}/{tag}[/yellow]\n")
    
    # Load questions
    questions = load_qas_questions(DATA_FILE)
    
    if limit:
        questions = questions[:limit]
        console.print(f"[yellow]Limited to {limit} questions for testing[/yellow]\n")
    
    # Run evaluation
    results, results_path = run_evaluation(questions, tag)
    
    # Save results
    save_results(results, results_path)


if __name__ == "__main__":
    main()