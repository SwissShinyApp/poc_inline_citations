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
    get_openai_client,
    generate_plan,
    generate_draft,
    generate_final_answer,
    load_prompts,
    build_evidence_block,
    extract_used_citations,
    extract_answer_and_references,
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
    Evaluate a single question through CoT RAG pipeline:
    Retrieve → Plan → Draft → Insurance (fix citations)
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
                "plan_latency_ms": 0,
                "draft_latency_ms": 0,
                "insurance_latency_ms": 0,
                "total_latency_ms": 0,
                "documents_retrieved": 0,
            },
            "answer": None,
            "error": None,
        }
        
        try:
            total_start = time.time()
            
            # Load prompts
            prompts = load_prompts()
            
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
            
            # Build evidence block from retrieved documents
            evidence, evidence_map = build_evidence_block(documents)
            
            # Step 3: Generate Plan (hidden step)
            plan_start = time.time()
            plan = generate_plan(question_text, evidence, prompts)
            plan_latency = (time.time() - plan_start) * 1000
            result["metrics"]["plan_latency_ms"] = round(plan_latency, 2)
            
            if not plan:
                result["error"] = "Failed to generate plan"
                result["metrics"]["total_latency_ms"] = round((time.time() - total_start) * 1000, 2)
                return result
            
            # Step 4: Generate Draft with inline citations
            draft_start = time.time()
            draft = generate_draft(question_text, plan, evidence, prompts)
            draft_latency = (time.time() - draft_start) * 1000
            result["metrics"]["draft_latency_ms"] = round(draft_latency, 2)
            
            if not draft:
                result["error"] = "Failed to generate draft"
                result["metrics"]["total_latency_ms"] = round((time.time() - total_start) * 1000, 2)
                return result
            
            # Step 5: Insurance pass (fix and validate citations)
            insurance_start = time.time()
            final_answer = generate_final_answer(evidence, draft, prompts)
            insurance_latency = (time.time() - insurance_start) * 1000
            result["metrics"]["insurance_latency_ms"] = round(insurance_latency, 2)
            
            if not final_answer:
                result["error"] = "Failed to generate final answer"
                result["metrics"]["total_latency_ms"] = round((time.time() - total_start) * 1000, 2)
                return result
            
            # Parse answer and extract citations
            answer_text, references_text = extract_answer_and_references(final_answer)
            used_citations = extract_used_citations(answer_text)
            
            result["answer"] = answer_text
            result["references"] = references_text
            result["citations_count"] = len(used_citations)
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
        console.print(f"  Average Total Latency: {results['average_latency_ms']}ms")
        console.print()
        
        # Print breakdown of average latencies across all results
        if results['results']:
            valid_results = [r for r in results['results'] if not r.get('error')]
            if valid_results:
                avg_embedding = sum(r['metrics']['embedding_latency_ms'] for r in valid_results) / len(valid_results)
                avg_retrieval = sum(r['metrics']['retrieval_latency_ms'] for r in valid_results) / len(valid_results)
                avg_plan = sum(r['metrics']['plan_latency_ms'] for r in valid_results) / len(valid_results)
                avg_draft = sum(r['metrics']['draft_latency_ms'] for r in valid_results) / len(valid_results)
                avg_insurance = sum(r['metrics']['insurance_latency_ms'] for r in valid_results) / len(valid_results)
                
                console.print("[bold cyan]CoT Pipeline Latency Breakdown[/bold cyan]")
                console.print(f"  Embedding:  {round(avg_embedding, 2)}ms")
                console.print(f"  Retrieval:  {round(avg_retrieval, 2)}ms")
                console.print(f"  Plan:       {round(avg_plan, 2)}ms")
                console.print(f"  Draft:      {round(avg_draft, 2)}ms")
                console.print(f"  Insurance:  {round(avg_insurance, 2)}ms")
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
