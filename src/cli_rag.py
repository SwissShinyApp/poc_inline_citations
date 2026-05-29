#!/usr/bin/env python3
"""
RAG CLI with Chain-of-Thought (CoT) pipeline:
Retrieve → Plan → Draft → Insurance (fix citations)
Outputs with inline [k] citations and "References:" section.
"""
import os
import sys
import re
from typing import List, Dict, Any, Tuple

import click
import weaviate
from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from langsmith import traceable
from langsmith.wrappers import wrap_openai
import json

console = Console()
client_openai = None

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4-turbo")
TOP_K = int(os.getenv("TOP_K", "5"))  # Increased for better CoT performance

# System prompts will be loaded from prompts.json
SYSTEM_PROMPTS = {}


def load_prompts():
    """Load all prompts from prompts.json file."""
    global SYSTEM_PROMPTS
    try:
        with open("src/prompts.json", "r") as f:
            prompts = json.load(f)
            prompts_dict = {p["name"]: p["prompt"] for p in prompts}
            # Extract system prompts
            SYSTEM_PROMPTS = {
                "planner": prompts_dict.get("SYSTEM_PROMPT_PLANNER", ""),
                "drafter": prompts_dict.get("SYSTEM_PROMPT_DRAFTER", ""),
                "insurer": prompts_dict.get("SYSTEM_PROMPT_INSURER", "")
            }
            return prompts_dict
    except Exception as e:
        console.print(f"[red]Error loading prompts: {e}[/red]")
        sys.exit(1)


def get_openai_client():
    """Get or initialize OpenAI client."""
    global client_openai
    if client_openai is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
            sys.exit(1)
        client_openai = wrap_openai(OpenAI(api_key=api_key))
    return client_openai


def get_weaviate_client():
    """Connect to local Weaviate instance."""
    try:
        client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051
        )
        return client
    except Exception as e:
        console.print(f"[red]Error connecting to Weaviate: {e}[/red]")
        sys.exit(1)


def get_embedding(text):
    """Get embedding from OpenAI for the given text."""
    try:
        client = get_openai_client()
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding
    except Exception as e:
        console.print(f"[red]Error getting embedding: {e}[/red]")
        return None


@traceable(run_type="retriever")
def retrieve_documents(query_embedding, weaviate_client, top_k=TOP_K):
    """Retrieve top-k similar documents from Weaviate."""
    try:
        collection = weaviate_client.collections.get("Document")
        
        results = collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
        )
        
        return results.objects
    except Exception as e:
        console.print(f"[red]Error retrieving documents: {e}[/red]")
        return []


def build_evidence_block(documents: List[Any]) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """Build numbered evidence block from retrieved documents.
    
    Returns:
        (evidence_text, evidence_map) where evidence_map maps doc_index to source info
    """
    lines = []
    evidence_map = {}
    
    for i, doc in enumerate(documents):
        props = doc.properties
        title = props.get("title", "Unknown")
        paper_id = props.get("paper_id", "Unknown")
        section_name = props.get("section", "main")  # Default to "main" if not available
        text = props.get("text", "")
        
        entry = f"[{i}] {title} ({paper_id}) - {section_name}: {text}"
        lines.append(entry)
        evidence_map[i] = {
            "title": title,
            "paper_id": paper_id,
            "section_name": section_name,
            "text": props.get("text", ""),
            "doc_index": i
        }
    
    return "\n".join(lines), evidence_map


@traceable(run_type="llm")
def generate_plan(query: str, evidence: str, prompts: Dict[str, str]):
    """Generate a plan for answering based on evidence (hidden from user)."""
    try:
        client = get_openai_client()
        prompt = prompts["PLAN_PROMPT"].format(evidence=evidence, question=query)
        
        full_response = ""
        with client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["planner"]},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300,
            stream=True,
        ) as stream:
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
        
        return full_response.strip()
    except Exception as e:
        console.print(f"[red]Error generating plan: {e}[/red]")
        return ""


@traceable(run_type="llm")
def generate_draft(query: str, plan: str, evidence: str, prompts: Dict[str, str]):
    """Generate draft answer with inline citations."""
    try:
        client = get_openai_client()
        prompt = prompts["DRAFT_PROMPT"].format(
            plan=plan,
            evidence=evidence,
            question=query
        )
        
        full_response = ""
        with client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["drafter"]},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000,
            stream=True,
        ) as stream:
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
        
        return full_response.strip()
    except Exception as e:
        console.print(f"[red]Error generating draft: {e}[/red]")
        return ""


@traceable(run_type="llm")
def generate_final_answer(evidence: str, draft: str, prompts: Dict[str, str]):
    """Insurance pass: Fix and validate citations in draft using token streaming."""
    try:
        client = get_openai_client()
        prompt = prompts["INSURANCE_PROMPT"].format(
            evidence=evidence,
            draft=draft
        )
        
        full_response = ""
        with client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPTS["insurer"]},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000,
            stream=True,
        ) as stream:
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    console.print(token, end="", highlight=False)
        
        console.print()  # Newline after streaming
        return full_response.strip()
    except Exception as e:
        console.print(f"[red]Error generating final answer: {e}[/red]")
        return ""


def extract_used_citations(text: str) -> List[Dict[str, Any]]:
    """Extract dict-based citations from text.
    
    Returns list of unique citation dicts found in the answer.
    """
    # Match pattern: [{'doc_index': X, 'section_name': '...', 'paper_id': '...'}]
    pattern = r"\[\{'doc_index':\s*(\d+),\s*'section_name':\s*'([^']*)',\s*'paper_id':\s*'([^']*)'\}\]"
    matches = re.findall(pattern, text)
    
    citations = []
    seen = set()
    for doc_index, section_name, paper_id in matches:
        key = (int(doc_index), section_name, paper_id)
        if key not in seen:
            citations.append({
                "doc_index": int(doc_index),
                "section_name": section_name,
                "paper_id": paper_id
            })
            seen.add(key)
    
    # Sort by doc_index
    citations.sort(key=lambda x: x["doc_index"])
    return citations


def extract_answer_and_references(text: str) -> Tuple[str, str]:
    """Split final output into answer and references sections."""
    if "References:" in text:
        parts = text.split("References:", 1)
        answer = parts[0].strip()
        references = "References:\n" + parts[1].strip()
        return answer, references
    return text.strip(), ""


def rag_query(query: str):
    """Execute CoT RAG pipeline: embed → retrieve → plan → draft → insurance."""
    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}\n")
    
    # Step 1: Embedding and retrieval
    with console.status("[yellow]Embedding query...[/yellow]"):
        query_embedding = get_embedding(query)
        if query_embedding is None:
            return
    
    weaviate_client = get_weaviate_client()
    
    with console.status("[yellow]Searching for relevant documents...[/yellow]"):
        documents = retrieve_documents(query_embedding, weaviate_client, top_k=TOP_K)
    
    weaviate_client.close()
    
    if not documents:
        console.print("[yellow]No documents found. Try a different query.[/yellow]\n")
        return
    
    console.print(f"[green]Found {len(documents)} relevant documents[/green]\n")
    
    # Build evidence block
    evidence, evidence_map = build_evidence_block(documents)
    
    # Load prompts
    prompts = load_prompts()
    
    # Step 2: Plan (hidden)
    with console.status("[yellow]Planning answer strategy...[/yellow]"):
        plan = generate_plan(query, evidence, prompts)
    
    if not plan:
        console.print("[red]Failed to generate plan[/red]\n")
        return
    
    # Step 3: Draft with inline citations
    with console.status("[yellow]Drafting answer with citations...[/yellow]"):
        draft = generate_draft(query, plan, evidence, prompts)
    
    if not draft:
        console.print("[red]Failed to generate draft[/red]\n")
        return
    
    # Step 4: Insurance pass (fix citations) - streamed
    console.print("[bold cyan]Final Answer:[/bold cyan]")
    final_answer = generate_final_answer(evidence, draft, prompts)
    
    if not final_answer:
        console.print("[red]Failed to generate final answer[/red]\n")
        return
    
    # Parse output
    answer_text, references_text = extract_answer_and_references(final_answer)
    used_citations = extract_used_citations(answer_text)
    
    # Display answer
    console.print("[bold cyan]Answer:[/bold cyan]")
    console.print(Markdown(answer_text))
    
    # Display references
    console.print("\n[bold cyan]References:[/bold cyan]")
    for citation in used_citations:
        doc_idx = citation["doc_index"]
        section = citation["section_name"]
        paper_id = citation["paper_id"]
        
        if doc_idx in evidence_map:
            info = evidence_map[doc_idx]
            console.print(
                f"  [{{'doc_index': {doc_idx}, 'section_name': '{section}', 'paper_id': '{paper_id}'}}]"
            )
            console.print(f"    → {info['title']} ({paper_id}) - {section}")
    
    console.print()


@click.command()
@click.argument('query', required=False)
@click.option(
    '--interactive',
    is_flag=True,
    help='Run in interactive mode (ask multiple questions)',
)
@click.option(
    '--top-k',
    type=int,
    default=TOP_K,
    help='Number of documents to retrieve',
)
def main(query, interactive, top_k):
    """RAG CLI with CoT: Query documents and get AI-generated answers with citations."""
    # Override TOP_K if provided
    global TOP_K
    TOP_K = top_k
    
    console.print(Panel(
        "[bold cyan]RAG Query Interface with Chain-of-Thought[/bold cyan]\n"
        "Retrieve → Plan → Draft → Validate\n"
        "[dim]Powered by Weaviate + OpenAI[/dim]",
        border_style="cyan"
    ))
    
    if interactive:
        # Interactive mode: ask multiple questions
        console.print("[yellow]Interactive mode. Type 'exit' to quit.[/yellow]\n")
        while True:
            try:
                user_query = console.input("[cyan]Enter your question:[/cyan] ").strip()
                if user_query.lower() in ("exit", "quit"):
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                if user_query:
                    rag_query(user_query)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
                break
    elif query:
        # Single query mode
        rag_query(query)
    else:
        # No query provided, show help
        console.print(Panel(
            "[yellow]Usage:\n\n"
            "  Single query:\n"
            "    python src/cli_rag.py \"What is affective events?\"\n\n"
            "  Interactive mode:\n"
            "    python src/cli_rag.py --interactive\n\n"
            "  Custom retrieval count:\n"
            "    python src/cli_rag.py \"question\" --top-k 12[/yellow]",
            title="[bold cyan]Examples[/bold cyan]",
            border_style="cyan"
        ))


if __name__ == "__main__":
    main()
