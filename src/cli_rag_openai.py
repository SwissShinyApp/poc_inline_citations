#!/usr/bin/env python3
"""
RAG CLI (OpenAI Vector Store): answer questions over QASPER paragraphs indexed in
an OpenAI Vector Store, using the Responses API + `file_search`.

Generation streams token-by-token and emits **inline citation markers** as
`file_search` annotation events arrive, so factual statements are cited in-text
rather than only at the end. The vector store is built by `index_openai.py`.

`retrieve_documents` (direct `vector_stores.search`) is also exposed here for
callers that want explicit retrieval without the model in the loop (e.g.
`evaluate_qas.py`).
"""
import json
import os
import sys
from pathlib import Path

import click
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from langsmith import traceable
from langsmith.wrappers import wrap_openai

console = Console()
client_openai = None

VECTOR_STORE_NAME = os.getenv("OPENAI_VECTOR_STORE_NAME", "qasper-paragraphs")
STORE_ID_FILE = Path("data/openai_vector_store.json")
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "5"))


def get_openai_client():
    """Get or initialize a (LangSmith-wrapped) OpenAI client."""
    global client_openai
    if client_openai is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
            sys.exit(1)
        client_openai = wrap_openai(OpenAI(api_key=api_key))
    return client_openai


def load_store_id(name):
    """Resolve a persisted vector store id by name (env var wins)."""
    env_id = os.getenv("OPENAI_VECTOR_STORE_ID")
    if env_id:
        return env_id
    if STORE_ID_FILE.exists():
        try:
            return json.loads(STORE_ID_FILE.read_text()).get(name)
        except Exception:
            return None
    return None


class RetrievedDoc:
    """Lightweight shim exposing `.properties` so retrieved chunks are drop-in
    compatible with `cli_rag.generate_answer` (which expects Weaviate-style objects)."""

    def __init__(self, properties, score=None):
        self.properties = properties
        self.score = score


@traceable(run_type="retriever", name="retrieve_documents_openai")
def retrieve_documents(query, vector_store_id, top_k=TOP_K):
    """Semantic search against the vector store; returns RetrievedDoc objects."""
    try:
        client = get_openai_client()
        results = client.vector_stores.search(
            vector_store_id=vector_store_id,
            query=query,
            max_num_results=top_k,
        )

        docs = []
        for item in results.data:
            text = "".join(
                part.text for part in item.content if getattr(part, "type", None) == "text"
            )
            attrs = item.attributes or {}
            docs.append(
                RetrievedDoc(
                    properties={
                        "text": text,
                        "title": attrs.get("title", "Unknown"),
                        "paper_id": attrs.get("paper_id", "Unknown"),
                        "section_name": attrs.get("section_name", "Unknown"),
                        "section_idx": int(attrs.get("section_idx", 0)),
                        "paragraph_idx": int(attrs.get("paragraph_idx", 0)),
                        "filename": item.filename,
                        "file_id": item.file_id,
                    },
                    score=item.score,
                )
            )
        return docs
    except Exception as e:
        console.print(f"[red]Error searching vector store: {e}[/red]")
        return []


def parse_source_filename(filename):
    """Parse `paper_id__sX__pY.txt` into (paper_id, section_idx, paragraph_idx)."""
    base = (filename or "").rsplit(".", 1)[0]
    parts = base.split("__")
    paper_id = parts[0] if parts else filename
    section_idx = paragraph_idx = None
    for part in parts[1:]:
        if part.startswith("s"):
            section_idx = part[1:]
        elif part.startswith("p"):
            paragraph_idx = part[1:]
    return paper_id, section_idx, paragraph_idx


@traceable(run_type="llm", name="generate_answer_file_search")
def generate_answer(query, vector_store_id, top_k=TOP_K, model=GENERATION_MODEL):
    """Stream an answer via the Responses API + file_search, printing inline
    citation markers as annotations arrive (in-text, not only at the end).

    Returns (answer_text, sources) where each source is a dict with its citation
    number and the file it points to.
    """
    client = get_openai_client()

    instructions = (
        "You are a research assistant answering questions about academic papers. "
        "Use the file_search tool to ground every factual statement in the retrieved "
        "passages, and cite the supporting source inline, immediately after each "
        "statement it supports (not only at the end). Answer concisely."
    )

    citations = {}   # file_id -> citation number, assigned in order of first appearance
    sources = []     # ordered list of {n, file_id, filename, paper_id, section_idx, paragraph_idx}
    full_answer = ""

    try:
        stream = client.responses.create(
            model=model,
            instructions=instructions,
            input=query,
            tools=[{
                "type": "file_search",
                "vector_store_ids": [vector_store_id],
                "max_num_results": top_k,
            }],
            stream=True,
        )

        for event in stream:
            etype = getattr(event, "type", "")

            if etype == "response.output_text.delta":
                token = event.delta
                full_answer += token
                console.print(token, end="", style="cyan")

            elif etype == "response.output_text.annotation.added":
                ann = event.annotation
                # The annotation may arrive as a typed object or a plain dict.
                if isinstance(ann, dict):
                    file_id = ann.get("file_id")
                    filename = ann.get("filename")
                else:
                    file_id = getattr(ann, "file_id", None)
                    filename = getattr(ann, "filename", None)
                if not file_id:
                    continue

                if file_id not in citations:
                    citations[file_id] = len(citations) + 1
                    paper_id, section_idx, paragraph_idx = parse_source_filename(filename)
                    sources.append({
                        "n": citations[file_id],
                        "file_id": file_id,
                        "filename": filename,
                        "paper_id": paper_id,
                        "section_idx": section_idx,
                        "paragraph_idx": paragraph_idx,
                    })

                marker = f"[{citations[file_id]}]"
                full_answer += marker
                console.print(marker, end="", style="bold green")

        console.print()  # newline after streaming
        return full_answer, sources
    except Exception as e:
        console.print(f"\n[red]Error generating answer: {e}[/red]")
        return None, []


def rag_query(query, vector_store_id, top_k=TOP_K):
    """Execute RAG pipeline: stream a file_search-grounded answer with inline citations."""
    console.print(f"\n[cyan]Query:[/cyan] {query}")
    console.print(f"[dim]Vector store:[/dim] {vector_store_id}\n")

    # Generate answer with streaming + inline citations (file_search does the retrieval)
    console.print("[bold cyan]Answer:[/bold cyan]")
    answer, sources = generate_answer(query, vector_store_id, top_k=top_k)

    if not answer:
        return

    if not sources:
        console.print("\n[yellow]No sources were cited.[/yellow]\n")
        return

    # Show the sources behind the inline [n] markers
    console.print("\n[dim]Sources:[/dim]")
    for src in sources:
        loc = f"section {src['section_idx']}, paragraph {src['paragraph_idx']}"
        console.print(f"  [{src['n']}] {src['paper_id']} ({loc}) — {src['filename']}")
    console.print()


@click.command()
@click.argument("query", required=False)
@click.option("--name", default=VECTOR_STORE_NAME, help="Vector store name to query.")
@click.option("--interactive", is_flag=True, help="Run in interactive mode (ask multiple questions).")
@click.option("--top-k", type=int, default=TOP_K, help="Number of paragraphs to retrieve.")
def main(query, name, interactive, top_k):
    """RAG CLI: query an OpenAI vector store and get a cited, streamed answer."""
    store_id = load_store_id(name)
    if not store_id:
        console.print(f"[red]No vector store found for '{name}'. Run index_openai.py first.[/red]")
        sys.exit(1)

    console.print(Panel(
        "[bold cyan]RAG Query Interface[/bold cyan]\nPowered by OpenAI Vector Store",
        border_style="cyan"
    ))

    if interactive:
        console.print("[yellow]Interactive mode. Type 'exit' to quit.[/yellow]\n")
        while True:
            try:
                user_query = console.input("[cyan]Enter your question:[/cyan] ").strip()
                if user_query.lower() in ("exit", "quit"):
                    console.print("[yellow]Goodbye![/yellow]")
                    break
                if user_query:
                    rag_query(user_query, store_id, top_k=top_k)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
                break
    elif query:
        rag_query(query, store_id, top_k=top_k)
    else:
        console.print(Panel(
            "[yellow]Usage:\n\n"
            "  Single query:\n"
            "    python src/cli_rag_openai.py \"What are affective events?\"\n\n"
            "  Interactive mode:\n"
            "    python src/cli_rag_openai.py --interactive[/yellow]",
            title="[bold cyan]Examples[/bold cyan]",
            border_style="cyan"
        ))


if __name__ == "__main__":
    main()
