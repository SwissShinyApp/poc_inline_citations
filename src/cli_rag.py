#!/usr/bin/env python3
"""
RAG CLI: Query interface for retrieving documents and generating answers with OpenAI.
"""
import os
import sys

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
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "gpt-4o-mini")
TOP_K = int(os.getenv("TOP_K", "5"))


def load_prompts():
    """Load prompts from prompts.json file."""
    try:
        with open("src/prompts.json", "r") as f:
            prompts = json.load(f)
            return {p["name"]: p["prompt"] for p in prompts}
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


@traceable(run_type="retriever", name="embedding")
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

@traceable(run_type="llm")
def generate_answer(query, documents, prompt_name='NAIVE_ALCE'):
    """Generate answer using OpenAI with retrieved documents as context, streamed."""
    if not documents:
        return "No relevant documents found. Please try a different query."
    
    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(documents, 1):
        props = doc.properties
        text = props.get("text", "")[:500]  # Truncate for brevity
        title = props.get("title", "Unknown")
        section_name = props.get("section_name", "Unknown")
        paragraph_idx = props.get("paragraph_idx", 0)
        context_parts.append(f"[{i}] {title} - {section_name} (paragraph {paragraph_idx}):\n{text}...")
    
    context = "\n\n".join(context_parts)
    prompts = load_prompts()
    
    prompt = f"""You are a helpful research assistant. Answer the following question based on the provided research papers.


                RELEVANT DOCUMENTS:
                {context}
                
                CITATION FORMAT: 
                {prompts['CITATION_FORMAT']}
                
                INSTRUCTIONS: 
                {prompts[prompt_name]}

                Question: {query}

                """
    
    try:
        client = get_openai_client()
        
        full_answer = ""
        stream = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful research assistant that answers questions based on academic papers."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_answer += token
                console.print(token, end="", style="cyan")
        
        console.print()  # New line after streaming
        return full_answer
    except Exception as e:
        console.print(f"[red]Error generating answer: {e}[/red]")
        return None


def rag_query(query, prompt='NAIVE_ALSE'):
    """Execute RAG pipeline: embed query, retrieve docs, generate answer."""
    console.print(f"\n[cyan]Query:[/cyan] {query}\n")
    
    # Get query embedding
    with console.status("[yellow]Embedding query...[/yellow]"):
        query_embedding = get_embedding(query)
        if query_embedding is None:
            return
    
    # Connect to Weaviate and retrieve documents
    weaviate_client = get_weaviate_client()
    
    with console.status("[yellow]Searching for relevant documents...[/yellow]"):
        documents = retrieve_documents(query_embedding, weaviate_client, top_k=TOP_K)
    
    weaviate_client.close()
    
    if not documents:
        console.print("[yellow]No documents found. Try a different query.[/yellow]\n")
        return
    
    console.print(f"[green]Found {len(documents)} relevant documents[/green]\n")
    
    # Generate answer with streaming
    console.print("[bold cyan]Answer:[/bold cyan]")
    answer = generate_answer(query, documents, prompt)
    
    if not answer:
        return
    
    # Show sources
    console.print("\n[dim]Sources:[/dim]")
    for i, doc in enumerate(documents, 1):
        props = doc.properties
        title = props.get("title", "Unknown")
        section_name = props.get("section_name", "Unknown")
        paragraph_idx = props.get("paragraph_idx", 0)
        console.print(f"  [{i}] {title} - {section_name} (paragraph {paragraph_idx})")
    
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
@click.option(
    '--prompt',
    type=str,
    default='NAIVE_ALCE',
    help='Prompt template to use for answer generation',
)
def main(query, interactive, top_k, prompt):
    """RAG CLI: Query documents and get AI-generated answers with citations."""
    # OpenAI client will be initialized on first use
    
    # Override TOP_K if provided
    global TOP_K
    TOP_K = top_k
    
    console.print(Panel(
        "[bold cyan]RAG Query Interface[/bold cyan]\nPowered by Weaviate + OpenAI",
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
                    rag_query(user_query, prompt)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
                break
    elif query:
        # Single query mode
        rag_query(query, prompt)
    else:
        # No query provided, show help
        console.print(Panel(
            "[yellow]Usage:\n\n"
            "  Single query:\n"
            "    python src/cli_rag.py \"What is affective events?\"\n\n"
            "  Interactive mode:\n"
            "    python src/cli_rag.py --interactive[/yellow]",
            title="[bold cyan]Examples[/bold cyan]",
            border_style="cyan"
        ))


if __name__ == "__main__":
    main()
