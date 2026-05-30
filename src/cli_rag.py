#!/usr/bin/env python3
"""
RAG CLI: Query interface for retrieving documents and generating answers with Cohere.
"""
import os
import sys

import click
import cohere
import weaviate
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from langsmith import traceable

console = Console()
client_cohere = None

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = os.getenv("GENERATION_MODEL", "command-r7b-12-2024")
TOP_K = int(os.getenv("TOP_K", "5"))


def get_cohere_client():
    """Get or initialize Cohere client."""
    global client_cohere
    if client_cohere is None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            console.print("[red]Error: COHERE_API_KEY environment variable not set[/red]")
            sys.exit(1)
        client_cohere = cohere.ClientV2(api_key=api_key)
    return client_cohere


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
    """Get embedding from Cohere for the given text."""
    try:
        client = get_cohere_client()
        response = client.embed(
            model=EMBEDDING_MODEL,
            texts=[text],
            input_type="search_query",
        )
        return response.embeddings[0]
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

def format_documents_for_cohere(documents):
    """Format retrieved documents for Cohere API with citations.
    
    Converts Weaviate documents to Cohere's expected format:
    {
        "data": {"title": "...", "snippet": "..."},
        "id": "doc_id"
    }
    """
    cohere_documents = []
    for i, doc in enumerate(documents):
        props = doc.properties
        text = props.get("text", "")[:500]  # Truncate for brevity
        title = props.get("title", "Unknown")
        paper_id = props.get("paper_id", "Unknown")
        
        cohere_doc = {
            "data": {
                "title": title,
                "snippet": text,
            },
            "id": paper_id,  # Use paper_id as the document ID
        }
        cohere_documents.append(cohere_doc)
    
    return cohere_documents


@traceable(run_type="llm")
def generate_answer(query, documents):
    """Generate answer using Cohere with retrieved documents as context, with fast citations and token-level streaming."""
    if not documents:
        return "No relevant documents found. Please try a different query."
    
    # Format documents for Cohere API
    cohere_documents = format_documents_for_cohere(documents)
    
    try:
        client = get_cohere_client()
        
        full_answer = ""
        citations_list = []
        
        # Use chat_stream with fast citations
        stream = client.chat_stream(
            model=GENERATION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful research assistant that answers questions based on academic papers."
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            documents=cohere_documents,
            citation_options={"mode": "fast"},
            temperature=0.7,
            max_tokens=1000,
        )
        
        # Process streaming response
        for chunk in stream:
            if chunk:
                # Handle text content (token-level streaming)
                if chunk.type == "content-delta":
                    if chunk.delta.message.content:
                        token = chunk.delta.message.content.text
                        full_answer += token
                        console.print(token, end="", style="cyan")
                
                # Handle citations
                elif chunk.type == "citation-start":
                    if chunk.delta.message.citations:
                        citation = chunk.delta.message.citations
                        citations_list.append(citation)
                        # Print citation marker inline
                        citation_id = citation.sources[0].id if citation.sources else "?"
                        console.print(f" [{citation_id}]", end="", style="yellow")
        
        console.print()  # New line after streaming
        return full_answer, citations_list
    except Exception as e:
        console.print(f"[red]Error generating answer: {e}[/red]")
        return None, []


def rag_query(query):
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
    
    # Generate answer with streaming and citations
    console.print("[bold cyan]Answer:[/bold cyan]")
    answer, citations_list = generate_answer(query, documents)
    
    if not answer:
        return
    
    # Show sources
    console.print("\n[dim]Sources:[/dim]")
    for i, doc in enumerate(documents, 1):
        props = doc.properties
        title = props.get("title", "Unknown")
        paper_id = props.get("paper_id", "Unknown")
        console.print(f"  [{paper_id}] {title}")
    
    # Show citation details if available
    if citations_list:
        console.print("\n[dim]Citations:[/dim]")
        for citation in citations_list:
            for source in citation.sources:
                console.print(f"  [{source.id}] {source.document.get('title', 'Unknown')}")
    
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
    """RAG CLI: Query documents and get AI-generated answers with citations."""
    # OpenAI client will be initialized on first use
    
    # Override TOP_K if provided
    global TOP_K
    TOP_K = top_k
    
    console.print(Panel(
        "[bold cyan]RAG Query Interface[/bold cyan]\nPowered by Weaviate + Cohere",
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
            "    python src/cli_rag.py --interactive[/yellow]",
            title="[bold cyan]Examples[/bold cyan]",
            border_style="cyan"
        ))


if __name__ == "__main__":
    main()
