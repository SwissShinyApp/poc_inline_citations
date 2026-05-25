#!/usr/bin/env python3
"""
Index documents from QASPER dataset into Weaviate.
Extracts full_text from each paper and creates embeddings using OpenAI.
"""
import json
import os
import sys
from pathlib import Path

import click
import weaviate
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress

console = Console()
client_openai = None

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 100


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


def delete_collection(client):
    """Delete the Document collection if it exists."""
    try:
        client.collections.delete("Document")
        console.print("[green]✓ Deleted existing Document collection[/green]")
    except Exception as e:
        console.print(f"[yellow]Collection does not exist or could not be deleted: {e}[/yellow]")


def create_schema(client):
    """Create or verify Document class schema in Weaviate."""
    # Check if collection already exists
    try:
        collection_list = [c.name for c in client.collections.list_all()]
        if "Document" in collection_list:
            console.print("[yellow]Document collection already exists.[/yellow]")
            return
    except Exception as e:
        console.print(f"[yellow]Could not verify existing collections: {e}[/yellow]")
    
    # Create the class if it doesn't exist
    try:
        from weaviate.collections.classes.config import Configure, Property, DataType
        
        schema = {
            "class": "Document",
            "properties": [
                {
                    "name": "text",
                    "dataType": ["text"],
                    "description": "Full text of the document section",
                },
                {
                    "name": "title",
                    "dataType": ["text"],
                    "description": "Title of the paper",
                },
                {
                    "name": "paper_id",
                    "dataType": ["text"],
                    "description": "Paper ID from QASPER dataset",
                    "indexInverted": True,
                },
                {
                    "name": "source",
                    "dataType": ["text"],
                    "description": "Source dataset",
                },
                {
                    "name": "section_name",
                    "dataType": ["text"],
                    "description": "Name of the section from the paper",
                },
                {
                    "name": "section_idx",
                    "dataType": ["int"],
                    "description": "Index of the section in the full_text array",
                },
            ],
            "vectorizer": "none",
            "vectorIndexConfig": {
                "distance": "cosine",
            },
        }
        
        client.collections.create_from_dict(schema)
        console.print("[green]✓ Created Document class in Weaviate[/green]")
    except Exception as e:
        console.print(f"[red]Error creating schema: {e}[/red]")
        raise


def format_section_text(section):
    """Format a single section into a string."""
    section_name = section.get("section_name", "")
    paragraphs = section.get("paragraphs", [])
    
    text_parts = []
    if section_name:
        text_parts.append(f"## {section_name}")
    
    for para in paragraphs:
        if para.strip():
            text_parts.append(para)
    
    return "\n\n".join(text_parts)


def get_openai_client():
    """Get or initialize OpenAI client."""
    global client_openai
    if client_openai is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            console.print("[red]Error: OPENAI_API_KEY environment variable not set[/red]")
            sys.exit(1)
        client_openai = OpenAI(api_key=api_key)
    return client_openai


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


def index_documents(json_file, delete_existing=False):
    """Index documents from JSON file into Weaviate."""
    # Load JSON
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        console.print(f"[green]✓ Loaded {len(data)} papers from {json_file}[/green]")
    except Exception as e:
        console.print(f"[red]Error loading JSON: {e}[/red]")
        sys.exit(1)
    
    # Connect to Weaviate
    weaviate_client = get_weaviate_client()
    
    # Delete existing collection if requested
    if delete_existing:
        delete_collection(weaviate_client)
    
    # Create schema
    create_schema(weaviate_client)
    
    # Get or create collection
    try:
        collection = weaviate_client.collections.get("Document")
    except Exception as e:
        console.print(f"[red]Error accessing Document collection: {e}[/red]")
        sys.exit(1)
    
    # Index documents
    console.print("[cyan]Indexing documents...[/cyan]")
    
    indexed_count = 0
    total_sections = sum(len(paper_data.get("full_text", [])) for paper_data in data.values())
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing...", total=total_sections)
        
        for paper_id, paper_data in data.items():
            try:
                title = paper_data.get("title", "")
                full_text_sections = paper_data.get("full_text", [])
                
                # Index each section separately
                for section_idx, section in enumerate(full_text_sections):
                    try:
                        section_name = section.get("section_name", "")
                        text = format_section_text(section)
                        
                        if not text.strip():
                            progress.update(task, advance=1)
                            continue
                        
                        # Get embedding
                        embedding = get_embedding(text)
                        if embedding is None:
                            progress.update(task, advance=1)
                            continue
                        
                        # Prepare object
                        obj = {
                            "text": text,
                            "title": title,
                            "paper_id": paper_id,
                            "source": "qasper-train-v0.1",
                            "section_name": section_name,
                            "section_idx": section_idx,
                        }
                        
                        # Add to Weaviate with vector
                        collection.data.insert(
                            properties=obj,
                            vector=embedding,
                        )
                        
                        indexed_count += 1
                        
                    except Exception as e:
                        console.print(f"[yellow]Warning: Failed to index section {section_idx} of {paper_id}: {e}[/yellow]")
                    
                    progress.update(task, advance=1)
                    
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to process {paper_id}: {e}[/yellow]")
    
    console.print(f"[green]✓ Indexed {indexed_count} documents successfully[/green]")
    
    weaviate_client.close()


@click.command()
@click.option(
    '--file',
    type=click.Path(exists=True),
    default='data/dummy_data/qasper-train-v0.1.json',
    help='Path to JSON file containing documents',
)
@click.option(
    '--delete',
    is_flag=True,
    default=False,
    help='Delete existing Document collection before indexing',
)
def main(file, delete):
    """Index QASPER documents into Weaviate."""
    # OpenAI client will be initialized on first use
    index_documents(file, delete_existing=delete)


if __name__ == "__main__":
    main()
