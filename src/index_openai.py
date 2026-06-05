#!/usr/bin/env python3
"""
Index QASPER paragraphs into an OpenAI Vector Store.

Each paragraph is uploaded as its own file (filename `paper_id__sX__pY.txt`) with
its metadata attached as vector-store file attributes, mirroring the granularity
of the Weaviate pipeline in `index_weaviate.py`. The created vector store id is
persisted to `data/openai_vector_store.json` so `cli_rag_openai.py` can reuse it.
"""
import io
import json
import os
import sys
from pathlib import Path

import click
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

console = Console()
client_openai = None

VECTOR_STORE_NAME = os.getenv("OPENAI_VECTOR_STORE_NAME", "qasper-paragraphs")
STORE_ID_FILE = Path("data/openai_vector_store.json")
DEFAULT_DATA_FILE = "data/dummy_data/qasper-train-v0.1.json"


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


def save_store_id(name, store_id):
    """Persist the vector store id keyed by name so it can be reused."""
    data = {}
    if STORE_ID_FILE.exists():
        try:
            data = json.loads(STORE_ID_FILE.read_text())
        except Exception:
            data = {}
    data[name] = store_id
    STORE_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    STORE_ID_FILE.write_text(json.dumps(data, indent=2))


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


def iter_paragraphs(data):
    """Yield (paper_id, title, section_idx, section_name, paragraph_idx, text)."""
    for paper_id, paper_data in data.items():
        title = paper_data.get("title", "")
        for section_idx, section in enumerate(paper_data.get("full_text", [])):
            section_name = section.get("section_name", "")
            for paragraph_idx, paragraph in enumerate(section.get("paragraphs", [])):
                text = (paragraph or "").strip()
                if not text:
                    continue
                yield paper_id, title, section_idx, section_name, paragraph_idx, text


def get_or_create_vector_store(client, name, recreate=False):
    """Create the vector store, optionally deleting any existing one with this id."""
    existing_id = load_store_id(name)
    if existing_id and recreate:
        try:
            client.vector_stores.delete(existing_id)
            console.print(f"[green]✓ Deleted existing vector store {existing_id}[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not delete {existing_id}: {e}[/yellow]")
        existing_id = None

    if existing_id and not recreate:
        try:
            store = client.vector_stores.retrieve(existing_id)
            console.print(f"[yellow]Reusing existing vector store {store.id}[/yellow]")
            return store
        except Exception:
            console.print("[yellow]Persisted vector store id is stale, creating a new one.[/yellow]")

    store = client.vector_stores.create(name=name)
    save_store_id(name, store.id)
    console.print(f"[green]✓ Created vector store '{name}' ({store.id})[/green]")
    return store


def index_paragraphs(json_file, name=VECTOR_STORE_NAME, recreate=False):
    """Index every paragraph as its own file in the OpenAI vector store."""
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        console.print(f"[green]✓ Loaded {len(data)} papers from {json_file}[/green]")
    except Exception as e:
        console.print(f"[red]Error loading JSON: {e}[/red]")
        sys.exit(1)

    client = get_openai_client()
    store = get_or_create_vector_store(client, name, recreate=recreate)

    paragraphs = list(iter_paragraphs(data))
    console.print(f"[cyan]Indexing {len(paragraphs)} paragraphs as files...[/cyan]")

    indexed_count = 0
    with Progress() as progress:
        task = progress.add_task("[cyan]Uploading...", total=len(paragraphs))
        for paper_id, title, section_idx, section_name, paragraph_idx, text in paragraphs:
            try:
                filename = f"{paper_id}__s{section_idx}__p{paragraph_idx}.txt"
                # Upload the paragraph text as a file...
                uploaded = client.files.create(
                    file=(filename, io.BytesIO(text.encode("utf-8"))),
                    purpose="assistants",
                )
                # ...then attach it to the vector store with paragraph metadata.
                client.vector_stores.files.create_and_poll(
                    vector_store_id=store.id,
                    file_id=uploaded.id,
                    attributes={
                        "paper_id": paper_id,
                        "title": title[:512],
                        "section_name": section_name[:512],
                        "section_idx": section_idx,
                        "paragraph_idx": paragraph_idx,
                        "source": "qasper-train-v0.1",
                    },
                )
                indexed_count += 1
            except Exception as e:
                console.print(
                    f"[yellow]Warning: failed to index {paper_id} s{section_idx} p{paragraph_idx}: {e}[/yellow]"
                )
            progress.update(task, advance=1)

    console.print(f"[green]✓ Indexed {indexed_count} paragraphs into {store.id}[/green]")
    return store.id


@click.command()
@click.option("--file", "json_file", type=click.Path(exists=True), default=DEFAULT_DATA_FILE,
              help="Path to QASPER JSON file.")
@click.option("--name", default=VECTOR_STORE_NAME, help="Vector store name.")
@click.option("--recreate", is_flag=True, default=False,
              help="Delete the existing vector store (if persisted) before indexing.")
def main(json_file, name, recreate):
    """Index QASPER paragraphs into an OpenAI vector store."""
    console.print(Panel("[bold cyan]OpenAI Vector Store Indexing[/bold cyan]", border_style="cyan"))
    index_paragraphs(json_file, name=name, recreate=recreate)


if __name__ == "__main__":
    main()
