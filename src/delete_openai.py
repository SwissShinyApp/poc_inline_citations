#!/usr/bin/env python3
"""
Delete an OpenAI Vector Store created by `index_openai.py`.

By default this also deletes the underlying uploaded files (one per paragraph),
since deleting only the vector store would leave those File objects orphaned in
your OpenAI account. Use `--keep-files` to remove just the vector store. The
store's entry in `data/openai_vector_store.json` is forgotten afterwards.
"""
import json
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

from index_openai import get_openai_client, load_store_id, STORE_ID_FILE, VECTOR_STORE_NAME

console = Console()


def forget_store_id(name):
    """Remove the persisted vector store id for `name`, if present."""
    if not STORE_ID_FILE.exists():
        return
    try:
        data = json.loads(STORE_ID_FILE.read_text())
    except Exception:
        return
    if name in data:
        del data[name]
        STORE_ID_FILE.write_text(json.dumps(data, indent=2))


def delete_vector_store(name, keep_files=False):
    """Delete the vector store for `name` and (optionally) its uploaded files."""
    client = get_openai_client()
    store_id = load_store_id(name)
    if not store_id:
        console.print(f"[yellow]No vector store found for '{name}'. Nothing to delete.[/yellow]")
        return

    # Delete the underlying uploaded files (one per paragraph) unless asked to keep them.
    if not keep_files:
        try:
            file_ids = [vs_file.id for vs_file in client.vector_stores.files.list(vector_store_id=store_id)]
            if file_ids:
                with Progress() as progress:
                    task = progress.add_task("[cyan]Deleting files...", total=len(file_ids))
                    for file_id in file_ids:
                        try:
                            client.files.delete(file_id)
                        except Exception as e:
                            console.print(f"[yellow]Could not delete file {file_id}: {e}[/yellow]")
                        progress.update(task, advance=1)
                console.print(f"[green]✓ Deleted {len(file_ids)} uploaded files[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not list/delete files: {e}[/yellow]")

    # Delete the vector store itself.
    try:
        client.vector_stores.delete(store_id)
        console.print(f"[green]✓ Deleted vector store '{name}' ({store_id})[/green]")
    except Exception as e:
        console.print(f"[red]Error deleting vector store {store_id}: {e}[/red]")
        sys.exit(1)

    forget_store_id(name)


@click.command()
@click.option("--name", default=VECTOR_STORE_NAME, help="Vector store name to delete.")
@click.option("--keep-files", is_flag=True, default=False,
              help="Delete only the vector store, keeping the uploaded paragraph files.")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip the confirmation prompt.")
def main(name, keep_files, yes):
    """Delete an OpenAI vector store and (by default) its uploaded files."""
    console.print(Panel("[bold red]Delete OpenAI Vector Store[/bold red]", border_style="red"))

    store_id = load_store_id(name)
    if not store_id:
        console.print(f"[yellow]No vector store found for '{name}'. Nothing to delete.[/yellow]")
        return

    console.print(f"Name: [yellow]{name}[/yellow]")
    console.print(f"ID:   [yellow]{store_id}[/yellow]")
    console.print(
        "[dim]Uploaded paragraph files will also be deleted.[/dim]"
        if not keep_files else
        "[dim]Uploaded files will be kept (only the vector store is removed).[/dim]"
    )

    if not yes and not click.confirm("Proceed with deletion?"):
        console.print("[yellow]Aborted.[/yellow]")
        return

    delete_vector_store(name, keep_files=keep_files)


if __name__ == "__main__":
    main()
