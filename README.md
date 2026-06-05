# poc_inline_citations
The goal of this repo is to test different approaches on the in-text citation of passages while keeping the token-level streaming of the generation.

This repo ships **two interchangeable retrieval backends** over the same QASPER
paragraph data:

- **Weaviate** (`index_weaviate.py` + `cli_rag.py`) — local vector DB, you provide the
  OpenAI embeddings. Requires Docker.
- **OpenAI Vector Store** (`index_openai.py` + `cli_rag_openai.py`) — a managed
  OpenAI Vector Store; OpenAI handles embedding and retrieval. No Docker required.
  Generation uses the **Responses API + `file_search`**, streaming token-by-token
  with **inline `[n]` citations** emitted as retrieval annotations arrive.

## RAG Pipeline with Weaviate & OpenAI

This project includes a simple Retrieval-Augmented Generation (RAG) pipeline that:
1. Indexes documents from the QASPER dataset into a local Weaviate instance
2. Embeds queries and retrieves relevant documents using OpenAI embeddings
3. Generates answers using OpenAI's GPT model with document context as citations

### Prerequisites

- Docker and Docker Compose
- Python 3.9+
- OpenAI API key

### Setup & Installation

1. **Clone/setup the environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   Create a `.env` file in the project root directory with the following variables:
   
   ```
   # Required variables
   OPENAI_API_KEY=your-openai-api-key
   WEAVIATE_URL=http://localhost:8080
   
   # Optional variables
   GENERATION_MODEL=gpt-4o-mini        # Model to use for generation (defaults to gpt-4o-mini)
   TOP_K=5                             # Number of documents to retrieve (defaults to 5)
   ```
   
   **Variable descriptions:**
   - `OPENAI_API_KEY` (required): Your OpenAI API key for embeddings and generation
   - `WEAVIATE_URL` (required): URL where Weaviate instance is running
   - `LANGSMITH_TRACING=true`(optional): Enable LangSmith tracing for debugging and analysis
   - `LANGSMITH_ENDPOINT=https://aws.api.smith.langchain.com` (optional): LangSmith endpoint for tracing data
   - `LANGSMITH_API_KEY`="your-api-key" (optional): LangSmith API key for tracing
   - `LANGSMITH_PROJECT`="your-project-name" (optional): LangSmith project name for organizing traces
   - `GENERATION_MODEL` (optional): OpenAI model for generating answers; defaults to `gpt-4o-mini`
   - `TOP_K` (optional): Number of documents to retrieve; defaults to `5`
   - `OPENAI_VECTOR_STORE_NAME` (optional): Name of the OpenAI Vector Store; defaults to `qasper-paragraphs` (OpenAI backend only)
   - `OPENAI_VECTOR_STORE_ID` (optional): Force a specific vector store id, overriding the one persisted in `data/openai_vector_store.json` (OpenAI backend only)

3. **Start Weaviate locally:**
   ```bash
   docker-compose up -d
   ```
   Verify it's running: `curl http://localhost:8080/v1/.well-known/ready`

4. **Index the dataset:**
   ```bash
   python src/index_weaviate.py --file data/dummy_data/qasper-train-v0.1.json
   ```
   This will embed and index all documents from the QASPER training set into Weaviate.
   
   **Optional flags:**
   - `--delete`: Delete the existing Document collection before creating a new one. Use this if you want to re-index from scratch:
     ```bash
     python src/index_weaviate.py --file data/dummy_data/qasper-train-v0.1.json --delete
     ```

### Usage

**Single Query:**
```bash
python src/cli_rag.py "What is affective events?"
```

**Interactive Mode:**
```bash
python src/cli_rag.py --interactive
```
Then ask multiple questions. Type `exit` or `quit` to exit.

**Customize retrieval:**
```bash
python src/cli_rag.py "Your question here?" --top-k 10
```

### Project Structure

```
.
├── docker-compose.yml          # Local Weaviate setup
├── requirements.txt            # Python dependencies
├── src/
│   ├── index_weaviate.py      # Index documents from JSON into Weaviate
│   ├── cli_rag.py             # CLI interface for querying (Weaviate backend)
│   ├── index_openai.py        # Index paragraphs into an OpenAI Vector Store
│   ├── cli_rag_openai.py      # CLI querying (OpenAI backend, file_search inline citations)
│   ├── delete_openai.py       # Delete an OpenAI Vector Store and its uploaded files
│   └── evaluate_qas.py        # Batch evaluation over QASPER QAS items (OpenAI backend)
├── data/
│   ├── openai_vector_store.json    # Persisted OpenAI Vector Store id(s) (generated)
│   └── dummy_data/
│       └── qasper-train-v0.1.json  # Dataset with full_text documents
└── README.md
```

### How It Works

1. **Indexing (`index_weaviate.py`):**
   - Reads the QASPER JSON file
   - Extracts the `full_text` field from each paper (flattens section structure)
   - Calls OpenAI Embeddings API to vectorize each document
   - Stores vectors and metadata in Weaviate

2. **Querying (`cli_rag.py`):**
   - Embeds the user query using OpenAI Embeddings
   - Performs vector similarity search in Weaviate (top-k retrieval)
   - Composes a prompt with retrieved documents as context
   - Calls OpenAI Chat/Completion API to generate an answer with citations
   - Displays the answer and source documents

## RAG Pipeline with OpenAI Vector Store

This backend stores each paragraph as its own file in a managed **OpenAI Vector
Store** — no Weaviate/Docker needed. Generation uses the **Responses API +
`file_search`**: the model retrieves passages itself and the answer streams
token-by-token with **inline `[n]` citations**.

1. **Index the dataset:**
   ```bash
   python src/index_openai.py --file data/dummy_data/qasper-train-v0.1.json
   ```
   This creates a vector store, uploads each paragraph as a file with its metadata
   (`paper_id`, `title`, `section_name`, `section_idx`, `paragraph_idx`, `source`) as
   file attributes, and persists the vector store id to `data/openai_vector_store.json`.

   **Optional flags:**
   - `--name`: Vector store name (defaults to `qasper-paragraphs`).
   - `--recreate`: Delete the persisted vector store before indexing, to rebuild from scratch:
     ```bash
     python src/index_openai.py --recreate
     ```

2. **Query:**
   ```bash
   # Single query
   python src/cli_rag_openai.py "What are affective events?"

   # Interactive mode
   python src/cli_rag_openai.py --interactive

   # Control how many passages file_search may retrieve
   python src/cli_rag_openai.py "Your question here?" --top-k 10
   ```

3. **Delete the vector store** (when you're done or want to rebuild):
   ```bash
   # Deletes the vector store AND its uploaded paragraph files, then clears
   # the entry in data/openai_vector_store.json (prompts for confirmation)
   python src/delete_openai.py

   python src/delete_openai.py --yes          # skip the prompt
   python src/delete_openai.py --keep-files   # remove only the vector store
   ```

> **Note:** Run these from the project root (`python src/...`), so the `data/` paths
> resolve and the `src/` modules import each other correctly.

### How It Works (OpenAI backend)

1. **Indexing (`index_openai.py`):** uploads one file per paragraph to an OpenAI
   Vector Store with metadata as file attributes. OpenAI embeds the content internally.
2. **Querying (`cli_rag_openai.py`):** calls the Responses API with the `file_search`
   tool pointed at the vector store and `stream=True`. As events arrive it prints text
   deltas (`response.output_text.delta`) and, on each citation annotation
   (`response.output_text.annotation.added`), assigns a stable `[n]` per source file and
   prints it **inline** — so factual statements are cited in-text, not only at the end.
   The numbered sources behind the markers are listed after the answer.

   > A direct `retrieve_documents` helper (raw `vector_stores.search`) is also exposed in
   > `cli_rag_openai.py` for callers that want retrieval without the model in the loop.

## Evaluating the Pipeline

Run automated evaluations on all QAS (Question-Answer Sets) from the QASPER dataset
through the **OpenAI `file_search`** pipeline. Build the vector store first
(`python src/index_openai.py`), since the evaluation reuses it.

```bash
# Run evaluation with auto-generated timestamp tag
python src/evaluate_qas.py

# Run with custom tag
python src/evaluate_qas.py --tag "experiment_v1"

# Test run with limited questions
python src/evaluate_qas.py --tag "test" --limit 3

# Target a specific vector store / retrieval depth
python src/evaluate_qas.py --tag "exp" --name qasper-paragraphs --top-k 8
```

Results are saved to `results/<TAG>/results.json` with:
- Question and the inline-cited answer for each QAS item
- Latency metrics (generation, total) and number of sources cited
- `sources`: the files behind each `[n]` marker (paper_id, section/paragraph, file_id)
- Any errors encountered

**Analyze results programmatically:**
```python
import json
with open("results/experiment_v1/results.json") as f:
    data = json.load(f)
print(f"Average latency: {data['average_latency_ms']}ms")
print(f"Success rate: {data['questions_completed']}/{data['total_questions']}")
```

See [README_EVALUATION.md](README_EVALUATION.md) for detailed evaluation documentation.

### Troubleshooting

- **Connection refused on port 8080:** Ensure `docker-compose up -d` has started Weaviate successfully.
- **OpenAI API errors:** Check that `OPENAI_API_KEY` is set and valid.
- **Embedding/generation timeouts:** Reduce `TOP_K` or check OpenAI API status.
- **No documents found:** Ensure indexing completed successfully; check Weaviate UI at `http://localhost:8080/console`.
- **"class name Document already exists" error:** The Document collection already exists in Weaviate from a previous run. Use the `--delete` flag to recreate it:
  ```bash
  python src/index_weaviate.py --file data/dummy_data/qasper-train-v0.1.json --delete
  ```
  Or manually delete it through the Weaviate UI if you prefer to keep it and add more documents.
