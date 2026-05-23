# poc_inline_citations
The goal of this repo is to test different approaches on the in-text citation of passages while keeping the token-level streaming of the generation.

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
   GENERATION_MODEL=gpt-4-turbo        # Model to use for generation (defaults to gpt-4-turbo)
   TOP_K=5                             # Number of documents to retrieve (defaults to 5)
   ```
   
   **Variable descriptions:**
   - `OPENAI_API_KEY` (required): Your OpenAI API key for embeddings and generation
   - `WEAVIATE_URL` (required): URL where Weaviate instance is running
   - `LANGSMITH_TRACING=true`(optional): Enable LangSmith tracing for debugging and analysis
   - `LANGSMITH_ENDPOINT=https://aws.api.smith.langchain.com` (optional): LangSmith endpoint for tracing data
   - `LANGSMITH_API_KEY`="your-api-key" (optional): LangSmith API key for tracing
   - `LANGSMITH_PROJECT`="your-project-name" (optional): LangSmith project name for organizing traces
   - `GENERATION_MODEL` (optional): OpenAI model for generating answers; defaults to `gpt-4-turbo`
   - `TOP_K` (optional): Number of documents to retrieve from Weaviate; defaults to `5`

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
│   └── cli_rag.py             # CLI interface for querying
├── data/
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
