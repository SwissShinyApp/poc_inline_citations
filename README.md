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

2. **Set environment variables:**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export WEAVIATE_URL="http://localhost:8080"
   export GENERATION_MODEL="gpt-4-turbo"  # Optional, defaults to gpt-4-turbo
   export TOP_K="5"  # Optional, number of documents to retrieve
   ```

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
