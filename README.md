# GraphAgenticRAG

A LangGraph-based agentic RAG system combining hybrid search, graph reasoning, and web search — with a synthesizer/verifier loop for grounded, accurate answers.

## Features

- **Hybrid Retrieval** — Dense vector search (Pinecone) + sparse keyword search (BM25) for balanced semantic and lexical relevance
- **Graph RAG** — NetworkX-based knowledge graph traversal for multi-hop reasoning over document relationships
- **Web Search Fallback** — Tavily integration for queries beyond the ingested knowledge base
- **Synthesizer/Verifier Loop** — LLM-generated answers are checked against retrieved context before being returned, reducing hallucination
- **Session Isolation** — Per-user, per-session scoped storage (BM25 index, graph state) using `user_id_session_id` keys
- **Streaming Responses** — Server-Sent Events (SSE) for real-time token streaming to the frontend
- **Authentication** — Supabase Auth with JWT/JWKS (ES256) verification
- **Web UI** — Single-page interface with ChatGPT-style session management and drag-and-drop PDF ingestion
- **Containerized** — Dockerfile included for consistent, portable deployment

## Architecture

User → FastAPI → LangGraph Workflow
├─ Hybrid Retrieval (Pinecone + BM25)
├─ Graph Reasoning (NetworkX)
├─ Web Search (Tavily)
└─ Synthesizer/Verifier Loop → Groq Inference
↓
Supabase (persistence + auth)


Ingestion and querying are handled as separate pipelines — documents are processed and indexed independently from the query-time retrieval flow.

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| Vector Search | Pinecone |
| Keyword Search | BM25 |
| Graph Reasoning | NetworkX |
| Web Search | Tavily |
| LLM Inference | Groq |
| Backend | FastAPI |
| Auth & DB | Supabase |
| Package Management | uv |
| Containerization | Docker |

## Getting Started

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (optional, for containerized deployment)

### Local Setup

```bash
git clone https://github.com/Mohit-Baraiya11/GraphAgenticRAG.git
cd GraphAgenticRAG
uv sync
```

Create a `.env` file in the project root with the following variables:

GROQ_API_KEY=
LANGCHAIN_TRACING_V2=
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=
SUPABASE_URL=
SUPABASE_KEY=
TAVILY_API_KEY=


Run the server:

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000`.

### Docker Setup

Build the image:

```bash
docker build -t graphagenticrag .
```

Run the container:

```bash
docker run -p 8000:8000 --env-file .env graphagenticrag
```

Visit `http://localhost:8000` once the container is running.

> **Note:** Docker's `--env-file` requires unquoted values (`KEY=value`, not `KEY="value"`) and no spaces around the `=` sign.

## API

Interactive API docs are available at `/docs` (Swagger UI) once the server is running.

## Status

Actively in development. Core RAG pipeline, hybrid retrieval, session management, and Docker deployment are functional. RAGAS-based evaluation is planned next to quantify retrieval and answer quality.

## License

MIT
