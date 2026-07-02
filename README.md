## Production RAG Observability & Evaluation Pipeline

A Retrieval-Augmented Generation (RAG) system built with FastAPI, LangChain, Ollama, and Qdrant, equipped with prompt-injection guardrails, continuous monitoring (Prometheus & Grafana), and automated LLM evaluations (DeepEval + GitHub Actions).

### Tech Stack

| Component | Implementation |
| --- | --- |
| API | FastAPI, Uvicorn |
| RAG Orchestration | LangChain |
| Vector Database | Qdrant |
| LLM & Embeddings | Ollama (llama3.2:1b & nomic-embed-text) |
| Observability | Prometheus & Grafana |
| Evaluation & CI/CD | DeepEval, Pytest, GitHub Actions |

### Prerequisites

Before running the application, ensure you have the following installed:
- Docker & Docker Compose
- Python 3.11+

### How to Run

1. **Start all services:**
   ```bash
   docker compose up -d
   ```

2. **Pull required Ollama models:**
   ```bash
   docker compose exec ollama ollama pull llama3.2:1b
   docker compose exec ollama ollama pull nomic-embed-text
   docker compose exec ollama ollama pull llama-guard3:1b
   docker compose exec ollama ollama pull qwen2.5:14b
   ```

3. **Ingest sample data into vector database:**
   ```bash
   docker compose exec fastapi_app python src/vectorstore/ingest.py
   ```

4. **Run evaluation tests:**
   ```bash
   docker compose exec fastapi_app python -m pytest src/evals/test_rag_eval.py -v -s
   ```

### Sample Queries

> **For macOS / Linux Terminal:**
```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "What is meant by Retrieval Augmented Generation?"}'
```

> **For Windows PowerShell:**
```powershell
$body = @{ prompt = "What is meant by Retrieval Augmented Generation?" } | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/query" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

### Observability

Once services are running, access the user interfaces and metrics via your browser:

| Service | Endpoint | Description |
| --- | --- | --- |
| FastAPI Swagger UI | http://localhost:8000/docs | Interactive API documentation |
| Prometheus Metrics | http://localhost:9090 | Raw application metrics |
| Grafana | http://localhost:3000 | Visual observability dashboards |
| Qdrant Dashboard | http://localhost:6333/dashboard | Inspect vector collections |

### Evaluations

This project uses **DeepEval** to test for faithfulness and answer relevancy. Evaluations run against a local Ollama model rather than OpenAI, so no external API keys are required. 

The GitHub Actions workflow (`.github/workflows/ci-eval.yml`) automatically executes these evaluation tests on every `push` or `pull request` to the `main` branch.

### Project Structure

```text
.
├── .github
│   └── workflows
│       └── ci-eval.yml
├── Dockerfile
├── README.md
├── config
│   └── prometheus.yml
├── docker-compose.yml
├── requirements.txt
├── sample_knowledge.txt
└── src
    ├── __init__.py
    ├── app
    │   ├── __init__.py
    │   ├── guardrails.py
    │   ├── main.py
    │   └── rag_chain.py
    ├── evals
    │   └── test_rag_eval.py
    └── vectorstore
        └── ingest.py
```

