## Production RAG Observability & Evaluation Pipeline

A Retrieval-Augmented Generation (RAG) system built with FastAPI, LangChain, Ollama, and Qdrant, equipped with prompt-injection guardrails, continuous monitoring (Prometheus & Grafana), and automated LLM evaluations (DeepEval + GitHub Actions), deployable with Docker Compose or on Kubernetes.

### Tech Stack

| Component | Implementation |
| --- | --- |
| API | FastAPI, Uvicorn |
| RAG Orchestration | LangChain |
| Vector Database | Qdrant |
| LLM & Embeddings | Ollama (qwen2.5:14b for answers, llama-guard3:1b for guardrails, nomic-embed-text for embeddings; llama3.2:1b in CI) |
| Observability | Prometheus & Grafana |
| Deployment | Docker, Docker Compose, Kubernetes (kind), Kustomize |
| Evaluation & CI/CD | DeepEval, Pytest, GitHub Actions (LLM evals + Kubernetes smoke test) |


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

### Run on Kubernetes (kind)

Requires Docker Desktop, [kind](https://kind.sigs.k8s.io/) and kubectl.

```bash
kind create cluster --name rag
docker build -t rag-api:local .
kind load docker-image rag-api:local --name rag
kubectl apply -k k8s/
kubectl -n rag get pods -w        # wait for rag-api to be 1/1 Running
kubectl -n rag port-forward svc/rag-api 8000:8000
```

The stack runs in the `rag` namespace: FastAPI, Qdrant, Ollama, Prometheus and Grafana. Jobs pull the models and seed the vector store. The API waits for the collection before starting.

The default model (`qwen2.5:14b`) needs ~10 GB RAM. On a laptop, set `MODEL_NAME` in `k8s/config.yaml` to a smaller model such as `qwen2.5:3b`.

Every push to `main` deploys the stack to a fresh kind cluster in GitHub Actions (`k8s-smoke-test.yml`).

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

On Kubernetes, port-forward each service first, e.g. `kubectl -n rag port-forward svc/grafana 3000:3000` (likewise `prometheus 9090`, `qdrant 6333`, `rag-api 8000`). Grafana comes with Prometheus already configured as a data source.


### Evaluations

This project uses **DeepEval** to test for faithfulness and answer relevancy. Evaluations run against a local Ollama model rather than OpenAI, so no external API keys are required. 

The GitHub Actions workflow (`.github/workflows/ci-eval.yml`) automatically executes these evaluation tests on every `push` or `pull request` to the `main` branch.

### Project Structure

```text
.
├── .github
│   └── workflows
│       ├── ci-eval.yml
│       └── k8s-smoke-test.yml
├── .dockerignore
├── Dockerfile
├── README.md
├── config
│   └── prometheus.yml
├── docker-compose.yml
├── k8s
│   ├── kustomization.yaml
│   ├── namespace.yaml
│   ├── config.yaml
│   ├── qdrant.yaml
│   ├── ollama.yaml
│   ├── model-pull-job.yaml
│   ├── ingest-job.yaml
│   ├── api.yaml
│   └── monitoring.yaml
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

