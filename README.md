# TDS GA5 Q1 — RAG Colosseum

## Files
- `app.py`: FastAPI service with POST `/query`
- `knowledge_base.json`: your 50 personalized passages
- `requirements.txt`: Python dependencies

## Run locally
Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

First startup downloads `sentence-transformers/all-MiniLM-L6-v2`, so internet is
needed for the first run unless the model is already cached.

## Test
```bash
curl -X POST http://localhost:7860/query \\
  -H "Content-Type: application/json" \\
  -d '{"question":"Which consensus protocol uses a quorum of 2 nodes?","top_k":3}'
```

Open `http://localhost:7860/docs` to test interactively.

## Hugging Face Spaces
Create a new **Docker** Space, upload these three files, and use this Dockerfile:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV HF_HOME=/data
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

The public base URL is your Space URL, without `/query`.

## Important grading note
The initial abstention threshold is a starting point, not a verified optimum.
Use the assignment's 20 probe questions to tune `ABSTAIN_THRESHOLD`, then run
Check while the server is live. This version uses extractive answers to avoid
inventing facts. It has not yet been tested against the private evaluation probes.
