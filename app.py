import json
import os
from functools import lru_cache
from typing import Optional

import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

KB_PATH = os.getenv("KB_PATH", "knowledge_base.json")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Tune this using the official probe questions. Too high => false abstentions;
# too low => answers to unrelated questions.
ABSTAIN_THRESHOLD = float(os.getenv("ABSTAIN_THRESHOLD", "0.34"))

with open(KB_PATH, "r", encoding="utf-8") as f:
    PASSAGES = json.load(f)

if len(PASSAGES) != 50:
    raise RuntimeError(f"Expected 50 passages, found {len(PASSAGES)}")

model = SentenceTransformer(MODEL_NAME)
vectors = model.encode(
    PASSAGES,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=False
).astype("float32")

index = faiss.IndexFlatIP(vectors.shape[1])
index.add(vectors)

app = FastAPI(title="TDS GA5 RAG Colosseum", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=3, ge=1, le=10)

class QueryResponse(BaseModel):
    answer: Optional[str]
    sources: list[int]
    scores: list[float]
    confidence: float

@app.get("/")
def root():
    return {"service": "TDS GA5 RAG", "status": "ready"}

@app.get("/health")
def health():
    return {"status": "ok", "passages": len(PASSAGES), "embedding_model": MODEL_NAME}

@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    q = model.encode(
        [req.question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    k = min(req.top_k, len(PASSAGES))
    distances, indices = index.search(q, k)

    pairs = [
        (int(i), float(max(0.0, min(1.0, s))))
        for i, s in zip(indices[0], distances[0])
        if int(i) >= 0
    ]
    if not pairs:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.0)

    best_idx, best_score = pairs[0]
    if best_score < ABSTAIN_THRESHOLD:
        return QueryResponse(answer=None, sources=[], scores=[], confidence=0.15)

    # Extractive answer: return the retrieved source passage verbatim.
    # This avoids hallucinating facts not present in the personalized KB.
    confidence = round(min(0.99, max(0.61, best_score)), 4)
    return QueryResponse(
        answer=PASSAGES[best_idx],
        sources=[i for i, _ in pairs],
        scores=[round(s, 4) for _, s in pairs],
        confidence=confidence
    )
