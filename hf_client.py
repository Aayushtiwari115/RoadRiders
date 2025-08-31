# hf_client.py
# ---------------------------------------------------------------------
# Hugging Face API Client for PBS What-if Simulator
#
# Provides a wrapper for text-generation, question-answering,
# and sentence-transformers embeddings.
# ---------------------------------------------------------------------

import os
import requests
from typing import Dict, Any, List

HF_API_KEY = os.getenv("HF_API_KEY", "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")
BASE_URL = "https://api-inference.huggingface.co/models"

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}


def hf_text_generation(prompt: str,
                       model: str = "gpt2",
                       max_new_tokens: int = 200) -> str:
    """Text generation using Hugging Face API"""
    try:
        response = requests.post(
            f"{BASE_URL}/{model}",
            headers=HEADERS,
            json={"inputs": prompt,
                  "parameters": {"max_new_tokens": max_new_tokens}}
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return str(data)
    except Exception as e:
        return f"[HF Error: text-generation] {e}"


def hf_question_answering(question: str,
                          context: str,
                          model: str = "distilbert-base-uncased-distilled-squad") -> str:
    """Extractive QA from context"""
    try:
        response = requests.post(
            f"{BASE_URL}/{model}",
            headers=HEADERS,
            json={"inputs": {"question": question, "context": context}}
        )
        response.raise_for_status()
        data = response.json()
        return data.get("answer", str(data))
    except Exception as e:
        return f"[HF Error: QA] {e}"


def hf_sentence_embedding(texts: List[str],
                          model: str = "sentence-transformers/all-MiniLM-L6-v2") -> List[List[float]]:
    """Embed sentences into vectors"""
    try:
        response = requests.post(
            f"{BASE_URL}/{model}",
            headers=HEADERS,
            json={"inputs": texts}
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        return [[0.0]]  # fallback


def semantic_search(query: str, corpus: Dict[str, str]) -> str:
    """Find most semantically similar entry in corpus"""
    texts = [query] + list(corpus.values())
    embeddings = hf_sentence_embedding(texts)
    if not embeddings or len(embeddings) < 2:
        return "[HF Error: embeddings]"
    query_emb = embeddings[0]
    scores = []
    for i, emb in enumerate(embeddings[1:], start=0):
        score = sum(q * d for q, d in zip(query_emb, emb))
        scores.append((list(corpus.keys())[i], score))
    scores.sort(key=lambda x: x[1], reverse=True)
    best_key, best_score = scores[0]
    return f"Closest match: {best_key} (score={best_score:.2f}) → {corpus[best_key]}"
