"""Search the chunks by embedding: the steps Lab 01 builds by hand, kept for the labs after it.

chunk_vectors() embeds the chunks once and saves them in .cache/, in a file named after the embedding model and a
hash of the endpoint, the model, EMBED_DIMENSIONS and every chunk text, so a changed setting or chunk never reuses
old vectors. search() ranks every chunk for one question; hit_at_k() checks a whole question set.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from shared.settings import ROOT, embed_client, embed_url, stamp


def embed(texts: list[str], settings: dict, batch_size: int = 100) -> np.ndarray:
    """One vector per text, in input order, from EMBED_MODEL (shortened to EMBED_DIMENSIONS when it is set)."""
    client = embed_client(settings)
    options = {"model": settings["EMBED_MODEL"]}
    if settings.get("EMBED_DIMENSIONS"):
        options["dimensions"] = int(settings["EMBED_DIMENSIONS"])
    rows = []
    for start in range(0, len(texts), batch_size):
        response = client.embeddings.create(input=texts[start:start + batch_size], **options)
        rows += [item.embedding for item in response.data]
    return np.array(rows)


def chunk_vectors(chunks: list[dict], settings: dict) -> np.ndarray:
    """The chunks' vectors, row i for chunks[i]: embedded the first time, read from .cache/ after that.

    A chunk with a "header" (a context header, Lab 03) is embedded with the header in front of its text.
    """
    texts = [chunk.get("header", "") + chunk["text"] for chunk in chunks]
    key = json.dumps([embed_url(settings), settings["EMBED_MODEL"], settings.get("EMBED_DIMENSIONS"), texts])
    name = settings["EMBED_MODEL"].replace("/", "_")
    path = ROOT / ".cache" / f"chunks-{name}-{hashlib.sha256(key.encode()).hexdigest()[:12]}.npz"
    if path.exists():
        return np.load(path)["vectors"]
    vectors = embed(texts, settings)
    path.parent.mkdir(exist_ok=True)
    np.savez(path, vectors=vectors, chunk_ids=[chunk["chunk_id"] for chunk in chunks],
             stamp=json.dumps(stamp(settings)))
    return vectors


def search(question: str, chunks: list[dict], vectors: np.ndarray, settings: dict, k: int = 3) -> list:
    """The k chunks closest to the question, best first, as (score, chunk) pairs."""
    scores = vectors @ embed([question], settings)[0]
    return [(float(scores[i]), chunks[i]) for i in np.argsort(-scores, kind="stable")[:k]]


def hit_at_k(questions: list[dict], chunks: list[dict], vectors: np.ndarray, settings: dict, k: int = 3) -> list:
    """For each question, True when one of its top k chunks comes from a paragraph in its known_support."""
    scores = embed([question["question"] for question in questions], settings) @ vectors.T
    hits = []
    for question, row in zip(questions, scores):
        top = np.argsort(-row, kind="stable")[:k]
        hits.append(any(chunks[i]["source_id"] in question["known_support"] for i in top))
    return hits
