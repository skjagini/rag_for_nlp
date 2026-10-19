"""Hybrid search, reranking and paired comparisons: the steps Lab 08 builds by hand, kept for the labs after it.

bm25_index() and bm25_search() rank records by the words they share with a question (BM25, through bm25s, set up
like the hand version: every word, no stop-word list, k1=1.2, b=0.75). rrf() fuses ranked ID lists by reciprocal
rank. rerank() asks RERANK_MODEL to order the candidates and keeps the search order unless the reply holds exactly the
candidate IDs. paired() and sign_test() compare a change with the baseline question by question.
"""
from __future__ import annotations

import time
from math import comb

import bm25s
from pydantic import BaseModel

from shared.settings import lc_chat_model

WORDS = r"(?u)\b\w+\b"      # every word, one letter or more, as terms() in Lab 08

RERANK_RULES = (
    "Rank the passages by how directly they answer the question."
    ' Reply in JSON shaped like {"ranking": ["ID", ...]}, with every passage ID once, best first.'
)


class Ranking(BaseModel):
    ranking: list[str]


def bm25_index(records: list[dict], k1: float = 1.2, b: float = 0.75) -> bm25s.BM25:
    """A BM25 index of the records' texts that returns their chunk IDs."""
    index = bm25s.BM25(k1=k1, b=b, corpus=[record["chunk_id"] for record in records])
    tokens = bm25s.tokenize([record["text"] for record in records], token_pattern=WORDS, stopwords=None,
                            show_progress=False)
    index.index(tokens, show_progress=False)
    return index


def bm25_search(index: bm25s.BM25, question: str, k: int = 10) -> list[str]:
    """The chunk IDs of the k records with the highest BM25 scores, best first."""
    tokens = bm25s.tokenize([question], token_pattern=WORDS, stopwords=None, show_progress=False)
    ids, _ = index.retrieve(tokens, k=k, show_progress=False)
    return [str(chunk_id) for chunk_id in ids[0]]


def rrf(lists: list[list[str]], c: int = 60) -> list[str]:
    """The IDs of every list, ordered by their summed 1 / (c + rank); an ID missing from a list adds nothing."""
    scores = {}
    for ranked in lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (c + rank)
    return sorted(scores, key=lambda chunk_id: -scores[chunk_id])


def rerank(question: str, hits: list, settings: dict) -> dict:
    """One RERANK_MODEL call that orders the (Document, score) hits; the search order is kept if the reply is not
    exactly the candidate IDs. Returns the order used, the reply's order, whether it was accepted, tokens and seconds."""
    candidates = [document.id for document, _ in hits]
    passages = "\n".join(f"[{document.id}] {document.page_content}" for document, _ in hits)
    method = "json_schema" if settings["JSON_MODE"] == "json_schema" else "json_mode"
    model = lc_chat_model("RERANK_MODEL", settings).with_structured_output(Ranking, method=method, include_raw=True)
    start = time.perf_counter()
    reply = model.invoke([("system", RERANK_RULES), ("user", f"Question: {question}\nPassages:\n{passages}")])
    seconds = time.perf_counter() - start
    returned = reply["parsed"].ranking if reply["parsed"] else []
    accepted = sorted(returned) == sorted(candidates)
    usage = reply["raw"].usage_metadata
    return {"order": returned if accepted else candidates, "returned": returned, "accepted": accepted,
            "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"], "seconds": seconds}


def paired(before: list[bool], after: list[bool]) -> dict:
    """Per-question pass/fail before and after a change: how many held or stayed failed, and which positions moved."""
    pairs = list(zip(before, after))
    return {"both": sum(b and a for b, a in pairs), "neither": sum(not b and not a for b, a in pairs),
            "gained": [i for i, (b, a) in enumerate(pairs) if a and not b],
            "lost": [i for i, (b, a) in enumerate(pairs) if b and not a]}


def sign_test(gains: int, losses: int) -> float:
    """Two-sided sign test: the chance of a split at least this uneven if each changed question were a coin flip."""
    n = gains + losses
    tail = sum(comb(n, i) for i in range(min(gains, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)
