"""Trace records for the pipeline: the steps Lab 04 builds by hand, kept for the labs after it.

trace() runs one question through search, packing and the answer call, and returns what each step produced:
the hits, the packed IDs, a hash of the prompt, the raw reply, the tokens billed and the stamp. baseline() traces
the 30 development questions once and saves them in .cache/, in a file named after the chat model and a hash of
the settings, the rules and the questions, so a changed model or endpoint never reuses old traces.
"""
from __future__ import annotations

import hashlib
import json

from shared.answer import RULES, pack, response_format
from shared.search import chunk_vectors, search
from shared.settings import DATA, ROOT, chat_client, chat_options, stamp


def corpus() -> list[dict]:
    """The 272 chunks plus the five policies, each policy as one chunk whose ID is its source ID."""
    chunks = json.loads((DATA / "squad-chunks.json").read_text(encoding="utf-8"))
    policies = json.loads((DATA / "support-case.json").read_text(encoding="utf-8"))
    return chunks + [{**policy, "chunk_id": policy["source_id"]} for policy in policies]


def trace(question: str, chunks: list[dict], vectors, settings: dict, k: int = 5) -> dict:
    """Search, pack and answer one question, and record what each step produced. The reply is kept unparsed."""
    hits = search(question, chunks, vectors, settings, k=k)
    lines = pack(hits, settings)
    user = f"Question: {question}\nEvidence:\n" + "\n".join(lines)
    messages = [{"role": "system", "content": RULES}, {"role": "user", "content": user}]
    reply = chat_client(settings).chat.completions.create(
        **chat_options("CHAT_MODEL", settings), response_format=response_format(settings), messages=messages)
    return {
        "question": question,
        "hits": [[chunk["chunk_id"], round(score, 4)] for score, chunk in hits],
        "packed": [line[1:line.index("]")] for line in lines],
        "prompt_sha": hashlib.sha256(json.dumps(messages).encode()).hexdigest()[:12],
        "reply": reply.choices[0].message.content,
        "usage": {"input": reply.usage.prompt_tokens, "output": reply.usage.completion_tokens},
        "stamp": stamp(settings),
    }


def baseline(settings: dict) -> list[dict]:
    """Traces of the 30 development questions: run the first time, read from .cache/ after that."""
    questions = json.loads((DATA / "development-questions.json").read_text(encoding="utf-8"))
    run = {name: value for name, value in stamp(settings).items() if name != "run_date"}
    key = json.dumps([run, settings.get("EMBED_DIMENSIONS"), settings["JSON_MODE"], RULES,
                      [question["question"] for question in questions]])
    name = settings["CHAT_MODEL"].replace("/", "_")
    path = ROOT / ".cache" / f"baseline-{name}-{hashlib.sha256(key.encode()).hexdigest()[:12]}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    chunks = corpus()
    vectors = chunk_vectors(chunks, settings)
    traces = [trace(question["question"], chunks, vectors, settings) for question in questions]
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(traces, indent=1, ensure_ascii=False), encoding="utf-8")
    return traces
