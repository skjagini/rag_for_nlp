"""Cited answers from retrieved evidence: the steps Lab 02 builds by hand, kept for the labs after it.

pack() turns search hits into "[ID] text" evidence lines inside a token budget. answer() asks CHAT_MODEL for a JSON
answer whose claims each name an evidence line and quote it, checks the reply's shape and retries once.
quote_found() checks one quote against the evidence line it names (a quote cut with "..." passes when its parts
are there, in order).
"""
from __future__ import annotations

import json
import re

import tiktoken

from shared.settings import chat_client, chat_options

RULES = (
    "Answer the question in one sentence, using only the evidence."
    " List each claim the answer makes, with the ID of the evidence line it comes from and a quote: words copied"
    " exactly from that line."
    ' If the evidence does not answer the question, the answer is "Insufficient evidence." and the claims list is empty.'
    ' Reply in JSON shaped like {"answer": "...", "claims": [{"claim": "...", "source_id": "...", "quote": "..."}]}.'
)

CLAIM = {
    "type": "object",
    "properties": {"claim": {"type": "string"}, "source_id": {"type": "string"}, "quote": {"type": "string"}},
    "required": ["claim", "source_id", "quote"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "claims": {"type": "array", "items": CLAIM}},
    "required": ["answer", "claims"],
    "additionalProperties": False,
}


def pack(hits: list, settings: dict, budget: int = 400) -> list[str]:
    """Evidence lines "[ID] text" from (score, chunk) hits, best first, stopping before the token budget is passed.

    A chunk with a "header" (a context header, Lab 03) has it packed in front of its text.
    """
    encoding = tiktoken.get_encoding(settings["TOKEN_ENCODING"])
    lines, used = [], 0
    for _, chunk in hits:
        line = f"[{chunk['chunk_id']}] {chunk.get('header', '')}{chunk['text']}"
        tokens = len(encoding.encode(line))
        if used + tokens > budget:
            break
        lines.append(line)
        used += tokens
    return lines


def response_format(settings: dict) -> dict:
    """A strict JSON schema, or plain JSON for servers without strict schemas (JSON_MODE in models.env)."""
    if settings["JSON_MODE"] == "json_schema":
        return {"type": "json_schema", "json_schema": {"name": "cited_answer", "strict": True, "schema": SCHEMA}}
    return {"type": "json_object"}


def parse(text: str) -> dict:
    """The reply as {"answer": str, "claims": [{"claim", "source_id", "quote"}]}, or a ValueError."""
    data = json.loads(text)
    ok = (isinstance(data, dict) and isinstance(data.get("answer"), str) and isinstance(data.get("claims"), list)
          and all(isinstance(c, dict) and all(isinstance(c.get(k), str) for k in CLAIM["required"])
                  for c in data["claims"]))
    if not ok:
        raise ValueError("the reply does not have the answer's shape: " + text[:200])
    return data


def answer(question: str, lines: list[str], settings: dict, rules: str = RULES) -> dict:
    """One chat call for a cited answer; a second call if the first reply does not parse."""
    user = f"Question: {question}\nEvidence:\n" + "\n".join(lines)
    messages = [{"role": "system", "content": rules}, {"role": "user", "content": user}]
    for attempt in (1, 2):
        reply = chat_client(settings).chat.completions.create(
            **chat_options("CHAT_MODEL", settings), response_format=response_format(settings), messages=messages)
        try:
            return parse(reply.choices[0].message.content)
        except ValueError:                           # bad JSON (JSONDecodeError is a ValueError) or the wrong shape
            if attempt == 2:
                raise


def words(text: str) -> str:
    """The text as lowercase words, without punctuation or extra spaces."""
    return " ".join(re.findall(r"\w+", text.lower()))


def quote_found(claim: dict, lines: list[str]) -> bool:
    """True when the claim's quote, word for word, is in the evidence line its source_id names.

    A quote cut with "..." passes when each part is in that line, in order.
    """
    evidence = {line[1:line.index("]")]: line[line.index("]") + 2:] for line in lines}
    if claim["source_id"] not in evidence:
        return False
    text, start = words(evidence[claim["source_id"]]), 0
    parts = [words(part) for part in re.split(r"\.\.\.|…", claim["quote"]) if words(part)]
    for part in parts:
        start = text.find(part, start)
        if start < 0:
            return False
        start += len(part)
    return bool(parts)
