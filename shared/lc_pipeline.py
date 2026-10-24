"""The pipeline of Labs 01 and 02, rebuilt with LangChain in Lab 05: the baseline for the labs after it.

load_corpus() is Lab 04's corpus(): the 272 chunks and the five policies. to_documents() turns its records into
LangChain Documents. build_store() embeds them into an InMemoryVectorStore. retrieve() searches it, with k=3 by
default (the store's own default is 4). evidence() writes the hits as "[ID] text" lines, and answer() asks for a
CitedAnswer with Lab 02's rules, so shared.answer.quote_found() checks its claims unchanged.
"""
from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel

from shared.answer import RULES
from shared.settings import lc_chat_model, lc_embeddings
from shared.trace import corpus as load_corpus


class Claim(BaseModel):
    claim: str
    source_id: str
    quote: str


class CitedAnswer(BaseModel):
    answer: str
    claims: list[Claim]


def to_documents(records: list[dict]) -> list[Document]:
    """One Document per record: the text as page_content, the chunk ID as the ID, every other field as metadata."""
    return [Document(id=record["chunk_id"], page_content=record["text"],
                     metadata={name: value for name, value in record.items() if name != "text"})
            for record in records]


def build_store(documents: list[Document], settings: dict) -> InMemoryVectorStore:
    """An InMemoryVectorStore holding the documents, embedded with EMBED_MODEL through lc_embeddings()."""
    store = InMemoryVectorStore(lc_embeddings(settings))
    store.add_documents(documents)
    return store


def retrieve(store: InMemoryVectorStore, question: str, k: int = 3, filter=None) -> list:
    """The k closest documents, best first, as (Document, cosine score) pairs; filter is a function of a Document."""
    return store.similarity_search_with_score(question, k=k, filter=filter)


def evidence(hits: list) -> list[str]:
    """The hits as "[ID] text" lines, the format shared.answer.pack() writes and quote_found() reads."""
    return [f"[{document.id}] {document.page_content}" for document, _ in hits]


def answer(question: str, lines: list[str], settings: dict, rules: str = RULES) -> CitedAnswer:
    """One chat call through lc_chat_model() that returns a CitedAnswer (JSON_MODE picks a strict schema or JSON)."""
    method = "json_schema" if settings["JSON_MODE"] == "json_schema" else "json_mode"
    model = lc_chat_model("CHAT_MODEL", settings).with_structured_output(CitedAnswer, method=method)
    user = f"Question: {question}\nEvidence:\n" + "\n".join(lines)
    return model.invoke([("system", rules), ("user", user)])
