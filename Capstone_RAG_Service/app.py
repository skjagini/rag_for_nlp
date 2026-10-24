"""The RAG service: the pipeline of Labs 05 and 11 behind a sign-in page (Gradio).

The capstone notebook builds and tests each function here first. Run it from the repository folder:

    python Capstone_RAG_Service/app.py

It prints a new password for each demo sign-in, then serves the page at http://127.0.0.1:7860. Every request
appends one trace record to .cache/capstone/traces.jsonl (gitignored).
"""
import hashlib
import json
import secrets
import sys
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
from langchain_core.callbacks import get_usage_metadata_callback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))             # so that "shared" is found from any folder

from shared import lc_pipeline  # noqa: E402
from shared.answer import RULES, quote_found  # noqa: E402
from shared.evals import REFUSAL  # noqa: E402
from shared.settings import load_key, load_settings, stamp  # noqa: E402

settings = load_settings()
if not load_key("OPENAI_API_KEY"):
    sys.exit("OPENAI_API_KEY not found: put it in .env, or set it in your shell or as a Codespaces secret.")

CONSTRUCTED = json.loads((ROOT / "11_Release_Gate" / "resources" / "constructed-records.json").read_text(encoding="utf-8"))
USERS = CONSTRUCTED["users"]                             # which access labels each kind of user may read
SIGN_INS = {"casey": "customer", "alex": "agent"}       # demo sign-ins; a real service uses your company's sign-in
TRACES = ROOT / ".cache" / "capstone" / "traces.jsonl"   # gitignored: traces hold what users asked
HELD = {"answer": "This answer could not be checked against its sources, so it is not shown. Read the sources instead.",
        "claims": []}


def build_store():
    """Every record with its access label (public unless it has one), embedded fresh."""
    records = [{"access": "public", **record} for record in lc_pipeline.load_corpus()] + CONSTRUCTED["restricted"]
    return lc_pipeline.build_store(lc_pipeline.to_documents(records), settings)


def can_read(user):
    """A filter for one user: True for a document whose access label the user may read."""
    return lambda document: document.metadata["access"] in USERS[user]


def answer_and_trace(question, user, hits):
    """Answer from the hits, check the quotes, and append the request's trace record to TRACES."""
    lines = lc_pipeline.evidence(hits)
    start = time.perf_counter()
    with get_usage_metadata_callback() as usage:
        reply = lc_pipeline.answer(question, lines, settings)
    claims = [claim.model_dump() for claim in reply.claims]
    refused = reply.answer.strip() == REFUSAL
    tokens = usage.usage_metadata.values()
    record = {
        "time": datetime.now().isoformat(timespec="seconds"), "user": user, "question": question,
        "hits": [[document.id, round(score, 4)] for document, score in hits],
        "packed": [document.id for document, _ in hits], "evidence": lines,
        "prompt_sha": hashlib.sha256((RULES + question + "".join(lines)).encode()).hexdigest()[:12],
        "reply": {"answer": reply.answer, "claims": claims},
        "checked": refused or (bool(claims) and all(quote_found(claim, lines) for claim in claims)),
        "usage": {"input": sum(t["input_tokens"] for t in tokens), "output": sum(t["output_tokens"] for t in tokens)},
        "seconds": round(time.perf_counter() - start, 2), "stamp": stamp(settings),
    }
    TRACES.parent.mkdir(parents=True, exist_ok=True)
    with TRACES.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def shown(record):
    """The record as the page shows it: the reply only when its quotes passed the check, HELD otherwise."""
    return record if record["checked"] else {**record, "reply": HELD}


def sources_text(hits):
    """The hits as the sources panel shows them: ID, title, access label, score and text."""
    return "\n\n".join(f"**[{document.id}]** {document.metadata.get('title', '')} · {document.metadata['access']} · "
                       f"{score:.3f}\n\n{document.page_content}" for document, score in hits)


def answer_text(record):
    """The answer with its quotes."""
    quotes = [f"- [{claim['source_id']}] *{claim['quote']}*" for claim in record["reply"]["claims"]]
    return "\n\n".join([record["reply"]["answer"], *quotes])


store = build_store()                     # the service's index, built when it starts


def respond(question, request: gr.Request):
    """The page's function: the user comes from the sign-in, never from the visitor's input."""
    user = SIGN_INS[request.username]
    yield "*Searching...*", "", None
    hits = lc_pipeline.retrieve(store, question, filter=can_read(user))
    yield f"*Answering from {len(hits)} sources...*", sources_text(hits), None
    record = shown(answer_and_trace(question, user, hits))
    yield answer_text(record), sources_text(hits), record


def build_page():
    return gr.Interface(
        respond, gr.Textbox(label="Question"),
        [gr.Markdown(label="Answer"), gr.Markdown(label="Sources"), gr.JSON(label="Trace")],
        title="Ask with sources", flagging_mode="never",
        description="Answers quote the documents you may read, or say *Insufficient evidence.*",
        examples=["How long does an Atlas reset link remain valid?",
                  "How long can a support agent make an Atlas reset link last?",
                  "In what year was Nikola Tesla born?"])


if __name__ == "__main__":
    passwords = {login: secrets.token_urlsafe(8) for login in SIGN_INS}
    for login, password in passwords.items():
        print(f"Sign in as {login} ({SIGN_INS[login]}) with the password {password}")
    build_page().launch(auth=list(passwords.items()))
