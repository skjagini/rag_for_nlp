"""Golden records and the measures that score a RAG pipeline on them: the steps Lab 06 builds by hand.

em_f1() and found() compare an answer with its reference answers; first_rank() finds the first retrieved source a
record names as support. record_from_dev() turns a development question into a golden record, and golden_set() reads
the whole development golden set: the 30 development questions and the reviewed policy cases in Lab 06's resources/.
score() runs one record through shared.lc_pipeline and returns its row; version() names the set, the rules and the
data a run used.
"""
from __future__ import annotations

import hashlib
import json
import re
import string
from collections import Counter

from shared.answer import RULES
from shared.lc_pipeline import answer, evidence, retrieve
from shared.settings import DATA, ROOT, stamp

REFUSAL = "Insufficient evidence."
POLICY_CASES = ROOT / "06_Evaluation_Set" / "resources" / "golden-policy-cases.json"


def normalize(text: str) -> str:
    """SQuAD's normalisation: lower case, no punctuation, no articles (a, an, the), single spaces."""
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def em_f1(prediction: str, references: list[str]) -> tuple[int, float]:
    """Exact match and token F1 against the closest reference: the best of each over all references."""
    best_em, best_f1 = 0, 0.0
    for reference in references:
        a, b = normalize(prediction).split(), normalize(reference).split()
        overlap = sum((Counter(a) & Counter(b)).values())
        f1 = 0.0
        if overlap:
            precision, recall = overlap / len(a), overlap / len(b)
            f1 = 2 * precision * recall / (precision + recall)
        best_em, best_f1 = max(best_em, int(a == b)), max(best_f1, f1)
    return best_em, best_f1


def found(prediction: str, references: list[str]) -> bool:
    """True when a whole reference, normalised, appears in the normalised answer."""
    answer_text = f" {normalize(prediction)} "
    return any(f" {normalize(reference)} " in answer_text for reference in references)


def first_rank(sources: list[str], support: list[str]) -> int | None:
    """The rank (1 = first) of the first retrieved source in support, or None when none is."""
    return next((rank for rank, source in enumerate(sources, 1) if source in support), None)


def record_from_dev(question: dict) -> dict:
    """A golden record from one development question: its text, its accepted answers and its known support."""
    return {"question_id": question["question_id"], "split": "development", "question": question["question"],
            "type": "answerable", "known_support": question["known_support"],
            "references": sorted({a["text"] for a in question["answers"]}), "refuse": False}


def golden_set() -> list[dict]:
    """The 30 development questions as records, then the reviewed policy cases (G01 to G09)."""
    questions = json.loads((DATA / "development-questions.json").read_text(encoding="utf-8"))
    cases = json.loads(POLICY_CASES.read_text(encoding="utf-8"))["cases"]
    return [record_from_dev(question) for question in questions] + cases


def score(record: dict, store, settings: dict) -> dict:
    """Run one record through the pipeline (or its fixed pack) and return its row."""
    if "pack" in record:
        hits = [(document, None) for document in store.get_by_ids(record["pack"])]
    else:
        hits = retrieve(store, record["question"])
    lines = evidence(hits)
    reply = answer(record["question"], lines, settings)
    row = {"question_id": record["question_id"], "type": record["type"], "refuse": record["refuse"],
           "retrieved": [document.id for document, _ in hits],
           "rank": first_rank([document.metadata["source_id"] for document, _ in hits], record["known_support"]),
           "answer": reply.answer, "refused": reply.answer.strip() == REFUSAL,
           "claims": [claim.model_dump() for claim in reply.claims], "evidence": lines}
    if record["refuse"]:
        row["passed"] = row["refused"]
    else:
        row["em"], row["f1"] = em_f1(reply.answer, record["references"])
        row["passed"] = found(reply.answer, record["references"])
    return row


def version(records: list[dict], settings: dict) -> dict:
    """Short hashes of the records, the answer rules and the data files, with the models and endpoints."""
    def sha(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    data = "".join((DATA / name).read_text(encoding="utf-8") for name in ("squad-chunks.json", "support-case.json"))
    return {"records": sha(json.dumps(records, sort_keys=True)), "rules": sha(RULES), "data": sha(data),
            **stamp(settings)}
