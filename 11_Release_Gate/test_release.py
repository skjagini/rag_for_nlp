"""Release tests for a RAG change. They read saved results only: no model call, no API key.

    pytest 11_Release_Gate/test_release.py -q
    GATE_CANDIDATE=path/to/candidate.json pytest 11_Release_Gate/test_release.py -q

A results file holds the build manifest ("manifest") and one row per question ("rows"); the Lab 11 notebook
writes them. The baseline is the release that runs today, the candidate the change. Every test must pass.
"""
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from shared.answer import quote_found  # noqa: E402

BASELINE = Path(os.environ.get("GATE_BASELINE", HERE / "results" / "baseline.json"))
CANDIDATE = Path(os.environ.get("GATE_CANDIDATE", HERE / "results" / "candidate.json"))

RECORDS = json.loads((HERE / "resources" / "constructed-records.json").read_text(encoding="utf-8"))
POLICIES = json.loads((ROOT / "data" / "support-case.json").read_text(encoding="utf-8"))
USERS = {user: set(levels) for user, levels in RECORDS["users"].items()}     # what each user may read
CURRENT = {doc["source_id"]: doc["version"] for doc in POLICIES + RECORDS["restricted"]}
MIN_QUOTES_FOUND = 0.95
REVIEWED = {}   # question_id: why losing it is accepted, written after reading its rows


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def baseline():
    return load(BASELINE)


@pytest.fixture(scope="module")
def candidate():
    return load(CANDIDATE)


def test_no_restricted_source_in_pack(candidate):
    access = {doc["id"]: doc["access"] for doc in candidate["manifest"]["documents"]}
    leaks = [f"{row['question_id']} ({row['user']}) read {doc_id} ({access[doc_id]})"
             for row in candidate["rows"] for doc_id in row["packed"] if access[doc_id] not in USERS[row["user"]]]
    assert not leaks, f"{len(leaks)} packed documents the user may not read: " + "; ".join(leaks)


def test_only_current_versions_in_index(candidate):
    stale = [f"{doc['id']} ({doc['source_id']} {doc['version']}, current {CURRENT[doc['source_id']]})"
             for doc in candidate["manifest"]["documents"]
             if doc["version"] and doc["version"] != CURRENT[doc["source_id"]]]
    assert not stale, "outdated versions in the index: " + "; ".join(stale)


def test_refusal_slice_held(baseline, candidate):
    held = {row["question_id"] for row in baseline["rows"] if row["refuse"] and row["passed"]}
    broken = [f"{row['question_id']}: {row['answer']}" for row in candidate["rows"]
              if row["question_id"] in held and not row["passed"]]
    assert not broken, "refusals the baseline held, now answered: " + "; ".join(broken)


def test_regressions_reviewed(baseline, candidate):
    passed_before = {row["question_id"] for row in baseline["rows"] if row["passed"]}
    lost = [f"{row['question_id']}: {row['answer']}" for row in candidate["rows"]
            if row["question_id"] in passed_before and not row["passed"] and row["question_id"] not in REVIEWED]
    assert not lost, "passed in the baseline, fail now, not reviewed: " + "; ".join(lost)


def test_quotes_found(candidate):
    claims = [(claim, row["evidence"]) for row in candidate["rows"] for claim in row["claims"]]
    found = sum(quote_found(claim, evidence) for claim, evidence in claims)
    assert found >= MIN_QUOTES_FOUND * len(claims), f"{found} of {len(claims)} quotes found"
