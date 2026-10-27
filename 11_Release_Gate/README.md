# Lab 11: How to gate a RAG change before release

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/11_Release_Gate/11_release_gate.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls; the release tests need none) | Practises: access
filtering, versioned indexes and release tests run on saved results | In class**

## Goal

Keep the documents a user may not read out of that user's evidence, and check the filter by reading the replies, not
only the hits. The moment that teaches it: with no filter, a customer's question about reset links finds three
documents customers may not read, and every answer repeats the runbook's rule. Filtered inside the search, the
customer gets only public policies; yet asked how long a support agent can make a reset link last, the model answers
with Atlas's public 30 minutes, in our runs often as if a support agent sets it. **The hits were right, and the reply
was still wrong.**

Parts C to F (self-study) stop a change before release with tests that read its saved results: the change passes more
questions than today's release, and the tests still fail it, because a note customers may not read reached their
evidence. No reply repeated the note, so only a check on the evidence catches it.

Your pipeline now answers follow-ups, and you can measure any change question by question, but every user can read
everything in the index, and nothing stops a change that scores better while breaking a rule. This lab adds
documents customers may not read, and writes the tests a change must pass before release.

## What you will learn

- How to label every document with who may read it, and list what each user may read by hand.
- How to make one filter per user with a function that returns a function, and use it after the search and inside it.
- Why a permission filter is checked by reading the replies, not only the hits.
- How to find an outdated version of a policy in the index and remove it by its ID.
- How to save what a change did, one row per question, with a record of what produced the run.
- How to write release checks as pytest tests, run them on saved results with no key, read a failing assertion, and
  say why a better average cannot pass a broken rule.

**In class:** Parts A and B. Parts C to F are self-study.

## Coming from

[Lab 06](../06_Evaluation_Set/README.md) wrote the golden set with its refusal questions, and its Part D shows the
pipeline this lab runs (`shared/lc_pipeline.py`: `build_store()`, `retrieve()`, `evidence()`, `answer()`).
[Lab 02](../02_Cited_Answers/README.md) showed that every answer check trusts the evidence, so the defence is control
over what enters the index and who may read it. This lab adds documents that not every user may read and an outdated
policy, then writes the checks a change must pass before it ships, as tests a CI job can run with no key.

## What you will do

**Part A: Label who may read each document.** Four restricted records and the allow-list of what each user may read,
an access label on all 281 records, each user's allowed records counted by hand, and the label in the store's metadata.
**Part B: Filter by permission after the search, then inside it.** A customer's question answered from the runbook;
`can_read(user)`, a function that returns each user's filter (a closure), built and called step by step; a post-filter
that leaves no evidence and its count on the 38 golden questions; a pre-filter inside the store's search; and the same
runbook question asked by an agent and by a customer.
**Part C: Replace an outdated policy in the index (self-study).** April's H02 left beside today's, the answers it
gives, a version check written as a plain loop, `delete` and `get_by_ids`, and an add with the same ID.
**Part D: Save the rows of a change, with its manifest (self-study).** 42 release questions, one row each, a build
manifest, a deny-list filter beside the allow-list, and today's release and the change saved as results files and
compared question by question.
**Part E: Write the release tests and run them with pytest (self-study).** The access check by hand, then
`test_release.py`: the first test, `gate()` running pytest in a new process, the other four tests, the quotes-found rate
as a formula and computed by hand, and all five on the change.
**Part F: Fix the change and pass the gate (self-study).** The allow-list filter, the gate passing, and the paired
comparison.
**Going further.** LangChain's indexing API removing the old version on update (incremental indexing).
**Challenge.** Ship the fixed change with April's H02 left in. Optional: check the citations instead of the packs.

## Checkpoints

| After | You should see |
|---|---|
| Part A | 281 records: 277 `public`, 3 `staff`, 1 `internal`; a customer may read 277, an agent 281; R04's metadata holds `access: internal` |
| Part B | no filter: R01, R04, R02 and *Yes* with the runbook's condition, 3 of 3; the filters print `customer: False True` and `agent: True True`; post-filter: no hits, *Insufficient evidence.*, and three hits left for 31 of the 38 golden questions; pre-filter: H02, H01, H04 and *Insufficient evidence.* 3 of 3; the runbook question: *4 hours* for the agent 3 of 3, *... last for 30 minutes.* for the customer (3 of 3 in the saved run, 1 to 3 of 3 across our runs) |
| Part C | both versions: *Insufficient evidence.* and *24 hours*, 3 of 3 each; `outdated: ['H02-2026-04-01']`, then `[]`; *30 minutes* 3 of 3 |
| Part D | today's release 35 of 42, the change 37; gained A01, A02, A03; lost G02 |
| Part E | R04 in six customer packs; `1 failed, 4 deselected`; quotes found 37 of 37 (rate 1.000, threshold 0.95); then `2 failed, 3 passed` (access and regressions) |
| Part F | fixed change 38 of 42; `5 passed`; gained A01 to A03, lost none |

**About the saved outputs.** They come from `gpt-4o-mini` and `text-embedding-3-small`. Model replies vary from run to
run, so compare the pattern (which tests fail and which questions moved), not every word. Over eleven runs the
baseline passed 34 or 35 of 42, the change 37 and the fixed change 38 every time, and R04 reached the same six customer
packs, because the search, not the model, puts it there. The customer's answer to the runbook question credited agents
with the 30 minutes in 3 of 3 in most runs and 1 of 3 in one.

## Before you start

- **Lab 00 first,** then Labs 05 and 06: this lab uses `shared/lc_pipeline.py` (Lab 05, shown in Lab 06 Part D),
  `shared/evals.py` (Lab 06) and `paired()` from `shared/hybrid.py` (Lab 08).
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `11_release_gate.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** `data/squad-chunks.json`, `data/support-case.json` and Lab 06's golden cases, read in place, plus
  `resources/constructed-records.json`, written for this lab (who may read what, four restricted records, the April
  version of H02 and four release cases). They are not real policies; the files in `data/` are unchanged.
- **Files:** `test_release.py` holds the release tests; the notebook writes `results/baseline.json`,
  `results/candidate.json` and `results/fixed.json`, and the saved ones come from our run.
- **Cost:** 148 short chat calls and one embedding of the corpus on `gpt-4o-mini` (13 of the calls in Parts A and B);
  task 1 of the challenge adds 42.

**Run the gate without the notebook.** From the repository folder, on the saved results, with no key:

```bash
python -m pytest 11_Release_Gate/test_release.py -q                     # baseline and candidate: 2 failed
GATE_CANDIDATE=11_Release_Gate/results/fixed.json python -m pytest 11_Release_Gate/test_release.py -q   # 5 passed
```

`GATE_BASELINE` and `GATE_CANDIDATE` name any two results files with the same shape: a `manifest` with a
`documents` list (`id`, `source_id`, `version`, `access`) and `rows` with `question_id`, `user`, `refuse`, `packed`,
`answer`, `passed`, `claims` and `evidence`.

The notebook ends with a troubleshooting table: a store built without access labels, a results file pytest cannot
find, and pytest started from the wrong folder.

## Resources

- [LangChain: vector stores](https://docs.langchain.com/oss/python/integrations/vectorstores): `add_documents` with
  IDs, `delete`, `get_by_ids` and metadata filters
- [LangChain: indexing API](https://reference.langchain.com/python/langchain-core/indexing/api/index): the record
  manager and `cleanup` modes that remove old versions
- [pytest: get started](https://docs.pytest.org/en/stable/getting-started.html) and
  [fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html): test discovery, `assert` messages and fixtures
- [Python: subprocess.run](https://docs.python.org/3/library/subprocess.html#subprocess.run): how `gate()` starts
  pytest in a new process
- [OWASP: LLM02 sensitive information disclosure](https://genai.owasp.org/llmrisk/llm022025-sensitive-information-disclosure/):
  why access control belongs before the model, not in the prompt

Next: [the capstone](../Capstone_RAG_Service/README.md), *How to ship a RAG service with its release gate*. Back to [the lab list](../README.md).
