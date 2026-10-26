# Lab 11: How to gate a RAG change before release

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/11_Release_Gate/11_release_gate.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls; the release tests need none) | Practises: access
filtering, versioned indexes and release tests run on saved results | In class**

## Goal

Stop a change before release with tests that read its saved results. The moment that teaches it: a change that lets
support agents use the assistant scores better than today's release, and the tests still fail it, because a note
customers may not read reached the evidence given to customers. Before that, an outdated policy left in the index
after an update keeps answering with the old rule.

Your pipeline now answers follow-ups, and you can measure any change question by question, but every user can read
everything in the index, and nothing stops a change that scores better while breaking a rule. This lab adds
documents customers may not read, and writes tests a change must pass before release.

## What you will learn

- How to label every document with who may read it, and list a user's allowed records by hand.
- How to filter by permission after the search and before it (a metadata filter on the store), and what each does
  to a customer's results.
- How to replace an outdated policy in the index by ID, and check that no old version survives.
- How to write a build manifest and save one row per question for a change.
- How to write release checks as pytest tests (access, versions, refusals, reviewed regressions, quotes), run them on
  saved results and read a failing assertion.
- Why an average cannot pass a hard boundary.

**In class:** Parts A and B. Parts C to F are self-study.

## Coming from

[Lab 05](../05_Rebuild_With_LangChain/README.md) left documents out with a metadata filter, and
[Lab 06](../06_Evaluation_Set/README.md) wrote the golden set and its refusal slice. [Lab 02](../02_Cited_Answers/README.md)
showed that every answer check trusts the evidence, so the defence is control over what enters the index and who may
read it. This lab adds documents that not every user may read and an outdated policy, then writes the checks a change
must pass before it ships, as tests a CI job can run with no key.

## What you will do

**Part A: Label who may read each document.** Four restricted records, an access label on all 281 records, and each
user's allowed records counted by hand.
**Part B: Filter by permission after the search, then before it.** A customer's question answered from the runbook,
a post-filter that leaves no evidence, a pre-filter on the store, and the same runbook question asked by an agent and
by a customer.
**Part C: Replace an outdated policy in the index (self-study).** April's H02 left beside today's, the answers it gives, a version
check, `delete` and `get_by_ids`, and an add with the same ID.
**Part D: Save the rows of a change, with its manifest (self-study).** 42 release questions, one row each, a build manifest, and
today's release and the change saved as results files and compared question by question.
**Part E: Write the release tests and run them with pytest (self-study).** The access check by hand, then `test_release.py`: the
first test, the failure read, the other four tests, and all five on the change.
**Part F: Fix the change and pass the gate (self-study).** The allow-list filter, the gate passing, and the paired comparison.
**Going further.** LangChain's indexing API removing the old version on update.
**Challenge.** Ship the fixed change with April's H02 left in. Optional: check the citations instead of the packs.

## Checkpoints

| After | You should see |
|---|---|
| Part A | 281 records: 277 `public`, 3 `staff`, 1 `internal`; a customer may read 277, an agent 281 |
| Part B | no filter: R01, R04, R02 and *Yes* with the runbook's condition, 3 of 3; post-filter: no hits; pre-filter: H02, H01, H04 and *Insufficient evidence.* 3 of 3; the runbook question: *4 hours* for the agent, *... last for 30 minutes.* for the customer, 3 of 3 each |
| Part C | both versions: *Insufficient evidence.* and *24 hours*, 3 of 3 each; `outdated: ['H02-2026-04-01']`, then `[]`; *30 minutes* 3 of 3 |
| Part D | today's release 35 of 42, the change 37; gained A01, A02, A03; lost G02 |
| Part E | R04 in six customer packs; `1 failed, 4 deselected`, then `2 failed, 3 passed` (access and regressions) |
| Part F | fixed change 38 of 42; `5 passed`; gained A01 to A03, lost none |

The saved outputs come from `gpt-4o-mini` and `text-embedding-3-small` at `https://api.openai.com/v1` on 24 Oct 2026
(macOS, Python 3.13.5, a fresh `uv sync`; `langchain-core` 1.6.12, `langchain-openai` 1.7.2, `pytest` 9.1.1). The same day
the notebook ran on the `pip` route (Python 3.12.11) and with `gpt-6-luna` and `REASONING_EFFORT=none`. Model replies vary
between runs: compare which tests fail and which questions moved, not every word.

**Across our runs.** Nine runs of the release set on `gpt-4o-mini`, the saved run among them. The baseline passed 35 of 42 in eight runs and 34
in one, the change 37 in all nine, the fixed change 38 in all nine. R04 reached the same six customer packs and G02 was the change's one loss every time, because
the search, not the model, puts R04 in those packs.

**On the other routes.** The `pip` run gave the same numbers in every part. With `gpt-6-luna` the lab runs unchanged and the gate
catches the change, but the totals move between runs: on 24 Oct the baseline passed 35 of 42, the change 35 and the
fixed change 37 (18 Oct: 35, 37 and 39). The fixed change still failed one test: it lost two questions the baseline
passed (a complexity question answered in other words, and G09, now refused), and nobody had reviewed them, which is
what that test is for. Four things differ from `gpt-4o-mini`. Answering from the unfiltered runbook, it repeated the
incident note too (*support should check the request time in the admin console*), 3 of 3. Asked the runbook question as
a customer, it did not credit agents with anything (*An Atlas reset link lasts 30 minutes.*, or *Insufficient
evidence.* once). With both versions of H02 in the evidence it answered *24 hours* to both questions, 6 of 6, instead of
refusing the first. And R04 changed one of the change's answers: A04, the customer question that should be refused,
became *Yes, support can send you a new Atlas reset link.*, so the gate failed three tests, the refusal slice as well.
The gate caught the change on both models.

## Before you start

- **Lab 00 first,** then Labs 05 and 06: this lab uses `shared/lc_pipeline.py` (Lab 05), `shared/evals.py` (Lab 06)
  and `paired()` from `shared/hybrid.py` (Lab 08).
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `11_release_gate.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** `data/squad-chunks.json`, `data/support-case.json` and Lab 06's golden cases, read in place, plus
  `resources/constructed-records.json`, written for this lab (who may read what, four restricted records, the April
  version of H02 and four release cases). They are not real policies; the files in `data/` are unchanged.
- **Files:** `test_release.py` holds the release tests; the notebook writes `results/baseline.json`,
  `results/candidate.json` and `results/fixed.json`, and the saved ones come from our run.
- **Cost:** 148 short chat calls and one embedding of the corpus on `gpt-4o-mini`; task 1 of the challenge adds 42.

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
- [OWASP: LLM02 sensitive information disclosure](https://genai.owasp.org/llmrisk/llm022025-sensitive-information-disclosure/):
  why access control belongs before the model, not in the prompt

Next: [the capstone](../Capstone_RAG_Service/README.md), *How to ship a RAG service with its release gate*. Back to [the lab list](../README.md).
