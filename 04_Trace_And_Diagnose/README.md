# Lab 04: How to find the stage where a RAG answer failed

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/04_Trace_And_Diagnose/04_trace_and_diagnose.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls) | Practises: tracing a RAG pipeline and locating the
failing stage with oracle context | In class**

## Goal

Follow every question through the pipeline, find the wrong answers, and name the step where each one went wrong. The
moment that teaches it: the wrong answers all fail at the same step, after the right passage was found and given to
the model. Giving the model that passage alone shows three different causes: one answer was misled by the other
passages, one was right all along in words the answer key did not list, and one the model still refuses. One failing
step, three different repairs.

## What you will learn

- What a trace record holds, and how to make one for a question by hand.
- How to trace a whole question set and count what it cost.
- How to screen answers against expected answer strings, and read the answers the screen flags.
- How to build a stage table that follows the supporting passage through the hits, the pack and the answer.
- How to use oracle context to tell a failure caused by the other evidence from one caused by the answer step, and
  how an incomplete label turns a right answer into a "miss".
- How to save a baseline run that later labs compare against.

**In class:** Parts A and D. Parts B, C, E and F are self-study. Run the cells of Parts B and C in class without
stopping on them, because Part D reads the traces and screen results they make.

## Coming from

[Lab 02](../02_Cited_Answers/README.md) built the answer step and `shared/answer.py`: the model answers from packed
evidence with quoted citations, or says *Insufficient evidence.* This lab runs that pipeline on every question and
keeps a record of each step, so a wrong answer can be traced to the step that caused it. It stays in plain Python;
Lab 05 then rebuilds the same pipeline with LangChain.

## What you will do

**Part A: Trace one question by hand.** Search, pack, hash the prompt, call the model and keep the raw reply, the
tokens billed and the stamp in one record.
**Part B: Trace every question (self-study).** The same steps as a function, run on the 30 development questions and four Atlas
questions written for this lab.
**Part C: Screen the answers (self-study).** A whole-word check against the expected answer strings, and the three answers it flags.
**Part D: Find the stage where the support was lost.** A stage table that follows the supporting passage through the
corpus, the hits, the pack and the answer.
**Part E: Give the model only the labelled passage (self-study).** Oracle context on the three failures, and the abstention line
tested on its own.
**Part F: Save the baseline in `shared/trace.py` (self-study).** The trace step as a module, and the 30 development questions traced
once, saved and read back.
**Challenge.** Find the packed line that misled the model. Optional: name the stage in three traces written by hand.

## Checkpoints

| After | You should see |
|---|---|
| Part A | 5 hits and 4 packed lines for the Tesla question; a JSON reply; 500 input tokens billed |
| Part B | 34 traces; 17,641 input tokens |
| Part C | passed 31 of 34; flagged: the complexity, temporary-accounts and mobile-app questions |
| Part D | support in the corpus, hits and pack 34 of 34; all three failures at `generation` |
| Part E | with oracle context: mobile app right 5 of 5; complexity right in other words, screen 0 of 5; temporary accounts still refused 5 of 5, answered 5 of 5 without the abstention line |
| Part F | 30 traces read back the same; `baseline-gpt-4o-mini-<hash>.json`; 29 of 30 pass the screen |

The saved outputs come from `gpt-4o-mini` and `text-embedding-3-small` at `https://api.openai.com/v1` on 18 Oct 2026
(macOS, Python 3.13.5, a fresh `uv sync`, `openai` 3.35.1). The same notebook ran on the `pip` route (Python 3.12.11)
and with `gpt-6-luna` and `REASONING_EFFORT=none`. Model replies vary between runs: compare which questions fail and
at which stage, not every word.

**On the other routes.** The `pip` run gave the same three failures, stages and oracle results. With `gpt-6-luna` the
lab runs unchanged and fails elsewhere: it answered the temporary-accounts and mobile-app questions correctly, so the
screen flagged only two of 34, and both answers were right in other words. The complexity question got *the amount of
resources needed to solve it* even with the full pack, and the Vistula question got *otters, beavers* where the label
says *otter, beaver*. On that model, Part E's oracle shows two incomplete labels and no pipeline failure, and the
baseline passes 28 of 30. Which failures you see depends on the model; the way to find their stage does not.

## Before you start

- **Lab 00 first,** then Labs 01 and 02: this lab uses `shared/search.py`, `shared/answer.py` and the cached chunk
  vectors.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `04_trace_and_diagnose.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** `data/squad-chunks.json`, `data/squad-passages.json`, `data/support-case.json` and
  `data/development-questions.json`, read in place. The four Atlas questions and their expected answers are written in
  the notebook; the data files are not changed.
- **Cost:** about 90 short chat calls (120 with the challenge) and a few hundred embeddings, well under one US cent on
  `gpt-4o-mini`.
- **Saved files:** `shared/trace.py` writes the baseline to `.cache/` (not committed), one file per model and endpoint.

The notebook ends with a troubleshooting table: a stage table that compares chunk IDs with paragraph IDs, a chunk ID
looked up as a passage, and a trace whose reply is not JSON.

## Resources

- [OpenAI: create chat completion](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create):
  `usage`, the token counts a trace records
- [Python: `hashlib`](https://docs.python.org/3/library/hashlib.html): SHA-256 and `hexdigest()`
- [OpenTelemetry: traces](https://opentelemetry.io/docs/concepts/signals/traces/): how tracing tools record each step
  of a request as a span, the production form of these records
- [Joren et al., *Sufficient Context*, ICLR 2025](https://arxiv.org/abs/2411.06037): separating "the context was not
  enough" from "the model failed with enough context"

Next: [Lab 05](../05_Rebuild_With_LangChain/README.md), *How to rebuild a RAG pipeline with LangChain*. Back to [the lab list](../README.md).
