# Lab 04: How to find the stage where a RAG answer failed

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/04_Trace_And_Diagnose/04_trace_and_diagnose.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls) | Practises: tracing a RAG pipeline and locating the
failing stage with oracle context | In class**

## Goal

Follow every question through the pipeline, find the wrong answers, and name the stage where each one went wrong. The
moment that teaches it: the three wrong answers all failed at the same stage, generation, after the right passage had
been found and packed every time, so the fix is not in the search. Part E (self-study) then gives the model that
passage alone and tells the three causes apart: one answer misled by the other passages, one right in words the answer
key did not list, one the model still refuses.

## What you will learn

- What a trace record holds, and how to make one by hand.
- How to trace a question set and read its cost from the records.
- How to screen answers against expected strings, and read what the screen flags.
- How to build a stage table that follows the support through the hits, the pack and the answer.
- How to use oracle context to separate a failure caused by the other evidence from one caused by the answer step,
  and to spot an incomplete label.
- How to save a baseline run that later labs compare against.

**In class:** Parts A and D; run the cells of Parts B and C without stopping on them, because Part D reads what they
make. Parts B, C, E and F are self-study.

## Coming from

[Lab 02](../02_Cited_Answers/README.md) built the answer step and `shared/answer.py`: the model answers from packed
evidence with quoted citations, or says *Insufficient evidence.* This lab runs that pipeline on every question and
keeps a record of each step, so a wrong answer can be traced to the step that caused it. It stays in plain Python;
Lab 05 then rebuilds the same pipeline with LangChain.

## What you will do

**Part A: Trace one question by hand.** Search, pack, print the system and user messages, call the model and keep the raw reply, the
tokens billed and the stamp in one record.
**Part B: Trace every question (self-study).** The prompt hash, Part A's steps as a function, and a run on the 30
development questions and four Atlas questions written for this lab.
**Part C: Screen the answers (self-study).** An exact-match screen against the expected answer strings, and the
answers it flags.
**Part D: Find the stage where the support was lost.** The flagged answers read first, then a stage table that
follows the supporting passage through the corpus, the hits, the pack and the answer.
**Part E: Give the model only the labelled passage (self-study).** Oracle context on the three failures, and the
abstention line tested on its own.
**Part F: Save the baseline in `shared/trace.py` (self-study).** The trace step as a module, and the 30 development
questions traced once, saved and read back.
**Challenge.** Find the packed line that misled the model. Optional: name the stage in three traces written by hand.

## Checkpoints

| After | You should see |
|---|---|
| Part A | 5 hits and 4 packed lines for the Tesla question; a JSON reply; 500 input tokens billed |
| Part B | prompt hash `c2aa8c003db7`; 34 traces; 17,641 input tokens |
| Part C | passed 31 of 34; flagged: the complexity, temporary-accounts and mobile-app questions |
| Part D | support in the corpus, hits and pack 34 of 34; all three failures at `generation` |
| Part E | with oracle context: mobile app right 5 of 5; complexity right in other words 5 of 5, screen 0 of 5; temporary accounts still refused 5 of 5, answered 5 of 5 without the abstention line |
| Part F | 30 traces read back the same; `baseline-gpt-4o-mini-<hash>.json`; 29 of 30 pass the screen |

**About the saved outputs.** They come from `gpt-4o-mini` and `text-embedding-3-small`. Model replies vary from run to
run, so compare the pattern (which questions fail and at which stage), not every word. On another model the failures can move to other questions; the checks, and the way to measure them,
stay the same.

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
