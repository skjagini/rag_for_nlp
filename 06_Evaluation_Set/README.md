# Lab 06: How to build an evaluation set for RAG

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/06_Evaluation_Set/06_evaluation_set.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls) | Practises: writing a golden set and scoring
retrieval, answers and refusals per question | In class**

## Goal

Write a test set for your pipeline, score it one question at a time, and report each kind of question on its own. The
moment that teaches it: the standard word-overlap score gives half credit to a wrong number of minutes and little
credit to a right answer written as a full sentence. And the pipeline looks good on the whole set while failing some
of the policy questions written to test it, one of them answering a Boreal question from the Atlas policy.

Lab 04 named the failing stage for a few wrong answers, one trace at a time, and Lab 05 rebuilt the same pipeline
with LangChain. Before you change a stage, you need to measure every question the same way each time, so you can
see what a change fixes and what it breaks: this lab writes that question set and scores the pipeline on it.

## What you will learn

- How to compute hit@3 and mean reciprocal rank (MRR) with every question in the denominator.
- What exact match (EM) and token F1 measure, and where they mislead on generated answers.
- How to write golden records with a type, known support, and references or an expected refusal, and how to version
  a set with hashes.
- How to run a pipeline on a golden set and keep one row per question.
- How to score refusals on their own, and report each kind of question as its own slice.

**In class:** Parts A to D. Parts E and F are self-study.

## Coming from

[Lab 05](../05_Rebuild_With_LangChain/README.md) rebuilt the pipeline with LangChain and kept it in
`shared/lc_pipeline.py`, the baseline for the labs after it. Before you change that pipeline, you need questions to
measure every change on. This lab writes them, as a golden set kept outside the index, and scores the baseline on it.
The measures stay in plain Python, so every number can be checked by hand.

## What you will do

**Part A: Score a small example by hand.** hit@3 and MRR@3 on three ranks, then EM and token F1 with precision and
recall.
**Part B: Write the measures as functions.** `em_f1()` with several references and `first_rank()`, checked against
Part A, and four answers to predict.
**Part C: Write ten golden records.** One record by hand, seven from the development questions, three about the
policies (a paraphrase, an exception, a refusal), versioned with hashes and saved outside the index.
**Part D: Run the pipeline on the ten records.** One record by hand, then one row per record.
**Part E: Report each measure with its denominator (self-study).** hit@3, MRR@3, EM and F1, the rows behind them, a reference
check that fits generated sentences, and the refusals on their own lines.
**Part F: Score the course golden set by slice (self-study).** The reviewed policy cases, `shared/evals.py`, 38 records scored,
the slices, three runs of the policy cases, and the rows saved for later labs.
**Challenge.** Write records for a slice the set lacks. Optional: version a change to the set.

## Checkpoints

| After | You should see |
|---|---|
| Part A | MRR@3 0.444 (0.667 if the miss leaves the denominator); *about 30 minutes*: F1 0.800, EM 0 |
| Part B | the wrong *30 minutes* F1 0.500, the right Tesla sentence 0.222; the best of two references 0.800 |
| Part C | ten records, 1 to refuse; `.cache/golden-<hash>.json` |
| Part D | G02 retrieved H01, H04, H02 (rank 3), F1 0.333; ten rows, EM 1 only for *1884*, the weather question refused |
| Part E | hit@3 9 of 9 (1 not counted), MRR@3 0.870; EM 1 of 9, F1 0.371; `found()` 7 of 9; refused 1 of 1 and 0 of 9 |
| Part F | passed 34 of 38: development 29 of 30, policy cases 5 of 8; exception 0 of 1, wrong product 0 of 2, refusal slice 3 of 4; G05, G08, G09 failed 3 of 3 runs |

These counts compare versions of this pipeline on the same 38 records; they do not estimate how often it is right on
new questions (Part F says why).

The saved outputs come from `gpt-4o-mini` and `text-embedding-3-small` at `https://api.openai.com/v1` on 18 Oct 2026
(macOS, Python 3.13.5, a fresh `uv sync`; `langchain-core` 1.6.12, `langchain-openai` 1.7.2). The same notebook ran on
the `pip` route (Python 3.12.11) and with `gpt-6-luna` and `REASONING_EFFORT=none`. Model replies vary between runs:
compare which records pass and fail, not every word.

**On the other routes.** The `pip` run gave the same rows and counts: 34 of 38, the same four failures, G05, G08 and G09 failing in 3 of 3
runs, and the challenge's billing record 0 of 3 (one Norman answer was worded differently, so the F1 mean was
0.374). With `gpt-6-luna` the lab runs unchanged and the failures move. Every policy case passed in 3 of 3 runs:
the refusal slice was 4 of 4, G05 answered *No; temporary Atlas accounts require a support agent rather than the
usual email route.*, and the challenge's billing record was refused 3 of 3. The whole set passed 35 of 38, and
the three failures were development answers that `found()` missed although they were right or close: *otters,
beavers* where the reference says *otter, beaver*; curly quotation marks around *Norseman, Viking* (SQuAD's
normalisation removes only ASCII punctuation); and the complexity answer in other words (*the resources required
to solve it*). Every answer was a full sentence, *1884* included, so EM was 0 of 9 and the F1 mean 0.275. On that
model the string checks, not the pipeline, are what fail: read the failing rows before you count them.

## Before you start

- **Lab 00 first,** then Lab 05: this lab uses `shared/lc_pipeline.py` and Lab 02's `RULES`.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `06_evaluation_set.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** `data/development-questions.json`, `data/squad-chunks.json` and `data/support-case.json`, read in place.
  The eight policy cases are in `resources/golden-policy-cases.json`, constructed for these labs and reviewed (see
  Part F); they are not human annotations and not a benchmark.
- **Packages:** none beyond Lab 00's setup; the measures use only Python's standard library.
- **Cost:** one embedding of the corpus (about 21,000 tokens) and about 65 short chat calls, well under one US cent on
  `gpt-4o-mini`.

The notebook writes two files to `.cache/` (gitignored): your ten records and the golden set's rows, each named by a
hash of its records. It ends with a troubleshooting table: an F1 that divides by zero, chunk IDs compared with source
IDs, a reciprocal rank of a miss, and an average over refusal records.

## Resources

- [Rajpurkar et al., *SQuAD*, 2016](https://arxiv.org/abs/1606.05250): the dataset of the development questions, its
  several answers per question, and EM and F1
- [Gao et al., *Enabling Large Language Models to Generate Text with Citations* (ALCE), 2023](https://arxiv.org/abs/2305.14627):
  exact match recall, a short reference found inside a long generated answer
- [LangSmith: evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts): datasets, reference
  outputs, evaluators and experiments as a hosted service

**The same idea in three RAG books' code**

- [*Hands-On RAG for Production*, `chapter6/metrics.ipynb`](https://github.com/ofermend/hands-on-rag/blob/main/chapter6/metrics.ipynb): context precision and recall from retrieved and
  relevant chunk IDs, and BLEU and ROUGE-L for the answers, on three worked examples
- [*RAG from First Principles*, `09-Evaluation/`](https://github.com/PacktPublishing/RAG-from-First-Principles/tree/main/09-Evaluation): one script each for RAGAS, TruLens, DeepEval and LlamaIndex's
  evaluators
- [*Unlocking Data with Generative AI and RAG* (2nd ed.), `CHAPTER_09/`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/tree/main/CHAPTER_09): RAGAS generating a ten-question test set from a PDF, then
  scoring a vector-search chain against a hybrid one

Next: [Lab 07](../07_LLM_Judge_Calibration/README.md) (check an LLM judge against your own labels). Back to [the lab list](../README.md).
