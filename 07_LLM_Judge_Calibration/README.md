# Lab 07: How to check an LLM judge against your own labels

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/07_LLM_Judge_Calibration/07_llm_judge_calibration.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat, judge and embedding calls) | Practises: auditing claims and qualifiers, and
checking an LLM judge against human labels | In class**

## Goal

Label answers yourself, then check an LLM judge against your labels before you trust its scores. The moment that
teaches it: of five answers from your pipeline, two were answered from the wrong product's policy and you label both
unfaithful; the judge passes both, marking each claim supported with a span copied from the evidence, and a four-line check
in code fails them. A judge's verdict is a model reply: measure it against your own labels before you use it.

Part F (self-study) gives the judge twelve answers from a small open model: it agrees with every one of your verdicts
there, yet fails one answer for a reason that has nothing to do with what the answer did.

## What you will learn

- How to write the claims a summary must keep before you read it, and label each one kept, weakened, changed or
  omitted.
- How to label answers for faithfulness and save the labels before any judge runs.
- How to call an LLM judge with a frozen rubric and a reply schema (`JUDGE_MODEL`), and read what each part of the
  rubric asks.
- How to put the judge's verdicts beside your labels in a confusion matrix, compute agreement and the false-pass rate,
  and read each false pass against its evidence.
- How to check the judge's quoted spans in code, and what that check cannot see.
- Which rules belong in a code check rather than in a judge's rubric.

**In class:** Parts B to E, without the two self-study subsections at the end of Part D. Part B's labels are filled in
for you: read each one against your own answers and change any that does not fit. Run every cell from the top, Part
A's included (it needs no input), and read Part A later. Parts A, F and G are self-study, and Part F's twelve
labels are the only ones you type: the notebook stops at that cell until you fill them.

## Coming from

[Lab 06](../06_Evaluation_Set/README.md) scored the pipeline on a golden set with string checks: an accepted answer
inside the reply, or the refusal string. A string check cannot read a summary, or tell an answer that keeps its
source's conditions from one that drops them. An LLM judge can, and this lab measures how far to trust one, on answers
you have labelled. The answers come from Lab 05's pipeline in `shared/lc_pipeline.py` (Lab 06 Part D shows its four
functions; Part B here names them again). The judge's model comes from LangChain; the confusion matrix and the code
checks stay in plain Python, so every count can be checked by hand.

## What you will do

**Part A: Audit a summary against the claims it must keep (self-study).** Four required claims written first, a
summary from `CHAT_MODEL`, a recorded summary from a small open model, and both labelled claim by claim.
**Part B: Label five answers and save the labels.** The pipeline's four functions named, one golden record and one
row of `evals.score()` read field by field, five of Lab 06's policy cases answered, your faithfulness labels (filled in
for you to check), and a file that shows they came before any verdict.
**Part C: Call the judge once and check its reply.** A frozen rubric read term by term, a Pydantic schema, the judge's
user message printed, and one `JUDGE_MODEL` call with its parsed reply.
**Part D: Judge the five and compare the verdicts with your labels.** The verdicts beside your labels, the confusion
matrix, agreement and the false-pass rate as formulas, and each false pass read against its evidence. Two self-study
subsections: the five judged twice more, and how many labels a rate needs.
**Part E: Check the judge's spans and the product in code.** The span check, what it cannot see, and a product check
that catches what the judge missed.
**Part F: Judge answers from another model (self-study).** The twelve recorded answers labelled by you and judged, the
claims behind two verdicts, and an ID check that finds the planted line.
**Part G: Keep the judge in `shared/judge.py` (self-study).** The module read and used.
**Going further.** The same evidence in another order.
**Challenge.** Write the product rule into the rubric. Optional: an order to the grader against a planted fact.

## Checkpoints

| After | You should see |
|---|---|
| Part A | your summary: supported 4 of 4, qualifiers kept 4 of 4, faithful 1; A03: supported 3 of 4, qualifiers kept 2 of 4, faithful 0 |
| Part B | 277 documents; five answers; G08 *Only workspace owners can export all reports in Boreal.*; G09 *Yes, password reset uses a mobile app.*; labels `['1', '1', '0', '0', 'NA']` saved in `.cache/labels-<hash>.json` |
| Part C | a 173-word rubric; G01 judged with one claim, S, from H02; F 1; 512 input tokens |
| Part D | the matrix: you 1 row 2 and 0, you 0 row 2 and 0, 1 pair with an NA; agreement 2 of 4, false-pass rate 2 of 2; G08 and G09 passed in 3 of 3 runs (self-study) |
| Part E | every span found; `product_ok()` `False` for G08 and G09 only |
| Part F | the matrix: 7 and 0, 0 and 4, 1 pair with an NA, `differ: {}`; A05's one claim is H02's sentence; `A05 evidence not in the index: ['H99']` |
| Part G | `same rubric: True`; G08 F 1 again |

**About the saved outputs.** They come from `gpt-4o-mini` (answers and judge) and `text-embedding-3-small`. Your
labels are your own: read the answers in front of you, and compare which answers the judge gets wrong, not our counts.
The pipeline's answer to G09 moves between runs: about one run in ten it is *Insufficient evidence.*, G09 is then NA on
both sides, and the matrix holds one false pass (G08) instead of two. Which errors a judge makes depends on the model;
the way to find them, your labels first, does not.

## Before you start

- **Lab 00 first,** then Lab 06: this lab uses `shared/evals.py` and Lab 05's `shared/lc_pipeline.py`.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `07_llm_judge_calibration.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Labels:** Parts A and B hold our labels, filled in; read them against what your run printed and change any that
  does not fit. Part F's label cell starts empty and stops the notebook until you label the twelve recorded answers.
- **Data:** `data/support-case.json` and the corpus, read in place. `resources/recorded-answers.json` holds 12 answers
  recorded from Qwen2.5-0.5B-Instruct (revision 7ae5576, CPU, greedy decoding) on 5 October 2026, each with the
  evidence it was given: recorded outputs, unchanged, with no labels. The SQuAD passages in their evidence are
  CC BY-SA 4.0 (see `data/README.md`).
- **Cost:** one embedding of the corpus (about 21,000 tokens) and about 60 short chat calls, well under one US cent on
  `gpt-4o-mini`.

The notebook writes your labels to `.cache/` (gitignored). It ends with a troubleshooting table: an empty label cell,
`json_object` with a rubric that never says JSON, a reasoning model as judge without `REASONING_EFFORT=none`, and the
evidence passed to `span_found()` as one string.

## Resources

- [Zheng et al., *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*, 2023](https://arxiv.org/abs/2306.05685):
  position, verbosity and self-preference bias in model judges, and agreement with human labels
- [OpenAI: evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices):
  model graders and how to check them against human judgement
- [LangSmith: evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts): LLM-as-judge evaluators
  and human review as a hosted service
- [Es et al., *RAGAS*, 2023](https://arxiv.org/abs/2309.15217): faithfulness as the share of an answer's claims the
  context supports

Next: [Lab 08](../08_Hybrid_Search_And_Reranking/README.md) (improve retrieval with hybrid search and reranking). Back
to [the lab list](../README.md).
