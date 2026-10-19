# Lab 07: How to check an LLM judge against your own labels

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/07_LLM_Judge_Calibration/07_llm_judge_calibration.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat, judge and embedding calls) | Practises: auditing claims and qualifiers, and
checking an LLM judge against human labels | In class**

## Goal

Label answers yourself, save the labels, then run an LLM judge with a frozen rubric on the same answers and count
where it agrees with you before you trust its scores. The moment that teaches it: of five answers from your pipeline,
you fail the two answered from the wrong product's policy, a Boreal question from the Atlas policy and an Atlas
question from the Boreal one. The judge passes both in 3 of 3 runs, quoting spans that really are in the evidence, so
the span check passes them too; a four-line check in code fails both. On twelve answers recorded from a small open
model the judge agrees with you on all 11 that are not refusals, yet its reason for failing the one that obeyed a
planted order never mentions the order.

## What you will learn

- How to write the claims a summary must keep before you read it, and label each one kept, weakened, changed or
  omitted.
- How to label answers for faithfulness yourself and save the labels before any judge runs.
- How to call a judge with a frozen rubric and a schema (`JUDGE_MODEL`), and check that its reply parsed.
- How to fill a 2×2 table of your labels against the judge's, and read each false pass against the evidence.
- How to check the judge's quoted spans in code, and what that check cannot see.
- Which rules belong in a code check rather than in a judge's rubric.

## Coming from

[Lab 06](../06_Evaluation_Set/README.md) scored the pipeline on a golden set with string checks: an accepted answer
inside the reply, or the refusal string. A string check cannot read a summary, or tell an answer that keeps its
source's conditions from one that drops them. An LLM judge can, and this lab measures how far to trust one, on answers
you have labelled yourself. The judge's model comes from LangChain; the 2×2 table and the code checks stay in plain
Python, so every count can be checked by hand.

## What you will do

**Part A: Audit a summary against the claims it must keep.** Four required claims written first, a summary from
`CHAT_MODEL`, a recorded summary from a small open model, and both labelled claim by claim.
**Part B: Label five answers and save the labels.** Five of Lab 06's policy cases through Lab 05's pipeline, your
faithfulness labels, and a file that shows they came before any verdict.
**Part C: Call the judge once and check its reply.** A frozen rubric, a Pydantic schema, one `JUDGE_MODEL` call and
its parsed reply.
**Part D: Judge the five and fill the 2×2 table.** The verdicts beside your labels, the table, each false pass read
against its evidence, and three runs.
**Part E: Check the judge's spans and the product in code.** The span check, what it cannot see, and a product check
that catches what the judge missed.
**Part F: Judge answers from another model.** The twelve recorded answers labelled and judged, the claims behind two
verdicts, and an ID check that finds the planted line.
**Part G: Keep the judge in `shared/judge.py`.** The module read and used.
**Going further.** The same evidence in another order.
**Challenge.** Write the product rule into the rubric. Optional: an order to the grader against a planted fact.

## Checkpoints

| After | You should see |
|---|---|
| Part A | your summary: supported 4 of 4, qualifiers kept 4 of 4, faithful 1; A03: supported 3 of 4, qualifiers kept 2 of 4, faithful 0 |
| Part B | five answers; G09 *Yes, password reset uses a mobile app.*; labels `['1', '1', '0', '0', 'NA']` saved in `.cache/labels-<hash>.json` |
| Part C | a 173-word rubric; G01 judged with one claim, S, from H02; F 1; 512 input tokens |
| Part D | the 2×2: you 1 row 2 and 0, you 0 row 2 and 0, 1 pair with an NA; G08 and G09 passed in 3 of 3 runs |
| Part E | every span found; `product_ok()` `False` for G08 and G09 only |
| Part F | the 2×2: 7 and 0, 0 and 4, 1 pair with an NA, `differ: {}`; A05's one claim is H02's sentence; `A05 evidence not in the index: ['H99']` |
| Part G | `same rubric: True`; G08 F 1 again |

The saved outputs come from `gpt-4o-mini` (answers and judge) and `text-embedding-3-small` at
`https://api.openai.com/v1` on 18 Oct 2026 (macOS, Python 3.13.5, a fresh `uv sync`; `langchain-core` 1.6.12,
`langchain-openai` 1.7.2). The same notebook ran on the `pip` route (Python 3.12.11) and with `gpt-6-luna` and
`REASONING_EFFORT=none`. Your labels are your own: label the answers in front of you, and compare which answers the
judge gets wrong, not our counts.

**On the other routes.** The pipeline's answer to G09 moves between runs. On the `pip` route, and in the worked
solution's run, it was *Insufficient evidence.*, so G09 was NA on both sides, and the 2×2 held one false pass (G08, 3
of 3 runs) instead of two. Everything else matched: the same summary labels, the twelve recorded answers 11 of 11 in
agreement, A05 judged on H02's sentence. With `gpt-6-luna` as answer model and judge the lab runs unchanged and the
false passes disappear for another reason: the pipeline refused G08, and answered G09 *No, Atlas passwords can be
reset by email.* We labelled that 1; the judge failed it in 3 of 3 runs because the evidence does not say the app is
not needed, a false fail you may well agree with. On the recorded answers it also agreed 11 of 11, and read them more
closely: it labelled A03's *typically* claim U, and gave A05 the claim the answer makes, *Atlas reset links remain
valid for 24 hours*, contradicted. In Going further one G09 call in the reversed order passed (1, 0, 0). Which errors
a judge makes depends on the model; the way to find them, your labels first, does not.

## Before you start

- **Lab 00 first,** then Lab 06: this lab uses `shared/evals.py` and Lab 05's `shared/lc_pipeline.py`.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `07_llm_judge_calibration.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Labels:** three cells start empty and stop the notebook until you fill them (two summaries, your five answers,
  the twelve recorded answers). Label from what your run printed.
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
