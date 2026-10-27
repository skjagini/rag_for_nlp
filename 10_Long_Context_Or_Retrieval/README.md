# Lab 10: How to choose between long context and retrieval

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/10_Long_Context_Or_Retrieval/10_long_context_or_retrieval.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls) | Practises: choosing between long context and
retrieval from measured answers, tokens, cost and seconds | Self-study**

## Goal

Decide between putting the whole corpus in the prompt and retrieving a few chunks, from answers, tokens, cost and
time measured on the same questions. The moment that teaches it: the whole corpus fits in one prompt, yet it answers
fewer questions right than retrieval, at many times the cost. The model answers the questions it missed when it gets
the right passage alone, but not when that passage sits inside the whole corpus.

Every lab so far has paid for retrieval: an index to build and keep current, a search before every answer, and in
Lab 08 a second index that added work for no gain. This corpus is small enough to fit in one prompt, so this lab
asks what retrieval buys here, in answers, tokens and dollars.

## What you will learn

- How to answer the same questions three ways (no evidence, three retrieved chunks, the whole corpus) and record the
  tokens and seconds of every answer.
- How to compare the three routes with one answer check, and read every answer it flags.
- How to price each route per 1,000 questions, and what a cached prefix changes (an OpenAI-only feature).
- How an answer changes when the passage that holds it gets more text around it, or moves.
- How to write a decision rule from your own table, and see where the corpus stops fitting in the prompt.

## Coming from

[Lab 05](../05_Rebuild_With_LangChain/README.md) built `shared/lc_pipeline.py`, the pipeline the labs after it
measure against, and [Lab 04](../04_Trace_And_Diagnose/README.md) the answer screen and oracle context. This corpus
is small enough to fit in one prompt, so this lab asks whether the pipeline needs its retrieval step at all, and
answers from measurements: the same questions with no evidence, with three retrieved chunks and with the whole corpus.

## What you will do

**Part A: Answer one question three ways.** The corpus as 125 evidence lines, Lab 05's store, an answer function that
keeps the tokens billed, and the three routes on a Tesla question and an Atlas question.
**Part B: Answer every question three ways.** 34 questions, one screen for all three routes, every flagged answer
printed, and the citations checked.
**Part C: Price each route.** Tokens, dollars per 1,000 questions and seconds beside the screen count; then a cached
prefix (OpenAI only), and what moving the corpus first did to the answers.
**Part D: Give the passage more text, or move it.** The questions the whole corpus lost, asked with their passage
alone, inside its article, inside the corpus, and moved first, to the middle and last.
**Part E: Write the decision rule.** The cost as the corpus grows, the context window limit, a rule written from the
table, and what the notebook cost.
**Going further.** The whole corpus without the abstention line.
**Challenge.** Move a passage the model already answers from. Optional: price a workload.

## Checkpoints

| After | You should see |
|---|---|
| Part A | 125 lines, 19,466 tokens; Tesla 1856 from all three routes; Atlas *24 hours* with no evidence, *30 minutes* from the other two |
| Part B | screen: no evidence 19, retrieval 31, whole corpus 23 of 34; refused 0, 1 and 6; claims 34 (34 quotes found) and 29 (28 found) |
| Part C | 0.02, 0.11 and 2.99 dollars per 1,000 questions; 489 against 19,700 input tokens; corpus first: 607,104 of 669,798 tokens cached, 1.65 dollars, 23 passed |
| Part D | eight questions lost; passage alone 24 of 24, its article 21, whole corpus 0; moved first 24, to the middle 11, last 18 |
| Part E | 97,564 tokens a question at five times the corpus; seven copies (136,262 tokens) refused at 128,000; the notebook used 3,315,159 input tokens, 0.36 dollars |

**About the saved outputs.** They come from `gpt-4o-mini` and `text-embedding-3-small`. Model replies vary from run to
run, so compare the pattern (the counts and which answers fail), not every word. Never take one run's seconds as
the latency. Over six runs, retrieval passed 31 of 34 every time, the whole corpus 23 to 25 and no evidence 18 or 19,
and each notebook run cost about 0.30 to 0.36 US dollars. OpenAI keeps a prompt cache for a while, so a run within the
hour after another reports far more cached tokens. The dollars printed use the prices in `models.env`: change the
`CHAT_PRICE_*` lines when you change the model.

## Before you start

- **Lab 00 first,** then Labs 02, 04 and 05: this lab uses `shared/lc_pipeline.py` (Lab 05), `shared/answer.py` (Lab 02)
  and the screen from Lab 04.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `10_long_context_or_retrieval.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** `data/squad-passages.json`, `data/squad-chunks.json`, `data/support-case.json` and
  `data/development-questions.json`, read in place. The four Atlas questions are written in the notebook, as in Lab 04.
- **Cost:** ⚠️ more than the other labs. More than 150 calls send the whole corpus (about 20,000 tokens each); our run
  used 3,315,159 input tokens, about 0.36 US dollars on `gpt-4o-mini`, plus about 21,000 embedding tokens.
  Task 1 of the challenge adds 15 whole-corpus calls. The cost table reads the prices in `models.env`; change them with the model.

The notebook ends with a troubleshooting table: a prompt longer than the context window, the dictionary that
`include_raw=True` returns, and an empty price line.

## Resources

- [OpenAI: prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching): which prompts are cached,
  the minimum prefix length for each model, and `cached_tokens` in `usage`
- [OpenAI: pricing](https://developers.openai.com/api/docs/pricing): input, cached input and output prices per model
- [LangChain: models, structured output](https://docs.langchain.com/oss/python/langchain/models#structured-output):
  `include_raw=True` and the token counts in `usage_metadata`
- [Hong, Troynikov and Huber, *Context Rot*, Chroma, 2025](https://www.trychroma.com/research/context-rot): 18 models
  answering worse as the input grows, and how related text distracts them
- [Liu et al., *Lost in the Middle*, TACL 2023](https://arxiv.org/abs/2307.03172): how answers depended on where the
  relevant passage sat in a long prompt, on the models of 2023

Next: [Lab 11](../11_Release_Gate/README.md), *How to gate a RAG change before release*. Back to [the lab list](../README.md).
