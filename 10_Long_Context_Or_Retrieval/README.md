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

The saved outputs come from `gpt-4o-mini` and `text-embedding-3-small` at `https://api.openai.com/v1` on 24 Oct 2026
(macOS, Python 3.13.5, a fresh `uv sync`; `langchain-core` 1.6.12, `langchain-openai` 1.7.2). The same day the notebook ran on
the `pip` route (Python 3.12.11) and with `gpt-6-luna` and `REASONING_EFFORT=none`. Model replies vary between runs:
compare the counts and which answers fail, not every word, and never one run's seconds as the latency.

**Across our runs.** Six runs on `gpt-4o-mini`: four of the whole notebook, the saved run among them, and two of Part B
alone. Retrieval passed 31 of 34 in all six; the whole corpus passed 23 to 25, and the no-evidence
route 18 or 19. With the corpus first, 21 to 23 passed. In Part D the passage alone and the passage moved first passed
in every run; inside the whole corpus, in its own place, it passed in 0 to 2 runs per notebook run, and the middle (6
to 12) and last (12 to 18) moved between runs. Median seconds: retrieval 1.22 to 1.43, whole corpus 1.38 to 1.70.
Each notebook run cost 0.29 to 0.36 dollars. On 24 Oct 2026 three more runs (the saved one, the solved notebook and
pip) gave the whole corpus 23 of 34 each and lost eight questions in Part D instead of six; run within the hour after
another, they cached far more of the question-first prompts (470,016 and 432,896 tokens against 19,584).

**On the other routes.** The `pip` run gave the same pattern: 19, 31 and 23 of 34; eight questions lost; in Part D the passage alone 24 of 24,
inside the whole corpus 1 of 24, moved first 24 of 24. With `gpt-6-luna` the lab runs unchanged, and four things
differ. It answers better from the whole corpus: 28 of 34 against 32 for retrieval (16 with no evidence), 2 refused;
four questions were lost, and in Part D the whole corpus passed 1 of 12 runs, moved last 12 of 12, first 8 of 12, the
middle 8 of 12. It cached nothing with the corpus first (0 of 669,696 input tokens), so that order saved nothing (30
of 34 passed). It answered the seven copies (136,262 tokens) instead of raising the error: its window is larger. And
the dollars it prints are `gpt-4o-mini`'s prices from `models.env`; `gpt-6-luna` is listed at 0.10 and 0.50 dollars
per million input and output tokens, so set the `CHAT_PRICE_*` lines when you switch. In Going further, without the
abstention line, it passed 10 of 12 runs and still declined the Moon question.

## Before you start

- **Lab 00 first,** then Labs 02, 04 and 05: this lab uses `shared/lc_pipeline.py` (Lab 05), `shared/answer.py` (Lab 02)
  and the screen from Lab 04.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `10_long_context_or_retrieval.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** `data/squad-passages.json`, `data/squad-chunks.json`, `data/support-case.json` and
  `data/development-questions.json`, read in place. The four Atlas questions are written in the notebook, as in Lab 04.
- **Cost:** ⚠️ more than the other labs. More than 150 calls send the whole corpus (about 20,000 tokens each); our run
  used 2,836,809 input tokens, about 0.31 US dollars on `gpt-4o-mini`, plus about 21,000 embedding tokens.
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

**The same idea in three RAG books' code**

- [*Hands-On RAG for Production*, `chapter4/caching.ipynb`](https://github.com/ofermend/hands-on-rag/blob/main/chapter4/caching.ipynb): exact-match and semantic caching of search results
  in Redis, and a reworded question that only the semantic cache recognises
- [*Unlocking Data with Generative AI and RAG* (2nd ed.), `CHAPTER_15/`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/tree/main/CHAPTER_15): a semantic cache that masks amounts, years and tickers before matching,
  and checks a match with a cross-encoder before reusing it

Next: [Lab 11](../11_Release_Gate/README.md), *How to gate a RAG change before release*. Back to [the lab list](../README.md).
