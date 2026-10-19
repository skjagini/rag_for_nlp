# Lab 01: How to search documents with embeddings

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/01_Search_With_Embeddings/01_search_with_embeddings.ipynb)

**CPU | API key: `OPENAI_API_KEY` (embedding calls only, no chat model) | Practises: retrieving evidence by
meaning and measuring retrieval | In class**

## Goal

Rank a corpus for a question through an embeddings API, measure how often the right paragraph comes back, and know
what a similarity score cannot tell you. The moment that teaches it: ten questions labelled answerable or not before
any search. *Who sang the national anthem at Super Bowl 50?*, which no paragraph answers, scores 0.6554, higher than
three of the five questions the corpus does answer, so no score cutoff can separate the two kinds.

## What you will learn

- How to build an embeddings client, embed a text and read the vector that comes back.
- Why cosine similarity is a plain dot product for the unit-length vectors the API returns.
- How to embed 272 chunks in batches and keep every row tied to its chunk ID.
- How to rank every chunk for 30 questions with one matrix product and count known-source hit@3.
- How to test, on questions labelled before the search, whether a score cutoff tells answerable questions from
  unanswerable ones.
- How to search the corpus with `shared/search.py`, which keeps these steps for the labs after this one.

## Coming from

You have called a language model and you know what a held-out set is; Lab 00 checked that your setup reaches the
embeddings endpoint. You have not yet built the search that finds evidence for a question, or measured one. This lab
builds it in plain Python (the OpenAI SDK and NumPy), so every step is visible before later labs hand parts of it to a
library.

## What you will do

**Part A: Embed one question.** Build the client by hand, embed one question and read the vector, then switch to
`embed_client()` after reading its source.
**Part B: Measure closeness with cosine similarity.** Dot product and cosine on two-number vectors, then on three real
sentences, and why the dot product is enough for unit-length vectors.
**Part C: Search the five product policies.** Rank five Atlas and Boreal policies for an Atlas question, and see a
policy for the wrong product come second.
**Part D: Embed the corpus in batches.** 272 chunks in batches of 100, with the IDs kept in the same order as the rows.
**Part E: Rank the corpus for 30 questions and count hit@3.** One matrix product for all questions, hit@3 and hit@1,
and why the one hit@1 miss happened.
**Part F: Test whether a score cutoff separates answerable questions.** A cutoff chosen from two good examples, then
ten questions labelled before the search and five cutoffs counted.
**Part G: Keep the search in `shared/search.py`.** The same steps as functions, with a cache, used from here on.
**Going further.** Shorter vectors with `EMBED_DIMENSIONS` (OpenAI only), measured on hit@3 and hit@1.
**Challenge.** Test the cutoff on four questions of your own. Optional: reword the Atlas question and see where the
answering policy lands.

## Checkpoints

| After | You should see |
|---|---|
| Part A | `text-embedding-3-small`, 1,536 numbers, 8 tokens billed |
| Part B | toy cosines A 1.0, B 0.7071; real sentences 1–2 0.7307, 1–3 0.3137, 2–3 0.2729 |
| Part C | H02 0.6970, H04 (Boreal) 0.4882, H01 0.4254, H05 0.2401, H03 0.1775 |
| Part D | three batches, 21,539 tokens, `D: (272, 1536)` |
| Part E | Tesla's top chunk `S03-P00-C00` (0.7192); hit@3 30/30; hit@1 29/30, missing *Who was the Super Bowl 50 MVP?* |
| Part F | anthem question 0.6555, above the 0.60 cutoff; lowest answerable 0.5700, highest unanswerable 0.6554; 0.55 makes one mistake on the ten and refuses 3 of the 30 development questions |
| Part G | the module's vectors within a few thousandths of `D`, hit@3 30 of 30 |
| Going further | 256 numbers: hit@3 30 of 30, hit@1 27 of 30 |

The saved outputs come from `text-embedding-3-small` at `https://api.openai.com/v1` on 18 Oct 2026 (macOS, Python
3.13.5, a fresh `uv sync`). The same notebook gave the same counts with `pip` on Python 3.12.14 and with
`CHAT_MODEL`, `JUDGE_MODEL` and `RERANK_MODEL` set to `gpt-6-luna` and `REASONING_EFFORT=none` (this lab calls no
chat model, so only the stamp changed). Scores can move by a few thousandths between runs (up to 0.0044 in ours);
check the order and the counts.

## Before you start

- **Lab 00 first.** It checks that your key reaches the embeddings endpoint, on your route.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `01_search_with_embeddings.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** `data/support-case.json`, `data/squad-chunks.json`, `data/development-questions.json`, read in place. The
  chunk vectors are saved in `.cache/` in the repository folder, which git ignores.
- **Cost:** about 70,000 embedding tokens in all, well under one US cent. No chat model.

The notebook ends with a troubleshooting table: no key, vectors of two different lengths, a batch that is too large,
and rows out of step with their IDs.

## Resources

- [OpenAI: embeddings guide](https://developers.openai.com/api/docs/guides/embeddings): the models, vector lengths,
  normalisation and the `dimensions` parameter
- [OpenAI: create embeddings](https://developers.openai.com/api/reference/resources/embeddings/methods/create): the
  request, its limits and the `usage` field
- [NumPy: `argsort`](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html): sorting order and
  `kind="stable"`
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard): how embedding models compare on public tasks;
  test a candidate on your own questions before you switch

Next: [Lab 03](../03_Cited_Answers/README.md) (cited answers), or the self-study [Lab 02](../02_Chunking_With_Lineage/README.md) (chunking). Back to [the lab list](../README.md).
