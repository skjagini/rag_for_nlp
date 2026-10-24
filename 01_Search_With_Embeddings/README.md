# Lab 01: How to search documents with embeddings

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/01_Search_With_Embeddings/01_search_with_embeddings.ipynb)

**CPU | API key: `OPENAI_API_KEY` (embedding calls only, no chat model) | Practises: retrieving evidence by
meaning and measuring retrieval | In class**

## Goal

Search a set of documents for a question with an embeddings API, check how often the right paragraph comes back, and
learn what a similarity score cannot tell you. The moment that teaches it: a question that no paragraph answers
scores higher than questions the paragraphs do answer, so no score cutoff can tell the two kinds apart.

Lab 00 checked that your setup reaches the API. Before a model can answer from your documents, something has to
find the passages that hold the answer, and you need a way to tell whether it found them: this lab builds that
search and measures it.

## What you will learn

- How to build an embeddings client, embed a text and read the vector that comes back.
- Why cosine similarity is a plain dot product for the unit-length vectors the API returns.
- How to embed 272 chunks in batches and keep every row tied to its chunk ID.
- How to rank every chunk for 30 questions with one matrix product and count known-source hit@3.
- How to test, on questions labelled before the search, whether a score cutoff tells answerable questions from
  unanswerable ones.
- How to search the corpus with `shared/search.py`, which keeps these steps for the labs after this one.

**In class:** Parts A, B, C and E. Parts D, F and G are self-study. Run Part D's cells in class without stopping on
them, because Part E ranks the vectors Part D builds.

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
**Part D: Embed the corpus in batches (self-study).** 272 chunks in batches of 100, with the IDs kept in the same order as the rows.
**Part E: Rank the corpus for 30 questions and count hit@3.** One matrix product for all questions, hit@3 and hit@1,
and why the one hit@1 miss happened.
**Part F: Test whether a score cutoff separates answerable questions (self-study).** A cutoff chosen from two good
examples, then ten questions labelled before the search: what that one cutoff answers and refuses, then four cutoffs
compared.
**Part G: Keep the search in `shared/search.py` (self-study).** The same steps as functions, with a cache, used from here on.
**Going further.** Shorter vectors with `EMBED_DIMENSIONS` (OpenAI only), measured on hit@3 and hit@1.
**Challenge.** Test the cutoff on four questions of your own. Optional: reword the Atlas question and see where the
answering policy lands.

## Checkpoints

| After | You should see |
|---|---|
| Part A | `text-embedding-3-small`, 1,536 numbers, 8 tokens billed |
| Part B | toy cosines A 1.0, B 0.7071; real sentences 1–2 about 0.73, 1–3 about 0.31, 2–3 about 0.27 |
| Part C | H02 about 0.70, H04 (Boreal) 0.49, H01 0.43, H05 0.24, H03 0.18 |
| Part D | three batches, 21,539 tokens, `D: (272, 1536)` |
| Part E | Tesla's top chunk `S03-P00-C00` (about 0.72); hit@3 30/30; hit@1 29/30, missing *Who was the Super Bowl 50 MVP?* |
| Part F | anthem question about 0.66, above the 0.60 cutoff; at 0.60, 8 of the 10 decided right (the anthem answered, Apollo 1 refused); lowest answerable about 0.57, highest unanswerable about 0.66; 0.55 makes one mistake on the ten and refuses 3 of the 30 development questions |
| Part G | the module's vectors within a few thousandths of `D`, hit@3 30 of 30 |
| Going further | 256 numbers: hit@3 30 of 30, hit@1 27 of 30 |

The saved outputs come from `text-embedding-3-small` at `https://api.openai.com/v1` on 24 Oct 2026 (macOS, Python
3.13.5, a fresh `uv sync`). The same day the notebook gave the same counts with `pip` on Python 3.12.11 and with
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

**The same idea in three RAG books' code**

- [*Hands-On RAG for Production*, `chapter2/embedding.ipynb`](https://github.com/ofermend/hands-on-rag/blob/main/chapter2/embedding.ipynb): four sentences embedded with a local
  Sentence Transformers model, their cosine similarities as a heatmap, and the same scores on a random slice of the
  dimensions
- [*RAG from First Principles*, `03-Embedding/`](https://github.com/PacktPublishing/RAG-from-First-Principles/tree/main/03-Embedding): scripts for OpenAI embeddings in a small game recommender,
  clustering, BM25, and BGE-M3's dense, sparse and multi-vector embeddings
- [*Unlocking Data with Generative AI and RAG* (2nd ed.), `CHAPTER_08/CHAPTER8-1_DISTANCEMETRICS.ipynb`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/blob/main/CHAPTER_08/CHAPTER8-1_DISTANCEMETRICS.ipynb):
  Euclidean distance, dot product and cosine distance written by hand for three sentences

Next: [Lab 02](../02_Cited_Answers/README.md) (cited answers). Back to [the lab list](../README.md).
