# Lab 05: How to rebuild a RAG pipeline with LangChain

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/05_Rebuild_With_LangChain/05_rebuild_with_langchain.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls) | Practises: building a RAG pipeline from a framework's
components and checking it against your own | In class**

## Goal

Rebuild the pipeline of Labs 01 and 03 (documents, search, a cited answer) from LangChain's current components, and
show on the same questions that it gives the same results. The moment that teaches it: on the 30 development
questions, hit@3 is 30 of 30 for your own search and 30 of 30 for LangChain's vector store, yet the two return the
same chunks for 0 of the 30 questions. The store returns four chunks unless you ask for three, and the count could
not see it. With `k=3` set, the lists agree for 30 of 30 and the answers pass the same checks.

## What you will learn

- How to turn chunk records into LangChain `Document`s that keep their IDs and fields.
- How LangChain's embeddings and chat model are built from `models.env` (`lc_embeddings()`, `lc_chat_model()`), and
  what `OpenAIEmbeddings` sends to the API.
- How to index documents in `InMemoryVectorStore` and search them with scores.
- How to check a framework's results against your own, question by question, and find the default that made them
  differ.
- How to get a cited answer with `with_structured_output` and check it with Lab 03's quote check.
- How to leave documents out of a search with a metadata filter.

## Coming from

You built the search (Lab 01) and the cited answer with its checks (Lab 03) in plain Python, and you can measure
both. This lab moves the same pipeline onto LangChain, the framework the labs after it use, and proves on your own
questions that nothing measurable changed before you build on it.

## What you will do

**Part A: Turn the chunks into documents.** One chunk record as a `Document` by hand, then all 277.
**Part B: Build the embeddings from the settings.** `OpenAIEmbeddings` by hand, the same vector as Lab 01's
`embed()`, what it sends (token IDs), then `lc_embeddings()` read and used.
**Part C: Index the documents and search with scores.** `InMemoryVectorStore`, and one question searched by both
pipelines.
**Part D: Check the two searches on 30 questions.** hit@3, the per-question lists, the default `k=4` found in the
signature, its cost in tokens, and parity once `k=3` is set.
**Part E: Get a cited answer with `with_structured_output`.** `ChatOpenAI` by hand, then `lc_chat_model()`; a
Pydantic class for Lab 03's schema; a cited answer checked with `quote_found()`.
**Part F: Check the two answer steps on 30 questions.** Refusals, claims and quotes in both pipelines, a second run as
a control, the Moon question, and the parity table.
**Part G: Leave documents out with a metadata filter.** A question that names no product, answered with and without
an Atlas-only filter.
**Part H: Keep the pipeline in `shared/lc_pipeline.py`.** The same steps as functions, the baseline for the next labs.
**Challenge.** Check a retriever against your own search. Optional: write a filter for a Boreal customer.

## Checkpoints

| After | You should see |
|---|---|
| Part A | 277 documents; H05 with `version` and `access` in its metadata |
| Part B | 1,536 numbers, largest difference from `embed()` 0.000000; `input sent: [[644, 1148, ...]]` |
| Part C | the same top three for the Tesla question, scores at most 0.0007 apart; the store prints a fourth row |
| Part D | hit@3 30 and 30 of 30; same lists 0 of 30; `k: int = 4`; 9,946 evidence tokens against 7,764; with `k=3`, same lists 30 of 30, largest score difference 0.0009 |
| Part E | `temperature: 0.0`; the same required fields as Lab 03's `SCHEMA`; *Nikola Tesla was born in the year 1856.*, quote found |
| Part F | both refuse 0 of 30, 31 claims each, all quotes found; same text 25 of 30 (plain against itself 28 of 30); the Moon question abstained 3 of 3 in each |
| Part G | without the filter H02, H04, H01 and both products' times; with `atlas_only` H02, H01, H05 and Atlas only, 3 of 3 |
| Part H | the module: Tesla with its quote found, *Insufficient evidence.* for the Moon |

The saved outputs come from `gpt-4o-mini` and `text-embedding-3-small` at `https://api.openai.com/v1` on 18 Oct 2026
(macOS, Python 3.13.5, a fresh `uv sync`; `langchain-core` 1.6.12, `langchain-openai` 1.7.2). The same notebook ran on
the `pip` route (Python 3.12.11) and with `gpt-6-luna` and `REASONING_EFFORT=none`. Model replies vary between runs:
compare the counts and which checks pass, not every word.

**On the other routes.** Parts A to D do not depend on the chat model, and every route gave the same lists, counts and
tokens; the largest score difference after `k=3` was between 0.0009 and 0.0021. The `pip` run matched the saved
pattern in Part F (31 claims each, all found, same text 26 of 30). With `gpt-6-luna` the lab runs unchanged and the
measures still agree (0 refusals in each pipeline, 35 and 32 claims, every quote found, the Moon question abstained 3 of
3), but its wording varies more: the two pipelines gave the same text for 18 of 30 questions, and two plain runs for 20
of 30, with one quote of 36 not found in the second. Compare pipelines on one model, with a second run of one of them
as the control.

## Before you start

- **Lab 00 first,** then Labs 01 and 03: this lab uses `shared/search.py`, `shared/answer.py` and the cached chunk
  vectors.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `05_rebuild_with_langchain.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** `data/squad-chunks.json`, `data/support-case.json` and `data/development-questions.json`, read in place.
- **Packages:** `langchain-core` and `langchain-openai`, already installed by Lab 00's setup. No `langchain-community`.
- **Cost:** about 110 short chat calls and two or three embeddings of the corpus (about 21,000 tokens each), well
  under one US cent on `gpt-4o-mini`.

The notebook ends with a troubleshooting table: an import from an older tutorial, a missing key, `temperature` on a
reasoning model, a dictionary schema without a name, and a filter that reads a field some documents lack.

## Resources

- [LangChain: build a semantic search engine](https://docs.langchain.com/oss/python/langchain/knowledge-base):
  documents, embeddings, vector stores and retrievers
- [LangChain: models, structured output](https://docs.langchain.com/oss/python/langchain/models#structured-output):
  `with_structured_output` with a Pydantic class, its methods, and `include_raw=True` for the token counts
- [LangChain reference: `InMemoryVectorStore`](https://reference.langchain.com/python/langchain-core/vectorstores/in_memory/InMemoryVectorStore):
  every search method, its defaults and the filter argument
- [LangChain reference: `OpenAIEmbeddings`](https://reference.langchain.com/python/langchain-openai/embeddings/base/OpenAIEmbeddings):
  `check_embedding_ctx_length` and the settings for other OpenAI-compatible servers

Next: [Lab 06](../06_Evaluation_Set/README.md) (build an evaluation set). Back to [the lab list](../README.md).
