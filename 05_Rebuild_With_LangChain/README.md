# Lab 05: How to rebuild a RAG pipeline with LangChain

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/05_Rebuild_With_LangChain/05_rebuild_with_langchain.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls) | Practises: building a RAG pipeline from a framework's
components and comparing it with your own | Demo**

## Goal

Rebuild the search of Labs 01 and 02 with LangChain and check it against your own, question by question. The moment
that teaches it: both searches find the right paragraph for all 30 questions, yet they return different chunks for
every one of them, because a LangChain default returns four chunks where your search returns three, and the overall
score cannot show it. Compare a framework with your own code question by question, not by a total. Parts E to H
(self-study) rebuild the answer step, a metadata filter and the module the next labs use, with the same comparison.

## What you will learn

- How to turn chunk records into LangChain documents that keep their IDs and fields.
- How LangChain's embeddings and chat model are built from `models.env`, and what the embeddings object sends to the
  API.
- How to index documents in a vector store and search them with scores.
- How to compare a framework's results with your own question by question, and find the default that made them
  differ.
- How to get a cited answer back as a Python object, check it with Lab 02's quote check, and explain why its answers
  can differ from your own.
- How to leave documents out of a search with a metadata filter.

**In class:** Parts A, C and D, shown as a demo, with the `lc_embeddings()` cell at the start of Part B, which
builds the embeddings Part C's store uses. The rest of Part B and Parts E to H are self-study.

## Coming from

You built the search (Lab 01) and the cited answer with its checks (Lab 02) in plain Python, and you can measure
both. This lab moves the same pipeline onto LangChain, the framework the labs after it use, and compares the two on
your own questions, step by step, so you know what changed before you build on it.

## What you will do

**Part A: Turn the chunks into documents.** One chunk record as a `Document` by hand, then all 277.
**Part B: Build the embeddings from the settings.** `lc_embeddings()` read and used; then, as self-study, the same
vector as Lab 01's `embed()` and the token IDs LangChain sends in place of the text.
**Part C: Index the documents and search with scores.** `InMemoryVectorStore`, and one question searched by both
pipelines.
**Part D: Check the two searches on 30 questions.** hit@3, the per-question lists, the default `k=4` found in the
signature, its cost in tokens, and the same lists once `k=3` is set.
**Part E: Get a cited answer with `with_structured_output` (self-study).** `ChatOpenAI` by hand, then `lc_chat_model()`;
a Pydantic class for Lab 02's schema and how the two schemas differ in text; a cited answer checked with
`quote_found()`.
**Part F: Check the two answer steps on 30 questions (self-study).** Both answer steps given the same evidence lines;
refusals, claims and quotes in both pipelines, a second run as a control, the Moon question, and the comparison table.
**Part G: Leave documents out with a metadata filter (self-study).** A question that names no product, answered with
and without an Atlas-only filter.
**Part H: Keep the pipeline in `shared/lc_pipeline.py` (self-study).** The same steps as functions, the baseline for
the next labs, and what each pipeline gives its answer step: `pack()`'s lines (Labs 02 and 04) against `evidence()`'s
(Labs 06 to 11).
**Challenge.** Check a retriever against your own search. Optional: write a filter for a Boreal customer.

## Checkpoints

| After | You should see |
|---|---|
| Part A | 277 documents; H05 with `version` and `access` in its metadata |
| Part B | `send token IDs: True`; as self-study, 1,536 numbers with largest difference from `embed()` 0.000000, and the question as eight token IDs `[644, 1148, ...]` |
| Part C | the same top three for the Tesla question, scores within a thousandth; the store prints a fourth row |
| Part D | hit@3 30 and 30 of 30; same lists 0 of 30; `k: int = 4`; 9,946 evidence tokens against 7,764; with `k=3`, same lists 30 of 30, largest score difference 0.0007 |
| Part E | `temperature: 0.0`; the same required fields as Lab 02's `SCHEMA`, with `$defs` and `title` in Pydantic's; *Nikola Tesla was born in the year 1856.*, quote found |
| Part F | both refuse 0 of 30 and every quote is found (31 and 32 claims); same cited IDs 29 of 30, same text 29 of 30 (plain against itself 30 of 30); the Moon question abstained 3 of 3 in each |
| Part G | without the filter H02, H04, H01 and both products' times; with `atlas_only` H02, H01, H05 and Atlas only, 3 of 3 |
| Part H | the module: Tesla with its quote found, *Insufficient evidence.* for the Moon; `pack()` 125 lines and 10,761 tokens, `evidence()` 90 lines and 8,382 tokens, more lines from `pack()` for 23 of 30 |

**About the saved outputs.** They come from `gpt-4o-mini` and `text-embedding-3-small`. Model replies vary from run to
run, so compare the pattern (the counts and which checks pass), not every word. Parts A to D and H give the same
lists, counts and tokens every run. Part F's wording moves: the two pipelines gave the same text for about 28 of 30
questions, and two runs of the same pipeline for 26 to 30 of 30. So compare two pipelines against a second run of one
of them, on one model.

## Before you start

- **Lab 00 first,** then Labs 01 and 02: this lab uses `shared/search.py`, `shared/answer.py` and the cached chunk
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
- [Pydantic: models](https://docs.pydantic.dev/latest/concepts/models/): fields, validation and `model_dump()`

Next: [Lab 06](../06_Evaluation_Set/README.md) (build an evaluation set). Back to [the lab list](../README.md).
