# Optional A: How to build the same pipeline with LlamaIndex

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/Optional_LlamaIndex_Pipeline/optional_llamaindex_pipeline.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls) | Practises: building a RAG pipeline from a framework's
components, finding the defaults it sets for you, and checking it against your own | Self-study**

## Goal

Rebuild the pipeline of Labs 01, 02 and 05 with LlamaIndex, a framework made for RAG, and check it question by question
against your own code. The moment that teaches it: both pipelines find the right paragraph for every question, yet
they return the same three chunks in the same order for only about half the questions. LlamaIndex embeds each chunk's metadata with its
text unless you tell it not to, and the overall score could not show it.

## What you will learn

- How to turn chunk records into LlamaIndex `TextNode`s that keep their IDs, and print the text the embedding model and
  the LLM read from each node.
- How to build LlamaIndex's chat model and embeddings from `models.env` with the `openai-like` integrations, and what
  LlamaIndex uses when you set nothing.
- How to index nodes in a `VectorStoreIndex`, search them with a retriever, and find its default number of results.
- How to check a framework's search against your own, question by question, and find the setting that made them differ.
- How to read a query engine's prompt, and get a cited answer with `output_cls` that Lab 02's quote check can check.

## Coming from

[Lab 05](../05_Rebuild_With_LangChain/README.md) rebuilt the pipeline of Labs 01 and 02 with LangChain and showed it
gives the same results once one default (`k=4`) is set. This lab does the same with LlamaIndex, the other framework
many RAG tutorials use, so you can compare how much each one does for you and what it hides. Nothing later depends on
it.

## What you will do

**Part A: Turn the chunks into nodes.** One chunk record as a `TextNode` by hand, then all 277.
**Part B: Build the models from the settings.** What `Settings` holds when you set nothing, `OpenAILikeEmbedding`
against Lab 01's `embed()`, and `OpenAILike` on the Completions endpoint, then on Chat Completions.
**Part C: Index the nodes and retrieve.** A `VectorStoreIndex`, a retriever with its default `similarity_top_k`, and
the Tesla question searched by both pipelines.
**Part D: Check the two searches on 30 questions.** hit@3, the same lists count, the text the embedding model read,
and parity once the metadata is left out.
**Part E: Ask through a query engine.** Its retriever, its four prompts, its default system message, and an answer
with its source nodes.
**Part F: Get a cited answer with `output_cls`.** The 400 it gives by default, then `should_use_structured_outputs=True`,
Lab 02's rules and the chunk ID for the model, checked with `quote_found()`.
**Part G: Compare the two pipelines on 30 questions.** Refusals, claims and quotes in both, the parity table beside
Lab 05's LangChain results, and the defaults each framework sets for you.
**Challenge.** Choose which metadata the embedding model reads, and measure what the title does.

## Checkpoints

| After | You should see |
|---|---|
| Part A | 277 nodes; H05 with `version` and `access` in its metadata |
| Part B | `Settings` defaults `gpt-3.5-turbo` and `text-embedding-ada-002`; 1,536 numbers, largest difference 0.000000; `is_chat_model: False` and a reply that stops mid-sentence; then a full answer |
| Part C | 277 nodes in a `SimpleVectorStore`; two rows and `similarity_top_k: int = 2`; with 3, the same top three as `search()` but scores of about 0.67, 0.62, 0.59 against 0.72, 0.54, 0.47 |
| Part D | hit@3 30 and 30 of 30; same lists 15 of 30; six metadata lines on top of the embedded text; with them left out, same lists 30 of 30, largest score difference a few thousandths (0.0008) |
| Part E | `RetrieverQueryEngine`, top k 2; *Never directly reference the given context in your answer.*; *Nikola Tesla was born on 10 July 1856.* from two nodes |
| Part F | `BadRequestError: Error code: 400 ... 'tool_choice' ...`; *Nikola Tesla was born in the year 1856.*, quote found; *Insufficient evidence.* for the Moon |
| Part G | both refuse 0 of 30, 30 and 32 claims, every quote found; same cited IDs 28 of 30, same text 24 of 30 |

The saved outputs come from `gpt-4o-mini` and `text-embedding-3-small` at `https://api.openai.com/v1` on 24 Oct 2026
(macOS, Python 3.13.5, this lab's own environment made with `uv`: `llama-index-core` 0.14.25,
`llama-index-llms-openai-like` 0.8.1, `llama-index-embeddings-openai-like` 0.4.0, `openai` 2.54.0). Model replies vary
between runs: compare the counts and which checks pass, not every word.

**On the other routes.** The `pip` route (Python 3.12.11, `openai` 2.54.0) gave the same search results and the same checks. With `CHAT_MODEL`, `JUDGE_MODEL` and `RERANK_MODEL` set to `gpt-6-luna` and `REASONING_EFFORT=none`, the Completions cell in Part B prints `TypeError: Completions.create() got an unexpected keyword argument 'reasoning_effort'` instead of a reply; every later part runs unchanged. Its answers vary more in wording and in the number of claims, as in Lab 05.

## Before you start

- **Labs 01, 02 and 05 first:** this lab uses `shared/search.py` and `shared/answer.py` and compares with Lab 05's
  results. Run Lab 00 for your key and route.
- **Its own environment.** LlamaIndex's OpenAI integration (`llama-index-llms-openai` 0.8.3) needs `openai` below
  version 3, while the repository's environment installs `openai` 3. So this lab has its own `requirements.txt` and its
  own `.venv` inside this folder, which git ignores. Create it once, from the repository folder, on your route:

  With `uv` (Python 3.12 or 3.13; deactivate any other environment first, so `uv` installs into the new one):

  ```bash
  cd Optional_LlamaIndex_Pipeline
  uv venv --python 3.13
  uv pip install -r requirements.txt ipykernel
  ```

  With `pip` (Python 3.12 or 3.13):

  ```bash
  cd Optional_LlamaIndex_Pipeline
  python3.12 -m venv .venv
  source .venv/bin/activate          # Windows: .venv\Scripts\activate
  pip install -r requirements.txt ipykernel
  ```

  Either way you get `openai` 2.54.0 and the LlamaIndex versions above. In a codespace, use the `uv` commands in a
  terminal; on an Ubuntu VM too (`uv` fetches its own Python; the VM's Python 3.10 is too old for the pip route). On
  Colab, the notebook's Setup cell installs this folder's `requirements.txt`. If Setup prints an `openai` version other
  than 2.54.0, choose Runtime, Restart session and run Setup again.
- **Open the notebook:** locally or in a codespace, open `optional_llamaindex_pipeline.ipynb` and choose the kernel
  from `Optional_LlamaIndex_Pipeline/.venv`, not the repository's `.venv`; Setup prints `openai 2.54.0` when the kernel
  is right. On Colab, use the badge above (add `OPENAI_API_KEY` under Secrets first). Run Setup and Config first.
- **Data:** `data/squad-chunks.json`, `data/support-case.json` and `data/development-questions.json`, read in place.
- **Cost:** four embeddings of the corpus (about 21,000 tokens each) and about 70 short chat calls, well under one US
  cent on `gpt-4o-mini`; the challenge adds one more embedding of the corpus.

We ran the `uv` and `pip` routes on macOS. We could not test the Colab and Codespaces steps here (no Colab session or
codespace from this machine).

The notebook ends with a troubleshooting table: the repository's kernel chosen by mistake, a model built without
`api_key`, `OpenAILike` left on the Completions endpoint (two errors), `output_cls` without structured outputs, and the
retriever's default of two nodes.

## Resources

- [LlamaIndex: documents and nodes](https://developers.llamaindex.ai/python/framework/module_guides/loading/documents_and_nodes/usage_documents/):
  metadata, and what the LLM and the embedding model read
- [LlamaIndex: Settings](https://developers.llamaindex.ai/python/framework/module_guides/supporting_modules/settings/):
  the global defaults and how to set them
- [LlamaIndex reference: `OpenAILike`](https://developers.llamaindex.ai/python/framework-api-reference/llms/openai_like/)
  and [`OpenAILikeEmbedding`](https://developers.llamaindex.ai/python/framework-api-reference/embeddings/openai_like/):
  the settings for OpenAI-compatible servers
- [LlamaIndex: prompts](https://developers.llamaindex.ai/python/framework/module_guides/models/prompts/usage_pattern/):
  `get_prompts()` and replacing a query engine's prompts
- [LlamaIndex: structured outputs from a query engine](https://developers.llamaindex.ai/python/framework/module_guides/querying/structured_outputs/query_engine/):
  `output_cls`

**The same idea in three RAG books' code**

- [*RAG from First Principles*, `00-SimpleRAG/`](https://github.com/PacktPublishing/RAG-from-First-Principles/tree/main/00-SimpleRAG): one small RAG written many ways: LlamaIndex in five lines with the
  models swapped, LangChain, LCEL, LangGraph and plain Python

Back to [Lab 05](../05_Rebuild_With_LangChain/README.md) or to [the lab list](../README.md).
