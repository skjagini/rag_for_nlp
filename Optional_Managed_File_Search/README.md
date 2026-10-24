# Optional B: How to use a managed search service instead of your own index

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/Optional_Managed_File_Search/optional_managed_file_search.ipynb)

**CPU | API key: `OPENAI_API_KEY` (embedding calls; with `MANAGED_SEARCH=on`, uploads, searches and model calls to
OpenAI's file search, OpenAI only) | Practises: choosing between building your own retrieval and buying a hosted one,
from measured results and what each lets you inspect | Self-study**

## Goal

Hand the 120 paragraphs to OpenAI's hosted vector store, search it with the 30 development questions, and weigh what
you gain against what you can no longer see or change. The moment that teaches it: the hosted search matches Lab 05's
pipeline at hit@3, and part of that comes from a ranker you cannot inspect. Switch it off and the right paragraph
drops out of the top three for about 9 of the 30 questions.

## What you will learn

- How to upload documents to a hosted vector store, one file per paragraph, and keep track of every ID you create.
- How to search the store directly and score it with hit@3 and hit@1 against Lab 05's pipeline on the same questions.
- Which settings the service lets you change (chunk size, number of results, the ranker, a score threshold, query
  rewriting) and what stays hidden (the embedding model, the vectors, the ranker).
- How to give the store to a model with the `file_search` tool in the Responses API, and read its query, results and
  citations.
- How to count what the tool's search results cost in input tokens, at 20 results and at 3.
- How to delete every file and store you created, and check from the account listing that none is left.

## Coming from

[Lab 05](../05_Rebuild_With_LangChain/README.md) built the pipeline you can read end to end: your chunks, your
embeddings, your store, and Lab 08 measured what hybrid search and a reranker add to it. This lab tries the other
route, a service that does all of that for you, and asks what you give up for the code you no longer write.

## What you will do

**Part A: Measure Lab 05's pipeline on the 30 questions.** Lab 05's store, its top three counted by paragraph, hit@3 and
hit@1.
**Part B: Upload the paragraphs to a hosted vector store.** One `.txt` file per paragraph, a store, a batch, and what the
store reports about itself.
**Part C: Search the store and compare it with Lab 05.** One question's ten results, then the 30 questions side by side.
**Part D: See what the ranker does, and what you can change.** The ranker switched off, the scores it leaves behind, and
a table of what you can set and what stays hidden.
**Part E: Let a model search with the `file_search` tool.** One question through the Responses API, then 20 results
against 3 on ten questions, in tokens and found answers.
**Challenge.** Let the service rewrite the questions. Optional: keep only results above a score.
**Delete what this notebook created.** The store first, then its files, then the account listing.

## Checkpoints

| After | You should see |
|---|---|
| Part A | Lab 05: hit@3 30, hit@1 29 of 30, then the run line with the embedding model and endpoint |
| Part B | 120 files uploaded; `completed`, 120 of 120 indexed; chunks of 800 tokens with 400 overlap; 214,148 bytes |
| Part C | 10 results, `S00-P00.txt` first at 0.9599, the fourth at 0.0384; managed hit@3 30, hit@1 28 (gains the MVP question, loses two complexity theory questions) |
| Part D | ranker `none`: hit@3 22, hit@1 13; top score 0.0325, the same as `1/61 + 1/62` |
| Part E | `['file_search_call', 'message']`, 20 results read, *Denver Broncos* citing `S00-P00.txt`, 5,515 input tokens; 51,532 against 18,422 input tokens, answer found 8 and 8 of 10 |
| Last section | 128 files and 3 stores in the account; the store deleted and 120 of 120 files still there; then 8 files and 2 stores, 0 of our 120 files and 0 of our 1 stores left |

The saved outputs come from `gpt-4o-mini` and `text-embedding-3-small` at `https://api.openai.com/v1` on 24 Oct 2026
(macOS, Python 3.13.5, a fresh `uv sync`; `openai` 3.35.1), with `MANAGED_SEARCH=on`. The same day the notebook ran
on the `pip` route (Python 3.12.11, `openai` 3.37.0), with `gpt-6-luna` and `REASONING_EFFORT=none`, and once with
the switch empty, where it finished with no error and the hosted cells printed nothing. Model replies and the
service's scores vary a little between runs: compare the counts, not every number.

**Across runs.** In seven runs that day (one exploration run, two runs each of the student and solved notebooks,
`pip` and `gpt-6-luna`) the store with its ranker gave hit@3 30 and hit@1 28 every time, and with `ranker='none'`
hit@3 20 to 22 and hit@1 12 or 13. The two failed answers in Part E were the same two questions in every `gpt-4o-mini` run.

**On the other routes.** The `pip` route gave the same counts as the saved run (`ranker='none'` at hit@3 22), and
51,808 against 18,436 input tokens, 8 and 8 answers found. With `CHAT_MODEL`, `JUDGE_MODEL` and `RERANK_MODEL` set to
`gpt-6-luna` and `REASONING_EFFORT=none` the lab runs unchanged; Parts A to D do not use the chat model and gave the
same results (`ranker='none'`: hit@3 21, hit@1 12). In Part E luna read more tokens (6,567 for the first question;
63,546 against 25,791 for the ten) and found the accepted answer for 9 of 10 either way: it named the Krasiński Palace
Garden, and still answered the computational question with *computational complexity theory*. The dollar column uses
the price in `models.env`, which is `gpt-4o-mini`'s: change `CHAT_PRICE_INPUT` with the model.

## Before you start

- **Lab 00 first,** then Lab 05: this lab uses `shared/lc_pipeline.py` as the baseline and `found()` from
  `shared/evals.py` (Lab 06).
- **The switch:** the hosted cells run only with `MANAGED_SEARCH=on`, in your `.env` (or `models.env`). With another
  provider's endpoint there is no Files API: leave `MANAGED_SEARCH` empty and read the saved results.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `optional_managed_file_search.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** `data/squad-passages.json`, `data/squad-chunks.json`, `data/support-case.json` and
  `data/development-questions.json`, read in place. Uploaded copies of the 120 paragraphs are deleted at the end.
- **Packages:** none beyond Lab 00's setup.
- **Cost:** one embedding of the corpus for Lab 05's store (about 21,000 tokens). With the switch on, about 214 KB of
  storage (free under 1 GB), about 60 store searches (not billed) and 21 model calls with the `file_search` tool, at
  2.50 US dollars per 1,000 tool calls: about 0.07 US dollars a run on `gpt-4o-mini`. The challenge adds only
  store searches.

The notebook ends with a troubleshooting table: the switch left empty, the switch turned on halfway, results from a
removed file, and files left behind by a failed run.

## Resources

- [OpenAI: retrieval guide](https://developers.openai.com/api/docs/guides/retrieval): vector stores, search, ranking
  options, query rewriting, chunking and storage pricing
- [OpenAI: file search tool](https://developers.openai.com/api/docs/guides/tools-file-search): the `file_search` tool
  in the Responses API, `max_num_results` and `include`
- [OpenAI: pricing](https://developers.openai.com/api/docs/pricing): file search storage and tool call prices
- [OpenAI API reference: vector stores](https://developers.openai.com/api/reference/resources/vector_stores): every
  method, including `delete` for stores and for files

**The same idea in three RAG books' code**

- [*Hands-On RAG for Production*, `chapter5/`](https://github.com/ofermend/hands-on-rag/tree/main/chapter5): Vectara, a managed RAG platform: a PDF uploaded, queried with its hybrid
  search and reranker, and the answer given a factual consistency score

Next: [Optional A: How to build the same pipeline with LlamaIndex](../Optional_LlamaIndex_Pipeline/README.md). Back to
[Lab 05](../05_Rebuild_With_LangChain/README.md), the pipeline this lab compares with, or to [the lab list](../README.md).
