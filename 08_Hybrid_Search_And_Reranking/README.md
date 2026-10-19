# Lab 08: How to improve retrieval with hybrid search and reranking

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/08_Hybrid_Search_And_Reranking/08_hybrid_search_and_reranking.ipynb)

**CPU | API key: `OPENAI_API_KEY` (embedding and chat calls) | Practises: testing a retrieval change on paired
questions, with its cost | In class**

## Goal

Add keyword search, fuse it with the vector search, rerank the candidates with a chat model, and decide from a
question-by-question comparison whether each change helps and what it costs. The moment that teaches it: on the 30
development questions, hybrid search, a common architecture, loses one question at hit@3 and, at hit@1, gains 1 and
loses 3. The reranker gains 1 and loses 1, costs 961 tokens and 1.2 seconds per question, and for one question named a
chunk that was not a candidate in 4 of 6 calls. It fixes a ranking both searches miss only on a question worded unlike
its answer.

## What you will learn

- How BM25 scores a chunk by the words it shares with a question, by hand and with `bm25s`.
- How reciprocal rank fusion combines two ranked lists without adding scores on different scales.
- How to compare a change with the baseline question by question in a paired table, and what a sign test says.
- How to rerank candidates with a chat model and check the returned IDs in code.
- How to price a retrieval stage in tokens, dollars and seconds.
- How to write a decision from the paired results and the cost.

## Coming from

Lab 05 rebuilt the pipeline with LangChain and made its vector store the baseline; Lab 06 builds an evaluation set and
Lab 07 checks a judge against your labels. This lab changes one stage, retrieval, and measures each change against
Lab 05's baseline on the same questions, one row per question.

## What you will do

**Part A: Score the five policies with BM25 by hand.** The formula in plain Python, the words each policy shares with
a paraphrased question, then `bm25s` set up to give the same order.
**Part B: Search the corpus with BM25.** A BM25 index of the 277 records beside the vector store, the MVP question,
and both searches on the 30 questions.
**Part C: Fuse the two lists with reciprocal rank fusion.** A three-item example by hand, `rrf()`, and why fusion
pushes the Norman answer down.
**Part D: Compare hybrid search with the baseline, question by question.** The paired table at hit@3 and hit@1, the
moved questions, and the sign test.
**Part E: Rerank the candidates with a chat model.** One listwise rerank call with `RERANK_MODEL`, its tokens and
seconds, and the check that the reply holds exactly the candidate IDs.
**Part F: Rerank the 30 questions and count the cost.** Every reply checked, one question repeated five times, the
paired table, the lost question read, and the cost per question.
**Part G: Try both changes on Atlas questions and decide.** Five Atlas questions from Labs 01 and 04, then the
decision table and line.
**Part H: Keep the pieces in `shared/hybrid.py`.** The module read and checked against the hand versions.
**Going further.** Maximal marginal relevance on the store.
**Challenge.** Repair a rejected ranking. Optional: check whether the candidate order sways the reranker.

## Checkpoints

| After | You should see |
|---|---|
| Part A | H01 1.705, H02 1.519, H04 1.412; H02 shares `['atlas', 'the']`; `bm25s` ratio 2.20 for every policy |
| Part B | 4,083 distinct words; BM25 puts `S00-P02-C00` first for the MVP question; vector hit@1 29, hit@3 30; BM25 22 and 28 |
| Part C | A 0.032522, B 0.032266, C 0.016129; the Norman answer fourth with 0.0164 |
| Part D | hybrid hit@1 27, hit@3 29; hit@3 lost 1 (Norman); hit@1 gained 1, lost 3; p = 1.000 and 0.625 |
| Part E | reranked `H02`, `H01`, `H04`; 561 tokens in; `exactly the candidate IDs: True` |
| Part F | 29 of 30 replies pass; the Saxon Garden reply names `S02-P00-C02`, and fails in 3 of 5 repeats; hit@1 gained 1 (MVP), lost 1; 961 tokens and about 1,179 ms per question against about 220 ms for the vector search |
| Part G | right first: vector 3 of 5, hybrid 3 of 5, reranked 4 of 5; the mobile-app reply fails the check (left out `['H03']`) |
| Part H | same BM25 and fused lists 30 of 30; p = 0.625; `['H02', 'H01', 'H04']` |

The saved outputs come from `gpt-4o-mini` and `text-embedding-3-small` at `https://api.openai.com/v1` on 18 Oct 2026
(macOS, Python 3.13.5, a fresh `uv sync`; `bm25s` 0.3.13, `langchain-core` 1.6.12, `langchain-openai` 1.7.2). BM25,
fusion and the sign test give the same numbers on every run; the vector search can move a score by a few thousandths;
the reranker's replies vary. Compare the counts and which checks pass, not every millisecond.

**Across runs.** We ran the 30-question rerank pass 11 times on `gpt-4o-mini` while building the lab: 0 to 3 replies
failed the ID check per pass, and the Saxon Garden reply named a chunk that was not a candidate in 9 of the 11 (in 11
of 15 repeats). The paired result was the same in every pass: hit@3 unchanged, hit@1 gained the MVP question and lost
the computational measure question.

**On the other routes.** The `pip` route (Python 3.12.11, which installed `bm25s` 0.3.14) gave the same BM25, fusion
and paired numbers; 28 of 30 replies passed, and 4 of 5 Saxon Garden repeats failed. With `CHAT_MODEL`, `JUDGE_MODEL`
and `RERANK_MODEL` set to `gpt-6-luna` and `REASONING_EFFORT=none` the lab runs unchanged, and the reranker behaves
better: every reply passed the ID check (30 of 30, and 5 of 5 repeats), it gained the MVP question and lost none
(hit@1 30 of 30), and it put the right policy first for 5 of 5 Atlas questions, the mobile-app one included. It is
slower, 1,645 ms per question, for about the same tokens (960 in, 80 out). The dollar line uses the prices in
`models.env`, which are `gpt-4o-mini`'s: change `CHAT_PRICE_INPUT` and `CHAT_PRICE_OUTPUT` with the model.

## Before you start

- **Lab 00 first,** then Lab 05: this lab uses `shared/lc_pipeline.py` as the baseline.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `08_hybrid_search_and_reranking.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** `data/squad-chunks.json`, `data/support-case.json` and `data/development-questions.json`, read in place.
- **Packages:** `bm25s` and LangChain's core packages, already installed by Lab 00's setup.
- **Cost:** one embedding of the corpus (about 21,000 tokens) and about 40 rerank calls of about 1,000 tokens each,
  well under one US cent on `gpt-4o-mini`.

The notebook ends with a troubleshooting table: `bm25s` defaults that change the order, a `k` larger than the index,
JSON mode without the word JSON, and `REASONING_EFFORT` sent to a reranker that does not take it.

## Resources

- [`bm25s`](https://github.com/xhluca/bm25s): the BM25 package, its tokenizer options and scoring variants
- [Cormack, Clarke and Büttcher (2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf): reciprocal rank
  fusion
- [Elasticsearch: reciprocal rank fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
  and [Qdrant: hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/): hybrid search in a search
  engine and a vector database
- [Sun and others (2023), RankGPT](https://arxiv.org/abs/2304.09542): listwise reranking with a chat model
- [Cohere: rerank overview](https://docs.cohere.com/docs/rerank-overview): a production cross-encoder reranker

Next: [Lab 09: How to answer follow-up questions with retrieval](../09_Follow_Up_Questions/README.md). Back to
[the lab list](../README.md).
