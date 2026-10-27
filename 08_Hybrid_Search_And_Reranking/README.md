# Lab 08: How to improve retrieval with hybrid search and reranking

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/08_Hybrid_Search_And_Reranking/08_hybrid_search_and_reranking.ipynb)

**CPU | API key: `OPENAI_API_KEY` (embedding and chat calls) | Practises: testing a retrieval change on paired
questions, with its cost | In class**

## Goal

Add keyword search and a reranker to the vector search, and decide from a question-by-question comparison whether
each change helps. The moment that teaches it: **hybrid search**, a common design that runs a keyword search beside
the vector search and merges the two lists, gains one of the 30 development questions at rank 1 and loses three, a split that
chance explains as well as a real effect. Then one model call reads the candidates for a customer's question that
both searches and their merge rank wrong, and puts the answering policy first.

You can now measure the pipeline question by question and know how far to trust a judge. What you cannot yet say
is whether a common retrieval change, keyword search or a reranker, makes this pipeline better: this lab tries
both and decides from the questions each one gains and loses.

## What you will learn

- How BM25, the classic keyword ranking, scores a chunk from the words it shares with a question.
- How reciprocal rank fusion merges two ranked lists by their ranks instead of their scores.
- How to compare a change with the baseline question by question, and read a sign test on the questions that moved.
- How to rerank candidates with a chat model, and check in code that the reply names only the candidates.
- How to price a retrieval stage in tokens, dollars and seconds.
- How to write a decision from the paired results and the cost.

**In class:** Parts B to D, then Part E up to Checkpoint E (one rerank call). Part A, the ID check after
Checkpoint E, and Parts F to H are self-study.

## Coming from

Lab 05 rebuilt the pipeline with LangChain and made its vector store the baseline; Lab 06 builds an evaluation set and
Lab 07 checks a judge against your labels. This lab changes one stage, retrieval, and measures each change against
Lab 05's baseline on the same questions, one row per question.

## What you will do

**Part A: Score the five policies with BM25 by hand (self-study).** The BM25 formula in plain Python, the words each
policy shares with a paraphrased question, then `bm25s` set up to give the same order.
**Part B: Search the corpus with BM25.** Dense and sparse retrieval, the BM25 formula read term by term, a BM25 index of
the 277 records beside the vector store, the MVP question, and both searches on the 30 questions.
**Part C: Fuse the two lists with reciprocal rank fusion.** The RRF formula, a three-item example by hand, `rrf()`, and
why fusion pushes the Norman answer down.
**Part D: Compare hybrid search with the baseline, question by question.** The paired table at hit@3 and hit@1, the
moved questions, and the sign test as a formula and in code.
**Part E: Rerank the candidates with a chat model.** Two-stage retrieval, where the recovery question's answer sits in
each list, one listwise rerank call with `RERANK_MODEL` that puts it first, and its tokens and seconds. Self-study at
the end: the check that the reply holds exactly the candidate IDs.
**Part F: Rerank the 30 questions and count the cost (self-study).** Every reply checked, one question repeated five
times, the paired table, the lost question read, and the cost per question.
**Part G: Try both changes on Atlas questions and decide (self-study).** Five Atlas questions from Labs 04 and 06, then
the decision table and line.
**Part H: Keep the pieces in `shared/hybrid.py` (self-study).** The module read and checked against the hand versions.
**Going further.** Maximal marginal relevance on the store.
**Challenge.** Repair a rejected ranking. Optional: check whether the candidate order sways the reranker.

## Checkpoints

| After | You should see |
|---|---|
| Part A | H01 1.705, H02 1.519, H04 1.412; H02 shares `['atlas', 'the']`; `bm25s` ratio 2.20 for every policy |
| Part B | 4,083 distinct words; for the MVP question vector ranks the answer 2nd, BM25 1st (9.805); vector hit@1 29, hit@3 30; BM25 22 and 28 |
| Part C | A 0.032522, B 0.032266, C 0.016129; the Norman answer fourth with 0.0164 |
| Part D | hybrid hit@1 27, hit@3 29; hit@3 lost 1 (Norman); hit@1 gained 1, lost 3; p = 1.000 and 0.625 |
| Part E | H02 at rank 3 (vector), 6 (BM25), 4 (hybrid); reranked `H02`, `H01`, `H04`; 561 tokens in; self-study: `exactly the candidate IDs: True` |
| Part F | 28 of 30 replies pass (the Tesla reply names `S03-P00-C01`, an ID not in the corpus); the Saxon Garden question fails 4 of 5 repeats; hit@1 gained 1 (MVP), lost 1; 961 tokens and about 1,600 ms per question against about 230 ms for the vector search |
| Part G | right first: vector 3 of 5, hybrid 3 of 5, reranked 4 of 5; the mobile-app reply fails the check (left out `['H03']`) |
| Part H | same BM25 and fused lists 30 of 30; p = 0.625; `['H02', 'H01', 'H04']` |

**About the saved outputs.** They come from `gpt-4o-mini` and `text-embedding-3-small`. BM25, fusion and the sign test
give the same numbers every run; the vector search can move a score by a few thousandths; the reranker's replies vary.
The rerank call in Part E put H02 first in every call we made (18 of 18). In Part F's 30-question pass, 0 to 3 replies
fail the ID check per pass: most name a real chunk that was not a candidate, and one named an ID that does not exist.
The dollar line uses the prices in `models.env`: change `CHAT_PRICE_INPUT` and `CHAT_PRICE_OUTPUT` when you change the
model.

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
- [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25): the formula, IDF, `k1` and `b`
- [Cormack, Clarke and Büttcher (2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf): reciprocal rank
  fusion and `c = 60`
- [Sign test](https://en.wikipedia.org/wiki/Sign_test): the test on the questions that moved
- [Elasticsearch: reciprocal rank fusion](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)
  and [Qdrant: hybrid queries](https://qdrant.tech/documentation/concepts/hybrid-queries/): hybrid search in a search
  engine and a vector database
- [Sun and others (2023), RankGPT](https://arxiv.org/abs/2304.09542): listwise reranking with a chat model
- [Cohere: rerank overview](https://docs.cohere.com/docs/rerank-overview): a production cross-encoder reranker

Next: [Lab 09: How to answer follow-up questions with retrieval](../09_Follow_Up_Questions/README.md). Back to
[the lab list](../README.md).
