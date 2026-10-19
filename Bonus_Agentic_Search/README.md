# Bonus: How to let the model decide when to search

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/Bonus_Agentic_Search/bonus_agentic_search.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls) | Practises: choosing between a fixed search-then-answer
pipeline and an agent that searches when it needs to, from measured answers and costs | Self-study**

## Goal

Give the model a search tool, let it decide whether to search, how often and for what, and judge from answers, model
calls and tokens when that beats the fixed pipeline that searches once with the question. The moment that teaches it:
asked *Who ruled the duchy of Normandy, and when was the Latin version of the word Norman first recorded?*, the fixed
pipeline answers the second half with *the 10th and 11th centuries*, citing a passage whose quote passes Lab 03's check
(3 of 3 runs). The passage that answers it ranked fourth. The agent splits the question into two searches and answers
*the 9th century* (5 of 5 runs). On a one-fact question it makes two model calls where the pipeline makes one, and on
*Hi there!* it skips the search the pipeline always makes.

## What you will learn

- How a fixed search-then-answer pipeline handles a question with two parts, and a message that asks nothing.
- How to give the model a search tool and run its calls by hand, two calls in one reply included.
- How to add a stopping rule to the loop, and what a cap set too low costs.
- How to build the same loop with LangChain's `create_agent`, and read the graph it returns.
- How to compare two designs on the same questions in answers, model calls, searches and tokens, and decide which
  kind of question each one suits.

## Coming from

[Lab 05](../05_Rebuild_With_LangChain/README.md) built the fixed pipeline in `shared/lc_pipeline.py`: one search with
the question, one cited answer from the top three passages. [Lab 09](../09_Follow_Up_Questions/README.md) gave the model
a search tool so it could write a follow-up's query, and ran its calls in a loop. This lab hands the model the whole
decision (whether to search, how often and for what) and measures what that buys.

## What you will do

**Part A: Run the fixed pipeline on three kinds of question.** One fact, two parts, and a greeting; the two-part
question's wrong half passes the quote check, and the passage it needed ranks fourth.
**Part B: Give the model a search tool and run its calls by hand.** One reply with two tool calls, both run and sent
back, then the loop that repeats it until the model answers.
**Part C: Add a stopping rule.** A question the documents do not answer, a limit on searches in code, and the limit
tested where it bites.
**Part D: Build the same loop with `create_agent`.** The graph it returns, the same run message by message, and the
limit as `ToolCallLimitMiddleware`.
**Part E: Compare the two designs on the same questions.** Eight messages of three kinds, three runs each through each
design: right or not, model calls, searches and tokens.
**Challenge.** A question with three parts. Optional: a line that lets the agent answer small talk, tested where it
should change nothing.

## Checkpoints

| After | You should see |
|---|---|
| Part A | 277 documents; the Super Bowl question answered from S00-P02-C00; the Normandy question: Richard I and *the 10th and 11th centuries*, both quotes found, 3 of 3; S02-P02-C00 (*9th century*) fourth; *Hi there!* searched and refused |
| Part B | two tool calls in one reply; Richard I and *the 9th century*; the Super Bowl question in 2 calls with one search; *Hi there!* in 1 call with none |
| Part C | the refund question: two searches, three calls, refused, 3 of 3; with one search allowed, the Normandy answer gives the Latin form no date, 3 of 3 |
| Part D | `CompiledStateGraph`, recursion limit 9999; the same two queries and *the 9th century*; with `run_limit=1`, *Tool call limit exceeded. ...* for one call |
| Part E | one fact: both right 9 of 9, 9 against 18 model calls, 3,981 against 4,056 tokens in; two parts: fixed 3 of 9, agent 9 of 9, 4,242 against 6,123 tokens in; no search: fixed 0 of 6, agent 3 of 6 (it refuses the thank-you) |

The saved outputs come from `gpt-4o-mini` and `text-embedding-3-small` at `https://api.openai.com/v1` on 18 Oct 2026
(macOS, Python 3.13.5, a fresh `uv sync`; `langchain` 1.4.6, `langchain-core` 1.6.12, `langchain-openai` 1.7.2,
`langgraph` 1.2.16). The same notebook ran on the `pip` route (Python 3.12.11) and with `gpt-6-luna` and
`REASONING_EFFORT=none`. Model replies vary between runs: compare which answers are right and how many calls each
design makes, not every word.

**On the other routes.** The `pip` run (the same package versions) gave Part E's table with the same rights, calls,
searches and tokens in, and the same failures in every part; with `run_limit=1` its agent asked for the second search
again and was refused twice. With `gpt-6-luna` the lab runs unchanged (tools work at `REASONING_EFFORT=none`), and
both designs behave differently. The fixed pipeline answers the Normandy question with Richard I and says the
evidence does not give the date (3 of 3), so Part A's wrong half with a found quote does not appear; it is still
right on only 3 of 9 two-part runs. On the Normandy question the agent first searches with the whole question, then
for the missing part, one search per reply instead of two in one (2 or 3 searches, 3 or 4 calls), and the two-part
questions cost it 11,733 tokens in against the pipeline's 4,215, nearly three times. One-fact questions cost it 5,662
tokens in against 3,954. It also searches for *Hi there!* and *Thanks, that is all I needed.* (6 of 6, with queries
such as *greeting hello hi there*) before it replies politely, so it is right 6 of 6 there at twice the pipeline's
calls. In the challenge, the small-talk line stopped those searches and every one-fact question still searched and
answered. Which design wins on which question depends on the model: measure it on the one you deploy.

## Before you start

- **Lab 00 first,** then Lab 05; Lab 09 helps but is not required. This lab uses `shared/lc_pipeline.py`,
  `shared/answer.py` and `shared/evals.py`.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `bonus_agentic_search.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** `data/squad-chunks.json` and `data/support-case.json`, read in place. The eight test messages are written
  in the notebook.
- **Packages:** `langchain`, `langchain-core`, `langchain-openai` and `langgraph` (installed with `langchain`), already
  installed by Lab 00's setup.
- **Network:** the graph picture in Part D is drawn by the mermaid.ink web service; if your network blocks it, the
  notebook's troubleshooting table prints the same graph as text.
- **Cost:** one embedding of the corpus and about a hundred short chat calls, well under one US cent on `gpt-4o-mini`.

The notebook ends with a troubleshooting table: a tool call sent back without its result, a recursion limit set too
low, the graph picture on a blocked network, and tools on a reasoning model at the wrong effort.

## Resources

- [LangChain: retrieval](https://docs.langchain.com/oss/python/langchain/retrieval): 2-step RAG against agentic RAG,
  where the model decides when to retrieve
- [LangChain: agents](https://docs.langchain.com/oss/python/langchain/agents): `create_agent`, a model calling tools in
  a loop, and the middleware around it
- [LangChain: built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in): the tool call
  limit and the model call limit
- [OpenAI: function calling](https://developers.openai.com/api/docs/guides/function-calling): tool calls, parallel calls
  in one reply, and the tool results the API expects back
- [Asai et al., *Self-RAG*, ICLR 2024](https://arxiv.org/abs/2310.11511): a model trained to decide when to retrieve

Back to [the lab list](../README.md).
