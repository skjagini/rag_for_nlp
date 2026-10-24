# Lab 09: How to answer follow-up questions with retrieval

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/09_Follow_Up_Questions/09_follow_up_questions.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls) | Practises: turning a follow-up question into a search
that finds the right evidence | In class**

## Goal

Answer a follow-up question by searching with a query that stands on its own, and check that query in code before you
trust it. The moment that teaches it: after a question about Boreal, the user asks *How long is the link valid?*
Searched as typed, it finds the Atlas policy, and the answer gives Atlas's time with a correct citation. Rewriting
the question to name Boreal fixes it; letting the model write its own search drops Boreal again, and a two-line check
catches both.

## What you will learn

- Why a follow-up searched as typed can find the wrong evidence, or leave the answer with no question, while every
  citation check passes.
- How to rewrite a follow-up into a standalone query with one chat call, and check the query in code: the product
  kept, no number the user did not type.
- How to let the model write the query itself as a search-tool call (`@tool` and `bind_tools`), and run the call.
- How to test both routes on a set of conversations over several runs, and see where an earlier answer leaks in.
- When to ask the user which product they mean instead of guessing, and why that rule belongs in code.

**In class:** Parts A to C. Parts D, E and F are self-study.

## Coming from

[Lab 05](../05_Rebuild_With_LangChain/README.md) rebuilt the pipeline with LangChain: a vector store, a cited answer
and Lab 02's quote check, all for one question at a time. People ask in conversations, and their second question
often names nothing. This lab adds the step between the conversation and the search.

## What you will do

**Part A: Search a follow-up as typed.** Seven constructed conversations; the last turn searched and answered on its
own, which finds Atlas for a Boreal question, or gives the answer step no question.
**Part B: Rewrite the follow-up as a standalone query.** One chat call with a short instruction, then the same search
and answer.
**Part C: Check the rewrite in code.** Two checks, the product kept and no new number, each shown to pass and to fail.
**Part D: Let the model write the query as a tool call (self-study).** `@tool`, what the model is sent, `bind_tools`, the tool call
run by hand and in a loop, and a tool description that names the product.
**Part E: Test both routes on the seven conversations (self-study).** Four ways of writing the query, three runs each, one row per
conversation, and the earlier answer that leaks into a query.
**Part F: Ask which product instead of guessing (self-study).** An ask line in the instruction against a rule in code, and the
steps joined into one function.
**Challenge.** Add a conversation and test it on every route. Optional: check a rewrite instruction from an older
tutorial.

## Checkpoints

| After | You should see |
|---|---|
| Part A | *How long is the link valid?*: H02 first (about 0.59); Atlas's 30 minutes with H02, quote found, 7 of 10, both products' times 2 of 10, *Insufficient evidence.* 1 of 10. *What about Boreal?*: H04 about 0.33, just above a rainforest chunk (0.32); *Insufficient evidence.* 3 of 3 |
| Part B | the rewrite *Boreal password reset link validity duration.*, H04 first (about 0.78), Boreal's 10 minutes 3 of 3; *What about Boreal?* rewritten as *How long does a Boreal reset link remain valid?* |
| Part C | the rewrite passes both checks; *How long is the link valid?* fails the product check; a query with *30 minutes* fails the number check with `['30']` |
| Part D | the tool call's query *link validity*, product kept `False`; Atlas's 30 minutes with H02, 3 of 3; with the product rule, *Boreal password reset link validity* and Boreal's 10 minutes, 3 of 3 |
| Part E | the rewrite keeps the product 21 of 21 and copies *[H05]* into 3 of 3 `earlier_answer` queries; one `topic_switch` rewrite answered instead of rewriting; the first tool description drops it on three conversations; the product rule passes every column; `evidence_gap` refuses 3 of 3 |
| Part F | the ask line asks on 5 of 7 conversations; `needs_question()` asks on `two_products` only |

The saved outputs come from `gpt-4o-mini` and `text-embedding-3-small` at `https://api.openai.com/v1` on 24 Oct 2026
(macOS, Python 3.13.5, a fresh `uv sync`; `langchain` 1.4.6, `langchain-core` 1.6.12, `langchain-openai` 1.7.2). The
same day the notebook ran on the `pip` route (Python 3.12.11) and with `gpt-6-luna` and `REASONING_EFFORT=none`. Model replies
vary between runs: compare which checks pass, not every word.

**On the other routes.** The `pip` run kept the pattern: one of its three rewrites of *What about Boreal?* was again
*Boreal reset links expire after 30 minutes [H02].*, caught by the number check, with the wrong policy first. As
typed, *How long is the link valid?* gave both products' times 7 of 10 and Atlas's 30 minutes 3 of 10; the ask line
asked on 5 of 7. With `gpt-6-luna` the lab runs unchanged (tool calls work at `REASONING_EFFORT=none`), and the
failures move. As typed, it answers Atlas's 30 minutes 10 of 10, and answers *What about Boreal?* with Boreal's 10
minutes. Its own tool query keeps the product on `link_after_boreal`, so Part D's dead end does not appear; but on
`two_products` the product-rule query names Atlas only (3 of 3), as does one of three rewrites, and two of three tool
queries name one product, and the ask line asks on 0 of 7, guessing Atlas instead. Its rewrites copy *monthly* into the
`earlier_answer` query (3 of 3) with no number, which the checks cannot see. `needs_question()` gives the same result on every model. In the challenge, `gpt-6-luna` refused the as-typed question
instead of saying *yes*. Check the routes on the model you deploy; the code checks are what hold across models.

## Before you start

- **Lab 00 first,** then Lab 05: this lab uses `shared/lc_pipeline.py` and `shared/answer.py`.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `09_follow_up_questions.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** `data/squad-chunks.json` and `data/support-case.json`, read in place, and this lab's
  `resources/conversations.json`: seven constructed conversations about the five policies, with the earlier assistant
  turns written out and an answer key (product and policy) for each.
- **Packages:** `langchain`, `langchain-core` and `langchain-openai`, already installed by Lab 00's setup.
- **Cost:** one embedding of the corpus (about 21,000 tokens) and about 120 short chat calls, well under one US cent
  on `gpt-4o-mini`.

The notebook ends with a troubleshooting table: a tool without a docstring, a reply with no tool call, a tool call sent
back without its result, and tools on a reasoning model at the wrong effort.

## Resources

- [LangChain: retrieval](https://docs.langchain.com/oss/python/langchain/retrieval): 2-step RAG, agentic RAG with
  retrieval as a tool, and query rewriting as a step between them
- [LangChain: tools](https://docs.langchain.com/oss/python/langchain/tools): `@tool`, descriptions and argument schemas
- [LangChain: models, tool calling](https://docs.langchain.com/oss/python/langchain/models#tool-calling): `bind_tools`,
  `tool_calls`, the execution loop and `ToolMessage`
- [OpenAI: function calling](https://developers.openai.com/api/docs/guides/function-calling): what the API receives for
  a tool and returns for a call

Next: [Lab 10](../10_Long_Context_Or_Retrieval/README.md) (choose between long context and retrieval). Back to [the lab list](../README.md).
