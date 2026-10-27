# Lab 09: How to answer follow-up questions with retrieval

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/09_Follow_Up_Questions/09_follow_up_questions.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls) | Practises: turning a follow-up question into a search
that finds the right evidence | In class**

## Goal

Answer a follow-up question by searching with a query that stands on its own, and check that query in code before you
trust it. The moment that teaches it: the user asks how to reset a Boreal password, then *How long is the link
valid?* Searched as typed, the question finds the Atlas policy first, and the answer gives Atlas's reset-link time with
a citation whose quote really is in the evidence, so a quote check passes. One chat call that rewrites
the question to name Boreal fixes it, and two short checks in code say whether a rewrite kept the product and added
no number the user did not type.

Every question in the labs so far has stood on its own. In a
conversation the second question often names nothing (*How long is the link valid?*), and the search reads only the
words it is given, so this lab adds a step between the conversation and the search. The self-study parts let the model
write the query itself, as a search-tool call, and test both ways on seven conversations.

## What you will learn

- Why a follow-up searched as typed can find the wrong policy while the answer's citation passes a quote check.
- How to rewrite a follow-up into a question that stands on its own, with one chat call (query rewriting).
- How to check a rewritten query in code: the product kept, and no number the user did not type.
- How to let the model write the query itself as a search-tool call (`@tool` and `bind_tools`), and run the call.
- How to test both ways on a set of conversations over several runs, and see where an earlier answer leaks into a
  query.
- When to ask the user which product they mean instead of guessing, and why that rule belongs in code.

**In class:** Parts A to C, except the ten-run count at the end of Part A. That subsection and Parts D, E and F are
self-study.

## Coming from

[Lab 05](../05_Rebuild_With_LangChain/README.md) rebuilt the pipeline with LangChain, and Lab 06 Part D shows it: a
vector store, a cited answer and Lab 02's quote check, all for one question at a time. People ask in conversations, and their second question
often names nothing. This lab adds the step between the conversation and the search.

## What you will do

**Part A: Search a follow-up as typed.** Seven constructed conversations; the last turn searched and answered on its
own, one call at a time: the evidence lines, one cited answer, Lab 02's quote check on it, then three answers. It finds
Atlas for a Boreal question, or gives the answer step no question. A self-study subsection counts the answers over ten
runs.
**Part B: Rewrite the follow-up as a standalone question.** Query rewriting: one chat call with a short instruction,
the `(role, text)` messages shown before the call, then the same search and answer.
**Part C: Check the rewrite in code.** Two checks, the product kept and no new number, each shown to pass and to fail.
**Part D: Let the model write the query as a tool call (self-study).** `@tool`, what the model is sent, `bind_tools`,
the tool call run by hand and in a loop, and a tool description that names the product.
**Part E: Test both routes on the seven conversations (self-study).** Four ways of writing the query, three runs each,
one row per conversation, and the earlier answer that leaks into a query.
**Part F: Ask which product instead of guessing (self-study).** An ask line in the instruction against a rule in code,
and the steps joined into one function.
**Challenge.** Add a conversation and test it on every route. Optional: check a rewrite instruction from LangChain's
older history-aware retriever.

## Checkpoints

| After | You should see |
|---|---|
| Part A | *How long is the link valid?*: H02 first (about 0.59), H04 second (about 0.38); one answer *Atlas reset links expire after 30 minutes.* citing H02, quote found; three answers: Atlas's 30 minutes alone 1 of 3, both products' times 2 of 3, every quote found. *What about Boreal?*: H04 about 0.33, just above a rainforest chunk (0.32); *Insufficient evidence.* 3 of 3 |
| Part A, ten runs (self-study) | Atlas's 30 minutes alone 8 of 10, both products' times 2 of 10 (3 to 10 of 10 Atlas alone across our runs) |
| Part B | the rewrite *Boreal password reset link validity duration.*, H04 first (about 0.78), Boreal's 10 minutes 3 of 3; *What about Boreal?* rewritten as *How long does a Boreal reset link remain valid?*, Boreal's 10 minutes 3 of 3 |
| Part C | the rewrite passes both checks; *How long is the link valid?* fails the product check; a query with *30 minutes* fails the number check with `['30']` |
| Part D | the tool call's query *link validity*, product kept `False`; Atlas's 30 minutes with H02, 3 of 3; with the product rule, *Boreal password reset link validity* and Boreal's 10 minutes, 3 of 3 |
| Part E | the rewrite keeps the product 21 of 21, right policy first 15 of 15, and copies *[H05]* into 2 of 3 `earlier_answer` queries; the first tool description drops the product on three conversations; the product rule passes every column; `evidence_gap` refuses 3 of 3 |
| Part F | the ask line asks on 5 of 7 conversations; `needs_question()` asks on `two_products` only |

**About the saved outputs.** They come from `gpt-4o-mini` and `text-embedding-3-small`. Model replies vary from run to
run, so compare the pattern (which checks pass), not every word. The three-run count in Part A moves most:
over runs of ten, Atlas's time alone came back 3 to 10 times; the rest gave both products' times or, now and then,
*Insufficient evidence.*, and never Boreal's time alone. On another model the failures can move to other questions; the checks, and the way to measure them,
stay the same.

## Before you start

- **Lab 00 first.** This lab uses the pipeline in `shared/lc_pipeline.py` that Lab 05 rebuilt and Lab 06 Part D
  shows, and Lab 02's quote check from `shared/answer.py`.
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
