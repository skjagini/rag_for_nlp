# Lab 02: How to generate an answer that cites its sources

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/02_Cited_Answers/02_cited_answers.ipynb)

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls) | Practises: grounded answers with citations and
abstention, and checking them in code | In class**

## Goal

Answer a question from retrieved evidence with citations, check the citations in code, and learn what each check
cannot see. The moment that teaches it: asked who first walked on the Moon, with evidence about the Apollo program
that never names him, the model answers *Neil Armstrong* and cites a real chunk, so the check passes. What stops it is
one rule line that lets the model say the evidence does not answer.

Lab 01 ranked passages for a question; a ranked list is not an answer. This lab adds the **answer step**: the model
answers from the passages and cites them, and code checks the citations.

## What you will learn

- How to pack retrieved chunks as `[ID] text` evidence lines inside a token budget.
- How to call a chat model by hand, read the reply object, and set `temperature` through `chat_options()`.
- How to ask for citations and check the cited IDs in code, and why a valid ID does not prove a claim.
- How to let the model answer *Insufficient evidence.*, and what that rule costs on a question it should answer.
- How to get claims with quotes as JSON, check the reply's shape, and check each quote against its evidence line.
- How a line planted in the index gets past every one of these checks.

**In class:** Parts A to D, except the two subsections marked self-study inside them (what `chat_options()` sends to
a reasoning model, in Part B; the rule measured on the 30 development questions, in Part D). Parts E, F and G are
self-study.

## Coming from

Lab 01 built the search and `shared/search.py`; you can rank the corpus for a question and you know a score is not
evidence. This lab adds the step after the search: the model answers from the chunks, and code checks what it cites.
It stays in plain Python with the OpenAI SDK, so the request, the JSON schema and every check are visible.

## What you will do

**Part A: Pack the evidence for one question.** Search the chunks and the five policies, then pack the best chunks as
`[ID] text` lines inside a 400-token budget.
**Part B: Answer from the evidence by hand.** `chat_client()` read and built, the two messages printed before they are
sent, one Chat Completions call with the reply object read field by field, your token count beside `usage`, and
`chat_options()` used. Self-study subsection: what `chat_options()` sends to a reasoning model.
**Part C: Ask for citations and check the IDs.** One rule line with an example, a regular expression that checks the
cited IDs, and the Moon question, answered with a valid ID.
**Part D: Let the model say the evidence does not answer.** The abstention line, measured on three unanswerable
questions and one question that needs a step of reasoning. Self-study subsection: the rule on the 30 development
questions.
**Part E: Quote the evidence and check the quotes in code (self-study).** A JSON schema, the `JSON_MODE` switch, `parse()`, a quote
check that also accepts a quote cut with `...` when its parts are in order, and the Moon question again with every
quote found.
**Part F: Test a line planted in the index (self-study).** A fake policy update, ranked first and followed in every run, with or
without tags; what each check reads.
**Part G: Keep the answer step in `shared/answer.py` (self-study).** The same steps as functions, used from here on.
**Going further.** An instruction planted in the evidence, which the model refuses to follow and still cannot answer
past.
**Challenge.** Test the answer step on four questions of your own. Optional: flag names the evidence does not contain.

## Checkpoints

| After | You should see |
|---|---|
| Part A | 277 chunks; Tesla's evidence: four lines, 316 tokens |
| Part B | *Nikola Tesla was born on 10 July 1856.*; your count 342, billed 353; `chat_options` sends `temperature: 0` |
| Part C | Tesla with `[S03-P00-C00]`; the Moon question: *Neil Armstrong* with a valid ID in 5 of 5 |
| Part D | unanswerable questions abstain 5 of 5 each; the temporary-accounts question refused 5 of 5 with the rule, 0 of 5 without; self-study: 0 of 30 development questions refused |
| Part E | the cut quote `Nikola Tesla ... 10 July 1856` found, its parts swapped not found; 32 claims on the 30 development questions, 0 quotes not found; without the abstention line, the Moon question names Armstrong in 5 of 5 with all 9 quotes found |
| Part F | the planted H99 ranks first; *24 hours* in 5 of 5, every quote found in H99; with tags, 5 of 5 |
| Part G | the module: Tesla answered with its quote found, *Insufficient evidence.* for the Moon |

The saved outputs come from `gpt-4o-mini` and `text-embedding-3-small` at `https://api.openai.com/v1` on 26 Oct 2026
(macOS, Python 3.13.5, a fresh `uv sync`). On 24 Oct 2026 the notebook ran on the `pip` route (Python 3.12.11) and with
`gpt-6-luna` and `REASONING_EFFORT=none`. Model replies vary between runs: compare the pattern (which answers
abstain, which checks pass), not every word.

**On the other routes.** The `pip` run showed the same pattern in every part. With
`gpt-6-luna` the lab runs unchanged, and the model fails in different places: the Moon question named Armstrong in 4 of
5 runs in Part C, each time without a citation, so the ID check failed it (1 to 3 of 5 cited in earlier runs); the abstention line did not refuse the temporary-accounts question (0 of 5);
without the abstention line its JSON answers to the Moon question named him in 3 of 5, only once with a quote (in
earlier runs 0 of 5), so Part E's blind spot appeared once; and it ignored the planted order in Going further (30 minutes, 5 of 5). The planted
policy update in Part F fooled both models in 5 of 5 runs. Which failures you see depends on the model; the checks and
the way to measure them do not.

## Before you start

- **Lab 00 first,** then Lab 01: this lab uses `shared/search.py` and the cached chunk vectors.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `02_cited_answers.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** `data/squad-chunks.json`, `data/support-case.json` and `data/development-questions.json`, read in place.
  The planted line in Part F exists only in the notebook; the data files are not changed.
- **Cost:** about 120 short chat calls and 280 embeddings, well under one US cent on `gpt-4o-mini`.

The notebook ends with a troubleshooting table: `temperature` on a reasoning model, a strict schema without
`additionalProperties`, `json_object` without the word JSON, and a reply that is not JSON.

## Resources

- [OpenAI: create chat completion](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create):
  messages, `temperature`, `reasoning_effort`, `response_format` and `usage`
- [OpenAI: structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs): strict JSON schemas,
  what they support, and JSON mode
- [OWASP: LLM01 prompt injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/): indirect injection through
  retrieved content, and the layered controls
- [Joren et al., *Sufficient Context*, ICLR 2025](https://arxiv.org/abs/2411.06037): why models answer instead of
  abstaining when the context is not enough

Next: [Lab 03](../03_Chunking_With_Lineage/README.md) Part A (in class; the rest of Lab 03 is self-study), then [Lab 04](../04_Trace_And_Diagnose/README.md) (find the stage where an answer failed). Back to [the lab list](../README.md).
