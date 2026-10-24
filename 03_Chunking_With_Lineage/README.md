# Lab 03: How to split documents into chunks that keep their facts

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/03_Chunking_With_Lineage/03_chunking_with_lineage.ipynb)

**CPU | API key: `OPENAI_API_KEY` (embedding and chat calls) | Practises: preparing chunks that keep their source and
their qualifications | Self-study**

## Goal

Split documents into chunks that point back to their source, and choose how to split them by measuring, not by habit.
The moment that teaches it: cut five short product policies into single sentences, and answers about one product pick
up another product's rule or miss an exception, because a sentence on its own no longer says which product it belongs
to. The usual test questions find the right paragraph whatever the split, so on their own they would have chosen the
split that broke these answers.

## What you will learn

- How to cut a document into sentences or token windows, and keep each chunk's source ID and character offsets.
- How to check that every chunk slices back to its source and that no text was lost.
- How to split a Markdown file at its headings and blank lines, keeping each chunk's file and offsets.
- How to split by structure first: whole documents, then sentences, then windows, under one token budget.
- How to compare chunkings on hit@3, tokens per question and answers to questions that cross a boundary.
- How a title header on each chunk changes what a search finds.

## Coming from

[Lab 01](../01_Search_With_Embeddings/) searched 272 chunks that were already cut; this lab shows how chunks are cut,
and what a cut can lose. It uses Lab 01's `shared/search.py` and [Lab 02](../02_Cited_Answers/)'s `shared/answer.py`
by name, so read those labs first if the functions are new to you. Nothing in the labs after this one depends on it:
they keep the frozen 272 chunks as their baseline.

## What you will do

**Part A: Split the policies into sentences.** Two policy questions answered from sentence chunks and from whole
policies, five runs each, and the rankings that explain the difference.
**Part B: Cut token windows and check their offsets.** Windows of 96 tokens every 80, what their overlap repeats,
offsets from the tokenizer, and a check that every chunk slices back with no text lost.
**Part C: Split a Markdown handbook at its headings.** A real document file, split into sections at its headings and
into paragraphs at its blank lines, each chunk with its file and offsets, and the two splits measured on 24 labelled
questions.
**Part D: Split by structure first.** Whole documents, then sentences joined up to a budget, then windows for a
sentence that is too long.
**Part E: Compare four chunkings.** Passages, windows, sentences and structured chunks on hit@3, tokens per question and
the policy answers.
**Part F: Add the title to each chunk.** The same four with a title header, the handbook's paragraphs with their
headings, which one to keep, and which chunks the pipeline keeps.
**Going further.** LangChain's `RecursiveCharacterTextSplitter` beside the hand version, and context written for each
chunk by a chat model, with its calls counted.
**Challenge.** Keep the sentence before each sentence and measure it. Optional: change the structure-first budget.

## Checkpoints

| After | You should see |
|---|---|
| Part A | sentences: 0 of 5 right for each question (first replies *by email or through a mobile app* and *Insufficient evidence.*); whole policies: 5 of 5 each |
| Part B | the Tesla paragraph, 114 tokens, as two windows (0 to 509 and 417 to 607); 274 windows, 612 sentences and the frozen 272, none losing a character |
| Part C | the handbook, 1,590 tokens: 20 sections and 67 paragraphs, *Escalate when:* alone in a chunk; right first 21 and 23 of 24, top 3 24 of 24 for both; 241 and 61 tokens a question |
| Part D | H01 kept whole; Tesla in two chunks of whole sentences; 288 chunks, 16 documents kept whole |
| Part E | hit@3 30/30 for all four; tokens per question 435, 256, 110 and 228; policy answers 10, 10, 0 and 10 of 10 |
| Part F | with titles, policy answers 10 of 10 for all four; sentences with titles 122 tokens per question; handbook paragraphs with headings 23 of 24 first, 71 tokens |
| Going further | 7 chunks when sized in characters, 2 from the tokenizer; 10 context calls (869 tokens), policy answers 10 of 10 |

The saved outputs come from `gpt-4o-mini` and `text-embedding-3-small` at `https://api.openai.com/v1` on 24 Oct 2026
(macOS, Python 3.13.5, a fresh `uv sync`). Model replies vary between runs: the abstention line now and then refuses a question that takes a step of reasoning
(Lab 02 measured it), so a row that is 10 of 10 here can be 9 of 10 in your run (the structured row was, in our
run of the solved notebook). Compare the pattern, which chunking fails which question, not every word.

**On the other routes.** The `pip` runs (Python 3.12.11, the same package versions, 18 and 24 Oct 2026) showed the same pattern in every part; there the abstention line's occasional refusal of the temporary-account question also came once in Part A's whole-policy run (4 of 5). With `CHAT_MODEL`, `JUDGE_MODEL` and `RERANK_MODEL` set to `gpt-6-luna` and `REASONING_EFFORT=none`, the lab runs unchanged and the sentence chunks fail differently. The first question is right 5 of 5: the model leaves the mobile-app sentence out. The temporary-account question is wrong 5 of 5 in a worse way; the first reply reads *Reset the password by email; the reset link expires after 30 minutes.*, an answer where `gpt-4o-mini` refused. Sentences score 5 of 10 in Part E; with titles every chunking gets 10 of 10 on both models, and hit@3, tokens and the lineage checks are identical. Which failure a cut causes depends on the model; that a cut can lose an exception does not. Part C does not depend on the chat model, and gave the same counts on all three runs (24 Oct 2026): sections first for 21 of 24 and paragraphs for 23, all 24 in the top 3, 241 and 61 tokens a question.

## Before you start

- **Lab 00 first,** then Labs 01 and 02: this lab uses `shared/search.py` and `shared/answer.py` by name.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `03_chunking_with_lineage.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Data:** `data/squad-passages.json`, `data/squad-chunks.json`, `data/support-case.json` and
  `data/development-questions.json`, read in place and never changed. Part C reads this lab's `resources/`:
  `support_handbook.md`, the support handbook of Chinook, a made-up music shop, copied unchanged from an earlier
  course by the same author (customer and staff names from the MIT-licensed [Chinook sample database](https://github.com/lerocha/chinook-database)),
  and `handbook-questions.json`, 24 questions constructed for this lab and labelled before any search. Each chunking's vectors are saved in `.cache/`,
  which git ignores.
- **Cost:** about 120 short chat calls and 180,000 embedding tokens, well under one US cent on `gpt-4o-mini`.

The notebook ends with a troubleshooting table: a misspelt `TOKEN_ENCODING`, a chunk whose text no longer matches its
offsets, a chunk longer than the embedding model accepts, and a splitter sized in characters.

## Resources

- [OpenAI: tiktoken](https://github.com/openai/tiktoken): encodings, `encode` and `decode_with_offsets`
- [LangChain: text splitters](https://docs.langchain.com/oss/python/integrations/splitters): the recursive splitter
  and splitting by tokens
- [LangChain: split Markdown by headers](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter):
  `MarkdownHeaderTextSplitter`, Part C's split at headings with each heading kept as metadata
- [Chroma: evaluating chunking strategies for retrieval](https://www.trychroma.com/research/evaluating-chunking): a
  measured comparison of token, recursive and semantic chunkers
- [Qu et al., *Is semantic chunking worth the computational cost?*, NAACL Findings 2025](https://arxiv.org/abs/2410.13070):
  semantic splitting against fixed-size chunks
- [Anthropic: contextual retrieval](https://www.anthropic.com/news/contextual-retrieval): context written for each
  chunk before embedding, with the vendor's measurements

Next: [Lab 04](../04_Trace_And_Diagnose/README.md) (find the stage where an answer failed). Back to [the lab list](../README.md).
