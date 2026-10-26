# Lab 03: How to split documents into chunks that keep their facts

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/03_Chunking_With_Lineage/03_chunking_with_lineage.ipynb)

**CPU | API key: `OPENAI_API_KEY` (embedding and chat calls) | Practises: loading source files and preparing chunks
that keep their source and their qualifications | In class**

## Goal

Load source files into records that say where each piece of text came from, and check what the parser handed over
before anything is searched. One moment teaches it. A one-page PDF holds a small table, and the parser gives the table
back as loose lines: the row *Boreal Mobile app 10 minutes* keeps its words and loses its header. Asked how long a
Boreal reset link is valid, the search ranks that row below the header line, which has no number in it. Written as one
sentence with its header, the same row ranks first. **The parser decides what the search can find.** The self-study
parts then cut text into chunks and measure what a cut can lose.

## What you will learn

- How to load a Markdown page and a PDF, page by page, into records that keep their file, heading or page and their
  character offsets (their lineage), and check that each policy sentence came through word for word.
- How a table read out of a PDF loses its header, and how writing each row as a sentence with its header changes what
  a search finds.
- How to cut a document into sentences or token windows, keep each chunk's source ID and offsets, and check that no
  text was lost.
- How to split a Markdown handbook at its headings and blank lines, and how to split by structure first under one
  token budget.
- How to compare chunkings on hit@3, tokens per question and answers to questions that cross a boundary.
- How a title header on each chunk changes what a search finds.

**In class:** Part A up to Checkpoint A. The word-for-word check at the end of Part A and Parts B to G are self-study.

## Coming from

[Lab 01](../01_Search_With_Embeddings/) searched 272 chunks that were already cut; this lab starts from source files,
then shows how chunks are cut and what a cut can lose. It uses Lab 01's `shared/search.py` and [Lab 02](../02_Cited_Answers/)'s `shared/answer.py`
by name, so read those labs first if the functions are new to you; Part A needs only Lab 01's search. Nothing in the
labs after this one depends on it: they keep the frozen 272 chunks as their baseline.

## What you will do

**Part A: Load a Markdown page and a PDF.** An Atlas help page split at its headings into four records, and a
one-page comparison sheet read with `pypdf` and split into lines, each record with its file, heading or page and
offsets; the sheet's table as loose lines, then its rows written as sentences with their header (table-to-text), and
both versions searched for *How long is a Boreal reset link valid?*, with embedding calls only. A self-study check at
the end finds every policy sentence in the two files word for word.
**Part B: Split the policies into sentences (self-study).** Two policy questions answered from sentence chunks and from whole
policies, five runs each, and the rankings that explain the difference.
**Part C: Cut token windows and check their offsets (self-study).** Windows of 96 tokens every 80, what their overlap repeats,
offsets from the tokenizer, and a check that every chunk slices back with no text lost.
**Part D: Split a Markdown handbook at its headings (self-study).** A real document file, split into sections at its headings and
into paragraphs at its blank lines, each chunk with its file and offsets, and the two splits measured on 24 labelled
questions.
**Part E: Split by structure first (self-study).** Whole documents, then sentences joined up to a budget, then windows for a
sentence that is too long.
**Part F: Compare four chunkings (self-study).** Passages, windows, sentences and structured chunks on hit@3, tokens per question and
the policy answers.
**Part G: Add the title to each chunk (self-study).** The same four with a title header, the handbook's paragraphs with their
headings, which one to keep, and which chunks the pipeline keeps.
**Going further.** LangChain's `RecursiveCharacterTextSplitter` beside the hand version, and context written for each
chunk by a chat model, with its calls counted.
**Challenge.** Keep the sentence before each sentence and measure it. Optional: change the structure-first budget.

## Checkpoints

| After | You should see |
|---|---|
| Part A | the help page as 4 records (`atlas-help-S0` 158 to 274 ...); the sheet as 8 lines, the table's header on its own; `vectors: (12, 1536)`; *Boreal Mobile app 10 minutes* fifth (0.4928) as a line, first (0.7109) as a sentence with its header |
| Part A, self-study check | 8 of 10 policy sentences in the help page, each in its section; H04's 2 on page 1 of the sheet |
| Part B | sentences: 0 of 5 right for each question (first replies *Atlas passwords can be reset by email or through a mobile app.* and *Insufficient evidence.*); whole policies: 5 of 5 each |
| Part C | the Tesla paragraph, 114 tokens, as two windows (0 to 509 and 417 to 607); 274 windows, 612 sentences and the frozen 272, none losing a character |
| Part D | the handbook, 1,590 tokens: 20 sections and 67 paragraphs, *Escalate when:* alone in a chunk; right first 21 and 23 of 24, top 3 24 of 24 for both; 241 and 61 tokens a question |
| Part E | H01 kept whole; Tesla in two chunks of whole sentences; 288 chunks, 16 documents kept whole |
| Part F | hit@3 30/30 for all four; tokens per question 435, 256, 110 and 228; policy answers 10, 10, 0 and 10 of 10 |
| Part G | with titles, policy answers 10 of 10 for all four; tokens per question passage 446, window 266, sentence 122, structured 232; handbook paragraphs with headings 23 of 24 first, 71 tokens |
| Going further | 7 chunks when sized in characters, 2 from the tokenizer; 10 context calls (about 880 tokens), policy answers 10 of 10 |

The saved outputs come from `gpt-4o-mini` and `text-embedding-3-small` at `https://api.openai.com/v1` on
26 Oct 2026 (macOS, Python 3.13.5, a fresh `uv sync`). Model replies vary between runs: the abstention line now and
then refuses a question that takes a step of reasoning (Lab 02 measured it), so a row that is 10 of 10 here can be 9 of
10 in your run. Compare the pattern, which chunking fails which question, not every word. Part A makes no chat call:
its records, checks and ranks do not depend on the chat model.

**On the other routes.** The `pip` runs (Python 3.12.11, 24 Oct 2026) showed the same pattern in every part, and Part
A's records, checks and ranks were the same within 0.0001. On `gpt-6-luna` (`REASONING_EFFORT=none`) the lab runs
unchanged; the sentence chunks fail differently (5 of 10 in Part F, the temporary-account question answered wrongly
instead of refused), and with titles every chunking is 10 of 10 on both models. Which failure a cut causes depends on
the model; that a cut can lose an exception does not.

## Before you start

- **Lab 00 first,** then Labs 01 and 02: this lab uses `shared/search.py` and `shared/answer.py` by name.
- **Open the notebook:** on Colab with the badge above (add `OPENAI_API_KEY` under Secrets first); locally or in a
  codespace, open `03_chunking_with_lineage.ipynb` and choose the `.venv` kernel. Run Setup and Config first.
- **Source documents (Part A):** `resources/source-documents/` holds two files constructed for this lab, whose every
  fact restates the five policies in `data/support-case.json`: `atlas-help.md`, an Atlas help page with the sentences
  of H01, H02, H03 and H05 word for word, and `password-reset-comparison.pdf`, a one-page sheet with H04's sentences and
  a table (*Product / Reset method / Link valid for*) whose cells restate H01, H02 and H04. `make_pdf.py` beside them
  writes the PDF (`uv run --no-project --with fpdf2 python make_pdf.py` from that folder; fpdf2 2.8.9 is not a lab
  dependency). SHA-256: `atlas-help.md` `ab4ce38f4b3cd195c2f0702bfc5d16933af1068da657441e674890936cb9a6c4`,
  `password-reset-comparison.pdf` `65c29812282a999c9cf3e47ca546724e4426ef6c7460014805a344d0294372eb`. Part A reads the
  PDF with [pypdf](https://pypi.org/project/pypdf/) (pure Python, BSD licence), installed with the other packages.
- **Data:** `data/squad-passages.json`, `data/squad-chunks.json`, `data/support-case.json` and
  `data/development-questions.json`, read in place and never changed. Part D reads this lab's `resources/`:
  `support_handbook.md`, the support handbook of Chinook, a made-up music shop, copied unchanged from another of the
  author's lab repositories (customer and staff names from the MIT-licensed [Chinook sample database](https://github.com/lerocha/chinook-database));
  its first line names the build script it came from, which is not in this repo (SHA-256
  `d27874b858aecefed0116e5b520d614592c408863dc3085ddd73671e34effaea`, the same as the original),
  and `handbook-questions.json`, 24 questions constructed for this lab and labelled before any search. Each chunking's vectors are saved in `.cache/`,
  which git ignores.
- **Cost:** Part A embeds a few hundred tokens and makes no chat call. The whole notebook makes about 120 short chat
  calls and 180,000 embedding tokens, well under one US cent on `gpt-4o-mini`.

The notebook ends with a troubleshooting table: a misspelt `TOKEN_ENCODING`, a chunk whose text no longer matches its
offsets, a chunk longer than the embedding model accepts, and a splitter sized in characters.

## Resources

- [pypdf: extract text from a PDF](https://pypdf.readthedocs.io/en/stable/user/extract-text.html): `extract_text()`
  and its `layout` mode, used in Part A
- [OpenAI: tiktoken](https://github.com/openai/tiktoken): encodings, `encode` and `decode_with_offsets`
- [LangChain: text splitters](https://docs.langchain.com/oss/python/integrations/splitters): the recursive splitter
  and splitting by tokens
- [LangChain: split Markdown by headers](https://docs.langchain.com/oss/python/integrations/splitters/markdown_header_metadata_splitter):
  `MarkdownHeaderTextSplitter`, Part D's split at headings with each heading kept as metadata
- [Chroma: evaluating chunking strategies for retrieval](https://www.trychroma.com/research/evaluating-chunking): a
  measured comparison of token, recursive and semantic chunkers
- [Qu et al., *Is semantic chunking worth the computational cost?*, NAACL Findings 2025](https://arxiv.org/abs/2410.13070):
  semantic splitting against fixed-size chunks
- [Anthropic: contextual retrieval](https://www.anthropic.com/news/contextual-retrieval): context written for each
  chunk before embedding, with the vendor's measurements

Next: [Lab 04](../04_Trace_And_Diagnose/README.md) (find the stage where an answer failed). Back to [the lab list](../README.md).
