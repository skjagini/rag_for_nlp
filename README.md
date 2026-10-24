# Retrieval-Augmented Generation for NLP: labs

These labs are for developers and data scientists who write Python and have called a language model, and now want
to build retrieval-augmented generation (RAG) that they can trust. Across the labs you build one system: it searches
a small corpus (120 Wikipedia paragraphs and five product policies), answers with citations or refuses, and comes
with the measurements that show how well it works. Each lab adds one stage, measures it on the same questions, and
keeps what it built in `shared/`, so you can open any lab on its own.

**Status: built, in review.** Labs 00 to 11, the capstone, the bonus lab, the two optional labs and the final challenge guide are built.

## The labs

| Lab | What you do | In class or self-study |
|---|---|---|
| [00 Environment check](00_Environment_Check/README.md) | Set up your route, check it in one run, and see which models and endpoint the labs use | Before Day 1 |
| [01 Search with embeddings](01_Search_With_Embeddings/README.md) | Rank the corpus for a question, measure the ranking, and see what a score cannot tell you | In class: A, B, C, E |
| [02 Cited answers](02_Cited_Answers/README.md) | Answer with quoted sources or refuse, check the citations in code, and see what the checks miss | In class: A to D |
| [03 Chunking](03_Chunking_With_Lineage/README.md) | Load a Markdown page and a PDF and keep where each line came from, split documents so a rule keeps its exception, and choose a chunking from measurements | In class: A |
| [04 Trace and diagnose](04_Trace_And_Diagnose/README.md) | Trace every question and find the stage where an answer failed | In class: A, D |
| [05 Rebuild with LangChain](05_Rebuild_With_LangChain/README.md) | Rebuild the same pipeline with LangChain, compare it with your own code step by step, and find the defaults that make its answers differ | Instructor demo: A, C, D |
| [06 Evaluation set](06_Evaluation_Set/README.md) | Write a golden set and score retrieval and answers per question | In class: A to D |
| [07 LLM judge](07_LLM_Judge_Calibration/README.md) | Label answers yourself, then check a judge against your labels | In class: B to E |
| [08 Hybrid search and reranking](08_Hybrid_Search_And_Reranking/README.md) | Add BM25, rank fusion and a reranker, and decide from paired results | In class: A to D, the decision in G |
| [09 Follow-up questions](09_Follow_Up_Questions/README.md) | Turn a follow-up into a search that finds the right evidence | In class: A to C |
| [10 Long context or retrieval](10_Long_Context_Or_Retrieval/README.md) | Compare the whole corpus in the prompt with retrieval, in answers, tokens and cost | Self-study |
| [11 Release gate](11_Release_Gate/README.md) | Filter by permission, update a policy, and stop a change that fails a release test | In class: A, B |
| [Capstone](Capstone_RAG_Service/README.md) | Put the system behind a small web page with its release gate | Self-study |
| [Bonus](Bonus_Agentic_Search/README.md) | Let the model decide when to search | Self-study |
| [Optional A: LlamaIndex](Optional_LlamaIndex_Pipeline/README.md) | Build the same pipeline with LlamaIndex and find the defaults it sets for you (its own environment) | Self-study |
| [Optional B: Managed file search](Optional_Managed_File_Search/README.md) | Try a managed search service instead of your own index, and see what it hides (OpenAI only) | Self-study |
| [Final challenge: rescue a failing assistant](Challenge_Rescue_A_Failing_Assistant/README.md) | Take one failing question, find the stage, change it, and show the before-and-after on all 38 golden records | Optional, after Day 2 |

The first five labs build every step in plain Python, so you can see what each one does. Lab 05 switches to
LangChain, which the later labs use for models, documents and search; measuring and checking stays in plain Python.

Every lab is a guided notebook you can work through alone; the last column is a suggestion. An in-class lab names its core parts (letters are the lab's parts); its other parts are marked *(self-study)* in the lab and can be done after class.

## Setup

You need Python 3.11 or later (3.12 or 3.13 recommended) and an OpenAI API key. Nothing downloads a model.

- **Local (VS Code), with uv:** clone the repository (or download the ZIP), open the folder, run `uv sync`, copy
  `.env.example` to `.env` and paste your key after `OPENAI_API_KEY=`. Choose the `.venv` kernel in each notebook.
- **Local, with pip** (for machines that block uv): `python -m venv .venv`, activate it, then
  `pip install -r requirements.txt ipykernel`, and the same `.env` step.
- **GitHub Codespaces:** create a codespace from the repository with **New with options** and enter your key under
  the recommended secret `OPENAI_API_KEY`. The container installs everything.
- **Ubuntu VM (22.04 or 24.04):** as local, with uv (22.04's own Python is too old for the pip route).
- **Google Colab:** use the badge in a lab's README; add your key under Secrets (the key icon) as `OPENAI_API_KEY`.

Lab 00's README has every step for each route and a troubleshooting table.

## Models and endpoint

Every model name and the API address are in [`models.env`](models.env), one line each. To use another model, or
another OpenAI-compatible server, change that line (or override it in your `.env`); no notebook names a model. Every
result a lab saves records the model and endpoint that produced it.

## How to read a lab

Each lab is a guided notebook. Read the markdown even if you skip the code: it explains what each step does and what
the output means. Each lab ends with a short challenge and a list of resources.

## How these labs relate to three RAG books and one app collection

Three RAG books publish their code on GitHub: [*Hands-On RAG for Production*](https://github.com/ofermend/hands-on-rag)
(O'Reilly), [*RAG from First Principles*](https://github.com/PacktPublishing/RAG-from-First-Principles) (Packt) and
[*Unlocking Data with Generative AI and RAG*, 2nd edition](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition)
(Packt). Their repositories are code companions: the book holds the explanation and the notebooks hold short code.
These notebooks carry the explanation themselves. [Awesome LLM Apps](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/rag_tutorials)
is a collection of small runnable RAG apps, each on its own stack, worth reading to see how others put the pieces
together. Each lab's Resources section links the parts that cover the same idea, with a line on what you will find:

| Lab | *Hands-On RAG* | *RAG from First Principles* | *Unlocking Data* | Awesome LLM Apps |
|---|---|---|---|---|
| 01 | [`chapter2/embedding.ipynb`](https://github.com/ofermend/hands-on-rag/blob/main/chapter2/embedding.ipynb) | [`03-Embedding`](https://github.com/PacktPublishing/RAG-from-First-Principles/tree/main/03-Embedding) | [`CHAPTER_08/CHAPTER8-1_DISTANCEMETRICS.ipynb`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/blob/main/CHAPTER_08/CHAPTER8-1_DISTANCEMETRICS.ipynb) |  |
| 02 | [`chapter1/sample-rag.ipynb`](https://github.com/ofermend/hands-on-rag/blob/main/chapter1/sample-rag.ipynb), [`chapter3/hallucinations.ipynb`](https://github.com/ofermend/hands-on-rag/blob/main/chapter3/hallucinations.ipynb) | [`08-Generation`](https://github.com/PacktPublishing/RAG-from-First-Principles/tree/main/08-Generation) | [`CHAPTER_03/CHAPTER3-1_SOURCES.ipynb`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/blob/main/CHAPTER_03/CHAPTER3-1_SOURCES.ipynb), [`CHAPTER_05`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/tree/main/CHAPTER_05) |  |
| 03 | [`chapter2/parse-pdf.ipynb`](https://github.com/ofermend/hands-on-rag/blob/main/chapter2/parse-pdf.ipynb), [`chapter8/multi-page-tables.ipynb`](https://github.com/ofermend/hands-on-rag/blob/main/chapter8/multi-page-tables.ipynb), [`chapter8/simple-table-chunking.ipynb`](https://github.com/ofermend/hands-on-rag/blob/main/chapter8/simple-table-chunking.ipynb) | [`01-DataLoading/04-PDFFileLoading`](https://github.com/PacktPublishing/RAG-from-First-Principles/tree/main/01-DataLoading/04-PDFFileLoading), [`02-DocChunking`](https://github.com/PacktPublishing/RAG-from-First-Principles/tree/main/02-DocChunking) | [`CHAPTER_11/CHAPTER11-1_DOCUMENT_LOADERS.ipynb`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/blob/main/CHAPTER_11/CHAPTER11-1_DOCUMENT_LOADERS.ipynb) |  |
| 04 |  | [`06-Indexing/01-FromSmallChunksToLargeContext`](https://github.com/PacktPublishing/RAG-from-First-Principles/tree/main/06-Indexing/01-FromSmallChunksToLargeContext), [`07-PostRetrieval/03-Correction`](https://github.com/PacktPublishing/RAG-from-First-Principles/tree/main/07-PostRetrieval/03-Correction) |  | [`rag_failure_diagnostics_clinic`](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/rag_tutorials/rag_failure_diagnostics_clinic) |
| 05 |  | [`00-SimpleRAG`](https://github.com/PacktPublishing/RAG-from-First-Principles/tree/main/00-SimpleRAG) | [`CHAPTER_02/CHAPTER2-1_RAG_PIPELINE.ipynb`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/blob/main/CHAPTER_02/CHAPTER2-1_RAG_PIPELINE.ipynb) |  |
| 06 | [`chapter6/metrics.ipynb`](https://github.com/ofermend/hands-on-rag/blob/main/chapter6/metrics.ipynb) | [`09-Evaluation`](https://github.com/PacktPublishing/RAG-from-First-Principles/tree/main/09-Evaluation) | [`CHAPTER_09`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/tree/main/CHAPTER_09) |  |
| 07 | [`chapter6/llm-as-a-judge.ipynb`](https://github.com/ofermend/hands-on-rag/blob/main/chapter6/llm-as-a-judge.ipynb), [`chapter6/umbrela.ipynb`](https://github.com/ofermend/hands-on-rag/blob/main/chapter6/umbrela.ipynb) |  |  |  |
| 08 | [`chapter3/reranking.ipynb`](https://github.com/ofermend/hands-on-rag/blob/main/chapter3/reranking.ipynb) | [`03-Embedding/03-BM25.py`](https://github.com/PacktPublishing/RAG-from-First-Principles/blob/main/03-Embedding/03-BM25.py), [`07-PostRetrieval/01-Reranking`](https://github.com/PacktPublishing/RAG-from-First-Principles/tree/main/07-PostRetrieval/01-Reranking) | [`CHAPTER_08`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/tree/main/CHAPTER_08) | [`hybrid_search_rag`](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/rag_tutorials/hybrid_search_rag) |
| 09 |  | [`05-PreRetrieval/02-QueryTranslation`](https://github.com/PacktPublishing/RAG-from-First-Principles/tree/main/05-PreRetrieval/02-QueryTranslation) | [`CHAPTER_12`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/tree/main/CHAPTER_12) |  |
| 10 | [`chapter4/caching.ipynb`](https://github.com/ofermend/hands-on-rag/blob/main/chapter4/caching.ipynb) |  | [`CHAPTER_15`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/tree/main/CHAPTER_15) |  |
| 11 | [`chapter4/redaction.ipynb`](https://github.com/ofermend/hands-on-rag/blob/main/chapter4/redaction.ipynb) | [`07-PostRetrieval/01-Reranking/06-RecencyWeightedReranking.py`](https://github.com/PacktPublishing/RAG-from-First-Principles/blob/main/07-PostRetrieval/01-Reranking/06-RecencyWeightedReranking.py) | [`CHAPTER_05`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/tree/main/CHAPTER_05) |  |
| Capstone |  |  | [`CHAPTER_06/CHAPTER6-1_GRADIO.ipynb`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/blob/main/CHAPTER_06/CHAPTER6-1_GRADIO.ipynb) | [`rag_chain`](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/rag_tutorials/rag_chain) |
| Bonus |  | [`10-AdvanceRAG/04-AgenticRAG`](https://github.com/PacktPublishing/RAG-from-First-Principles/tree/main/10-AdvanceRAG/04-AgenticRAG) | [`CHAPTER_12`](https://github.com/PacktPublishing/Unlocking-Data-with-Generative-AI-and-RAG-Second-Edition/tree/main/CHAPTER_12) |  |
| Optional A |  | [`00-SimpleRAG`](https://github.com/PacktPublishing/RAG-from-First-Principles/tree/main/00-SimpleRAG) |  |  |
| Optional B | [`chapter5`](https://github.com/ofermend/hands-on-rag/tree/main/chapter5) |  |  |  |

## Data

The Wikipedia paragraphs and the questions come from SQuAD 1.1, the Stanford Question Answering Dataset, a well-known public benchmark for question answering. The labs use a small sample of it (120 paragraphs and 30 questions), so their numbers are not benchmark scores. The five Atlas and Boreal product policies are made up for these labs. Both are in [`data/`](data/README.md), with their sources and licence (CC BY-SA 4.0 for the SQuAD passages).
