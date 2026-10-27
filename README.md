# Retrieval-Augmented Generation for NLP: labs

These labs are for developers and data scientists who write Python and have called a language model, and now want
to build retrieval-augmented generation (RAG) that they can trust. Across the labs you build one system: it searches
a small corpus (120 Wikipedia paragraphs and five product policies), answers with citations or refuses, and comes
with the measurements that show how well it works. Each lab adds one stage, measures it on the same questions, and
keeps what it built in `shared/`, so you can open any lab on its own.

## The labs

| Lab | What you do | Core parts |
|---|---|---|
| [00 Environment check](00_Environment_Check/README.md) | Set up your route, check it in one run, and see which models and endpoint the labs use | Run first |
| [01 Search with embeddings](01_Search_With_Embeddings/README.md) | Rank the corpus for a question, measure the ranking, and see what a score cannot tell you | A, B, C, E |
| [02 Cited answers](02_Cited_Answers/README.md) | Answer with quoted sources or refuse, check the citations in code, and see what the checks miss | A to D |
| [03 Chunking](03_Chunking_With_Lineage/README.md) | Load a Markdown page and a PDF and keep where each line came from, split documents so a rule keeps its exception, and choose a chunking from measurements | A |
| [04 Trace and diagnose](04_Trace_And_Diagnose/README.md) | Trace every question and find the stage where an answer failed | A, D |
| [05 Rebuild with LangChain](05_Rebuild_With_LangChain/README.md) | Rebuild the same pipeline with LangChain, compare it with your own code step by step, and find the defaults that make its answers differ | A, C, D (demo) |
| [06 Evaluation set](06_Evaluation_Set/README.md) | Write a golden set and score retrieval and answers per question | A to D |
| [07 LLM judge](07_LLM_Judge_Calibration/README.md) | Label answers yourself, then check a judge against your labels | B to E |
| [08 Hybrid search and reranking](08_Hybrid_Search_And_Reranking/README.md) | Add BM25, rank fusion and a reranker, and decide from paired results | B to E |
| [09 Follow-up questions](09_Follow_Up_Questions/README.md) | Turn a follow-up into a search that finds the right evidence | A to C |
| [10 Long context or retrieval](10_Long_Context_Or_Retrieval/README.md) | Compare the whole corpus in the prompt with retrieval, in answers, tokens and cost | Self-study |
| [11 Release gate](11_Release_Gate/README.md) | Filter by permission, update a policy, and stop a change that fails a release test | A, B |
| [Capstone](Capstone_RAG_Service/README.md) | Put the system behind a small web page with its release gate | Self-study |
| [Bonus](Bonus_Agentic_Search/README.md) | Let the model decide when to search | Self-study |
| [Optional A: LlamaIndex](Optional_LlamaIndex_Pipeline/README.md) | Build the same pipeline with LlamaIndex and find the defaults it sets for you (its own environment) | Self-study |
| [Optional B: Managed file search](Optional_Managed_File_Search/README.md) | Try a managed search service instead of your own index, and see what it hides (OpenAI only) | Self-study |
| [Final challenge: rescue a failing assistant](Challenge_Rescue_A_Failing_Assistant/README.md) | Take one failing question, find the stage, change it, and show the before-and-after on all 38 golden records | Optional, after Lab 11 |

The first five labs build every step in plain Python, so you can see what each one does. Lab 05 switches to
LangChain, which the later labs use for models, documents and search; measuring and checking stays in plain Python.

Every lab is a guided notebook you can work through alone. The core parts (letters are the lab's parts) are the ones worked
through in class; every other part is marked *(self-study)* and can be done at your own pace.

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

## Data

The Wikipedia paragraphs and the questions come from SQuAD 1.1, the Stanford Question Answering Dataset, a well-known public benchmark for question answering. The labs use a small sample of it (120 paragraphs and 30 questions), so their numbers are not benchmark scores. The five Atlas and Boreal product policies are made up for these labs. Both are in [`data/`](data/README.md), with their sources and licence (CC BY-SA 4.0 for the SQuAD passages).
