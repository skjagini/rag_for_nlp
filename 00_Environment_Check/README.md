# Lab 00: How to check your lab environment

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tatwan/rag_for_nlp/blob/main/00_Environment_Check/00_environment_check.ipynb)

**CPU | API key: `OPENAI_API_KEY` | Practises: the setup every lab needs | Before Day 1**

## Goal

Get a working environment for every lab in this repository on your route, and prove it in one run. This README is
the setup guide: your own machine with `uv` or `pip`, GitHub Codespaces, an Ubuntu VM, or Colab. The notebook is the
check at the end. When its last cell shows ✅ on every line, you are ready for Lab 01, and you know which models and
endpoint the labs will use.

## What you will learn

- How to set up the labs on your route, from getting the repository to choosing the notebook's kernel.
- Where your key goes on each route, and how the labs find it without printing it.
- Where the model names and the API address are set, and how to change them for every lab or only on your machine.
- How to tell a blocked network from a key problem.
- How to read a failed check and fix it with the troubleshooting tables.

**Before Day 1:** set up your route and run the notebook on the machine you will use in class. If a line in its
last cell shows ❌, bring its output.

## Before you start

You need an OpenAI API key (your own, or the class key your instructor gives you) and, on every route except Colab
and Codespaces, a terminal. No route downloads a model or needs a GPU. The notebook sends five short requests with
your key; together they cost a small fraction of a US cent.

## Choose a route

| Route | Use it when | You need |
|---|---|---|
| **A. Local with `uv`** (recommended on your own machine) | you can install a command-line tool | VS Code with its Python and Jupyter extensions, Git or a ZIP download |
| **B. Local with `pip`** | your laptop does not allow `uv` | Python 3.11 to 3.14 already installed (3.12 or 3.13 is best), VS Code, Git or a ZIP |
| **C. GitHub Codespaces** | you cannot install anything, or your organisation works in Codespaces | a GitHub account; everything runs in the browser |
| **D. Ubuntu VM** (22.04 or 24.04) | your class gives you a Linux machine | a terminal on the VM; it then follows route A or B |
| **E. Colab** | Colab is allowed for your class | a Google account |

Routes A, C and D install the exact versions in `uv.lock`, the ones every saved output in these labs came from.
Routes B and E install from `requirements.txt`, which allows a range of versions; Part A of the notebook prints what
you got.

---

## Route A: Local with `uv`

[`uv`](https://docs.astral.sh/uv/) installs the right Python and the pinned packages into a folder called `.venv`
inside the repository, so nothing else on your machine changes.

**1. Install `uv`** (once per machine). macOS or Linux, in a terminal:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows, in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close the terminal and open a new one, then check that `uv --version` prints a version.

**2. Get the repository.** With Git:

```bash
git clone https://github.com/tatwan/rag_for_nlp.git
cd rag_for_nlp
```

Without Git, download the [ZIP](https://github.com/tatwan/rag_for_nlp/archive/refs/heads/main.zip), unzip it and
open a terminal in the unzipped folder, `rag_for_nlp-main`. With Git, `git pull` fetches later updates.

**3. Install Python and the packages:**

```bash
uv sync
```

`uv sync` reads `.python-version` (3.13) and `uv.lock`, downloads Python 3.13 if your machine does not have it, and
installs every package at its locked version into `.venv`. Run it again whenever `uv.lock` changes.

**4. Add your key.** Copy the example file, then open `.env` in a text editor and paste your key after
`OPENAI_API_KEY=` (no quotes, no spaces):

```bash
cp .env.example .env
```

On Windows: `copy .env.example .env`. `.env` stays on your machine: `.gitignore` keeps it out of Git. Never paste a
key into a notebook cell.

**5. Open the notebook in VS Code.** Open the repository folder (**File → Open Folder**), then
`00_Environment_Check/00_environment_check.ipynb`. Click **Select Kernel** at the top right, choose **Python
Environments**, then the `.venv` entry (Python 3.13). If VS Code offers to install the Python or Jupyter extension,
accept.

If you prefer JupyterLab in the browser, start it from the repository folder instead:

```bash
uv run --with jupyterlab jupyter lab
```

**6. Run the notebook** from the top (**Run All**), then read the last cell.

---

## Route B: Local with `pip`

Use this route when you cannot install `uv`. You bring your own Python.

**1. Check your Python.** You need 3.11 to 3.14:

```bash
python3 --version
```

On Windows: `py --version`. If you have none, install 3.13 from [python.org](https://www.python.org/downloads/).

**2. Get the repository**, as in route A, step 2.

**3. Create a virtual environment and install the packages.** A virtual environment is a folder that keeps this
repository's packages apart from every other Python project. From the repository folder, on macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt ipykernel
```

On Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt ipykernel
```

Your prompt shows `(.venv)` while the environment is active; activate it again in every new terminal. `ipykernel`
lets VS Code run notebooks in this environment. It is not in `requirements.txt` because Colab ships its own.

**4. Add your key**, as in route A, step 4.

**5. Open the notebook** in VS Code and choose the `.venv` kernel, as in route A, step 5.

**6. Run the notebook** from the top, then read the last cell.

---

## Route C: GitHub Codespaces

A codespace is a Linux machine in the cloud with VS Code in your browser. The repository's
`.devcontainer/devcontainer.json` sets it up: Python 3.13, `uv sync`, and the Python and Jupyter extensions.

**1. Create the codespace with your key.** On the repository's GitHub page, click **Code**, open the **Codespaces**
tab, click **...** and choose **New with options**. At the bottom of the page, under the recommended secrets, paste
your key into `OPENAI_API_KEY`, then click **Create codespace**. GitHub saves the key as one of your Codespaces
secrets and gives it to the codespace as an environment variable; the labs read it from there. You do not need a
`.env` file.

**2. Wait for the setup to finish.** The first start installs `uv` and runs `uv sync`; the terminal shows its
progress. Later starts reuse the installed packages.

**3. Open the notebook** `00_Environment_Check/00_environment_check.ipynb`, click **Select Kernel**, choose
**Python Environments**, then `.venv`.

**4. Run the notebook** from the top, then read the last cell.

If you add or change the secret later (GitHub: your picture → **Settings** → **Codespaces** → **Codespaces
secrets**), stop the codespace and start it again: a running codespace does not see a new secret. A codespace stops
by itself after a while without use and keeps your files; personal accounts have a monthly allowance of free hours,
so stop it when you finish (**Codespaces** menu at the bottom left → **Stop Current Codespace**).

---

## Route D: Ubuntu VM (22.04 or 24.04)

A VM works like your own machine. Install Git and `curl` first:

```bash
sudo apt update
sudo apt install -y git curl
```

**Ubuntu 22.04: use route A (`uv`).** Its system Python is 3.10, which reaches its end of life in October 2026 and is
below the labs' 3.11 floor. `uv sync` downloads Python 3.13 for the repository and leaves the system Python alone.

**Ubuntu 24.04: route A or route B.** Its system Python, 3.12, works. For route B, install the `venv` module first,
which Ubuntu ships as a separate package:

```bash
sudo apt install -y python3.12-venv
```

Then follow route B from step 2, with `python3`. Always install inside the activated `.venv`: Ubuntu 24.04 refuses
`pip install` into the system Python.

To open the notebook, connect VS Code on your laptop to the VM with the
[Remote - SSH](https://code.visualstudio.com/docs/remote/ssh) extension, open the repository folder on the VM, and
choose the `.venv` kernel as in route A, step 5. If you work on the VM's own desktop, open VS Code there.

---

## Route E: Colab

1. Open the notebook with the **Open in Colab** badge at the top of this README. Every lab's README has the same
   badge.
2. Add your key: click the key icon (**Secrets**) in Colab's left bar, add a secret named `OPENAI_API_KEY` with your
   key as its value, and switch on **Notebook access**.
3. Run the notebook from the top (**Runtime → Run all**). Its first cell copies this repository into the Colab
   session and installs the packages.

Each Colab session starts empty, so every lab's first cell copies the repository and installs the packages again.
Your key stays in Secrets.

---

## What the notebook checks

The notebook is a checklist: run every cell from the top, and read the last one.

**Setup and Config.** The two cells every lab opens with: they find the repository folder (on Colab they copy it and
install the packages first), read the settings and find your key without printing it.
**Part A: Check Python and the packages.** Python 3.11 to 3.14, and each package inside its range in
`requirements.txt`.
**Part B: Check the data files.** Every file in `data/` against its SHA-256 hash.
**Part C: Check that you can reach the endpoint.** One request without your key, to test the network alone.
**Part D: Check the tokenizer.** `tiktoken` loads its encoding, a one-time download from a second address.
**Part E: Check the key with one embedding.** One request to the embedding model with your key.
**Part F: Check the chat models.** One question to each chat model in the settings.
**Part G: Check JSON replies.** One answer in the JSON shape the labs ask for.
**Check that you are ready.** One ✅ or ❌ per check.

## Checkpoints

| After | You should see |
|---|---|
| Config | the stamp (models, endpoints, date) and `OPENAI_API_KEY found in ...` |
| Part A | Python 3.11 to 3.14; `outside their range: none` |
| Part B | `5 of 5 files match` |
| Part C | `https://api.openai.com/v1 answered: HTTP 401` (no key is sent, so 401 is a pass) |
| Part D | `o200k_base: 8 tokens` |
| Part E | 1536 numbers |
| Part F | a reply from each chat model, and the model that answered |
| Part G | an answer with its claims |
| The last cell | ✅ on every line |

The replies in Parts F and G can change in wording from run to run and from model to model.

## The models and endpoint every lab uses

No notebook names a model in its code. [`models.env`](../models.env), in the repository folder, sets the chat,
embedding, judge and reranking models and the API address for every lab. The shipped values are `gpt-4o-mini` and
`text-embedding-3-small` on OpenAI. To switch every lab, change a line there. To try another model or endpoint on your
machine only, put the same line in your `.env`: it wins over `models.env`, and an environment variable wins over
both. `gpt-4o-mini` is due to shut down on 14 January 2027. To run on its replacement, set `CHAT_MODEL`,
`JUDGE_MODEL` and `RERANK_MODEL` to `gpt-6-luna` and `REASONING_EFFORT=none`: the effort goes to every chat model,
and `gpt-4o-mini` rejects it. We ran this lab on `gpt-6-luna` that way, and every check passed. Config prints the values in use, and every lab prints
the same stamp, so you can always tell which models produced a result.

## Troubleshooting the setup

The notebook ends with a table for failed checks. These are the problems that come before it:

| Symptom | Likely cause | Fix |
|---|---|---|
| `zsh: command not found: uv` or `bash: uv: command not found` (Windows: "not recognized") | the terminal was open before `uv` was installed | open a new terminal; on macOS or Linux you can also run `source $HOME/.local/bin/env`, the line the installer prints |
| `uv sync` stops with `Failed to download` and `Connection refused` or a certificate error; `pip install` stops with `Could not find a version that satisfies the requirement openai<4,>=2.54 (from versions: none)` | a company proxy or firewall blocks the package index | try another network, or ask IT for the proxy settings; routes C and E work from a browser |
| `The virtual environment was not created successfully because ensurepip is not available` (Ubuntu) | Ubuntu ships `venv` as a separate package | `sudo apt install -y python3.12-venv` (24.04) or `python3.10-venv` (22.04, though route A is better there), then create `.venv` again |
| `error: externally-managed-environment` (Ubuntu 24.04) | `pip install` ran outside the virtual environment | activate it first (`source .venv/bin/activate`), then install again |
| Part A shows Python 3.10 (Ubuntu 22.04, route B) | the system Python is below the labs' floor; `pip` installs anyway | use route A: delete `.venv`, run `uv sync`, choose the new `.venv` kernel |
| PowerShell refuses to run `.venv\Scripts\activate` | Windows' script policy | run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use Command Prompt and `.venv\Scripts\activate.bat` |
| The notebook stops at Config with `ModuleNotFoundError: No module named 'dotenv'` | the notebook runs on a kernel without the packages | choose the `.venv` kernel (step 5 of your route) |
| Config says the key is `not found` (local or VM) | `.env` is missing, misnamed (`.env.txt`) or outside the repository folder | `.env` sits next to `models.env`; on Windows, turn on file name extensions to see the real name |
| Config says the key is `not found` (Codespaces) | the secret was added after the codespace started, or not for this repository | stop the codespace and start it again; check the secret's repository access under **Settings → Codespaces** |

## Resources

- [uv: installation](https://docs.astral.sh/uv/getting-started/installation/) and
  [working on projects](https://docs.astral.sh/uv/guides/projects/): `uv sync`, `.python-version` and the lock file
- [Python: virtual environments](https://docs.python.org/3/library/venv.html): `python -m venv` and activation on
  each system
- [VS Code: Jupyter notebooks](https://code.visualstudio.com/docs/datascience/jupyter-notebooks): opening a notebook
  and choosing a kernel
- [GitHub: secrets for Codespaces](https://docs.github.com/en/codespaces/managing-your-codespaces/managing-your-account-specific-secrets-for-github-codespaces):
  adding a secret, repository access, and restarting a codespace to see it
- [Colab: secrets](https://colab.research.google.com/notebooks/snippets/secrets.ipynb): storing a key in Colab
- [OpenAI: API keys](https://platform.openai.com/api-keys): where to create a key

Next: [Lab 01: How to search documents with embeddings](../01_Search_With_Embeddings/README.md). Back to
[the lab list](../README.md).
