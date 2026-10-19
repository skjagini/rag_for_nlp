"""Model names, the API endpoint and the key for every lab.

Settings come from models.env in the repository folder (committed, no secrets). A value in .env (local,
gitignored) overrides it, and an environment variable overrides both. Keys never live in models.env:
load_key() looks in the environment (where Codespaces secrets arrive), then .env, then Colab Secrets.

Every lab talks to the model through two calls that OpenAI-compatible servers commonly support: Chat
Completions and Embeddings. chat_client() and embed_client() build the clients for them from these settings,
so changing a model or the endpoint is a one-line change in models.env or .env.

Nothing is read when this module is imported.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]          # the repository folder
DATA = ROOT / "data"
OPENAI_DEFAULT_URL = "https://api.openai.com/v1"


def load_settings(root: Path = ROOT) -> dict:
    """models.env, then .env, then environment variables; later sources win."""
    settings = dict(dotenv_values(root / "models.env"))
    local = dotenv_values(root / ".env")
    for name in settings:
        if local.get(name):
            settings[name] = local[name]
        if os.environ.get(name):
            settings[name] = os.environ[name]
    return settings


def load_key(name: str = "OPENAI_API_KEY", root: Path = ROOT) -> str | None:
    """Put a key into os.environ for the client.

    Returns where the key was found ("environment", ".env" or "Colab Secrets"), or None.
    Never returns or prints the key itself.
    """
    if os.environ.get(name):
        return "environment"
    value = dotenv_values(root / ".env").get(name)
    if value:
        os.environ[name] = value
        return ".env"
    try:
        from google.colab import userdata
    except ImportError:                              # not on Colab
        return None
    try:
        value = userdata.get(name)
    except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
        return None
    if value:
        os.environ[name] = value
        return "Colab Secrets"
    return None


def chat_url(settings: dict) -> str:
    """The Chat Completions endpoint: OPENAI_BASE_URL, or OpenAI's default when it is empty."""
    return settings.get("OPENAI_BASE_URL") or OPENAI_DEFAULT_URL


def embed_url(settings: dict) -> str:
    """The Embeddings endpoint: EMBED_BASE_URL, or the chat endpoint when it is empty."""
    return settings.get("EMBED_BASE_URL") or chat_url(settings)


def chat_client(settings: dict | None = None):
    """An OpenAI client for CHAT_MODEL, JUDGE_MODEL and RERANK_MODEL (Chat Completions)."""
    from openai import OpenAI
    settings = settings or load_settings()
    return OpenAI(base_url=chat_url(settings), api_key=os.environ.get("OPENAI_API_KEY"))


def embed_client(settings: dict | None = None):
    """An OpenAI client for EMBED_MODEL (Embeddings). EMBED_API_KEY is used only with EMBED_BASE_URL."""
    from openai import OpenAI
    settings = settings or load_settings()
    key = os.environ.get("OPENAI_API_KEY")
    if settings.get("EMBED_BASE_URL") and load_key("EMBED_API_KEY"):
        key = os.environ["EMBED_API_KEY"]
    return OpenAI(base_url=embed_url(settings), api_key=key)


def chat_options(name: str = "CHAT_MODEL", settings: dict | None = None, temperature: float = 0) -> dict:
    """The keyword arguments every chat call adds: the model named by `name`, and temperature or reasoning effort.

    With REASONING_EFFORT empty (gpt-4o-mini and most models), it sends temperature and no reasoning_effort.
    With REASONING_EFFORT set, it sends reasoning_effort, and temperature only when the effort is "none":
    reasoning models reject temperature at any other effort.
    """
    settings = settings or load_settings()
    options = {"model": settings[name]}
    effort = settings.get("REASONING_EFFORT")
    if effort:
        options["reasoning_effort"] = effort
    if not effort or effort == "none":
        options["temperature"] = temperature
    return options


def lc_chat_model(name: str = "CHAT_MODEL", settings: dict | None = None, temperature: float = 0):
    """The same chat model as chat_client() + chat_options(), as a LangChain ChatOpenAI (Lab 05 on).

    use_responses_api=False keeps it on Chat Completions, which every compatible server implements.
    """
    from langchain_openai import ChatOpenAI
    settings = settings or load_settings()
    options = chat_options(name, settings, temperature)
    return ChatOpenAI(base_url=chat_url(settings), api_key=os.environ.get("OPENAI_API_KEY"),
                      use_responses_api=False, **options)


def lc_embeddings(settings: dict | None = None):
    """The same embedding model as embed_client(), as LangChain OpenAIEmbeddings (Lab 05 on).

    LangChain splits long texts with tiktoken and sends token IDs by default; other OpenAI-compatible servers
    may reject token IDs, so that is switched off whenever the endpoint is not OpenAI's.
    """
    from langchain_openai import OpenAIEmbeddings
    settings = settings or load_settings()
    key = os.environ.get("OPENAI_API_KEY")
    if settings.get("EMBED_BASE_URL") and load_key("EMBED_API_KEY"):
        key = os.environ["EMBED_API_KEY"]
    return OpenAIEmbeddings(model=settings["EMBED_MODEL"], base_url=embed_url(settings), api_key=key,
                            check_embedding_ctx_length=embed_url(settings) == OPENAI_DEFAULT_URL)


def stamp(settings: dict | None = None) -> dict:
    """What every saved result records: the models, the reasoning effort, the endpoints and the date of the run."""
    settings = settings or load_settings()
    return {
        "chat_model": settings["CHAT_MODEL"],
        "reasoning_effort": settings.get("REASONING_EFFORT") or None,
        "embed_model": settings["EMBED_MODEL"],
        "judge_model": settings.get("JUDGE_MODEL"),
        "rerank_model": settings.get("RERANK_MODEL"),
        "chat_endpoint": chat_url(settings),
        "embed_endpoint": embed_url(settings),
        "run_date": date.today().isoformat(),
    }
