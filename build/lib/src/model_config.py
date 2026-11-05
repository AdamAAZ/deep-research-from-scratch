"""Model choices for each stage of the research workflow."""

import os


DEFAULT_MODELS = {
    "openai": {
        "scope": "openai:gpt-4.1-mini",
        "research": "openai:gpt-4.1",
        "supervisor": "openai:gpt-4.1",
        "summarize": "openai:gpt-4.1-mini",
        "compress": "openai:gpt-4.1-mini",
        "writer": "openai:gpt-4.1",
    },
    "anthropic": {
        "scope": "anthropic:claude-haiku-4-5",
        "research": "anthropic:claude-sonnet-4-6",
        "supervisor": "anthropic:claude-sonnet-4-6",
        "summarize": "anthropic:claude-haiku-4-5",
        "compress": "anthropic:claude-haiku-4-5",
        "writer": "anthropic:claude-sonnet-4-6",
    },
}


def model_for(role: str) -> str:
    """Return the selected provider's model for a workflow role."""
    provider = os.getenv("MODEL_PROVIDER", "openai").lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError("MODEL_PROVIDER must be 'openai' or 'anthropic'")
    return os.getenv(f"{role.upper()}_MODEL", DEFAULT_MODELS[provider][role])
