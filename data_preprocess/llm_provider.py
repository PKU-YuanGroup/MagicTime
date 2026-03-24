"""
LLM provider configuration for data preprocessing scripts.

Supports multiple LLM providers via OpenAI-compatible APIs:
- OpenAI (default): GPT-4V, GPT-4o, etc.
- MiniMax: MiniMax-M2.7, MiniMax-M2.5, etc.
- Any OpenAI-compatible provider via --base_url and --model

Usage:
    from llm_provider import add_provider_args, create_client, get_model_name

    # In argument parser setup:
    add_provider_args(parser)

    # In code:
    args = parser.parse_args()
    client = create_client(args)
    model = get_model_name(args)
"""

import os

from openai import OpenAI

# Provider presets: base_url, default model, env var for API key
PROVIDER_PRESETS = {
    "openai": {
        "base_url": None,  # OpenAI SDK default
        "default_model": "gpt-4-vision-preview",
        "env_key": "OPENAI_API_KEY",
    },
    "minimax": {
        "base_url": "https://api.minimax.io/v1",
        "default_model": "MiniMax-M2.7",
        "env_key": "MINIMAX_API_KEY",
    },
}


def add_provider_args(parser):
    """Add LLM provider arguments to an argparse parser."""
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=list(PROVIDER_PRESETS.keys()),
        help="LLM provider to use (default: openai).",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Custom API base URL (overrides provider default).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (overrides provider default).",
    )


def _resolve_api_key(args):
    """Resolve API key from args or environment variables."""
    # Explicit --api_key takes priority
    api_key = getattr(args, "api_key", None)
    if api_key:
        return api_key

    # Check provider-specific env var
    provider = getattr(args, "provider", "openai")
    preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["openai"])
    env_key = preset["env_key"]
    api_key = os.environ.get(env_key)
    if api_key:
        return api_key

    # Fallback to OPENAI_API_KEY for any provider
    return os.environ.get("OPENAI_API_KEY")


def create_client(args):
    """Create an OpenAI-compatible client based on provider args."""
    provider = getattr(args, "provider", "openai")
    preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["openai"])

    base_url = getattr(args, "base_url", None) or preset["base_url"]
    api_key = _resolve_api_key(args)

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    return OpenAI(**kwargs)


def get_model_name(args):
    """Get the model name from args or provider defaults."""
    model = getattr(args, "model", None)
    if model:
        return model

    provider = getattr(args, "provider", "openai")
    preset = PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["openai"])
    return preset["default_model"]


def clamp_temperature(temperature, provider="openai"):
    """Clamp temperature to provider-specific valid range.

    MiniMax accepts temperature in (0.0, 1.0].
    """
    if provider == "minimax":
        return max(0.01, min(temperature, 1.0))
    return temperature
