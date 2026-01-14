"""Provider registry."""

from typo_eval.providers.base import Provider
from typo_eval.providers.openai import OpenAIProvider
from typo_eval.providers.anthropic import AnthropicProvider
from typo_eval.providers.google import GoogleProvider

PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
}


def get_provider(name: str) -> Provider:
    """Get provider instance by name."""
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[name]()


__all__ = [
    "Provider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "get_provider",
    "PROVIDERS",
]
