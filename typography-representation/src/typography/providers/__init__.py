"""Provider registry."""

from typography.providers.anthropic import AnthropicProvider
from typography.providers.base import Provider
from typography.providers.gemini import GeminiProvider
from typography.providers.openai import OpenAIProvider


PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}


def get_provider(name: str) -> Provider:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}")
    return PROVIDERS[name]()
