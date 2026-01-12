"""Anthropic provider stub."""

from __future__ import annotations

from typography.providers.base import Provider


class AnthropicProvider(Provider):
    name = "anthropic"

    def infer_text(self, model: str, temperature: float, input_text: str, question: str) -> str:
        raise NotImplementedError("Anthropic provider not implemented yet.")

    def infer_image(self, model: str, temperature: float, image_path: str, question: str) -> str:
        raise NotImplementedError("Anthropic provider not implemented yet.")
