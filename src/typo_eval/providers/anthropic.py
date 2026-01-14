"""Anthropic provider adapter."""

from __future__ import annotations

import base64
from pathlib import Path

import anthropic

from typo_eval.providers.base import Provider
from typo_eval.prompts import make_text_prompt, make_image_prompt


class AnthropicProvider(Provider):
    """Anthropic API provider."""

    name = "anthropic"

    def __init__(self) -> None:
        self.client = anthropic.Anthropic()

    def infer_text(
        self,
        model: str,
        temperature: float,
        input_text: str,
        question: str,
        system_prompt: str,
    ) -> str:
        """Run inference on text input using Anthropic."""
        resp = self.client.messages.create(
            model=model,
            max_tokens=100,
            system=system_prompt,
            messages=[
                {"role": "user", "content": make_text_prompt(input_text, question)},
            ],
        )
        if resp.content and len(resp.content) > 0:
            return resp.content[0].text
        return ""

    def infer_image(
        self,
        model: str,
        temperature: float,
        image_path: str,
        question: str,
        system_prompt: str,
    ) -> str:
        """Run inference on image input using Anthropic."""
        b64_data = _encode_image_base64(image_path)
        resp = self.client.messages.create(
            model=model,
            max_tokens=100,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_data,
                            },
                        },
                        {"type": "text", "text": make_image_prompt(question)},
                    ],
                },
            ],
        )
        if resp.content and len(resp.content) > 0:
            return resp.content[0].text
        return ""


def _encode_image_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode("utf-8")
