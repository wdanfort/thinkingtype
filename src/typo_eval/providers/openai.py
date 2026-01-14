"""OpenAI provider adapter."""

from __future__ import annotations

import base64
from pathlib import Path

from openai import OpenAI

from typo_eval.providers.base import Provider
from typo_eval.prompts import make_text_prompt, make_image_prompt


class OpenAIProvider(Provider):
    """OpenAI API provider."""

    name = "openai"

    def __init__(self) -> None:
        self.client = OpenAI()

    def infer_text(
        self,
        model: str,
        temperature: float,
        input_text: str,
        question: str,
        system_prompt: str,
    ) -> str:
        """Run inference on text input using OpenAI."""
        resp = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": make_text_prompt(input_text, question)},
            ],
        )
        return resp.choices[0].message.content or ""

    def infer_image(
        self,
        model: str,
        temperature: float,
        image_path: str,
        question: str,
        system_prompt: str,
    ) -> str:
        """Run inference on image input using OpenAI."""
        data_url = _encode_image(image_path)
        resp = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": make_image_prompt(question)},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
        return resp.choices[0].message.content or ""


def _encode_image(image_path: str) -> str:
    """Encode image to base64 data URL."""
    data = Path(image_path).read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"
