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
        self.last_usage = None
        # Set by the inference loop from ProviderConfig.service_tier
        self.service_tier = None

    def _capture_usage(self, resp) -> None:
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.last_usage = {
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
            }
        else:
            self.last_usage = None

    def infer_text(
        self,
        model: str,
        temperature: float,
        input_text: str,
        question: str,
        system_prompt: str,
        seed: int | None = None,
    ) -> str:
        """Run inference on text input using OpenAI."""
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self.service_tier:
            kwargs["service_tier"] = self.service_tier
        resp = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": make_text_prompt(input_text, question)},
            ],
            **kwargs,
        )
        self._capture_usage(resp)
        return resp.choices[0].message.content or ""

    def infer_image(
        self,
        model: str,
        temperature: float,
        image_path: str,
        question: str,
        system_prompt: str,
        seed: int | None = None,
    ) -> str:
        """Run inference on image input using OpenAI."""
        data_url = _encode_image(image_path)
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self.service_tier:
            kwargs["service_tier"] = self.service_tier
        resp = self.client.chat.completions.create(
            model=model,
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
            **kwargs,
        )
        self._capture_usage(resp)
        return resp.choices[0].message.content or ""


def _encode_image(image_path: str) -> str:
    """Encode image to base64 data URL."""
    data = Path(image_path).read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"
