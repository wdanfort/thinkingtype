"""OpenAI provider adapter."""

from __future__ import annotations

import base64
from pathlib import Path

from openai import OpenAI

from typography.schemas import SYSTEM_PROMPT
from typography.providers.base import Provider


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self) -> None:
        self.client = OpenAI()

    def infer_text(self, model: str, temperature: float, input_text: str, question: str) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f'Message:\n"{input_text}"\n\nQuestion:\n{question}',
                },
            ],
        )
        return resp.choices[0].message.content or ""

    def infer_image(self, model: str, temperature: float, image_path: str, question: str) -> str:
        data_url = _encode_image(image_path)
        resp = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Please evaluate the message shown in the image.\n\nQuestion:\n{question}",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
        )
        return resp.choices[0].message.content or ""


def _encode_image(image_path: str) -> str:
    data = Path(image_path).read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"
