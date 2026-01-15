"""Google (Gemini) provider adapter."""

from __future__ import annotations

from pathlib import Path

import google.genai as genai
from PIL import Image

from typo_eval.providers.base import Provider
from typo_eval.prompts import make_text_prompt, make_image_prompt


class GoogleProvider(Provider):
    """Google Gemini API provider."""

    name = "google"

    def __init__(self) -> None:
        # API key should be set via GOOGLE_API_KEY env var
        pass

    def infer_text(
        self,
        model: str,
        temperature: float,
        input_text: str,
        question: str,
        system_prompt: str,
    ) -> str:
        """Run inference on text input using Gemini."""
        client = genai.Client()
        prompt = make_text_prompt(input_text, question)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_prompt,
            ),
        )
        return response.text if response.text else ""

    def infer_image(
        self,
        model: str,
        temperature: float,
        image_path: str,
        question: str,
        system_prompt: str,
    ) -> str:
        """Run inference on image input using Gemini."""
        client = genai.Client()
        img = Image.open(image_path)
        prompt = make_image_prompt(question)
        response = client.models.generate_content(
            model=model,
            contents=[prompt, img],
            config=genai.types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_prompt,
            ),
        )
        return response.text if response.text else ""
