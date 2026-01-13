"""Google (Gemini) provider adapter."""

from __future__ import annotations

from pathlib import Path

import google.generativeai as genai
from PIL import Image

from typo_eval.providers.base import Provider
from typo_eval.prompts import make_text_prompt, make_image_prompt


class GoogleProvider(Provider):
    """Google Gemini API provider."""

    name = "google"

    def __init__(self) -> None:
        # API key should be set via GOOGLE_API_KEY env var
        # genai.configure() is called automatically when API key is in env
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
        gemini_model = genai.GenerativeModel(
            model,
            system_instruction=system_prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature),
        )
        prompt = make_text_prompt(input_text, question)
        response = gemini_model.generate_content(prompt)
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
        gemini_model = genai.GenerativeModel(
            model,
            system_instruction=system_prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperature),
        )
        img = Image.open(image_path)
        prompt = make_image_prompt(question)
        response = gemini_model.generate_content([prompt, img])
        return response.text if response.text else ""
