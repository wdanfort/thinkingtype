"""Google (Gemini) provider adapter."""

from __future__ import annotations

import os
from pathlib import Path

from google import genai
from google.genai import types
from PIL import Image

from typo_eval.providers.base import Provider
from typo_eval.prompts import make_text_prompt, make_image_prompt


class GoogleProvider(Provider):
    """Google Gemini API provider using the new google-genai SDK."""

    name = "google"

    def __init__(self) -> None:
        # Initialize the client with API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        self.client = genai.Client(api_key=api_key)

    def infer_text(
        self,
        model: str,
        temperature: float,
        input_text: str,
        question: str,
        system_prompt: str,
    ) -> str:
        """Run inference on text input using Gemini."""
        prompt = make_text_prompt(input_text, question)

        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
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
        # Load image and convert to bytes
        img = Image.open(image_path)

        # Convert PIL Image to Part
        import io
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img.format or 'PNG')
        img_byte_arr = img_byte_arr.getvalue()

        prompt = make_image_prompt(question)

        response = self.client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=img_byte_arr, mime_type=f"image/{(img.format or 'png').lower()}"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=temperature,
                system_instruction=system_prompt,
            ),
        )
        return response.text if response.text else ""
