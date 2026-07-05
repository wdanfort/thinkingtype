"""Provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional


class Provider(ABC):
    """Base class for inference providers."""

    name: str

    # Token usage of the most recent call: {"input_tokens": int, "output_tokens": int}.
    # Updated after each infer_* call; None before the first call or if the
    # provider response carried no usage info.
    last_usage: Optional[Dict[str, int]] = None

    @abstractmethod
    def infer_text(
        self,
        model: str,
        temperature: float,
        input_text: str,
        question: str,
        system_prompt: str,
        seed: int | None = None,
    ) -> str:
        """Run inference on text input. seed is honored only by providers whose API supports it."""
        raise NotImplementedError

    @abstractmethod
    def infer_image(
        self,
        model: str,
        temperature: float,
        image_path: str,
        question: str,
        system_prompt: str,
        seed: int | None = None,
    ) -> str:
        """Run inference on image input. seed is honored only by providers whose API supports it."""
        raise NotImplementedError
