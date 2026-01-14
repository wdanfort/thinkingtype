"""Provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Provider(ABC):
    """Base class for inference providers."""

    name: str

    @abstractmethod
    def infer_text(
        self,
        model: str,
        temperature: float,
        input_text: str,
        question: str,
        system_prompt: str,
    ) -> str:
        """Run inference on text input."""
        raise NotImplementedError

    @abstractmethod
    def infer_image(
        self,
        model: str,
        temperature: float,
        image_path: str,
        question: str,
        system_prompt: str,
    ) -> str:
        """Run inference on image input."""
        raise NotImplementedError
