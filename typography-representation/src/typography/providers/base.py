"""Provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Provider(ABC):
    name: str

    @abstractmethod
    def infer_text(self, model: str, temperature: float, input_text: str, question: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def infer_image(self, model: str, temperature: float, image_path: str, question: str) -> str:
        raise NotImplementedError
