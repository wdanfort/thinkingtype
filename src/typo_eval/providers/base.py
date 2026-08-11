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

    def infer_images(
        self,
        model: str,
        temperature: float,
        image_paths: "list[str]",
        question: str,
        system_prompt: str,
        seed: int | None = None,
    ) -> str:
        """Run inference on a MULTI-PAGE document, one image block per page.

        hrsim1-3 truncated resumes to a single page, so single-image inference
        was enough. hrsim4 lays out full documents and runs 2-3 pages per
        resume (OpenDyslexic averages 3.1), and the pages cannot be stitched
        into one tall image without crossing Anthropic's 1568px limit and
        triggering the downscale that the render QC gate exists to prevent.

        Default implementation only handles the single-page case; providers
        that support multiple image blocks should override.
        """
        if len(image_paths) == 1:
            return self.infer_image(
                model, temperature, image_paths[0], question, system_prompt, seed=seed
            )
        raise NotImplementedError(
            f"{self.name} provider cannot run {len(image_paths)}-page documents"
        )
