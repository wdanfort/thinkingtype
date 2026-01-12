"""Schemas and constants used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

SYSTEM_PROMPT = (
    "You are evaluating the intent and meaning of a short message.\n"
    "Answer only with Yes or No.\n"
    "Do not explain your answer."
)

DIMENSIONS: Dict[str, str] = {
    "urgent": "Does this message create a sense of urgency?",
    "immediate_action": "Is the primary intent to prompt immediate action?",
    "formal": "Does the presentation of this message feel formal or institutional?",
    "trustworthy": "Does this message appear trustworthy?",
    "persuasive": "Is this message trying to persuade or influence the reader?",
    "emotional": "Does this message evoke a strong emotional response?",
    "professional": "Is this message intended for a professional or expert audience?",
    "high_risk": "Would misinterpreting this message likely cause harm?",
    "confident": "Is there enough information to confidently infer the intent of this message?",
    "form_dependent": "Would this message lose meaning if presented as plain text?",
}


@dataclass(frozen=True)
class InferenceRecord:
    run_id: str
    artifact_set_id: str
    provider: str
    model: str
    temperature: float
    item_id: str
    category: str
    variant_id: str
    representation: str
    dimension: str
    response_norm: str
    response_01: int | None
    raw_response: str
    error: str | None
    created_at: str

    def to_row(self) -> Dict[str, str]:
        return {
            "run_id": self.run_id,
            "artifact_set_id": self.artifact_set_id,
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "item_id": self.item_id,
            "category": self.category,
            "variant_id": self.variant_id,
            "representation": self.representation,
            "dimension": self.dimension,
            "response_norm": self.response_norm,
            "response_01": self.response_01,
            "raw_response": self.raw_response,
            "error": self.error,
            "created_at": self.created_at,
        }


RESULTS_COLUMNS: List[str] = [
    "run_id",
    "artifact_set_id",
    "provider",
    "model",
    "temperature",
    "item_id",
    "category",
    "variant_id",
    "representation",
    "dimension",
    "response_norm",
    "response_01",
    "raw_response",
    "error",
    "created_at",
]
