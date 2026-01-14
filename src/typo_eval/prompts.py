"""Prompt templates and utilities."""

from __future__ import annotations

import hashlib
from typing import Dict, Tuple

# System prompts
DIMENSIONS_SYSTEM_PROMPT = (
    "You are evaluating the intent and meaning of a short message.\n"
    "Answer only with Yes or No.\n"
    "Do not explain your answer."
)

DECISION_SYSTEM_PROMPT = (
    "You are helping route messages in a product system.\n"
    "Answer only with Yes or No.\n"
    "Do not explain your answer."
)

# Binary dimension questions
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

# Decision prompts
DECISION_PROMPTS: Dict[str, str] = {
    "escalation_v1": (
        "Should this message be escalated to a human reviewer "
        "for safety, compliance, or potential user harm?"
    ),
}


def get_dimension_question(dimension_id: str) -> str:
    """Get the question for a dimension."""
    if dimension_id not in DIMENSIONS:
        raise ValueError(f"Unknown dimension: {dimension_id}")
    return DIMENSIONS[dimension_id]


def get_decision_question(prompt_id: str) -> str:
    """Get the question for a decision prompt."""
    if prompt_id not in DECISION_PROMPTS:
        raise ValueError(f"Unknown decision prompt: {prompt_id}")
    return DECISION_PROMPTS[prompt_id]


def make_text_prompt(text: str, question: str) -> str:
    """Create user prompt for text-based evaluation."""
    return f'Message:\n"{text}"\n\nQuestion:\n{question}'


def make_image_prompt(question: str) -> str:
    """Create user prompt for image-based evaluation."""
    return f"Please evaluate the message shown in the image.\n\nQuestion:\n{question}"


def compute_prompt_hash(system_prompt: str, user_prompt: str) -> str:
    """Compute SHA256 hash of the full prompt."""
    full_prompt = f"{system_prompt}\n---\n{user_prompt}"
    return hashlib.sha256(full_prompt.encode()).hexdigest()[:16]


def parse_yes_no(raw: str | None) -> Tuple[int | None, str]:
    """
    Parse yes/no response from model.

    Returns (parsed_int, normalized_str):
    - parsed_int: 1 for yes, 0 for no, None if invalid
    - normalized_str: "yes", "no", or truncated raw response
    """
    if raw is None:
        return None, ""
    s = raw.strip().lower()
    if s.startswith("yes"):
        return 1, "yes"
    if s.startswith("no"):
        return 0, "no"
    return None, s[:200]


def coerce_to_binary(value) -> int | None:
    """
    Coerce various response formats to binary 0/1.

    Handles: int, float, bool, and string representations.
    """
    import math

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)) and float(value) in (0.0, 1.0):
            return int(value)
        return None
    s = str(value).strip().lower()
    if s in {"1", "yes", "y", "true", "t"}:
        return 1
    if s in {"0", "no", "n", "false", "f"}:
        return 0
    return None
