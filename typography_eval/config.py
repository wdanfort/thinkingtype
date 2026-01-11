from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
RESULTS_PATH = RUNS_DIR / "results.csv"
RUNS_REGISTRY_PATH = RUNS_DIR / "runs.csv"
IMAGES_DIR = ROOT / "images"
OCR_DIR = ROOT / "ocr"
METADATA_PATH = ROOT / "metadata.csv"

SENTENCE_SET_ID = "v1_simple"
VARIANT_SET_ID = "v1_plus_accessibility"

MODEL_TEXT = "gpt-4o-mini"
MODEL_IMAGE = "gpt-4o"
TEMPERATURE = 0.0

SYSTEM_PROMPT = (
    "You are evaluating the intent and meaning of a short message.\n"
    "Answer only with Yes or No.\n"
    "Do not explain your answer."
)

BINARY_QUESTIONS = {
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

OCR_VARIANT_ID = "ocr"
