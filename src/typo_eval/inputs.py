"""Input data generation and loading."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from typo_eval.config import TypoEvalConfig


# Default sentence dataset matching the original notebook
DEFAULT_SENTENCES: List[Tuple[int, str, str]] = [
    (1, "The account status was updated earlier today", "neutral"),
    (2, "Your recent activity has been successfully recorded", "neutral"),
    (3, "This message contains information about your account", "neutral"),
    (4, "The system processed the request without errors", "neutral"),
    (5, "Your preferences were saved for future use", "neutral"),
    (6, "The document reflects the most recent changes", "neutral"),
    (7, "Please review the details before continuing", "cta"),
    (8, "You may want to confirm the information provided", "cta"),
    (9, "Consider updating your settings if needed", "cta"),
    (10, "Additional steps may be required to proceed", "cta"),
    (11, "You can review available options on the next screen", "cta"),
    (12, "Further action may improve your experience", "cta"),
    (13, "This notice applies to all registered users", "authority"),
    (14, "Access is granted based on current eligibility criteria", "authority"),
    (15, "The following requirements must be acknowledged", "authority"),
    (16, "Compliance with these guidelines is expected", "authority"),
    (17, "This policy governs acceptable use of the service", "authority"),
    (18, "The information below outlines required procedures", "authority"),
    (19, "Failure to complete this step may affect access", "warning"),
    (20, "Incomplete information could result in delays", "warning"),
    (21, "Certain actions may lead to unintended consequences", "warning"),
    (22, "Errors in submission can cause processing issues", "warning"),
    (23, "This action may impact your current settings", "warning"),
    (24, "Some features may not function as expected", "warning"),
    (25, "Discover features designed to improve your workflow", "promo"),
    (26, "This update introduces new capabilities for users", "promo"),
    (27, "Explore tools built to support your goals", "promo"),
    (28, "Enhanced options are now available for you", "promo"),
    (29, "Unlock additional benefits with updated settings", "promo"),
    (30, "New functionality is available in this release", "promo"),
    (31, "Enter the required information in the fields below", "procedural"),
    (32, "Follow the steps outlined to complete the process", "procedural"),
    (33, "Select an option to continue", "procedural"),
    (34, "Review the information before submitting the form", "procedural"),
    (35, "Use the menu to navigate available sections", "procedural"),
    (36, "Complete each section before proceeding", "procedural"),
]


# Default artifact templates
DEFAULT_ARTIFACT_TEMPLATES = {
    "email": [
        "Subject: Important update regarding your account",
        "Your subscription renewal is approaching",
        "Action required: Please verify your information",
    ],
    "notification": [
        "New message from support",
        "Your request has been processed",
        "Update available for your application",
    ],
    "alert": [
        "Security notice: unusual activity detected",
        "System maintenance scheduled",
        "Important: Terms of service updated",
    ],
    "form": [
        "Please complete all required fields",
        "Enter your details to continue",
        "Submit your information for verification",
    ],
}


def generate_sentences(
    config: TypoEvalConfig,
    output_path: Path,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate sentences dataset.

    If source is "synthetic", uses DEFAULT_SENTENCES.
    If source is "file", loads from the specified path.
    """
    sentences_config = config.inputs.sentences

    if not sentences_config.get("enabled", True):
        return pd.DataFrame(columns=["sentence_id", "text", "category"])

    source = sentences_config.get("source", "synthetic")

    if source == "file":
        file_path = sentences_config.get("path")
        if file_path and Path(file_path).exists():
            return pd.read_csv(file_path)
        raise FileNotFoundError(f"Sentences file not found: {file_path}")

    # Synthetic generation - use default sentences
    rng = random.Random(seed or config.seed)

    n_sentences = sentences_config.get("n_sentences", 36)
    categories = sentences_config.get("categories", [])

    # Filter by categories if specified
    sentences = DEFAULT_SENTENCES
    if categories:
        sentences = [s for s in sentences if s[2] in categories]

    # Limit to n_sentences
    if len(sentences) > n_sentences:
        sentences = rng.sample(sentences, n_sentences)

    df = pd.DataFrame(sentences, columns=["sentence_id", "text", "category"])

    # Save to output path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def generate_artifacts(
    config: TypoEvalConfig,
    output_path: Path,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Generate artifacts dataset.

    If source is "synthetic", generates from templates.
    If source is "file", loads from the specified path.
    """
    artifacts_config = config.inputs.artifacts

    if not artifacts_config.get("enabled", False):
        return pd.DataFrame(columns=["artifact_id", "type", "text"])

    source = artifacts_config.get("source", "synthetic")

    if source == "file":
        file_path = artifacts_config.get("path")
        if file_path and Path(file_path).exists():
            return pd.read_csv(file_path)
        raise FileNotFoundError(f"Artifacts file not found: {file_path}")

    # Synthetic generation
    rng = random.Random(seed or config.seed)

    n_each_type = artifacts_config.get("n_each_type", 6)
    artifact_types = artifacts_config.get("types", list(DEFAULT_ARTIFACT_TEMPLATES.keys()))

    rows = []
    artifact_id = 1

    for artifact_type in artifact_types:
        templates = DEFAULT_ARTIFACT_TEMPLATES.get(artifact_type, [])

        # Sample or repeat templates to get n_each_type
        if len(templates) >= n_each_type:
            selected = rng.sample(templates, n_each_type)
        else:
            selected = templates * (n_each_type // len(templates) + 1)
            selected = selected[:n_each_type]

        for text in selected:
            rows.append({
                "artifact_id": f"artifact_{artifact_id:03d}",
                "type": artifact_type,
                "text": text,
            })
            artifact_id += 1

    df = pd.DataFrame(rows)

    # Save to output path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


def load_sentences(path: Path) -> pd.DataFrame:
    """Load sentences from CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Sentences file not found: {path}")
    return pd.read_csv(path)


def load_artifacts(path: Path) -> pd.DataFrame:
    """Load artifacts from CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Artifacts file not found: {path}")
    return pd.read_csv(path)
