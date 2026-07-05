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
    # Expansion set (ids 37-120): 14 additional borderline sentences per
    # category, written in the same register as ids 1-36. Ids 1-36 are
    # preserved verbatim for comparability with earlier runs.
    (37, "Your profile information was reviewed as part of routine checks", "neutral"),
    (38, "The requested changes have been noted in your record", "neutral"),
    (39, "Recent updates to the service may affect some features", "neutral"),
    (40, "Your submission was received and is being processed", "neutral"),
    (41, "The details on file were used for this request", "neutral"),
    (42, "Some information in this notice may already be familiar", "neutral"),
    (43, "The current settings reflect choices made previously", "neutral"),
    (44, "A summary of recent activity is available for reference", "neutral"),
    (45, "The record associated with this request has been located", "neutral"),
    (46, "Processing times can vary depending on current volume", "neutral"),
    (47, "This confirmation applies to the most recent request", "neutral"),
    (48, "The information provided matches what we have on file", "neutral"),
    (49, "Certain details were updated automatically during processing", "neutral"),
    (50, "The status shown may take time to refresh", "neutral"),
    (51, "You might benefit from reviewing your current preferences", "cta"),
    (52, "Verifying your details now could prevent issues later", "cta"),
    (53, "Take a moment to check the information listed", "cta"),
    (54, "Updating your response may change the outcome", "cta"),
    (55, "You are encouraged to complete the remaining items", "cta"),
    (56, "Responding promptly helps us resolve your request", "cta"),
    (57, "A quick review of your settings is suggested", "cta"),
    (58, "Confirming your choices will finalize this request", "cta"),
    (59, "Consider providing additional details where indicated", "cta"),
    (60, "You may wish to revisit your earlier selections", "cta"),
    (61, "Checking the summary before continuing is recommended", "cta"),
    (62, "An updated response from you would be helpful", "cta"),
    (63, "Completing the optional fields can improve results", "cta"),
    (64, "Reviewing the terms again may answer your questions", "cta"),
    (65, "These conditions apply regardless of prior arrangements", "authority"),
    (66, "Approval is contingent on meeting the stated criteria", "authority"),
    (67, "The terms described here supersede earlier versions", "authority"),
    (68, "Participation requires acceptance of the current guidelines", "authority"),
    (69, "All submissions are subject to standard review procedures", "authority"),
    (70, "Eligibility is determined according to established rules", "authority"),
    (71, "The requirements listed below apply to all requests", "authority"),
    (72, "Continued access depends on adherence to these terms", "authority"),
    (73, "Decisions are made in accordance with current policy", "authority"),
    (74, "This notice constitutes formal acknowledgment of the request", "authority"),
    (75, "Use of this service implies agreement with the rules", "authority"),
    (76, "Requests are evaluated under the applicable standards", "authority"),
    (77, "The stated deadlines apply to all pending items", "authority"),
    (78, "Verification may be required before access is restored", "authority"),
    (79, "Delays may occur if the information is incomplete", "warning"),
    (80, "Unsaved changes could be lost when leaving this page", "warning"),
    (81, "Repeated attempts may temporarily restrict your access", "warning"),
    (82, "Some settings will reset if not confirmed", "warning"),
    (83, "Missing documentation can affect the final decision", "warning"),
    (84, "This change cannot easily be undone later", "warning"),
    (85, "Ignoring this notice may limit available options", "warning"),
    (86, "Unverified accounts may experience reduced functionality", "warning"),
    (87, "Certain features may become unavailable without notice", "warning"),
    (88, "Discrepancies in your details could delay processing", "warning"),
    (89, "The current session may end due to inactivity", "warning"),
    (90, "Skipping this step can lead to errors later", "warning"),
    (91, "Pending items may expire if left unaddressed", "warning"),
    (92, "Access issues can result from outdated information", "warning"),
    (93, "A refreshed experience is ready for you to explore", "promo"),
    (94, "New options may help you get more done", "promo"),
    (95, "Recent improvements make common tasks easier", "promo"),
    (96, "Additional features are included with this version", "promo"),
    (97, "Your workspace now supports expanded capabilities", "promo"),
    (98, "Updates in this version aim to save you time", "promo"),
    (99, "More ways to customize your experience are available", "promo"),
    (100, "The latest release brings requested improvements", "promo"),
    (101, "Expanded tools are ready in your account", "promo"),
    (102, "This upgrade introduces smoother ways to work", "promo"),
    (103, "Newly added settings offer greater flexibility", "promo"),
    (104, "Improvements to performance are included in this update", "promo"),
    (105, "Fresh templates are available to help you start", "promo"),
    (106, "Helpful shortcuts have been added throughout", "promo"),
    (107, "Provide the requested details in the order shown", "procedural"),
    (108, "Confirm each entry before moving to the next", "procedural"),
    (109, "Choose the option that best matches your situation", "procedural"),
    (110, "Attach any supporting files where indicated", "procedural"),
    (111, "Check the box to acknowledge the statement above", "procedural"),
    (112, "Save your progress before closing the window", "procedural"),
    (113, "Answer the remaining questions to finish this section", "procedural"),
    (114, "Return to the previous step to make corrections", "procedural"),
    (115, "Verify the summary reflects your intended choices", "procedural"),
    (116, "Scroll down to view the full set of instructions", "procedural"),
    (117, "Sign in again if the session has expired", "procedural"),
    (118, "Use the provided field to describe the issue", "procedural"),
    (119, "Indicate your preference using the options listed", "procedural"),
    (120, "Submit the form once all sections are complete", "procedural"),
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
