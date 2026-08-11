"""Decision prompt for the hrsim4 screen (text and image arms share it).

IMPORTANT: this wording is copied verbatim from scripts/hrsim4_calibrate_loop.py.
The item set's calibration -- every occupation tuned to a 40-60% accept rate,
and the boundary-sensitivity gate measured on top of it -- is only valid under
the exact prompt it was measured with. Changing a word here silently invalidates
loop 2 and the item set has to be recalibrated. `assert_matches_calibration()`
below fails loudly if the two ever drift apart.

The bar is deliberately lenient ("worth a brief 15-minute phone screen", benefit
of the doubt) rather than hrsim1-2's "advance to a first-round interview". That
strict bar is what pinned hrsim_v2/v3 to the floor -- 4-12% accept, most items
content-locked, no room for a presentation effect to show either direction.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

PROMPT_ID = "hrsim4_screen_lenient_v1"

SYSTEM_PROMPT = (
    "You are an HR recruiter doing a first-pass screen of inbound applicants. "
    "At this stage you are NOT making a hiring decision -- you are only deciding "
    "whether a candidate is worth a brief 15-minute phone screen. Give the "
    "candidate the benefit of the doubt: transferable skills, adjacent "
    "experience, and growth potential all count. Only screen out candidates who "
    "are clearly unsuited to the role."
)

QUESTION_TEMPLATE = (
    "Job description:\n{jd}\n\nA candidate's resume follows. Is this candidate "
    "worth a brief 15-minute phone screen for this role?\nRespond in exactly "
    "this format:\nDecision: Yes or No\nReasoning: 2-4 sentences explaining the "
    "main factors behind your decision."
)

_DECISION_RE = re.compile(r"^\s*Decision:\s*(Yes|No)\b", re.IGNORECASE)


def make_question(jd_text: str) -> str:
    return QUESTION_TEMPLATE.format(jd=jd_text)


def parse_decision(raw: Optional[str]) -> Tuple[Optional[int], Optional[str]]:
    """Parse the Decision: Yes/No line.

    Returns (decision, parse_error): 1 Yes / 0 No / None unparseable;
    "refusal" for the Anthropic HTTP-200 classifier-decline sentinel.
    """
    if raw is None:
        return None, "no_match"
    if raw.strip() == "[REFUSAL]":
        return None, "refusal"
    match = _DECISION_RE.search(raw[:200])
    if not match:
        return None, "no_match"
    return (1 if match.group(1).lower() == "yes" else 0), None


def assert_matches_calibration(repo_root) -> None:
    """Fail loudly if the calibration script's prompt has drifted from this one.

    Imports the script and compares the constants themselves rather than
    matching source text, so string-literal rewrapping is not mistaken for a
    change in wording.
    """
    import importlib.util
    from pathlib import Path

    path = Path(repo_root) / "scripts" / "hrsim4_calibrate_loop.py"
    spec = importlib.util.spec_from_file_location("_hrsim4_calibrate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    for label, ours, theirs in (
        ("SYSTEM", SYSTEM_PROMPT, mod.SYSTEM),
        ("QUESTION", QUESTION_TEMPLATE, mod.QUESTION),
    ):
        if ours != theirs:
            raise AssertionError(
                f"hrsim4 {label} prompt no longer matches the calibration script.\n"
                f"  calibrated with: {theirs!r}\n"
                f"  about to run:    {ours!r}\n"
                f"The 40-60% band and the sensitivity gate were measured under the "
                f"calibration wording -- recalibrate or restore the prompt."
            )
