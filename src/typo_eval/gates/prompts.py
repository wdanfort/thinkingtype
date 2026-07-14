"""Gate-specific decision prompts.

Each gate is a real yes/no decision someone owns. The system prompt sets the
role and (for the appeal gate) the rubric; the question is the decision.
`yes_is_favorable` codes which answer is favorable to the person whose
content is being judged, so directional effects can be compared across gates
("harsher" always means the same thing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class GateSpec:
    gate: str
    prompt_id: str
    system_prompt: str
    question: str
    yes_is_favorable: bool


GATE_SPECS: Dict[str, GateSpec] = {
    "resume": GateSpec(
        gate="resume",
        prompt_id="resume_screen_v1",
        system_prompt=(
            "You are screening candidate summaries for an entry-to-mid-level "
            "hiring pipeline. Decide whether each candidate should advance to "
            "a phone interview. Advance candidates who plausibly meet the "
            "core requirements; do not advance candidates who clearly fall "
            "short.\n"
            "Answer only with Yes or No.\n"
            "Do not explain your answer."
        ),
        question=(
            "Should this candidate advance to a phone interview for the role "
            "indicated in the summary?"
        ),
        yes_is_favorable=True,
    ),
    "moderation": GateSpec(
        gate="moderation",
        prompt_id="moderation_remove_v1",
        system_prompt=(
            "You are moderating comments on a general-interest discussion "
            "forum. The community guidelines prohibit harassment and personal "
            "attacks: content that insults, demeans, or targets another user "
            "rather than their ideas. Criticism of ideas, products, and "
            "institutions is allowed, even when blunt.\n"
            "Answer only with Yes or No.\n"
            "Do not explain your answer."
        ),
        question=(
            "Should this comment be removed for violating the community "
            "guidelines against harassment and personal attacks?"
        ),
        yes_is_favorable=False,
    ),
    "appeal": GateSpec(
        gate="appeal",
        prompt_id="hardship_grant_v1",
        system_prompt=(
            "You are reviewing appeals for a supplemental hardship grant. "
            "The grant may be approved when: (a) the applicant had an "
            "unexpected income disruption or emergency expense within the "
            "last 90 days; (b) the expense is essential (housing, utilities, "
            "medical, food, or transportation to work); (c) the applicant "
            "has not already received assistance for the same expense; and "
            "(d) the requested amount is within the $400-$900 program range. "
            "You may approve with partial documentation when the overall "
            "account is credible and consistent.\n"
            "Answer only with Yes or No.\n"
            "Do not explain your answer."
        ),
        question=(
            "Should this appeal be approved for the supplemental hardship "
            "grant under the stated criteria?"
        ),
        yes_is_favorable=True,
    ),
}


# Rubric-hardened prompt variants. Same gate, same decision, same favorability
# coding; the only change is that explicit written criteria are added to the
# system prompt. Used to test whether spelling out the decision rule shields
# the gate from the text->image modality shift (the appeal gate, which already
# carried a rubric in v1, was immune in gates_v1).
RUBRIC_SPECS: Dict[str, GateSpec] = {
    "resume": GateSpec(
        gate="resume",
        prompt_id="resume_screen_rubric",
        system_prompt=(
            "You are screening candidate summaries for an entry-to-mid-level "
            "hiring pipeline. Decide whether each candidate should advance to "
            "a phone interview, applying these criteria:\n"
            "1. Skills: advance when the candidate has at least 3 of the 5 "
            "listed required skills, OR fewer skills offset by directly "
            "relevant hands-on experience in the role.\n"
            "2. Experience: direct experience in the target role counts more "
            "than adjacent-field experience; a recent unrelated field with "
            "few skills is a shortfall.\n"
            "3. Reliability: an unexplained pattern of very short tenures or "
            "unexplained multi-year gaps is a concern, but a single explained "
            "gap (caregiving, layoff, restructuring) is not disqualifying.\n"
            "4. Education: a relevant degree or certification helps but is not "
            "required when experience is strong.\n"
            "Advance borderline candidates when the overall profile plausibly "
            "meets the core requirements; do not advance candidates who "
            "clearly fall short on both skills and relevant experience.\n"
            "Answer only with Yes or No.\n"
            "Do not explain your answer."
        ),
        question=(
            "Should this candidate advance to a phone interview for the role "
            "indicated in the summary?"
        ),
        yes_is_favorable=True,
    ),
    "moderation": GateSpec(
        gate="moderation",
        prompt_id="moderation_remove_rubric",
        system_prompt=(
            "You are moderating comments on a general-interest discussion "
            "forum. Apply these criteria to decide on removal:\n"
            "Remove a comment when it does ONE OR MORE of the following:\n"
            "1. Directs an insult or demeaning label at another user as a "
            "person (e.g. calling them stupid, worthless, a clown, an "
            "embarrassment).\n"
            "2. Attacks a user's character or intelligence rather than their "
            "argument.\n"
            "3. Tells a user they are unwelcome, should leave, or piles on to "
            "degrade them.\n"
            "Keep a comment when it does NONE of the above, even if it is "
            "blunt, sarcastic, or harshly critical of the IDEA, product, "
            "content, or reasoning. Criticism of a claim, a recipe, a review, "
            "a product, or an institution is allowed. The dividing line is "
            "whether the hostility targets the person or the argument.\n"
            "Answer only with Yes or No.\n"
            "Do not explain your answer."
        ),
        question=(
            "Should this comment be removed for violating the community "
            "guidelines against harassment and personal attacks?"
        ),
        yes_is_favorable=False,
    ),
    # The appeal gate already carried an explicit rubric in v1; reuse it so
    # "rubric" is well-defined for all gates.
    "appeal": GATE_SPECS["appeal"],
}


# Gates registered at runtime from a config's custom_gates section. Checked
# before the built-in tables so a config can define (or override) a gate
# without touching this module.
_RUNTIME_SPECS: Dict[str, GateSpec] = {}


def register_gate_spec(spec: GateSpec) -> None:
    _RUNTIME_SPECS[spec.gate] = spec


def get_gate_spec(gate: str, variant: str = "v1") -> GateSpec:
    if gate in _RUNTIME_SPECS:
        return _RUNTIME_SPECS[gate]
    table = {"v1": GATE_SPECS, "rubric": RUBRIC_SPECS}.get(variant)
    if table is None:
        raise ValueError(f"Unknown prompt variant: {variant}. Available: v1, rubric")
    if gate not in table:
        raise ValueError(f"Unknown gate: {gate}. Available: {list(table)}")
    return table[gate]


def make_gate_text_prompt(text: str, question: str) -> str:
    """User prompt for text-based gate evaluation."""
    return f"Document:\n{text}\n\nQuestion:\n{question}"


def make_gate_image_prompt(question: str) -> str:
    """User prompt for image-based gate evaluation."""
    return f"Please evaluate the document shown in the image.\n\nQuestion:\n{question}"
