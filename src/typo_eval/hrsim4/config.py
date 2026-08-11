"""Configuration schema for hrsim4.

Much smaller than hrsim/hrsim2's because v4 splits corpus construction from
execution. Sampling and rendering already happened in loop 1 (and passed 14 QC
gates), and JD pairing already happened in loop 2, so this config describes only
what to run: which arms, which items, how many repeats, against which models.

Reuses hrsim.config's leg/inference schema unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field

from typo_eval.hrsim.config import HrsimInferenceConfig, HrsimLegConfig

__all__ = [
    "HrsimLegConfig", "HrsimInferenceConfig", "HrsimV4Config", "load_hrsim_v4_config",
]

# The eight rendered arms, as built and QC'd by loop 1. Five typefaces plus the
# three-rung Roboto ink ladder; the ladder brackets the typefaces on measured
# ink coverage (0.0255-0.0665 vs 0.0325-0.0532) so "how much ink is on the page"
# can be separated from "which typeface it is" -- the confound v1-v3 could not
# address.
TYPEFACE_ARMS = ["T1_serif", "T3_sans", "T5_mono", "T7_comic", "A1_opendyslexic"]
INK_LADDER_ARMS = ["W1_roboto_light", "W2_roboto_regular", "W3_roboto_bold"]
TEXT_ARM = "TEXT"


class HrsimV4Config(BaseModel):
    """Top-level configuration for the hrsim4 experiment."""

    run_tag: str = "hrsim_v4"
    seed: int = 42
    repeats: int = 1

    # Arms to run. TEXT is the modality baseline and is added separately via
    # include_text_arm so it can't be confused with a rendered variant.
    image_arms: List[str] = Field(
        default_factory=lambda: TYPEFACE_ARMS + INK_LADDER_ARMS
    )
    include_text_arm: bool = True

    # Item selection. "all" runs every calibrated item; "sensitive" runs only
    # the boundary-sensitive subset from gate 3. Content-locked items cannot
    # flip and so carry no signal, but running everything is what lets us test
    # whether JD-perturbation sensitivity actually predicts font sensitivity.
    items: str = "all"

    corpus_dir: Path = Path("data/inputs/hrsim4")
    calibration_dir: Path = Path("results/hrsim/hrsim_v4")

    # Boundary calibration is model-specific -- the Sonnet-calibrated ranks put
    # Sol at ~74% accept, well above the band. Point this at a per-model
    # calibrated_ranks.csv to run that model at its own boundary. Defaults to
    # calibration_dir/calibrated_ranks.csv.
    ranks_path: Optional[Path] = None

    legs: List[HrsimLegConfig] = Field(default_factory=list)
    inference: HrsimInferenceConfig = Field(default_factory=HrsimInferenceConfig)

    def arms(self) -> List[str]:
        return ([TEXT_ARM] if self.include_text_arm else []) + list(self.image_arms)


def load_hrsim_v4_config(path: str | Path) -> HrsimV4Config:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text()) or {}
    return HrsimV4Config(**data)
