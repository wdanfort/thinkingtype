"""Metrics and parsing utilities."""

from __future__ import annotations

import math
from typing import Tuple


def parse_yes_no(raw: str | None) -> Tuple[int | None, str]:
    if raw is None:
        return None, ""
    s = raw.strip().lower()
    if s.startswith("yes"):
        return 1, "yes"
    if s.startswith("no"):
        return 0, "no"
    return None, s[:200]


def coerce_to_binary(value) -> int | None:
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
