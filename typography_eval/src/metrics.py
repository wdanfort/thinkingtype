from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

TRUE_SET = {"1", "yes", "true", "y", "t"}
FALSE_SET = {"0", "no", "false", "n", "f"}


def parse_to_01(value) -> Tuple[Optional[int], str]:
    """
    Normalize a response to (0|1, normalized_str). Returns (None, normalized_str)
    when the value cannot be parsed.
    """
    if value is None:
        return None, ""

    if isinstance(value, (int, np.integer)):
        if int(value) in (0, 1):
            return int(value), str(int(value))
        return None, str(value)

    if isinstance(value, (float, np.floating)):
        if value in (0.0, 1.0):
            return int(value), str(int(value))
        return None, str(value)

    text = str(value).strip().lower()
    if text in TRUE_SET:
        return 1, "yes"
    if text in FALSE_SET:
        return 0, "no"
    return None, text[:200]


def coerce_to_01(value) -> Optional[int]:
    parsed, _ = parse_to_01(value)
    return parsed
