"""Step 02: Augment audio with silence or background noise."""

from __future__ import annotations

from typing import Any, Dict


def run_augmentation(config: Dict[str, Any]) -> None:
    """Apply augmentation to aligned audio segments.

    Args:
        config: Full pipeline configuration dictionary.
    """
    raise NotImplementedError("Augmentation step not yet implemented.")
