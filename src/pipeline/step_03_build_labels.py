"""Step 03: Build SFT and DPO labels from augmented audio."""

from __future__ import annotations

from typing import Any, Dict


def run_label_construction(config: Dict[str, Any]) -> None:
    """Generate labels required for SFT and DPO training.

    Args:
        config: Full pipeline configuration dictionary.
    """
    raise NotImplementedError("Label construction step not yet implemented.")
