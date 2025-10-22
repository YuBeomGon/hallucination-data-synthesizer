"""Smoke tests for the label generator placeholders."""

from __future__ import annotations

import pytest

from src.modules.label_generator import LabelGenerator


def test_label_generator_instantiation() -> None:
    """LabelGenerator can be instantiated without arguments."""
    generator = LabelGenerator()
    assert isinstance(generator, LabelGenerator)


@pytest.mark.skip(reason="Pending label generation implementation.")
def test_build_labels_placeholder() -> None:
    """Placeholder to be replaced once label generation is implemented."""
    generator = LabelGenerator()
    with pytest.raises(NotImplementedError):
        _ = generator.build_labels([])
