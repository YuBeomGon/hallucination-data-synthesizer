"""Smoke tests for the audio processor placeholders."""

from __future__ import annotations

import pytest

from src.modules.audio_processor import AudioProcessor


def test_audio_processor_init() -> None:
    """AudioProcessor stores the provided sample rate."""
    processor = AudioProcessor(sample_rate=16000)
    assert processor.sample_rate == 16000


@pytest.mark.skip(reason="Pending audio processing implementation.")
def test_insert_silence_placeholder() -> None:
    """Placeholder to be replaced once silence insertion is implemented."""
    processor = AudioProcessor(sample_rate=16000)
    with pytest.raises(NotImplementedError):
        _ = processor.insert_silence([], duration_ms=1000)
