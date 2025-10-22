"""Wrapper around WhisperX alignment utilities."""

from __future__ import annotations

from typing import Any, Dict, Iterable


class WhisperXWrapper:
    """Thin wrapper to keep WhisperX integration isolated."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device

    def align(self, audio_paths: Iterable[str], texts: Iterable[str]) -> Iterable[Dict[str, Any]]:
        """Align audio and text using WhisperX.

        Args:
            audio_paths: Paths to audio files that require alignment.
            texts: Reference transcripts for the audio files.

        Returns:
            Iterable of alignment metadata dictionaries.
        """
        raise NotImplementedError("WhisperX alignment logic pending implementation.")
