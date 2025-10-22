"""Wrapper around WhisperX alignment utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import whisperx


LOGGER = logging.getLogger(__name__)


class WhisperXWrapper:
    """Thin wrapper that encapsulates WhisperX transcription and alignment."""

    def __init__(
        self,
        model_name: str,
        device: str,
        language: Optional[str] = None,
        compute_type: Optional[str] = None,
        batch_size: int = 8,
        vad_backend: Optional[str] = None,
        diarize: bool = False,
    ) -> None:
        """Initialize WhisperX wrapper without pyannote (VAD/diarization disabled)."""
        self.model_name = model_name
        self.device = device
        self.language = language
        self.batch_size = batch_size

        # Compute type: pick safe defaults per device if not provided
        if compute_type is None:
            compute_type = "float16" if str(device).startswith("cuda") else "float32"
        self.compute_type = compute_type

        # 🔒 Force-disable VAD & diarization to avoid any pyannote path
        requested_vad = (vad_backend or "none").lower()
        if requested_vad != "none":
            LOGGER.warning("VAD disabled by wrapper (requested '%s'); forcing vad_backend='none'.", requested_vad)
        self.vad_backend = "none"

        if diarize:
            LOGGER.warning("Diarization disabled by wrapper; forcing diarize=False.")
        self.diarize = False
        self.diarization_pipeline = None  # never load pyannote pipeline

        # Load WhisperX base model
        load_kwargs: Dict[str, Any] = {"device": self.device, "compute_type": self.compute_type}
        if self.language:
            load_kwargs["language"] = self.language

        LOGGER.info(
            "Loading WhisperX model=%s device=%s compute_type=%s lang=%s (VAD=none, diarize=False)",
            self.model_name, self.device, self.compute_type, self.language or "<auto>",
        )
        self.model = whisperx.load_model(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            language=self.language,
            vad_method="silero",        # 🔑 여기가 핵심
            # asr_options={}            # 필요시 ASR 옵션을 dict로 넘길 수 있음
        )

        # Load alignment model (default to Korean if not provided)
        lang_code = self.language or "ko"
        LOGGER.info("Loading align model for language_code=%s on device=%s", lang_code, self.device)
        self.align_model, self.align_metadata = whisperx.load_align_model(
            language_code=lang_code,
            device=self.device,
        )

        # Version
        self._version = getattr(whisperx, "__version__", "unknown")

    @property
    def version(self) -> str:
        """Return the underlying WhisperX version string."""

        return self._version

    def transcribe_and_align(self, audio_path: Path) -> Dict[str, Any]:
        """Transcribe and align a single audio file.

        Args:
            audio_path: Path to the audio file to process.

        Returns:
            Dictionary containing transcription and alignment results.
        """

        audio_path = audio_path.resolve()
        LOGGER.debug("Loading audio %s", audio_path)
        audio = whisperx.load_audio(str(audio_path))

        LOGGER.debug("Transcribing %s", audio_path.name)
        transcription = self.model.transcribe(
            audio,
            batch_size=self.batch_size,
            language=self.language,
        )

        if self.vad_backend not in (None, "", "none"):
            LOGGER.warning(
                "Configured vad_backend=%s but current WhisperX version does not "
                "support external VAD injection via API; proceeding without VAD.",
                self.vad_backend,
            )

        LOGGER.debug("Aligning %s", audio_path.name)
        alignment = whisperx.align(
            transcription.get("segments", []),
            self.align_model,
            self.align_metadata,
            audio,
            device=self.device,
        )

        if self.diarize and self.diarization_pipeline is not None:
            LOGGER.debug("Assigning speakers for %s", audio_path.name)
            diarize_segments = self.diarization_pipeline(str(audio_path))
            alignment = whisperx.assign_word_speakers(diarize_segments, alignment)

        return {
            "transcription": transcription,
            "alignment": alignment,
        }
