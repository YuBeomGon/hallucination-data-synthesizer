"""Voice activity detection helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class SilenceStats:
    """Summary of leading/trailing silence information."""

    leading_samples: int
    trailing_samples: int
    total_samples: int
    sample_rate: int
    backend: str

    @property
    def leading_sec(self) -> float:
        return self.leading_samples / max(1, self.sample_rate)

    @property
    def trailing_sec(self) -> float:
        return self.trailing_samples / max(1, self.sample_rate)

    @property
    def speech_samples(self) -> int:
        return max(0, self.total_samples - self.leading_samples - self.trailing_samples)

    @property
    def speech_sec(self) -> float:
        return self.speech_samples / max(1, self.sample_rate)


def _load_silero_model():
    import torch  # type: ignore

    try:
        model, utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            force_reload=False,
            onnx=False,
        )
        (get_speech_ts, _, _, _) = utils  # type: ignore[misc]
        return model, get_speech_ts
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(f"Failed to load Silero VAD model: {exc}") from exc


def _silero_ts(samples: np.ndarray, sample_rate: int, window_sec: float) -> List[Tuple[int, int]]:
    model, get_speech_ts = _load_silero_model()
    import torch  # type: ignore

    audio_tensor = torch.from_numpy(samples).float()
    speech_ts = get_speech_ts(
        audio_tensor,
        model,
        sampling_rate=sample_rate,
        min_speech_duration_ms=int(window_sec * 1000),
        window_size_samples=int(window_sec * sample_rate),
    )
    return [(seg["start"], seg["end"]) for seg in speech_ts]


def _energy_vad(samples: np.ndarray, sample_rate: int, window_sec: float) -> List[Tuple[int, int]]:
    window_samples = max(1, int(window_sec * sample_rate))
    if window_samples >= samples.size:
        rms = np.sqrt(np.mean(np.square(samples)) + 1e-12)
        if rms < 1e-4:
            return []
        return [(0, samples.size)]

    energies = []
    for start in range(0, samples.size, window_samples):
        end = min(samples.size, start + window_samples)
        window = samples[start:end]
        rms = np.sqrt(np.mean(np.square(window)) + 1e-12)
        energies.append(rms)

    energy_arr = np.array(energies)
    threshold = max(1e-5, np.median(energy_arr) * 0.5)
    speech_segments: List[Tuple[int, int]] = []
    in_speech = False
    seg_start = 0
    for idx, value in enumerate(energy_arr):
        if value >= threshold and not in_speech:
            in_speech = True
            seg_start = idx * window_samples
        elif value < threshold and in_speech:
            in_speech = False
            seg_end = min(samples.size, (idx + 1) * window_samples)
            speech_segments.append((seg_start, seg_end))
    if in_speech:
        speech_segments.append((seg_start, samples.size))
    return speech_segments


def compute_silence_stats(
    samples: np.ndarray,
    sample_rate: int,
    backend: str = "silero",
    window_sec: float = 0.03,
) -> SilenceStats:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1:
        samples = samples.mean(axis=1)

    try_backend = backend.lower()
    speech_segments: List[Tuple[int, int]] = []
    backend_used = try_backend

    if try_backend == "silero":
        try:
            speech_segments = _silero_ts(samples, sample_rate, window_sec)
        except Exception as exc:  # pragma: no cover - depends on environment
            LOGGER.warning("Silero VAD unavailable (%s); falling back to energy-based VAD.", exc)
            backend_used = "energy"
            speech_segments = _energy_vad(samples, sample_rate, window_sec)
    else:
        backend_used = "energy"
        speech_segments = _energy_vad(samples, sample_rate, window_sec)

    if not speech_segments:
        return SilenceStats(
            leading_samples=samples.size,
            trailing_samples=samples.size,
            total_samples=samples.size,
            sample_rate=sample_rate,
            backend=backend_used,
        )

    speech_segments.sort()
    first_start = speech_segments[0][0]
    last_end = speech_segments[-1][1]
    leading = max(0, first_start)
    trailing = max(0, samples.size - last_end)

    return SilenceStats(
        leading_samples=leading,
        trailing_samples=trailing,
        total_samples=samples.size,
        sample_rate=sample_rate,
        backend=backend_used,
    )

