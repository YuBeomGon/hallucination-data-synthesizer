"""Audio processing utilities for the Hallucination Data Synthesizer."""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass
import math
from typing import Dict, Tuple

import numpy as np
import soundfile as sf

try:  # pragma: no cover - optional dependency
    import librosa
    import pyloudnorm as pyln
except ImportError:  # pragma: no cover - optional dependency
    librosa = None  # type: ignore
    pyln = None


LOGGER = logging.getLogger(__name__)


EPS = 1e-9


@dataclass
class LoudnessReport:
    """Summary of loudness and peak information after processing."""

    lufs_before: float
    lufs_after: float
    true_peak_dbfs: float
    clip_guard_applied: bool


class AudioProcessor:
    """Handle audio loading, augmentation, and smoothing operations."""

    def __init__(
        self,
        sample_rate: int = 16000,
        crossfade_sec: float = 0.05,
        loudness_target_lufs: float = -23.0,
        true_peak_dbfs: float = -1.0,
        context_window_sec: float = 0.75,
    ) -> None:
        self.sample_rate = sample_rate
        self.crossfade_sec = crossfade_sec
        self.crossfade_samples = max(1, int(round(crossfade_sec * sample_rate)))
        self.loudness_target_lufs = loudness_target_lufs
        self.true_peak_limit_dbfs = true_peak_dbfs
        self.context_window_sec = context_window_sec
        self.context_window_samples = max(1, int(round(context_window_sec * sample_rate)))
        self._meter = pyln.Meter(sample_rate) if pyln else None

    # ---------------------------------------------------------------------
    # Audio loading helpers
    # ---------------------------------------------------------------------
    def load_waveform(self, path: Path) -> np.ndarray:
        """Load an audio file into memory and convert to mono float32."""

        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        data = self._ensure_mono(data)
        if sr != self.sample_rate:
            data = self._resample(data, sr, self.sample_rate)
        return data.astype(np.float32, copy=False)

    def load_segment(self, path: Path, start_sec: float, duration_sec: float) -> np.ndarray:
        """Load a segment of audio starting at ``start_sec`` for ``duration_sec``."""

        with sf.SoundFile(str(path)) as sound_file:
            sr = sound_file.samplerate
            start_frame = max(0, int(start_sec * sr))
            frames = int(duration_sec * sr)
            sound_file.seek(start_frame)
            data = sound_file.read(frames, dtype="float32", always_2d=False)

        data = self._ensure_mono(data)
        if sr != self.sample_rate:
            data = self._resample(data, sr, self.sample_rate)
        return data.astype(np.float32, copy=False)

    # ---------------------------------------------------------------------
    # Loudness / RMS helpers
    # ---------------------------------------------------------------------
    def compute_rms(self, samples: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(samples)) + EPS))

    def integrated_loudness(self, samples: np.ndarray) -> float:
        if samples.size == 0:
            return float("-inf")
        if samples.size == 0:
            return float("-inf")
        if self._meter is None:
            rms = self.compute_rms(samples)
            return 20 * np.log10(rms + EPS)
        return float(self._meter.integrated_loudness(samples.astype(np.float64, copy=False)))

    def true_peak_dbfs(self, samples: np.ndarray) -> float:
        peak = float(np.max(np.abs(samples)) + EPS)
        return 20.0 * np.log10(peak)

    def match_snr(self, noise: np.ndarray, context: np.ndarray, target_snr_db: float) -> Tuple[np.ndarray, float]:
        """Scale ``noise`` to achieve ``target_snr_db`` relative to ``context`` RMS."""

        context_rms = self.compute_rms(context)
        noise_rms = self.compute_rms(noise)
        if noise_rms < EPS or context_rms < EPS:
            return noise, float("inf")
        desired_noise_rms = context_rms / (10 ** (target_snr_db / 20))
        scale = desired_noise_rms / noise_rms
        scaled = noise * scale
        achieved = 20 * np.log10((context_rms + EPS) / (self.compute_rms(scaled) + EPS))
        return scaled.astype(np.float32, copy=False), achieved

    def normalize_loudness(self, samples: np.ndarray) -> LoudnessReport:
        """Normalize loudness and enforce true peak guard."""

        current_lufs = self.integrated_loudness(samples)
        gain_db = self.loudness_target_lufs - current_lufs if math.isfinite(current_lufs) else 0.0
        scaled = samples * (10 ** (gain_db / 20))
        peak_dbfs = self.true_peak_dbfs(scaled)
        clip_guard = False
        if peak_dbfs > self.true_peak_limit_dbfs:
            clip_guard = True
            reduction_db = self.true_peak_limit_dbfs - peak_dbfs
            scaled = scaled * (10 ** (reduction_db / 20))
            peak_dbfs = self.true_peak_dbfs(scaled)

        np.clip(scaled, -1.0, 1.0, out=scaled)
        final_lufs = self.integrated_loudness(scaled)
        return LoudnessReport(
            lufs_before=current_lufs,
            lufs_after=final_lufs,
            true_peak_dbfs=peak_dbfs,
            clip_guard_applied=clip_guard,
        ), scaled.astype(np.float32, copy=False)

    # ---------------------------------------------------------------------
    # Crossfade / fade helpers
    # ---------------------------------------------------------------------
    def apply_fades(self, samples: np.ndarray, fade_sec: float = 0.02) -> np.ndarray:
        fade_samples = max(1, int(round(fade_sec * self.sample_rate)))
        fade_samples = min(fade_samples, samples.size // 2)
        if fade_samples <= 0:
            return samples
        fade_in_curve = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype=np.float32)
        fade_out_curve = np.linspace(1.0, 0.0, fade_samples, endpoint=False, dtype=np.float32)
        result = samples.copy()
        result[:fade_samples] *= fade_in_curve
        result[-fade_samples:] *= fade_out_curve
        return result

    def insert_with_crossfade(
        self,
        base: np.ndarray,
        insert: np.ndarray,
        insert_time_sec: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Insert ``insert`` into ``base`` at ``insert_time_sec`` with crossfade."""

        insert_index = max(0, int(round(insert_time_sec * self.sample_rate)))
        insert_index = min(insert_index, base.size)
        crossfade = self.crossfade_samples

        left_cross = min(crossfade, insert_index)
        right_available = base.size - insert_index
        right_cross = min(crossfade, right_available)

        # Split base waveform around insertion point
        pre = base[: insert_index - left_cross]
        left_base = base[insert_index - left_cross : insert_index]
        right_base = base[insert_index : insert_index + right_cross]
        post = base[insert_index + right_cross :]

        insert_len = insert.size
        if insert_len < left_cross + right_cross:
            padding = left_cross + right_cross - insert_len
            insert = np.pad(insert, (0, padding), mode="constant")
            insert_len = insert.size

        left_insert = insert[:left_cross]
        mid_insert = insert[left_cross : insert_len - right_cross]
        right_insert = insert[insert_len - right_cross :]

        fade_in = np.linspace(0.0, 1.0, left_cross, endpoint=False, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, right_cross, endpoint=False, dtype=np.float32)

        left_mix = left_base * (1.0 - fade_in) + left_insert * fade_in if left_cross > 0 else np.array([], dtype=np.float32)
        right_mix = right_insert * (1.0 - fade_out) + right_base * fade_out if right_cross > 0 else np.array([], dtype=np.float32)

        result = np.concatenate(
            [
                pre,
                left_mix,
                mid_insert,
                right_mix,
                post,
            ]
        )
        np.clip(result, -1.0, 1.0, out=result)

        info = {
            "insert_index": insert_index,
            "left_cross_samples": int(left_cross),
            "right_cross_samples": int(right_cross),
        }
        return result.astype(np.float32, copy=False), info

    # ------------------------------------------------------------------
    # Context utilities
    # ------------------------------------------------------------------
    def context_window(self, samples: np.ndarray, insert_time_sec: float) -> np.ndarray:
        insert_index = int(round(insert_time_sec * self.sample_rate))
        start = max(0, insert_index - self.context_window_samples)
        end = min(samples.size, insert_index + self.context_window_samples)
        if end <= start:
            return samples
        window = samples[start:end]
        if window.size == 0:
            return samples
        return window

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_mono(self, samples: np.ndarray) -> np.ndarray:
        if samples.ndim == 1:
            return samples
        return samples.mean(axis=1)

    def _resample(self, samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return samples
        if librosa is not None:
            try:
                return librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr)
            except Exception as exc:  # pragma: no cover - fallback to linear
                LOGGER.warning("librosa.resample failed (%s); falling back to linear interpolation", exc)
        duration = samples.size / float(orig_sr)
        target_len = max(1, int(round(duration * target_sr)))
        original_times = np.linspace(0.0, duration, samples.size, endpoint=False, dtype=np.float32)
        target_times = np.linspace(0.0, duration, target_len, endpoint=False, dtype=np.float32)
        resampled = np.interp(target_times, original_times, samples).astype(np.float32, copy=False)
        return resampled
