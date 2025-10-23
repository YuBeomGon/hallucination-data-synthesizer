"""Unit tests for audio processing helpers."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.modules.audio_processor import AudioProcessor


@pytest.fixture()
def processor() -> AudioProcessor:
    return AudioProcessor(sample_rate=16000, crossfade_sec=0.05)


def test_match_snr_scales_noise(processor: AudioProcessor) -> None:
    speech = np.ones(16000, dtype=np.float32) * 0.1  # -20 dBFS approx
    noise = np.ones(16000, dtype=np.float32) * 0.1
    scaled, achieved = processor.match_snr(noise, speech, target_snr_db=10.0)
    ratio = processor.compute_rms(speech) / (processor.compute_rms(scaled) + 1e-9)
    snr = 20 * math.log10(ratio)
    assert pytest.approx(snr, abs=0.5) == 10.0
    assert pytest.approx(achieved, abs=0.5) == 10.0


def test_insert_with_crossfade_length(processor: AudioProcessor, tmp_path: Path) -> None:
    base = np.zeros(16000, dtype=np.float32)
    insert = np.ones(4000, dtype=np.float32) * 0.2
    augmented, info = processor.insert_with_crossfade(base, insert, insert_time_sec=0.5)
    expected_len = base.size - info["left_cross_samples"] - info["right_cross_samples"] + insert.size
    assert augmented.size == expected_len
    assert np.max(np.abs(augmented)) <= 1.0


def test_normalize_loudness_reduces_peak(processor: AudioProcessor) -> None:
    loud = np.ones(8000, dtype=np.float32) * 0.8
    report, normalized = processor.normalize_loudness(loud)
    assert report.true_peak_dbfs <= processor.true_peak_limit_dbfs + 1e-3
    assert normalized.dtype == np.float32


def test_load_waveform_resamples(processor: AudioProcessor, tmp_path: Path) -> None:
    path = tmp_path / "sine.wav"
    sr = 8000
    t = np.linspace(0, 1, sr, endpoint=False)
    wave = 0.2 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(path), wave, sr)
    loaded = processor.load_waveform(path)
    assert loaded.dtype == np.float32
    assert loaded.shape[0] == processor.sample_rate
