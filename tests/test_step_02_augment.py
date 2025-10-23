"""Tests for the augmentation pipeline."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf

from src.modules.audio_processor import AudioProcessor
from src.modules.noise_selector import NoiseSelector
from src.pipeline.step_02_augment import AugmentationConfig, process_record


def write_noise_catalog(path: Path, audio_rel_path: Path, duration: float) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "dataset_name",
            "split",
            "category_dir",
            "subcategory_dir",
            "label_json_path",
            "audio_path",
            "label_name",
            "clip_start_sec",
            "clip_end_sec",
            "clip_duration_sec",
            "category_01",
            "category_02",
            "category_03",
            "sub_category",
            "decibel",
            "source_sample_rate_hz",
            "source_channels",
            "audio_duration_sec",
        ])
        writer.writerow([
            "noise_dataset",
            "train",
            "1.자동차",
            "",
            "",
            str(audio_rel_path),
            audio_rel_path.name,
            0.0,
            duration,
            duration,
            "교통소음",
            "자동차",
            "차량경적",
            "",
            70.0,
            16000,
            1,
            duration,
        ])


def build_alignment_record(audio_path: Path) -> dict:
    return {
        "sample_id": "sample001",
        "split": "train",
        "audio_path": str(audio_path),
        "alignment": {
            "words": [
                {"w": "hello", "start": 0.0, "end": 0.2},
                {"w": "world", "start": 0.8, "end": 1.0},
            ]
        },
        "speech_regions": [{"start": 0.0, "end": 0.2}, {"start": 0.8, "end": 1.0}],
    }


def test_process_record_creates_augmented_audio(tmp_path: Path) -> None:
    sample_rate = 16000
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    audio_path = audio_root / "sample.wav"
    base_wave = np.zeros(sample_rate, dtype=np.float32)
    sf.write(str(audio_path), base_wave, sample_rate)

    noise_root = tmp_path / "noises"
    noise_root.mkdir()
    noise_rel = Path("noise_sample.wav")
    noise_path = noise_root / noise_rel
    noise_wave = np.random.uniform(-0.1, 0.1, size=8000).astype(np.float32)
    sf.write(str(noise_path), noise_wave, sample_rate)

    catalog_path = tmp_path / "noise_catalog.csv"
    write_noise_catalog(catalog_path, noise_rel, duration=0.8)

    audio_processor = AudioProcessor(sample_rate=sample_rate, crossfade_sec=0.05)
    noise_selector = NoiseSelector(
        catalog_path=catalog_path,
        noise_root=noise_root,
        rng=random.Random(1234),
        allow_categories=None,
    )

    config = AugmentationConfig(
        min_gap_sec=0.5,
        insertion_min_sec=0.5,
        insertion_max_sec=0.7,
        crossfade_sec=0.05,
        context_window_sec=0.5,
        target_snr_db=10.0,
        loudness_target_lufs=-23.0,
        true_peak_dbfs=-1.0,
        insertions_per_file=1,
        rng_seed=1234,
        noise_categories=None,
    )

    augmented_dir = tmp_path / "aug"
    augmented_dir.mkdir()

    record = build_alignment_record(audio_path)
    child_seed = 5678
    result = process_record(
        record=record,
        audio_processor=audio_processor,
        noise_selector=noise_selector,
        rng=random.Random(child_seed),
        seed=child_seed,
        config=config,
        input_audio_root=audio_root,
        augmented_audio_dir=augmented_dir,
    )

    assert result["status"] == "ok"
    aug_path = Path(result["augmented_audio_path"])
    assert aug_path.exists()
    assert aug_path.stat().st_size > 0
    assert result["augmentation"] is not None
    assert len(result["offset_map"]) >= 3
