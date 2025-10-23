import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.pipeline.step_01_two_utterances import synthesize_split


SAMPLE_RATE = 16000


def _sine_wave(duration_sec: float, freq_hz: float = 440.0, amplitude: float = 0.2) -> np.ndarray:
    t = np.linspace(0.0, duration_sec, int(SAMPLE_RATE * duration_sec), endpoint=False, dtype=np.float32)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _noise_wave(duration_sec: float, amplitude: float = 0.05) -> np.ndarray:
    rng = np.random.default_rng(1234)
    samples = rng.standard_normal(int(SAMPLE_RATE * duration_sec)).astype(np.float32)
    return (samples * amplitude).astype(np.float32)


@pytest.mark.parametrize("allow_silence_prob", [0.0, 0.5])
def test_synthesize_split_creates_metadata_and_audio(tmp_path: Path, allow_silence_prob: float) -> None:
    assets_root = tmp_path / "assets"
    input_audio_dir = assets_root / "zeroth_v2" / "train" / "spk_000"
    noise_dir = assets_root / "noises"
    input_audio_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)

    wav_a = input_audio_dir / "utt_a.wav"
    wav_b = input_audio_dir / "utt_b.wav"
    sf.write(wav_a, _sine_wave(1.0, 440.0), SAMPLE_RATE)
    sf.write(wav_b, _sine_wave(1.2, 660.0), SAMPLE_RATE)

    noise_path = noise_dir / "background.wav"
    sf.write(noise_path, _noise_wave(5.0), SAMPLE_RATE)

    data_root = tmp_path / "data"
    raw_dir = data_root / "zeroth_v2"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_samples_path = raw_dir / "raw_samples_train.jsonl"
    samples = [
        {
            "sample_id": "sample_a",
            "speaker_id": "spk_000",
            "audio_path": "train/spk_000/utt_a.wav",
            "text": "안녕하세요.",
            "split": "train",
            "duration_sec": 1.0,
        },
        {
            "sample_id": "sample_b",
            "speaker_id": "spk_000",
            "audio_path": "train/spk_000/utt_b.wav",
            "text": "반갑습니다.",
            "split": "train",
            "duration_sec": 1.2,
        },
    ]
    with raw_samples_path.open("w", encoding="utf-8") as handle:
        for record in samples:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    noise_catalog_path = data_root / "noise" / "noise_catalog.csv"
    noise_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    noise_catalog_path.write_text(
        "audio_path,clip_start_sec,clip_end_sec,clip_duration_sec,category_01,category_02,category_03,source_sample_rate_hz,source_channels\n"
        "background.wav,0,5,5,,,ambient,16000,1\n",
        encoding="utf-8",
    )

    config = {
        "paths": {
            "input_audio_dir": str(assets_root / "zeroth_v2"),
            "noise_dir": str(noise_dir),
            "augmented_audio_dir": str(data_root / "augmented_audio_v2"),
            "metadata_dir": str(data_root / "labels_v2"),
            "raw_samples_template": str(raw_dir / "raw_samples_{split}.jsonl"),
            "noise_catalog": str(noise_catalog_path),
            "noise_resampled_dir": str(data_root / "noise" / "resampled"),
        },
        "selection": {
            "min_utterance_sec": 0.5,
            "max_utterance_sec": 5.0,
            "max_total_duration_sec": 10.0,
            "max_length_ratio": 3.0,
            "rng_seed": 321,
        },
        "synthesis": {
            "crossfade_sec": 0.05,
            "target_snr_db": 10.0,
            "loudness_target_lufs": -23.0,
            "true_peak_dbfs": -1.0,
            "transition": {
                "min_noise_sec": 1.0,
                "max_noise_sec": 1.2,
                "min_pause_sec": 0.1,
                "max_pause_sec": 0.1,
                "allow_silence_prob": allow_silence_prob,
                "fade_ms": 15,
                "context_window_sec": 0.2,
            },
            "time_stretch": {
                "enable": False,
                "min_ratio": 0.95,
                "max_ratio": 1.05,
            },
            "noise_categories": [],
        },
        "labelling": {
            "baseline_model_name": "openai/whisper-large-v3",
            "transition_token": "<SIL_TRANS>",
            "include_silence_token": True,
            "max_response_sec": 45.0,
        },
        "metadata": {
            "enable_tracking": True,
            "save_source_text": True,
            "save_transition_stats": True,
        },
    }

    stats = synthesize_split(config=config, split="train", limit=1)

    assert stats.generated == 1
    assert stats.skipped == 0
    assert stats.errors == 0

    meta_path = Path(config["paths"]["metadata_dir"]) / "train" / "paired_meta.jsonl"
    assert meta_path.exists()

    with meta_path.open("r", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle if line.strip()]

    assert len(lines) == 1
    record = lines[0]
    assert record["status"] == "ok"
    assert record["segments"]["utterance_a"]["sample_id"] == "sample_a"
    assert record["segments"]["utterance_b"]["sample_id"] == "sample_b"
    assert "<SIL_TRANS>" in record["text"]["combined_with_token"]

    transition = record["segments"]["transition"]
    if allow_silence_prob < 0.5:
        assert transition["type"] == "noise"
        assert transition["noise"]["source_path"] is not None
    else:
        assert transition["type"] in {"noise", "silence"}

    augmented_root = Path(config["paths"]["augmented_audio_dir"]).resolve()
    output_audio = augmented_root / record["audio"]["output_path"]
    assert output_audio.exists()

    out_data, _ = sf.read(str(output_audio))
    assert out_data.size > 0
