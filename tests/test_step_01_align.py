"""Tests for the alignment pipeline stage."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict

import pytest


def _install_whisperx_stub() -> None:
    module = ModuleType("whisperx")

    def _load_model(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - stub
        return None

    def _load_align_model(*args: Any, **kwargs: Any):  # pragma: no cover - stub
        return ({}, {})

    def _load_audio(*args: Any, **kwargs: Any) -> None:  # pragma: no cover - stub
        return None

    def _align(segments, *args: Any, **kwargs: Any):  # pragma: no cover - stub
        return {"segments": segments}

    class _DiarizationPipeline:  # pragma: no cover - stub
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, *args: Any, **kwargs: Any):
            return []

    def _assign_word_speakers(diarize_segments, alignment):  # pragma: no cover
        return alignment

    module.load_model = _load_model
    module.load_align_model = _load_align_model
    module.load_audio = _load_audio
    module.align = _align
    module.DiarizationPipeline = _DiarizationPipeline
    module.assign_word_speakers = _assign_word_speakers
    module.__version__ = "stub"

    sys.modules.setdefault("whisperx", module)


_install_whisperx_stub()

from src.pipeline import step_01_align


class DummyWrapper:
    """Test double for ``WhisperXWrapper``."""

    version = "test-version"
    model_name = "dummy-model"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def transcribe_and_align(self, audio_path):
        return {
            "transcription": {"text": "안녕하세요 반갑습니다."},
            "alignment": {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.4,
                        "words": [
                            {
                                "word": "안녕하세요",
                                "start": 0.0,
                                "end": 0.7,
                                "confidence": 0.9,
                            },
                            {
                                "word": "반갑습니다",
                                "start": 0.7,
                                "end": 1.4,
                                "confidence": 0.95,
                            },
                        ],
                        "tokens": [
                            {"token": "안", "start": 0.0, "end": 0.2},
                            {"token": "녕", "start": 0.2, "end": 0.4},
                        ],
                    }
                ],
            },
        }


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def test_run_alignment_writes_expected_output(tmp_path, monkeypatch):
    monkeypatch.setattr(step_01_align, "WhisperXWrapper", DummyWrapper)

    raw_samples = tmp_path / "raw_samples.jsonl"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    audio_rel_path = "sample.wav"
    (audio_dir / audio_rel_path).write_bytes(b"test")

    _write_jsonl(
        raw_samples,
        [
            {
                "sample_id": "zeroth_train_0001",
                "audio_path": audio_rel_path,
                "text": "안녕하세요 반갑습니다.",
                "split": "train",
                "sr_hz": 16000,
                "channels": 1,
            }
        ],
    )

    output_path = tmp_path / "aligned.jsonl"
    config: Dict[str, Any] = {
        "paths": {
            "input_audio_dir": str(audio_dir),
            "raw_samples_path": str(raw_samples),
            "alignment_output_path": str(output_path),
        },
        "aligner": {
            "language": "ko",
            "model_name": "large-v3",
            "device": "cuda",
            "compute_type": "float16",
            "batch_size": 8,
            "vad_backend": "silero",
            "diarize": False,
            "rng_seed": 42,
        },
    }

    args = SimpleNamespace(raw_samples=None, out=None, split="train", limit=None)
    step_01_align.run_alignment(config, args)

    records = list(_read_jsonl(output_path))
    assert len(records) == 1
    record = records[0]

    assert record["sample_id"] == "zeroth_train_0001"
    assert record["status"] == "ok"
    assert record["alignment"]["words"][0]["w"] == "안녕하세요"
    assert record["alignment"]["words"][1]["end"] == 1.4
    assert record["speech_regions"][0]["start"] == 0.0
    assert record["tool_version"]["whisperx"] == DummyWrapper.version
    assert record["auto_transcript"] == "안녕하세요 반갑습니다."
    coverage = record["alignment"]["coverage"]
    assert coverage["speech_coverage"] == pytest.approx(1.0)
    assert coverage["aligned_word_ratio"] == pytest.approx(1.0)


def test_run_alignment_respects_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(step_01_align, "WhisperXWrapper", DummyWrapper)

    raw_samples = tmp_path / "raw_samples.jsonl"
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "sample.wav").write_bytes(b"test")

    _write_jsonl(
        raw_samples,
        [
            {
                "sample_id": f"sample_{idx}",
                "audio_path": "sample.wav",
                "text": "안녕하세요 반갑습니다.",
                "split": "train",
            }
            for idx in range(5)
        ],
    )

    output_path = tmp_path / "aligned_limit.jsonl"
    config: Dict[str, Any] = {
        "paths": {
            "input_audio_dir": str(audio_dir),
            "raw_samples_path": str(raw_samples),
            "alignment_output_path": str(output_path),
        },
        "aligner": {},
    }

    args = SimpleNamespace(raw_samples=None, out=None, split="train", limit=2)
    step_01_align.run_alignment(config, args)

    records = list(_read_jsonl(output_path))
    assert len(records) == 2
    assert {rec["sample_id"] for rec in records} == {"sample_0", "sample_1"}
