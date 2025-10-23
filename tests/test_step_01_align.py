"""Tests for the alignment pipeline stage."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest


def _install_whisperx_stub() -> None:
    whisperx_module = ModuleType("whisperx")

    def load_align_model(*args: Any, **kwargs: Any):  # pragma: no cover - stub
        return ("align_model", {"language": kwargs.get("language_code", "ko")})

    def load_audio(path: str) -> np.ndarray:  # pragma: no cover - stub
        return np.zeros(16000, dtype=np.float32)

    def align(segments, *args: Any, **kwargs: Any):  # pragma: no cover - stub
        assert segments and "start" in segments[0]
        words = [
            {"word": "안녕하세요", "start": 0.0, "end": 0.7, "score": 0.9},
            {"word": "반갑습니다", "start": 0.7, "end": 1.4, "score": 0.95},
        ]
        return {
            "word_segments": words,
            "audio_duration_sec": 1.4,
            "segment_start": segments[0].get("start", 0.0),
            "segment_end": segments[0].get("end", 1.4),
        }

    whisperx_module.load_align_model = load_align_model
    whisperx_module.load_audio = load_audio
    whisperx_module.align = align
    whisperx_module.__version__ = "stub"

    sys.modules["whisperx"] = whisperx_module

    token_module = ModuleType("tokenizer_stub")

    class DummyTokenizer:  # pragma: no cover - stub
        def encode(self, text: str):
            tokens = list(range(len(text.split()))) or [0]
            return tokens, text

    def get_tokenizer(*args: Any, **kwargs: Any):  # pragma: no cover - stub
        return DummyTokenizer()

    token_module.get_tokenizer = get_tokenizer
    utils_module = ModuleType("whisperx.utils")
    utils_module.tokenization = token_module
    sys.modules["whisperx.utils"] = utils_module
    sys.modules["whisperx.utils.tokenization"] = token_module

    whisper_module = ModuleType("whisper")
    tokenizer_pkg = ModuleType("whisper.tokenizer")
    tokenizer_pkg.get_tokenizer = get_tokenizer
    whisper_module.tokenizer = tokenizer_pkg
    sys.modules["whisper"] = whisper_module
    sys.modules["whisper.tokenizer"] = tokenizer_pkg


_install_whisperx_stub()

from src.pipeline import step_01_align


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

    args = SimpleNamespace(
        raw_samples=str(raw_samples),
        out=str(output_path),
        split="train",
        limit=None,
    )
    step_01_align.run_alignment(config, args)

    records = list(_read_jsonl(output_path))
    assert len(records) == 1
    record = records[0]

    assert record["sample_id"] == "zeroth_train_0001"
    assert record["status"] == "ok"
    assert record["alignment"]["words"][0]["w"] == "안녕하세요"
    assert record["alignment"]["words"][1]["end"] == 1.4
    assert record["speech_regions"][0]["start"] == 0.0
    assert record["tool_version"]["whisperx"] == "stub"
    assert record["auto_transcript"] == "안녕하세요 반갑습니다."
    coverage = record["alignment"]["coverage"]
    assert coverage["speech_coverage"] == pytest.approx(1.0)
    assert coverage["aligned_word_ratio"] == pytest.approx(1.0)


def test_run_alignment_respects_limit(tmp_path, monkeypatch):
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
        },
        "aligner": {},
    }

    args = SimpleNamespace(
        raw_samples=str(raw_samples),
        out=str(output_path),
        split="train",
        limit=2,
    )
    step_01_align.run_alignment(config, args)

    records = list(_read_jsonl(output_path))
    assert len(records) == 2
    assert {rec["sample_id"] for rec in records} == {"sample_0", "sample_1"}
