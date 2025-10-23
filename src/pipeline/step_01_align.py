"""Step 01: Align transcripts using WhisperX alignment without ASR transcription."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from tqdm import tqdm

import whisper
import whisperx

from src.utils.config_loader import load_yaml
from src.utils.file_io import write_jsonl
from src.utils.logging_config import configure_logging


LOGGER = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class RawSample:
    sample_id: str
    audio_path: Path
    text: str
    split: str
    language: Optional[str] = None
    sr_hz: Optional[int] = None
    channels: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)


class TranscriptAligner:
    """Align fixed transcripts to audio using WhisperX CTC alignment."""

    def __init__(self, model_name: str, device: str, language: Optional[str]) -> None:
        self.model_name = model_name
        self.device = device
        self.language = language or "en"
        LOGGER.info("Loading tokenizer for %s (%s)", model_name, self.language)
        multilingual = not model_name.endswith(".en")
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=multilingual,
            language=self.language,
        )
        LOGGER.info("Loading align model (%s, %s)", self.language, device)
        self.align_model, self.metadata = whisperx.load_align_model(
            language_code=self.language,
            device=device,
        )
        self.version = getattr(whisperx, "__version__", "unknown")

    def align(self, audio_path: Path, text: str) -> Dict[str, Any]:
        audio = whisperx.load_audio(str(audio_path))
        tokens = self.tokenizer.encode(text)[0]
        duration = len(audio) / SAMPLE_RATE
        segments = [
            {
                "id": 0,
                "seek": 0,
                "start": 0.0,
                "end": duration,
                "text": text,
                "tokens": tokens,
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0,
            }
        ]
        result = whisperx.align(
            segments,
            self.align_model,
            self.metadata,
            audio,
            device=self.device,
        )
        result["audio_duration_sec"] = duration
        result["segment_start"] = 0.0
        result["segment_end"] = duration
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default_config.yaml"))
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--raw-samples", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_raw_samples(
    raw_path: Path,
    audio_root: Path,
    split_filter: Optional[str],
    limit: Optional[int],
) -> Iterator[RawSample]:
    emitted = 0
    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if split_filter and payload.get("split") != split_filter:
                continue

            sample_id = payload.get("sample_id")
            audio_rel = payload.get("audio_path")
            text = payload.get("text", "")
            if not sample_id or not audio_rel:
                LOGGER.warning("Skipping malformed sample entry: %s", payload)
                continue

            audio_path = Path(audio_rel)
            if not audio_path.is_absolute():
                audio_path = (audio_root / audio_path).resolve()

            extras = {
                key: value
                for key, value in payload.items()
                if key not in {"sample_id", "audio_path", "text", "language", "split", "sr_hz", "channels"}
            }

            yield RawSample(
                sample_id=sample_id,
                audio_path=audio_path,
                text=text,
                language=payload.get("language"),
                split=payload.get("split", "unknown"),
                sr_hz=payload.get("sr_hz"),
                channels=payload.get("channels"),
                extras=extras,
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                break


def build_alignment_record(
    sample: RawSample,
    align_result: Dict[str, Any],
    aligner: TranscriptAligner,
    aligner_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    words = align_result.get("word_segments", [])
    if not words:
        raise ValueError("Alignment produced no word segments")

    segment_start = float(align_result.get("segment_start", 0.0))
    segment_end = float(align_result.get("segment_end", segment_start))
    first_start = float(words[0].get("start", segment_start))
    last_end = float(words[-1].get("end", segment_end))
    audio_duration = align_result.get("audio_duration_sec") or max(segment_end, last_end)
    speech_duration = max(0.0, last_end - first_start)
    speech_coverage = min(1.0, speech_duration / audio_duration) if audio_duration else 0.0
    word_count = len(sample.text.split()) or 1
    avg_conf = np.mean([w.get("score") for w in words if w.get("score") is not None])

    alignment_words = [
        {
            "w": w.get("word"),
            "start": w.get("start"),
            "end": w.get("end"),
            "score": w.get("score"),
        }
        for w in words
    ]

    coverage = {
        "speech_coverage": speech_coverage,
        "aligned_word_ratio": 1.0,
        "avg_conf": float(avg_conf) if avg_conf is not None else None,
        "audio_duration_sec": audio_duration,
    }

    record = {
        "sample_id": sample.sample_id,
        "audio_path": str(sample.audio_path),
        "text": sample.text,
        "split": sample.split,
        "language": sample.language or aligner_cfg.get("language"),
        "alignment": {
            "words": alignment_words,
            "tokens": [],
            "coverage": coverage,
        },
        "speech_regions": [
            {
                "start": first_start,
                "end": last_end,
            }
        ],
        "tool_version": {"whisperx": aligner.version},
        "model_name": aligner.model_name,
        "rng_seed": aligner_cfg.get("rng_seed"),
        "auto_transcript": sample.text,
        "status": "ok",
        "error_msg": None,
    }

    if sample.sr_hz is not None:
        record["sr_hz"] = sample.sr_hz
    if sample.channels is not None:
        record["channels"] = sample.channels
    if sample.extras:
        record.update(sample.extras)

    return record


def run_alignment(config: Dict[str, Any], args: argparse.Namespace) -> None:
    paths = config.get("paths", {})
    aligner_cfg = config.get("aligner", {})

    raw_samples_path = Path(args.raw_samples) if args.raw_samples else Path(paths.get("raw_samples_path", "data/raw_samples.jsonl"))

    if args.out:
        output_path = Path(args.out)
    else:
        output_base = Path(paths.get("alignment_output_dir", paths.get("label_dir", "data/labels")))
        if args.split:
            output_path = output_base / args.split / "raw_alignment.jsonl"
        else:
            output_path = output_base / "raw_alignment.jsonl"

    audio_root = Path(paths.get("input_audio_dir", ".")).resolve()

    aligner = TranscriptAligner(
        model_name=aligner_cfg.get("model_name", "large-v3"),
        device=aligner_cfg.get("device", "cuda"),
        language=aligner_cfg.get("language"),
    )

    LOGGER.info("Loading raw samples from %s", raw_samples_path)
    samples = load_raw_samples(
        raw_path=raw_samples_path,
        audio_root=audio_root,
        split_filter=args.split,
        limit=args.limit,
    )

    records: List[Dict[str, Any]] = []
    for sample in tqdm(list(samples), desc="Aligning"):
        try:
            align_result = aligner.align(sample.audio_path, sample.text)
            record = build_alignment_record(sample, align_result, aligner, aligner_cfg)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Failed sample %s", sample.sample_id)
            record = {
                "sample_id": sample.sample_id,
                "audio_path": str(sample.audio_path),
                "text": sample.text,
                "split": sample.split,
                "language": sample.language or aligner_cfg.get("language"),
                "alignment": None,
                "speech_regions": [],
                "tool_version": {"whisperx": aligner.version},
                "model_name": aligner.model_name,
                "rng_seed": aligner_cfg.get("rng_seed"),
                "auto_transcript": sample.text,
                "status": "error",
                "error_msg": str(exc),
            }
        records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, records)
    LOGGER.info("Alignment records written to %s", output_path)


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_yaml(args.config)
    run_alignment(config, args)


if __name__ == "__main__":
    main()
