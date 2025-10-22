"""Step 01: Generate precise word-level alignments using WhisperX."""

from __future__ import annotations

import argparse
import json
import logging
from tqdm import tqdm
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from src.modules.whisperx_wrapper import WhisperXWrapper
from src.utils.config_loader import load_yaml
from src.utils.file_io import write_jsonl
from src.utils.logging_config import configure_logging


LOGGER = logging.getLogger(__name__)


@dataclass
class RawSample:
    """Container for raw sample metadata prior to alignment."""

    sample_id: str
    audio_path: Path
    text: str
    split: str
    language: Optional[str] = None
    sr_hz: Optional[int] = None
    channels: Optional[int] = None
    extras: Dict[str, Any] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the alignment step."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default_config.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional dataset split to process (e.g., train, validation).",
    )
    parser.add_argument(
        "--raw-samples",
        type=Path,
        default=None,
        help="Override path to the raw samples JSONL file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination path for the alignment JSONL output.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples to process (for smoke tests).",
    )
    return parser.parse_args()


def load_raw_samples(
    raw_path: Path,
    audio_root: Path,
    split_filter: Optional[str] = None,
    limit: Optional[int] = None,
) -> Iterator[RawSample]:
    """Yield raw samples from a JSONL file matching the optional split filter."""

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


def compute_alignment_metrics(
    alignment: Dict[str, Any],
    reference_text: str,
) -> Dict[str, Optional[float]]:
    """Compute coverage metrics from an alignment payload."""

    segments = alignment.get("segments", [])
    speech_duration = 0.0
    audio_end = 0.0
    aligned_word_count = 0
    confidence_total = 0.0
    confidence_count = 0

    for segment in segments:
        start = float(segment.get("start")) if segment.get("start") is not None else None
        end = float(segment.get("end")) if segment.get("end") is not None else None
        if start is not None and end is not None:
            speech_duration += max(0.0, end - start)
            audio_end = max(audio_end, end)

        for word in segment.get("words", []):
            if word.get("start") is not None and word.get("end") is not None:
                aligned_word_count += 1
            if word.get("confidence") is not None:
                confidence_total += float(word["confidence"])
                confidence_count += 1

    speech_coverage = speech_duration / audio_end if audio_end else 0.0
    reference_word_count = len(reference_text.split()) if reference_text else 0
    aligned_word_ratio = (
        aligned_word_count / reference_word_count if reference_word_count else None
    )
    avg_conf = confidence_total / confidence_count if confidence_count else None

    return {
        "speech_coverage": speech_coverage,
        "aligned_word_ratio": aligned_word_ratio,
        "avg_conf": avg_conf,
        "audio_duration_sec": audio_end or None,
    }


def convert_words(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten word-level information from aligned segments."""

    words: List[Dict[str, Any]] = []
    for segment in segments:
        for word in segment.get("words", []):
            words.append(
                {
                    "w": word.get("word"),
                    "start": word.get("start"),
                    "end": word.get("end"),
                    "conf": word.get("confidence"),
                }
            )
    return words


def convert_tokens(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten token-level information when available."""

    tokens: List[Dict[str, Any]] = []
    for segment in segments:
        for token in segment.get("tokens", []):
            tokens.append(
                {
                    "t": token.get("token") or token.get("text"),
                    "start": token.get("start"),
                    "end": token.get("end"),
                }
            )
    return tokens


def extract_speech_regions(segments: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Create a list of speech regions from aligned segments."""

    regions: List[Dict[str, float]] = []
    for segment in segments:
        start = segment.get("start")
        end = segment.get("end")
        if start is None or end is None:
            continue
        regions.append({"start": float(start), "end": float(end)})
    return regions


def build_alignment_record(
    sample: RawSample,
    wrapper: WhisperXWrapper,
    aligner_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Align a single sample and build the JSONL record."""

    try:
        result = wrapper.transcribe_and_align(sample.audio_path)
        alignment = result["alignment"]
        transcription = result["transcription"]
        segments: List[Dict[str, Any]] = alignment.get("segments", [])

        metrics = compute_alignment_metrics(alignment, sample.text)
        record = {
            "sample_id": sample.sample_id,
            "audio_path": str(sample.audio_path),
            "text": sample.text,
            "split": sample.split,
            "language": sample.language or aligner_config.get("language"),
            "alignment": {
                "words": convert_words(segments),
                "tokens": convert_tokens(segments),
                "coverage": metrics,
            },
            "speech_regions": extract_speech_regions(segments),
            "tool_version": {"whisperx": wrapper.version},
            "model_name": wrapper.model_name,
            "rng_seed": aligner_config.get("rng_seed"),
            "auto_transcript": transcription.get("text"),
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
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Failed to align sample %s", sample.sample_id)
        return {
            "sample_id": sample.sample_id,
            "audio_path": str(sample.audio_path),
            "text": sample.text,
            "split": sample.split,
            "language": sample.language or aligner_config.get("language"),
            "alignment": None,
            "speech_regions": [],
            "tool_version": {"whisperx": wrapper.version},
            "model_name": wrapper.model_name,
            "rng_seed": aligner_config.get("rng_seed"),
            "auto_transcript": None,
            "status": "error",
            "error_msg": str(exc),
        }


def run_alignment(config: Dict[str, Any], args: argparse.Namespace) -> None:
    """Execute the alignment pipeline using the provided configuration."""

    paths = config.get("paths", {})
    aligner_cfg = config.get("aligner", {})

    raw_samples_path = args.raw_samples or Path(paths.get("raw_samples_path", "data/raw_samples.jsonl"))
    output_path = args.out or Path(paths.get("alignment_output_path", "data/labels/raw_alignment.jsonl"))
    audio_root = Path(paths.get("input_audio_dir", ".")).resolve()

    wrapper = WhisperXWrapper(
        model_name=aligner_cfg.get("model_name", "large-v3"),
        device=aligner_cfg.get("device", "cuda"),
        language=aligner_cfg.get("language"),
        compute_type=aligner_cfg.get("compute_type"),
        batch_size=int(aligner_cfg.get("batch_size", 8)),
        vad_backend=aligner_cfg.get("vad_backend"),
        diarize=bool(aligner_cfg.get("diarize", False)),
    )

    LOGGER.info("Loading raw samples from %s", raw_samples_path)
    samples = load_raw_samples(
        raw_path=raw_samples_path,
        audio_root=audio_root,
        split_filter=args.split,
        limit=args.limit,
    )

    records = (build_alignment_record(sample, wrapper, aligner_cfg) for sample in tqdm(samples))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, records)
    LOGGER.info("Alignment records written to %s", output_path)


def main() -> None:
    """CLI entrypoint."""

    configure_logging()
    args = parse_args()
    config = load_yaml(args.config)
    run_alignment(config, args)


if __name__ == "__main__":
    main()
