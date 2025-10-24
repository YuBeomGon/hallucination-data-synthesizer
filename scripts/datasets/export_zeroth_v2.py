"""Export kresnik/zeroth_korean samples to structured WAV/JSONL assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, load_dataset
from tqdm import tqdm

from src.modules.vad import compute_silence_stats


LOGGER = logging.getLogger(__name__)
SAMPLE_RATE = 16000
AUDIO_EXTS = {".flac", ".wav", ".ogg", ".mp3", ".m4a"}
_AUDIO_INDEX_CACHE: Dict[Path, Dict[str, Path]] = {}


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="kresnik/zeroth_korean")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "test"],
        help="Dataset splits to export.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("assets/zeroth_v2"),
        help="Directory where WAV files will be stored.",
    )
    parser.add_argument(
        "--raw-samples-dir",
        type=Path,
        default=Path("data/zeroth_v2"),
        help="Directory that will contain raw_samples_<split>.jsonl files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of samples per split.",
    )
    parser.add_argument(
        "--vad-backend",
        type=str,
        default="silero",
        choices=["silero", "energy"],
        help="VAD backend to use when measuring leading/trailing silence.",
    )
    parser.add_argument(
        "--vad-window",
        type=float,
        default=0.03,
        help="Window size in seconds for VAD processing.",
    )
    return parser.parse_args()


def deterministic_sample_id(speaker_id: str, audio_rel: str, text: str) -> str:
    payload = f"{speaker_id}::{audio_rel}::{text}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return digest[:16]


def sanitize_component(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").strip()


def format_speaker_id(raw_speaker: int | str) -> str:
    try:
        return f"spk_{int(raw_speaker):03d}"
    except (TypeError, ValueError):
        return sanitize_component(str(raw_speaker))


def format_chapter_id(raw_chapter: int | str) -> str:
    try:
        return f"chap_{int(raw_chapter):03d}"
    except (TypeError, ValueError):
        return sanitize_component(str(raw_chapter))


def save_audio(samples: np.ndarray, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32)
    sf.write(path, samples, sample_rate)


@dataclass
class RawSampleRecord:
    sample_id: str
    audio_path: str
    text: str
    speaker_id: str
    chapter_id: str
    split: str
    sr_hz: int
    channels: int
    duration_sec: float
    dataset: str
    index: int
    source_id: str
    leading_silence_sec: float
    trailing_silence_sec: float
    leading_silence_samples: int
    trailing_silence_samples: int
    vad_backend: str

    def to_json(self) -> Dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "audio_path": self.audio_path,
            "text": self.text,
            "speaker_id": self.speaker_id,
            "chapter_id": self.chapter_id,
            "split": self.split,
            "sr_hz": self.sr_hz,
            "channels": self.channels,
            "duration_sec": round(self.duration_sec, 3),
            "dataset": self.dataset,
            "index": self.index,
            "source_id": self.source_id,
            "leading_silence_sec": round(self.leading_silence_sec, 3),
            "trailing_silence_sec": round(self.trailing_silence_sec, 3),
            "leading_silence_samples": self.leading_silence_samples,
            "trailing_silence_samples": self.trailing_silence_samples,
            "vad_backend": self.vad_backend,
        }


def load_split(dataset_name: str, split: str) -> Dataset:
    LOGGER.info("Loading %s split '%s'", dataset_name, split)
    dataset = load_dataset(dataset_name, split=split)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    return dataset


def iter_split_examples(dataset: Dataset, limit: int | None = None) -> Iterator[dict]:
    iterable: Iterable = dataset
    if limit is not None:
        iterable = dataset.select(range(min(limit, len(dataset))))
    for example in iterable:
        yield example


def resolve_audio_path(audio_candidates: Sequence[str | None], dataset: Dataset) -> Path | None:
    normalized_candidates: List[str] = []
    for candidate in audio_candidates:
        if not candidate:
            continue
        candidate_norm = candidate.replace("\\", "/")
        normalized_candidates.append(candidate_norm)

        path_obj = Path(candidate).expanduser()
        if path_obj.is_absolute() and path_obj.exists():
            return path_obj

    cache_files = dataset.cache_files
    datasets_root: Path | None = None
    for entry in cache_files:
        filename = entry.get("filename")
        if not filename:
            continue
        file_path = Path(filename).resolve()
        for parent in file_path.parents:
            if parent.name == "datasets":
                datasets_root = parent
                break
        if datasets_root:
            break

    if datasets_root is None:
        return None

    extracted_root = datasets_root.parent / "downloads" / "extracted"
    index = _AUDIO_INDEX_CACHE.get(extracted_root)
    if index is None:
        index = {}
        if extracted_root.exists():
            LOGGER.info("Indexing audio files under %s (one-time scan)", extracted_root)
            for file_path in extracted_root.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in AUDIO_EXTS:
                    continue
                rel_path = file_path.relative_to(extracted_root)
                rel_key = str(rel_path).replace("\\", "/")
                index.setdefault(rel_key, file_path)
                index.setdefault(file_path.name, file_path)
        _AUDIO_INDEX_CACHE[extracted_root] = index

    candidate_keys = []
    for candidate in normalized_candidates:
        candidate_keys.append(candidate)
        candidate_keys.append(candidate.lstrip("./"))
        candidate_keys.append(candidate.lstrip("/"))

    for key in candidate_keys:
        if key in index:
            return index[key]

    for key in candidate_keys:
        for prefix in ("train/", "test/", "validation/"):
            composite = prefix + key
            if composite in index:
                return index[composite]

    return None


def export_samples_for_split(
    dataset: Dataset,
    split: str,
    audio_root: Path,
    dataset_name: str,
    limit: int | None,
    vad_backend: str,
    vad_window_sec: float,
) -> List[RawSampleRecord]:
    records: List[RawSampleRecord] = []

    for idx, example in enumerate(tqdm(iter_split_examples(dataset, limit), desc=f"Exporting {split}")):
        text = example.get("text", "")
        speaker_raw = example.get("speaker_id")
        chapter_raw = example.get("chapter_id")
        source_id = str(example.get("id", f"{split}_{idx:06d}"))

        speaker_id = format_speaker_id(speaker_raw)
        chapter_id = format_chapter_id(chapter_raw)
        subdir = audio_root / split / speaker_id / chapter_id

        audio_info = example["audio"]
        resolved_path = resolve_audio_path(
            [
                audio_info.get("path"),
                example.get("path"),
                audio_info.get("file"),
            ],
            dataset,
        )
        samples = None
        sample_rate = SAMPLE_RATE

        if resolved_path and resolved_path.exists():
            samples, sample_rate = sf.read(resolved_path, always_2d=False)
        else:
            audio_bytes = audio_info.get("bytes")
            if audio_bytes:
                buffer = BytesIO(audio_bytes)
                samples, sample_rate = sf.read(buffer, always_2d=False)
            else:
                LOGGER.warning(
                    "Example %s missing resolvable audio path; candidates=%s",
                    source_id,
                    [
                        audio_info.get("path"),
                        example.get("path"),
                        audio_info.get("file"),
                    ],
                )
                continue

        samples = np.asarray(samples)
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)

        channels = 1 if samples.ndim == 1 else samples.shape[1]
        duration_sec = float(len(samples) / sample_rate)

        silence = compute_silence_stats(
            samples,
            sample_rate=sample_rate,
            backend=vad_backend,
            window_sec=vad_window_sec,
        )

        filename = sanitize_component(source_id) + ".wav"
        audio_path = subdir / filename
        save_audio(samples, sample_rate, audio_path)

        audio_rel = audio_path.relative_to(audio_root)
        sample_id = deterministic_sample_id(speaker_id, str(audio_rel), text)

        record = RawSampleRecord(
            sample_id=sample_id,
            audio_path=str(audio_rel),
            text=text,
            speaker_id=speaker_id,
            chapter_id=chapter_id,
            split=split,
            sr_hz=int(sample_rate),
            channels=int(channels),
            duration_sec=duration_sec,
            dataset=dataset_name,
            index=idx,
            source_id=source_id,
            leading_silence_sec=silence.leading_sec,
            trailing_silence_sec=silence.trailing_sec,
            leading_silence_samples=silence.leading_samples,
            trailing_silence_samples=silence.trailing_samples,
            vad_backend=silence.backend,
        )
        records.append(record)

    LOGGER.info("Collected %d records for split=%s", len(records), split)
    return records


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

def main() -> None:
    configure_logging()
    args = parse_args()

    audio_root = args.audio_dir.resolve()
    raw_dir = args.raw_samples_dir.resolve()

    audio_root.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        dataset = load_split(args.dataset, split)
        records = export_samples_for_split(
            dataset=dataset,
            split=split,
            audio_root=audio_root,
            dataset_name=args.dataset,
            limit=args.limit,
            vad_backend=args.vad_backend,
            vad_window_sec=args.vad_window,
        )

        raw_output = raw_dir / f"raw_samples_{split}.jsonl"
        write_jsonl(raw_output, (record.to_json() for record in records))
        LOGGER.info("Wrote raw samples to %s", raw_output)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
