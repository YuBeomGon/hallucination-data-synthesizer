"""Export Zeroth samples to WAV files and build raw_samples.jsonl."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Iterable, Iterator, List

import soundfile as sf
from datasets import Audio, load_dataset
from io import BytesIO


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Zeroth split to export.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of samples to export.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("assets/zeroth"),
        help="Directory where WAV files will be stored.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/zeroth/raw_samples_train.jsonl"),
        help="Destination JSONL file for raw sample metadata.",
    )
    return parser.parse_args()


def deterministic_sample_id(audio_rel: str, text: str) -> str:
    digest = hashlib.sha1(f"{audio_rel}::{text}".encode("utf-8")).hexdigest()
    return digest[:16]


def save_audio(array, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, array, sample_rate)


def iter_dataset(split: str, limit: int | None = None) -> Iterator[dict]:
    dataset = load_dataset("Bingsu/zeroth-korean", split=split)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    for idx, example in enumerate(dataset):
        yield example
        if limit is not None and idx + 1 >= limit:
            break


def build_record(
    split: str,
    index: int,
    text: str,
    audio_rel_path: Path,
    sample_rate: int,
) -> dict:
    return {
        "sample_id": deterministic_sample_id(str(audio_rel_path), text),
        "audio_path": str(audio_rel_path),
        "text": text,
        "language": "ko",
        "split": split,
        "sr_hz": sample_rate,
        "channels": 1,
        "dataset": "Bingsu/zeroth-korean",
        "index": index,
    }


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    configure_logging()
    args = parse_args()

    records: List[dict] = []
    audio_root = args.audio_dir.resolve()
    LOGGER.info("Exporting Zeroth %s split to %s", args.split, audio_root)

    for idx, example in enumerate(tqdm(iter_dataset(args.split, args.limit), desc=f"Exporting {args.split}")):
        text = example["text"]
        audio_info = example["audio"]

        samples = None
        sample_rate = None

        if audio_info.get("path"):
            source_path = Path(audio_info["path"]).expanduser()
            samples, sample_rate = sf.read(source_path, always_2d=False)
        elif audio_info.get("bytes"):
            buffer = BytesIO(audio_info["bytes"])
            samples, sample_rate = sf.read(buffer, always_2d=False)
        elif "array" in audio_info and audio_info["array"] is not None:
            samples = audio_info["array"]
            sample_rate = audio_info.get("sampling_rate")
        else:
            LOGGER.warning("Sample %s missing audio content; skipping.", idx)
            continue

        if sample_rate is None:
            sample_rate = audio_info.get("sampling_rate")

        if samples is None or sample_rate is None:
            LOGGER.warning("Sample %s has no decodable audio; skipping.", idx)
            continue

        if hasattr(samples, "dtype") and samples.dtype != "float32":
            samples = samples.astype("float32")

        subdir = audio_root / args.split
        filename = f"{args.split}_{idx:06d}.wav"
        audio_path = subdir / filename
        save_audio(samples, sample_rate, audio_path)

        audio_rel = audio_path.relative_to(audio_root)
        record = build_record(
            split=args.split,
            index=idx,
            text=text,
            audio_rel_path=audio_rel,
            sample_rate=int(sample_rate),
        )
        records.append(record)

    LOGGER.info("Writing %d records to %s", len(records), args.output)
    write_jsonl(args.output, records)
    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
