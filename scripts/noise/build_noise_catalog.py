"""Compile structured metadata for AI Hub noise datasets.

This script traverses the extracted noise resources under ``assets/noises``
and summarizes clip-level information into a CSV catalog. The catalog makes it
easy to sample noise segments when synthesizing hallucination datasets.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


LOGGER = logging.getLogger(__name__)


@dataclass
class NoiseRecord:
    """Structured representation of a single noise clip annotation."""

    dataset_name: str
    split: str
    category_dir: str
    subcategory_dir: str
    label_json_path: str
    audio_path: str
    label_name: str
    clip_start_sec: float
    clip_end_sec: float
    clip_duration_sec: float
    category_01: Optional[str]
    category_02: Optional[str]
    category_03: Optional[str]
    sub_category: Optional[str]
    decibel: Optional[float]
    source_sample_rate_hz: Optional[int]
    source_channels: Optional[int]
    audio_duration_sec: Optional[float]


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("assets/noises"),
        help="Root directory containing extracted AI Hub noise datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/noise/noise_catalog.csv"),
        help="Destination CSV file path for the compiled catalog.",
    )
    return parser.parse_args()


def discover_noise_records(root: Path) -> Iterator[NoiseRecord]:
    """Yield ``NoiseRecord`` entries discovered under ``root``."""

    if not root.exists():
        LOGGER.warning("Noise root %s does not exist; nothing to catalog.", root)
        return

    for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        dataset_name = dataset_dir.name
        data_dir = dataset_dir / "01.데이터"
        if not data_dir.exists():
            LOGGER.debug("Skipping %s (no 01.데이터 directory).", dataset_name)
            continue

        for split_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
            split_label = split_dir.name
            split = split_label.split(".", 1)[-1] if "." in split_label else split_label

            label_root = split_dir / "라벨링데이터"
            raw_root = split_dir / "원천데이터"
            if not label_root.exists() or not raw_root.exists():
                LOGGER.debug(
                    "Skipping %s/%s (label or raw directory missing).",
                    dataset_name,
                    split_label,
                )
                continue

            for json_path in sorted(label_root.rglob("*.json")):
                relative_category = json_path.relative_to(label_root).parent
                category_dir = relative_category.parts[0] if relative_category.parts else ""
                subcategory_dir = str(relative_category) if relative_category.parts else ""

                try:
                    payload = json.loads(json_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    LOGGER.error("Failed to parse %s: %s", json_path, exc)
                    continue

                audio_meta = payload.get("audio", {})
                annotations = payload.get("annotations", [])
                if not annotations:
                    LOGGER.debug("No annotations in %s; skipping.", json_path)
                    continue

                sample_rate_hz = parse_sample_rate(audio_meta.get("sampleRate"))
                channels = parse_channel_count(audio_meta.get("recodingType"))
                audio_duration_sec = safe_float(audio_meta.get("duration"))

                for annotation in annotations:
                    label_name = annotation.get("labelName")
                    if not label_name:
                        LOGGER.debug("Annotation without labelName in %s; skipping.", json_path)
                        continue

                    audio_rel_path = relative_category / label_name
                    audio_path = raw_root / audio_rel_path
                    if not audio_path.exists():
                        LOGGER.warning("Audio file missing for %s (expected %s).", label_name, audio_path)

                    area = annotation.get("area", {})
                    start = safe_float(area.get("start")) or 0.0
                    end = safe_float(area.get("end")) or 0.0
                    duration = max(0.0, end - start)

                    categories = annotation.get("categories", {})
                    try:
                        label_json_rel = str(json_path.relative_to(root))
                    except ValueError:
                        label_json_rel = str(json_path)

                    try:
                        audio_rel = str(audio_path.relative_to(root))
                    except ValueError:
                        audio_rel = str(audio_path)

                    record = NoiseRecord(
                        dataset_name=dataset_name,
                        split=split,
                        category_dir=category_dir,
                        subcategory_dir=subcategory_dir,
                        label_json_path=label_json_rel,
                        audio_path=audio_rel,
                        label_name=label_name,
                        clip_start_sec=start,
                        clip_end_sec=end,
                        clip_duration_sec=duration,
                        category_01=categories.get("category_01"),
                        category_02=categories.get("category_02"),
                        category_03=categories.get("category_03"),
                        sub_category=annotation.get("subCategory"),
                        decibel=safe_float(annotation.get("decibel")),
                        source_sample_rate_hz=sample_rate_hz,
                        source_channels=channels,
                        audio_duration_sec=audio_duration_sec,
                    )
                    yield record


def parse_sample_rate(value: Optional[str]) -> Optional[int]:
    """Convert sample rate descriptions like ``44.1kHz`` to integer Hz."""

    if not value:
        return None
    normalized = value.strip().lower()
    khz_match = re.match(r"([0-9]+(?:\.[0-9]+)?)\s*k(?:hz)?", normalized)
    hz_match = re.match(r"([0-9]+)\s*hz", normalized)
    try:
        if khz_match:
            return int(float(khz_match.group(1)) * 1000)
        if hz_match:
            return int(hz_match.group(1))
    except ValueError:
        LOGGER.debug("Unable to parse sample rate from %s", value)
    return None


def parse_channel_count(value: Optional[str]) -> Optional[int]:
    """Interpret channel descriptions such as ``Stereo`` or ``Mono``."""

    if not value:
        return None
    normalized = value.strip().lower()
    if "stereo" in normalized:
        return 2
    if "mono" in normalized:
        return 1
    return None


def safe_float(value: object) -> Optional[float]:
    """Parse ``value`` to float when possible."""

    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_catalog(records: Iterable[NoiseRecord], output_path: Path) -> int:
    """Write ``records`` to ``output_path`` in CSV format."""

    rows = [asdict(record) for record in records]
    if not rows:
        LOGGER.warning("No noise records discovered; nothing written.")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info("Wrote %d records to %s", len(rows), output_path)
    return len(rows)


def configure_logging() -> None:
    """Configure the default logging handler."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    """Entry point for the catalog builder."""

    configure_logging()
    args = parse_arguments()
    LOGGER.info("Scanning %s for noise annotations...", args.root)
    records = list(discover_noise_records(args.root))
    write_catalog(records, args.output)


if __name__ == "__main__":
    main()
