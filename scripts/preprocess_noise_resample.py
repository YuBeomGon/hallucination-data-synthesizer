"""Resample AI Hub noise clips to a canonical format for augmentation."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import librosa
import numpy as np
import soundfile as sf


LOGGER = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/noise_catalog.csv"),
        help="CSV catalog created by build_noise_catalog.py",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("assets/noises"),
        help="Root directory containing the original noise audio files.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path("data/noises_resampled"),
        help="Directory where resampled audio files will be stored.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/noise_catalog_resampled.csv"),
        help="Output CSV with resampled path metadata.",
    )
    parser.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Target sample rate in Hz.",
    )
    parser.add_argument(
        "--mono",
        dest="mono",
        action="store_true",
        default=True,
        help="Mix down to mono (default).",
    )
    parser.add_argument(
        "--keep-channels",
        dest="mono",
        action="store_false",
        help="Preserve original channel count instead of mixing to mono.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing resampled files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of catalog rows to process (for smoke tests).",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_catalog(catalog_path: Path, limit: int | None = None) -> List[Dict[str, str]]:
    """Load catalog rows into memory."""

    with catalog_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, str]] = []
        for idx, row in enumerate(reader):
            rows.append(row)
            if limit is not None and idx + 1 >= limit:
                break
    return rows


def resample_audio(
    source_path: Path,
    destination_path: Path,
    target_sr: int,
    to_mono: bool,
    overwrite: bool,
) -> Tuple[bool, int, int]:
    """Resample ``source_path`` and save to ``destination_path``."""

    if destination_path.exists() and not overwrite:
        LOGGER.debug("Skipping existing %s", destination_path)
        channels = 1 if to_mono else 0
        return True, target_sr, channels

    try:
        samples, orig_sr = librosa.load(source_path, sr=None, mono=False)
    except FileNotFoundError:
        LOGGER.warning("Missing audio file: %s", source_path)
        return False, 0, 0
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Failed to load %s: %s", source_path, exc)
        return False, 0, 0

    if samples.ndim > 1 and to_mono:
        samples = librosa.to_mono(samples)
    elif samples.ndim > 1:
        samples = np.asarray(samples)

    if orig_sr != target_sr:
        samples = librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr)
        new_sr = target_sr
    else:
        new_sr = orig_sr

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination_path, samples, new_sr)

    channels = 1 if to_mono or samples.ndim == 1 else samples.shape[0]
    return True, new_sr, channels


def update_rows(
    rows: Iterable[Dict[str, str]],
    root: Path,
    target_dir: Path,
    target_sr: int,
    to_mono: bool,
    overwrite: bool,
) -> Tuple[List[Dict[str, str]], int, int]:
    """Process catalog rows and update with resampled metadata."""

    processed: List[Dict[str, str]] = []
    success_count = 0
    failure_count = 0

    for row in rows:
        audio_rel = row.get("audio_path")
        row.setdefault("resampled_audio_path", "")
        row.setdefault("resampled_sample_rate_hz", "")
        row.setdefault("resampled_channels", "")
        if not audio_rel:
            failure_count += 1
            processed.append(row)
            continue

        source_path = root / audio_rel
        destination_path = target_dir / audio_rel
        ok, new_sr, channels = resample_audio(
            source_path,
            destination_path,
            target_sr,
            to_mono,
            overwrite,
        )

        if ok:
            success_count += 1
            rel_dest = destination_path
            try:
                rel_dest = destination_path.relative_to(target_dir.parent)
            except ValueError:
                rel_dest = destination_path

            row["resampled_audio_path"] = str(rel_dest)
            row["resampled_sample_rate_hz"] = str(new_sr)
            row["resampled_channels"] = str(channels)
        else:
            failure_count += 1
        processed.append(row)

    return processed, success_count, failure_count


def write_catalog(rows: List[Dict[str, str]], output_path: Path) -> None:
    """Write updated rows to ``output_path``."""

    if not rows:
        LOGGER.warning("No rows to write.")
        return

    fieldnames = list(rows[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    LOGGER.info("Wrote catalog with %d rows to %s", len(rows), output_path)


def main() -> None:
    configure_logging()
    args = parse_arguments()
    LOGGER.info("Loading catalog from %s", args.catalog)
    rows = load_catalog(args.catalog, limit=args.limit)
    LOGGER.info("Processing %d entries", len(rows))
    processed_rows, success_count, failure_count = update_rows(
        rows,
        root=args.audio_root,
        target_dir=args.target_dir,
        target_sr=args.target_sr,
        to_mono=args.mono,
        overwrite=args.overwrite,
    )
    LOGGER.info("Resampled %d files (failures: %d)", success_count, failure_count)
    write_catalog(processed_rows, args.output)


if __name__ == "__main__":
    main()
