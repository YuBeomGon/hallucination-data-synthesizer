"""Step 02: Augment audio with silence or background noise."""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm

from src.modules.audio_processor import AudioProcessor
from src.modules.noise_selector import NoiseSelector
from src.utils.config_loader import load_yaml
from src.utils.file_io import write_jsonl
from src.utils.logging_config import configure_logging


LOGGER = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    min_gap_sec: float
    insertion_min_sec: float
    insertion_max_sec: float
    crossfade_sec: float
    context_window_sec: float
    target_snr_db: float
    loudness_target_lufs: float
    true_peak_dbfs: float
    insertions_per_file: int
    rng_seed: int
    noise_categories: Optional[List[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default_config.yaml"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--alignment", type=Path, default=None, help="Override alignment JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of samples")
    parser.add_argument("--output", type=Path, default=None, help="Override augmented metadata output path")
    return parser.parse_args()


def load_alignment_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def collect_candidates(record: Dict[str, Any], min_gap: float, crossfade: float) -> List[Dict[str, float]]:
    candidates: List[Dict[str, float]] = []
    speech_regions = record.get("speech_regions", [])
    speech_regions = sorted(speech_regions, key=lambda r: float(r.get("start", 0.0)))

    for prev, nxt in zip(speech_regions, speech_regions[1:]):
        gap_start = float(prev["end"])
        gap_end = float(nxt["start"])
        gap_len = gap_end - gap_start
        if gap_len >= min_gap + 2 * crossfade:
            candidates.append({
                "type": "speech_gap",
                "start": gap_start,
                "end": gap_end,
                "length": gap_len,
            })

    words = record.get("alignment", {}).get("words", [])
    words = sorted(
        [w for w in words if w.get("start") is not None and w.get("end") is not None],
        key=lambda w: float(w["start"]),
    )
    for prev, nxt in zip(words, words[1:]):
        gap_start = float(prev["end"])
        gap_end = float(nxt["start"])
        gap_len = gap_end - gap_start
        if gap_len >= min_gap + 2 * crossfade:
            candidates.append({
                "type": "word_gap",
                "start": gap_start,
                "end": gap_end,
                "length": gap_len,
            })

    # Speech gaps first, then longest gaps
    candidates.sort(key=lambda c: (0 if c["type"] == "speech_gap" else 1, -c["length"]))
    return candidates


def choose_gap(rng: random.Random, candidates: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not candidates:
        return None
    return candidates[0]


def sanitize_path(audio_path: str, input_root: Path) -> Path:
    path = Path(audio_path)
    if path.is_absolute():
        return path
    return (input_root / path).resolve()


def uniform_duration(rng: random.Random, minimum: float, maximum: float, gap_length: float, crossfade: float) -> Optional[float]:
    max_allowed = gap_length - 2 * crossfade
    if max_allowed <= minimum:
        return None
    upper = min(maximum, max_allowed)
    if upper <= minimum:
        return None
    return rng.uniform(minimum, upper)


def offset_map_for_insertion(gap_start: float, gap_end: float, duration: float, original_duration: float) -> List[Dict[str, float]]:
    return [
        {"t0_src": 0.0, "t0_dst": 0.0},
        {"t0_src": gap_start, "t0_dst": gap_start},
        {"t0_src": gap_end, "t0_dst": gap_end + duration, "delta": duration},
        {"t0_src": original_duration, "t0_dst": original_duration + duration},
    ]


def shift_segments(words: List[Dict[str, Any]], shift_start: float, shift_amount: float) -> List[Dict[str, Any]]:
    updated: List[Dict[str, Any]] = []
    for word in words:
        start = float(word.get("start", 0.0))
        end = float(word.get("end", 0.0))
        new_word = dict(word)
        if start >= shift_start:
            new_word["start"] = start + shift_amount
            new_word["end"] = end + shift_amount
        updated.append(new_word)
    return updated


def run_augmentation(config: Dict[str, Any], args: argparse.Namespace) -> None:
    paths_cfg = config.get("paths", {})
    synthesis_cfg = config.get("synthesis", {})

    aug_cfg = AugmentationConfig(
        min_gap_sec=float(synthesis_cfg.get("min_gap_sec", 0.5)),
        insertion_min_sec=float(synthesis_cfg.get("insertion_duration_sec", {}).get("min", 1.5)),
        insertion_max_sec=float(synthesis_cfg.get("insertion_duration_sec", {}).get("max", 3.0)),
        crossfade_sec=float(synthesis_cfg.get("crossfade_sec", 0.05)),
        context_window_sec=float(synthesis_cfg.get("context_window_sec", 0.75)),
        target_snr_db=float(synthesis_cfg.get("target_snr_db", 12.0)),
        loudness_target_lufs=float(synthesis_cfg.get("loudness_target_lufs", -23.0)),
        true_peak_dbfs=float(synthesis_cfg.get("true_peak_dbfs", -1.0)),
        insertions_per_file=int(synthesis_cfg.get("insertions_per_file", 1)),
        rng_seed=int(synthesis_cfg.get("rng_seed", 4242)),
        noise_categories=synthesis_cfg.get("noise_categories"),
    )

    split = args.split
    input_audio_root = Path(paths_cfg.get("input_audio_dir", "assets/zeroth")).resolve()
    label_dir = Path(paths_cfg.get("label_dir", "data/labels")) / split
    label_dir.mkdir(parents=True, exist_ok=True)

    augmented_audio_dir = Path(paths_cfg.get("output_dir", "data/augmented_audio")) / split
    augmented_audio_dir.mkdir(parents=True, exist_ok=True)

    alignment_path = args.alignment
    if alignment_path is None:
        alignment_path = label_dir / "raw_alignment.jsonl"

    if not alignment_path.exists():
        raise FileNotFoundError(f"Alignment file not found: {alignment_path}")

    noise_catalog_path = Path(paths_cfg.get("noise_catalog", "data/noise/noise_catalog.csv"))
    noise_root = Path(paths_cfg.get("noise_dir", "assets/noises"))

    rng = random.Random(aug_cfg.rng_seed)

    audio_processor = AudioProcessor(
        sample_rate=16000,
        crossfade_sec=aug_cfg.crossfade_sec,
        loudness_target_lufs=aug_cfg.loudness_target_lufs,
        true_peak_dbfs=aug_cfg.true_peak_dbfs,
        context_window_sec=aug_cfg.context_window_sec,
    )

    noise_selector = NoiseSelector(
        catalog_path=noise_catalog_path,
        noise_root=noise_root,
        rng=rng,
        allow_categories=aug_cfg.noise_categories,
    )

    meta_output = args.output or (label_dir / "augmented_meta.jsonl")

    records = load_alignment_records(alignment_path)
    if args.limit is not None:
        records = (record for idx, record in enumerate(records) if idx < args.limit)

    augmented_records: List[Dict[str, Any]] = []

    buffered_records = list(records)
    for record in tqdm(buffered_records, desc=f"Augmenting {split}"):
        child_seed = rng.randint(0, 2 ** 31 - 1)
        child_rng = random.Random(child_seed)
        augmented_records.append(
            process_record(
                record=record,
                audio_processor=audio_processor,
                noise_selector=noise_selector,
                rng=child_rng,
                seed=child_seed,
                config=aug_cfg,
                input_audio_root=input_audio_root,
                augmented_audio_dir=augmented_audio_dir,
            )
        )

    write_jsonl(meta_output, augmented_records)
    LOGGER.info("Augmentation metadata written to %s", meta_output)


def process_record(
    record: Dict[str, Any],
    audio_processor: AudioProcessor,
    noise_selector: NoiseSelector,
    rng: random.Random,
    seed: int,
    config: AugmentationConfig,
    input_audio_root: Path,
    augmented_audio_dir: Path,
) -> Dict[str, Any]:
    sample_id = record.get("sample_id")
    split = record.get("split", "train")
    audio_path = sanitize_path(record.get("audio_path", ""), input_audio_root)

    result_meta: Dict[str, Any] = {
        "sample_id": sample_id,
        "split": split,
        "original_audio_path": str(audio_path),
        "augmentation": None,
        "offset_map": [],
        "updated_segments": [],
        "tool_version": {"synth": "0.1.0"},
        "rng_seed": seed,
        "status": "skip",
        "error_msg": None,
    }

    try:
        waveform = audio_processor.load_waveform(audio_path)
    except FileNotFoundError as exc:
        result_meta["status"] = "error"
        result_meta["error_msg"] = str(exc)
        return result_meta

    duration_sec = waveform.size / audio_processor.sample_rate
    candidates = collect_candidates(record, config.min_gap_sec, config.crossfade_sec)
    gap_choice = choose_gap(rng, candidates)
    if not gap_choice:
        result_meta["status"] = "skip"
        result_meta["error_msg"] = "no_gap"
        return result_meta

    gap_start = gap_choice["start"]
    gap_end = gap_choice["end"]
    gap_length = gap_choice["length"]

    insert_duration = uniform_duration(
        rng,
        config.insertion_min_sec,
        config.insertion_max_sec,
        gap_length,
        config.crossfade_sec,
    )

    if insert_duration is None:
        result_meta["status"] = "skip"
        result_meta["error_msg"] = "insufficient_gap"
        return result_meta

    insert_window = gap_start + (gap_length - insert_duration) / 2.0
    insert_time_sec = max(gap_start + config.crossfade_sec, insert_window)
    insert_time_sec = min(insert_time_sec, gap_end - insert_duration - config.crossfade_sec)
    insert_time_sec = max(0.0, insert_time_sec)

    try:
        noise_clip = noise_selector.sample_clip(insert_duration + config.crossfade_sec)
    except RuntimeError as exc:
        result_meta["status"] = "skip"
        result_meta["error_msg"] = str(exc)
        return result_meta

    clip_room = max(0.0, noise_clip.clip_duration_sec - insert_duration)
    clip_offset = rng.uniform(0.0, clip_room)
    noise_start = noise_clip.clip_start_sec + clip_offset
    noise_segment = audio_processor.load_segment(
        noise_clip.audio_path,
        start_sec=noise_start,
        duration_sec=insert_duration,
    )

    noise_segment = audio_processor.apply_fades(noise_segment)

    context = audio_processor.context_window(waveform, insert_time_sec)
    scaled_noise, achieved_snr = audio_processor.match_snr(
        noise_segment,
        context,
        config.target_snr_db,
    )

    augmented_wave, cross_info = audio_processor.insert_with_crossfade(
        base=waveform,
        insert=scaled_noise,
        insert_time_sec=insert_time_sec,
    )

    loudness_report, normalized_wave = audio_processor.normalize_loudness(augmented_wave)

    aug_id = f"{sample_id}_{rng.randint(0, 1_000_000):06d}"
    output_path = augmented_audio_dir / f"{aug_id}.wav"
    sf.write(str(output_path), normalized_wave, audio_processor.sample_rate, subtype="PCM_16")

    updated_words = shift_segments(
        record.get("alignment", {}).get("words", []),
        shift_start=gap_end,
        shift_amount=insert_duration,
    )

    offset_map = offset_map_for_insertion(
        gap_start=gap_start,
        gap_end=gap_end,
        duration=insert_duration,
        original_duration=duration_sec,
    )

    result_meta.update(
        {
            "aug_id": aug_id,
            "augmented_audio_path": str(output_path),
            "augmentation": {
                "events": [
                    {
                        "type": "noise",
                        "start_sec": gap_start,
                        "insert_sec": insert_time_sec,
                        "duration_sec": insert_duration,
                        "snr_db": achieved_snr,
                        "crossfade_sec": config.crossfade_sec,
                        "noise_src": str(noise_clip.audio_path),
                        "noise_offset_sec": noise_start,
                        "noise_duration_sec": insert_duration,
                    }
                ],
                "postprocess": {
                    "loudness_target_lufs": config.loudness_target_lufs,
                    "lufs_before": loudness_report.lufs_before,
                    "lufs_after": loudness_report.lufs_after,
                    "true_peak_dbfs": loudness_report.true_peak_dbfs,
                    "clip_guard_applied": loudness_report.clip_guard_applied,
                },
            },
            "offset_map": offset_map,
            "updated_segments": updated_words,
            "status": "ok",
        }
    )

    return result_meta


def main() -> None:
    configure_logging()
    args = parse_args()
    config = load_yaml(args.config)
    run_augmentation(config, args)


if __name__ == "__main__":
    main()
