"""Step 01: Synthesize two-utterance audio pairs with noise transitions."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import soundfile as sf
from pydub import AudioSegment

try:  # pragma: no cover - optional dependency
    import librosa
except ImportError:  # pragma: no cover - optional dependency
    librosa = None  # type: ignore

from src.modules.audio_processor import AudioProcessor
from src.modules.noise_selector import NoiseClip, NoiseSelector
from src.utils.config_loader import load_yaml
from src.utils.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)
SAMPLE_RATE = 16000
STEP_VERSION = "0.1.0"


@dataclass
class RawSample:
    """Representation of a Zeroth raw sample row."""

    sample_id: str
    speaker_id: str
    text: str
    audio_path: Path
    duration_sec: float
    split: str


@dataclass
class SynthesisStats:
    """Summary statistics for a synthesis run."""

    generated: int = 0
    skipped: int = 0
    errors: int = 0


class SkipPair(RuntimeError):
    """Raised to signal that a candidate pair should be skipped."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/two_utterances.yaml"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--meta-output", type=Path, default=None, help="Override metadata output path")
    parser.add_argument("--audio-output", type=Path, default=None, help="Override augmented audio output directory")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed override")
    parser.add_argument("--dry-run", action="store_true", help="Skip writing WAV files, emit metadata only")
    return parser.parse_args()


def load_raw_samples(raw_path: Path, audio_root: Path) -> List[RawSample]:
    samples: List[RawSample] = []
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw samples JSONL not found: {raw_path}")

    with raw_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            sample_id = payload.get("sample_id")
            speaker_id = payload.get("speaker_id")
            audio_rel = payload.get("audio_path")
            text = payload.get("text", "")
            duration = float(payload.get("duration_sec", payload.get("duration", 0.0)))
            split = payload.get("split", "train")

            if not sample_id or not speaker_id or not audio_rel:
                LOGGER.debug("Skipping malformed raw sample entry: %s", payload)
                continue

            audio_path = Path(audio_rel)
            if not audio_path.is_absolute():
                audio_path = (audio_root / audio_path).resolve()
            samples.append(
                RawSample(
                    sample_id=sample_id,
                    speaker_id=str(speaker_id),
                    text=str(text),
                    audio_path=audio_path,
                    duration_sec=float(duration),
                    split=str(split),
                )
            )

    if not samples:
        raise ValueError(f"No raw samples found in {raw_path}")
    return samples


def group_by_speaker(samples: Iterable[RawSample]) -> Dict[str, List[RawSample]]:
    grouped: Dict[str, List[RawSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.speaker_id, []).append(sample)
    return grouped


def numpy_to_segment(samples: np.ndarray, sample_rate: int) -> AudioSegment:
    clipped = np.clip(samples, -1.0, 1.0)
    int_samples = (clipped * 32767.0).astype(np.int16)
    return AudioSegment(
        int_samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )


def segment_to_numpy(segment: AudioSegment) -> np.ndarray:
    data = np.array(segment.get_array_of_samples()).astype(np.float32)
    return np.clip(data / 32768.0, -1.0, 1.0)


def apply_time_stretch(wave: np.ndarray, ratio: float) -> np.ndarray:
    if math.isclose(ratio, 1.0, rel_tol=1e-3):
        return wave
    if librosa is None:
        raise RuntimeError("librosa is required to use time stretching but is not installed.")
    rate = 1.0 / ratio
    stretched = librosa.effects.time_stretch(wave.astype(np.float32), rate=rate)
    return stretched.astype(np.float32, copy=False)


def build_context(audio_a: np.ndarray, audio_b: np.ndarray, context_samples: int) -> np.ndarray:
    if context_samples <= 0:
        return np.concatenate([audio_a, audio_b]) if audio_a.size or audio_b.size else np.zeros(1, dtype=np.float32)

    parts: List[np.ndarray] = []
    if audio_a.size:
        parts.append(audio_a[-context_samples:])
    if audio_b.size:
        parts.append(audio_b[:context_samples])

    if not parts:
        return np.zeros(1, dtype=np.float32)

    return np.concatenate(parts)


def sample_noise(
    selector: NoiseSelector,
    audio_processor: AudioProcessor,
    rng: random.Random,
    duration_sec: float,
    context: np.ndarray,
    target_snr_db: float,
    fade_ms: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    clip: NoiseClip = selector.sample_clip(duration_sec)
    available = max(0.0, clip.clip_duration_sec - duration_sec)
    offset = clip.clip_start_sec + (rng.uniform(0.0, available) if available > 0 else 0.0)
    noise_wave = audio_processor.load_segment(clip.audio_path, offset, duration_sec)
    noise_wave, achieved = audio_processor.match_snr(noise_wave, context, target_snr_db)
    noise_wave = audio_processor.apply_fades(noise_wave, fade_sec=fade_ms / 1000.0)
    return noise_wave, {
        "source_path": str(clip.audio_path),
        "offset_sec": round(offset, 3),
        "duration_sec": round(noise_wave.size / audio_processor.sample_rate, 3),
        "target_snr_db": target_snr_db,
        "achieved_snr_db": round(float(achieved), 3) if math.isfinite(achieved) else None,
    }


def synthesize_pair(
    sample_a: RawSample,
    sample_b: RawSample,
    config: Dict[str, Any],
    audio_processor: AudioProcessor,
    noise_selector: NoiseSelector,
    rng: random.Random,
    rng_seed: int,
) -> Tuple[Dict[str, Any], np.ndarray]:
    synthesis_cfg = config.get("synthesis", {})
    selection_cfg = config.get("selection", {})
    transition_cfg = synthesis_cfg.get("transition", {})
    labelling_cfg = config.get("labelling", {})

    crossfade_sec = float(synthesis_cfg.get("crossfade_sec", 0.1))
    crossfade_ms = max(0, int(round(crossfade_sec * 1000)))
    fade_ms = int(transition_cfg.get("fade_ms", 20))
    context_window_sec = float(transition_cfg.get("context_window_sec", 0.75))
    context_samples = max(1, int(round(context_window_sec * audio_processor.sample_rate)))

    time_stretch_cfg = synthesis_cfg.get("time_stretch", {})
    stretch_enabled = bool(time_stretch_cfg.get("enable", False))
    stretch_min = float(time_stretch_cfg.get("min_ratio", 0.95))
    stretch_max = float(time_stretch_cfg.get("max_ratio", 1.05))

    pause_sec = float(
        rng.uniform(
            float(transition_cfg.get("min_pause_sec", 0.0)),
            float(transition_cfg.get("max_pause_sec", 0.0)),
        )
    )
    silence_ms = max(0, int(round(pause_sec * 1000)))

    min_noise_sec = float(transition_cfg.get("min_noise_sec", 1.0))
    max_noise_sec = float(transition_cfg.get("max_noise_sec", 2.5))
    if max_noise_sec < min_noise_sec:
        raise ValueError("transition.max_noise_sec must be >= transition.min_noise_sec")
    noise_duration_sec = float(rng.uniform(min_noise_sec, max_noise_sec))
    allow_silence_prob = float(transition_cfg.get("allow_silence_prob", 0.0))

    noise_type = "silence" if rng.random() < allow_silence_prob else "noise"

    # Load audio
    audio_a = audio_processor.load_waveform(sample_a.audio_path)
    audio_b = audio_processor.load_waveform(sample_b.audio_path)

    ratio_a = rng.uniform(stretch_min, stretch_max) if stretch_enabled else 1.0
    ratio_b = rng.uniform(stretch_min, stretch_max) if stretch_enabled else 1.0
    if stretch_enabled:
        audio_a = apply_time_stretch(audio_a, ratio_a)
        audio_b = apply_time_stretch(audio_b, ratio_b)

    duration_a = audio_a.size / audio_processor.sample_rate
    duration_b = audio_b.size / audio_processor.sample_rate
    predicted_total = duration_a + duration_b + pause_sec + noise_duration_sec
    max_total = float(selection_cfg.get("max_total_duration_sec", 40.0))
    if predicted_total > max_total:
        raise SkipPair(
            f"Combined duration {predicted_total:.2f}s exceeds max_total_duration_sec={max_total:.2f} "
            f"for pair {sample_a.sample_id}/{sample_b.sample_id}"
        )

    context = build_context(audio_a, audio_b, context_samples)
    target_snr_db = float(synthesis_cfg.get("target_snr_db", 10.0))

    if noise_type == "noise":
        noise_wave, noise_meta = sample_noise(
            selector=noise_selector,
            audio_processor=audio_processor,
            rng=rng,
            duration_sec=noise_duration_sec,
            context=context,
            target_snr_db=target_snr_db,
            fade_ms=fade_ms,
        )
    else:
        noise_wave = np.zeros(int(round(noise_duration_sec * audio_processor.sample_rate)), dtype=np.float32)
        noise_meta = {
            "source_path": None,
            "offset_sec": None,
            "duration_sec": round(noise_duration_sec, 3),
            "target_snr_db": None,
            "achieved_snr_db": None,
        }

    silence_segment = AudioSegment.silent(duration=silence_ms)
    segment_a = numpy_to_segment(audio_a, audio_processor.sample_rate)
    segment_noise = numpy_to_segment(noise_wave, audio_processor.sample_rate).fade_in(fade_ms).fade_out(fade_ms)
    segment_b = numpy_to_segment(audio_b, audio_processor.sample_rate)

    combined = segment_a
    # Append silence (no crossfade)
    if silence_ms > 0:
        combined = combined.append(silence_segment, crossfade=0)

    prev_len_ms = len(combined)
    noise_len_ms = len(segment_noise)
    noise_crossfade_ms = min(crossfade_ms, prev_len_ms, noise_len_ms)
    combined = combined.append(segment_noise, crossfade=crossfade_ms if noise_crossfade_ms > 0 else 0)
    noise_start_ms = prev_len_ms - noise_crossfade_ms
    noise_end_ms = prev_len_ms + noise_len_ms - noise_crossfade_ms

    prev_len_ms = len(combined)
    b_len_ms = len(segment_b)
    b_crossfade_ms = min(crossfade_ms, prev_len_ms, b_len_ms)
    combined = combined.append(segment_b, crossfade=crossfade_ms if b_crossfade_ms > 0 else 0)
    b_start_ms = prev_len_ms - b_crossfade_ms
    total_len_ms = len(combined)

    # Convert back to numpy and normalize loudness
    combined_wave = segment_to_numpy(combined)
    loudness_report, normalized_wave = audio_processor.normalize_loudness(combined_wave)
    total_duration_sec = normalized_wave.size / audio_processor.sample_rate

    pair_id = f"{sample_a.speaker_id}_{sample_a.sample_id}_{sample_b.sample_id}"

    transition_token = labelling_cfg.get("transition_token", "<SIL_TRANS>")
    combined_text = f"{sample_a.text.strip()} {sample_b.text.strip()}".strip()
    combined_with_token = f"{sample_a.text.strip()} {transition_token} {sample_b.text.strip()}".strip()

    transition_start_sec = max(0.0, noise_start_ms / 1000.0)
    transition_end_sec = max(transition_start_sec, noise_end_ms / 1000.0)
    utterance_b_start_sec = max(transition_end_sec - b_crossfade_ms / 1000.0, 0.0)

    metadata = {
        "pair_id": pair_id,
        "speaker_id": sample_a.speaker_id,
        "split": sample_a.split,
        "audio": {
            "sample_rate": audio_processor.sample_rate,
            "duration_sec": round(total_duration_sec, 3),
            "loudness": asdict(loudness_report),
        },
        "segments": {
            "utterance_a": {
                "sample_id": sample_a.sample_id,
                "duration_sec": round(duration_a, 3),
                "stretch_ratio": round(ratio_a, 4),
                "output_start_sec": 0.0,
                "output_end_sec": round(duration_a, 3),
            },
            "transition": {
                "type": noise_type,
                "start_sec": round(transition_start_sec, 3),
                "end_sec": round(transition_end_sec, 3),
                "pause_sec": round(pause_sec, 3),
                "crossfade_in_sec": round(noise_crossfade_ms / 1000.0, 3),
                "crossfade_out_sec": round(b_crossfade_ms / 1000.0, 3),
                "noise": noise_meta,
            },
            "utterance_b": {
                "sample_id": sample_b.sample_id,
                "duration_sec": round(duration_b, 3),
                "stretch_ratio": round(ratio_b, 4),
                "output_start_sec": round(utterance_b_start_sec, 3),
                "output_end_sec": round(total_duration_sec, 3),
            },
        },
        "text": {
            "utterance_a": sample_a.text,
            "utterance_b": sample_b.text,
            "combined": combined_text,
            "combined_with_token": combined_with_token,
        },
        "source_samples": [sample_a.sample_id, sample_b.sample_id],
        "processing": {
            "rng_seed": rng_seed,
            "time_stretch": {
                "enabled": stretch_enabled,
                "ratio_a": round(ratio_a, 4),
                "ratio_b": round(ratio_b, 4),
            },
        },
        "tool_version": {"step_01_two_utterances": STEP_VERSION},
        "status": "ok",
        "error_msg": None,
    }

    return metadata, normalized_wave


def format_output_paths(
    config: Dict[str, Any],
    split: str,
    meta_override: Optional[Path],
    audio_override: Optional[Path],
) -> Tuple[Path, Path]:
    paths_cfg = config.get("paths", {})
    metadata_dir = Path(paths_cfg.get("metadata_dir", "data/labels_v2"))
    audio_dir = Path(paths_cfg.get("augmented_audio_dir", "data/augmented_audio_v2"))

    audio_output_dir = (audio_override or audio_dir).resolve() / split
    meta_output_path = (meta_override or metadata_dir.resolve() / split / "paired_meta.jsonl")

    audio_output_dir.mkdir(parents=True, exist_ok=True)
    meta_output_path.parent.mkdir(parents=True, exist_ok=True)
    return meta_output_path, audio_output_dir


def synthesize_split(
    config: Dict[str, Any],
    split: str,
    limit: Optional[int] = None,
    meta_override: Optional[Path] = None,
    audio_override: Optional[Path] = None,
    seed_override: Optional[int] = None,
    dry_run: bool = False,
) -> SynthesisStats:
    paths_cfg = config.get("paths", {})
    raw_template = paths_cfg.get("raw_samples_template")
    if not raw_template:
        raise KeyError("paths.raw_samples_template is required in the configuration")

    raw_samples_path = Path(raw_template.format(split=split)).resolve()
    input_audio_dir = Path(paths_cfg.get("input_audio_dir", "assets/zeroth_v2")).resolve()
    raw_samples = load_raw_samples(raw_samples_path, input_audio_dir)
    grouped = group_by_speaker(raw_samples)
    speakers = list(grouped.keys())

    if not speakers:
        raise ValueError("No speakers available for synthesis")

    selection_cfg = config.get("selection", {})
    seed = seed_override if seed_override is not None else selection_cfg.get("rng_seed", 777)
    rng = random.Random(seed)
    rng.shuffle(speakers)

    meta_path, audio_output_dir = format_output_paths(config, split, meta_override, audio_override)
    temp_meta_path = meta_path.with_suffix(".tmp")

    stats = SynthesisStats()
    audio_processor = AudioProcessor(
        sample_rate=SAMPLE_RATE,
        crossfade_sec=float(config.get("synthesis", {}).get("crossfade_sec", 0.1)),
        loudness_target_lufs=float(config.get("synthesis", {}).get("loudness_target_lufs", -23.0)),
        true_peak_dbfs=float(config.get("synthesis", {}).get("true_peak_dbfs", -1.0)),
        context_window_sec=float(config.get("synthesis", {}).get("transition", {}).get("context_window_sec", 0.75)),
    )

    noise_selector = NoiseSelector(
        catalog_path=Path(paths_cfg.get("noise_catalog", "data/noise/noise_catalog.csv")).resolve(),
        noise_root=Path(paths_cfg.get("noise_dir", "assets/noises")).resolve(),
        rng=rng,
        allow_categories=config.get("synthesis", {}).get("noise_categories"),
    )

    with temp_meta_path.open("w", encoding="utf-8") as meta_handle:
        for speaker in speakers:
            samples = grouped[speaker]
            if len(samples) < 2:
                continue
            rng.shuffle(samples)
            for idx in range(0, len(samples) - 1, 2):
                if limit is not None and stats.generated >= limit:
                    break
                sample_a = samples[idx]
                sample_b = samples[idx + 1]
                try:
                    metadata, waveform = synthesize_pair(
                        sample_a=sample_a,
                        sample_b=sample_b,
                        config=config,
                        audio_processor=audio_processor,
                        noise_selector=noise_selector,
                        rng=rng,
                        rng_seed=seed,
                    )
                except SkipPair as skip_exc:
                    stats.skipped += 1
                    LOGGER.debug("Skipping pair (%s, %s): %s", sample_a.sample_id, sample_b.sample_id, skip_exc)
                    continue
                except Exception as err:  # pragma: no cover - unexpected
                    stats.errors += 1
                    LOGGER.exception("Failed to synthesize pair (%s, %s): %s", sample_a.sample_id, sample_b.sample_id, err)
                    continue

                output_filename = f"{metadata['pair_id']}.wav"
                metadata["audio"]["output_path"] = str(Path(split) / output_filename)

                if not dry_run:
                    output_path = audio_output_dir / output_filename
                    sf.write(output_path, waveform, audio_processor.sample_rate, subtype="PCM_16")

                meta_handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                stats.generated += 1

            if limit is not None and stats.generated >= limit:
                break

    temp_meta_path.replace(meta_path)
    LOGGER.info(
        "Synthesis complete for split=%s | generated=%d skipped=%d errors=%d",
        split,
        stats.generated,
        stats.skipped,
        stats.errors,
    )
    return stats


def main() -> None:  # pragma: no cover - CLI entry
    configure_logging()
    args = parse_args()
    config = load_yaml(args.config)
    synthesize_split(
        config=config,
        split=args.split,
        limit=args.limit,
        meta_override=args.meta_output,
        audio_override=args.audio_output,
        seed_override=args.seed,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
