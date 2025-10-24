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
    leading_silence_sec: float = 0.0
    trailing_silence_sec: float = 0.0
    leading_silence_samples: int = 0
    trailing_silence_samples: int = 0


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
            leading_silence_sec = float(payload.get("leading_silence_sec", 0.0) or 0.0)
            trailing_silence_sec = float(payload.get("trailing_silence_sec", 0.0) or 0.0)
            leading_silence_samples = int(payload.get("leading_silence_samples", round(leading_silence_sec * SAMPLE_RATE)))
            trailing_silence_samples = int(payload.get("trailing_silence_samples", round(trailing_silence_sec * SAMPLE_RATE)))

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
                    leading_silence_sec=leading_silence_sec,
                    trailing_silence_sec=trailing_silence_sec,
                    leading_silence_samples=leading_silence_samples,
                    trailing_silence_samples=trailing_silence_samples,
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


def append_segment(base: np.ndarray, segment: np.ndarray, crossfade_samples: int) -> Tuple[np.ndarray, int, int]:
    if segment.size == 0:
        start = base.size
        return base, start, start
    if base.size == 0 or crossfade_samples <= 0:
        start = base.size
        new_base = np.concatenate([base, segment])
        end = start + segment.size
        return new_base.astype(np.float32, copy=False), start, end

    overlap = min(crossfade_samples, base.size, segment.size)
    if overlap <= 0:
        start = base.size
        new_base = np.concatenate([base, segment])
        end = start + segment.size
        return new_base.astype(np.float32, copy=False), start, end

    fade = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
    faded_base = base[-overlap:] * (1.0 - fade)
    faded_seg = segment[:overlap] * fade
    blended = faded_base + faded_seg

    new_base = np.concatenate([base[:-overlap], blended, segment[overlap:]])
    start = base.size - overlap
    end = start + segment.size
    return new_base.astype(np.float32, copy=False), start, end


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
    bandpass_cfg: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    clip: NoiseClip = selector.sample_clip(duration_sec)
    available = max(0.0, clip.clip_duration_sec - duration_sec)
    offset = clip.clip_start_sec + (rng.uniform(0.0, available) if available > 0 else 0.0)
    noise_wave = audio_processor.load_segment(clip.audio_path, offset, duration_sec)
    noise_wave, achieved = audio_processor.match_snr(
        noise_wave,
        context,
        target_snr_db,
        bandpass_cfg=bandpass_cfg,
    )
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

    sr = audio_processor.sample_rate
    crossfade_samples = max(0, int(round(float(synthesis_cfg.get("crossfade_sec", 0.1)) * sr)))
    fade_ms = int(transition_cfg.get("fade_ms", 20))
    context_before_samples = max(0, int(round(float(transition_cfg.get("context_window_before_sec", 0.5)) * sr)))
    context_after_samples = max(0, int(round(float(transition_cfg.get("context_window_after_sec", 0.5)) * sr)))
    bandpass_cfg = transition_cfg.get("bandpass_filter", {})

    allow_silence_prob = float(transition_cfg.get("allow_silence_prob", 0.0))
    concat_prob = float(transition_cfg.get("concat_without_noise_prob", 0.5))
    min_silence_concat_sec = float(transition_cfg.get("min_silence_for_direct_concat_sec", 2.5))
    short_noise_cfg = transition_cfg.get("short_noise_sec", {"min": 0.3, "max": 0.7})

    time_stretch_cfg = synthesis_cfg.get("time_stretch", {})
    stretch_enabled = bool(time_stretch_cfg.get("enable", True))
    stretch_min = float(time_stretch_cfg.get("min_ratio", 0.95))
    stretch_max = float(time_stretch_cfg.get("max_ratio", 1.05))
    target_max_after_stretch = float(synthesis_cfg.get("max_total_duration_after_stretch_sec", 30.0))

    pause_sec = float(
        rng.uniform(
            float(transition_cfg.get("min_pause_sec", 0.0)),
            float(transition_cfg.get("max_pause_sec", 0.0)),
        )
    )

    min_noise_sec = float(transition_cfg.get("min_noise_sec", 1.0))
    max_noise_sec = float(transition_cfg.get("max_noise_sec", 2.5))
    if max_noise_sec < min_noise_sec:
        raise ValueError("transition.max_noise_sec must be >= transition.min_noise_sec")

    noise_duration_sec = float(rng.uniform(min_noise_sec, max_noise_sec))

    long_silence = (
        sample_a.trailing_silence_sec >= min_silence_concat_sec
        and sample_b.leading_silence_sec >= min_silence_concat_sec
    )

    noise_mode = "noise"
    if long_silence and rng.random() < concat_prob:
        noise_mode = "silence_passthrough"
    elif long_silence:
        noise_mode = "short_noise"
    elif rng.random() < allow_silence_prob:
        noise_mode = "silence"

    if noise_mode == "short_noise":
        short_min = float(short_noise_cfg.get("min", 0.3))
        short_max = float(short_noise_cfg.get("max", short_min))
        if short_max < short_min:
            short_max = short_min
        noise_duration_sec = float(rng.uniform(short_min, short_max))
    elif noise_mode in {"silence", "silence_passthrough"}:
        noise_duration_sec = 0.0

    audio_a = audio_processor.load_waveform(sample_a.audio_path)
    audio_b = audio_processor.load_waveform(sample_b.audio_path)

    duration_a_raw = audio_a.size / sr
    duration_b_raw = audio_b.size / sr

    trailing_silence_samples = min(sample_a.trailing_silence_samples, audio_a.size)
    leading_silence_samples = min(sample_b.leading_silence_samples, audio_b.size)

    speech_a_end = audio_a.size - trailing_silence_samples
    speech_b_start = leading_silence_samples

    speech_a_use_sec = speech_a_end / sr
    speech_b_use_sec = (audio_b.size - speech_b_start) / sr

    if noise_mode == "silence_passthrough":
        predicted_total = duration_a_raw + duration_b_raw
    else:
        predicted_total = speech_a_use_sec + pause_sec + noise_duration_sec + speech_b_use_sec

    max_total = float(selection_cfg.get("max_total_duration_sec", 40.0))
    if predicted_total > max_total:
        raise SkipPair(
            f"Combined duration {predicted_total:.2f}s exceeds max_total_duration_sec={max_total:.2f} "
            f"for pair {sample_a.sample_id}/{sample_b.sample_id}"
        )

    ratio_applied = 1.0
    if (
        stretch_enabled
        and predicted_total > target_max_after_stretch
        and target_max_after_stretch > 0
    ):
        ratio_needed = predicted_total / target_max_after_stretch
        ratio_applied = min(max(ratio_needed, stretch_min), stretch_max)
        if ratio_applied > 1.001:
            audio_a = apply_time_stretch(audio_a, ratio_applied)
            audio_b = apply_time_stretch(audio_b, ratio_applied)
            trailing_silence_samples = int(trailing_silence_samples / ratio_applied)
            leading_silence_samples = int(leading_silence_samples / ratio_applied)
            sample_a.trailing_silence_sec /= ratio_applied
            sample_b.leading_silence_sec /= ratio_applied
            duration_a_raw = audio_a.size / sr
            duration_b_raw = audio_b.size / sr
            speech_a_end = max(0, audio_a.size - trailing_silence_samples)
            speech_b_start = min(audio_b.size, leading_silence_samples)
            speech_a_use_sec = speech_a_end / sr
            speech_b_use_sec = (audio_b.size - speech_b_start) / sr
            if noise_mode == "silence_passthrough":
                predicted_total = duration_a_raw + duration_b_raw
            else:
                predicted_total = speech_a_use_sec + pause_sec + noise_duration_sec + speech_b_use_sec

    ratio_a = ratio_applied
    ratio_b = ratio_applied

    context_before = audio_a[max(0, speech_a_end - context_before_samples) : speech_a_end]
    context_after = audio_b[speech_b_start : min(audio_b.size, speech_b_start + context_after_samples)]
    if context_before.size or context_after.size:
        context = np.concatenate([context_before, context_after])
    else:
        context = audio_a[-context_before_samples:] if audio_a.size else audio_b[:context_after_samples]

    pause_samples = 0 if noise_mode == "silence_passthrough" else int(round(pause_sec * sr))
    pause_wave = np.zeros(pause_samples, dtype=np.float32)

    target_snr_db = float(synthesis_cfg.get("target_snr_db", 10.0))
    noise_meta: Dict[str, Any]
    if noise_mode in {"noise", "short_noise"}:
        noise_wave, noise_meta = sample_noise(
            selector=noise_selector,
            audio_processor=audio_processor,
            rng=rng,
            duration_sec=noise_duration_sec,
            context=context,
            target_snr_db=target_snr_db,
            fade_ms=fade_ms,
            bandpass_cfg=bandpass_cfg,
        )
        noise_meta["duration_samples"] = noise_wave.size
    elif noise_mode == "silence":
        noise_wave = np.zeros(int(round(noise_duration_sec * sr)), dtype=np.float32)
        noise_meta = {
            "source_path": None,
            "offset_sec": None,
            "duration_sec": round(noise_duration_sec, 3),
            "target_snr_db": None,
            "achieved_snr_db": None,
            "duration_samples": noise_wave.size,
        }
    else:  # silence_passthrough
        noise_wave = np.zeros(0, dtype=np.float32)
        noise_meta = {
            "source_path": None,
            "offset_sec": None,
            "duration_sec": 0.0,
            "target_snr_db": None,
            "achieved_snr_db": None,
            "duration_samples": 0,
        }

    if bandpass_cfg:
        noise_meta["bandpass"] = bandpass_cfg

    if noise_mode == "silence_passthrough":
        base_wave = audio_a.copy()
        crossfade_noise_samples = 0
        crossfade_b_samples = 0
        transition_start_sample = max(0, len(audio_a) - trailing_silence_samples)
        transition_end_sample = transition_start_sample + trailing_silence_samples + leading_silence_samples
        timeline = base_wave
    else:
        speech_a = audio_a[:speech_a_end].copy()
        speech_b = audio_b[speech_b_start:].copy()
        timeline = speech_a
        transition_start_sample = len(timeline)
        timeline, _, _ = append_segment(timeline, pause_wave, 0)
        crossfade_noise_samples = min(crossfade_samples, timeline.size, noise_wave.size) if noise_wave.size else 0
        if noise_wave.size:
            timeline, noise_start_sample, noise_end_sample = append_segment(timeline, noise_wave, crossfade_noise_samples)
        else:
            noise_start_sample = len(timeline)
            noise_end_sample = len(timeline)
        transition_end_sample = noise_end_sample
        crossfade_b_samples = min(crossfade_samples, timeline.size, len(speech_b))
        timeline, b_start_sample, b_end_sample = append_segment(timeline, speech_b, crossfade_b_samples)
        audio_b = speech_b  # update for metadata durations
    if noise_mode == "silence_passthrough":
        timeline, b_start_sample, b_end_sample = append_segment(timeline, audio_b, 0)
        noise_start_sample = transition_start_sample
        noise_end_sample = transition_end_sample
        crossfade_b_samples = 0
        crossfade_noise_samples = 0

    total_samples = len(timeline)
    total_duration_sec = total_samples / sr

    loudness_report, normalized_wave = audio_processor.normalize_loudness(timeline)
    normalized_wave = audio_processor.apply_tpdf_dither(normalized_wave)
    np.clip(normalized_wave, -1.0, 1.0, out=normalized_wave)

    pair_id = f"{sample_a.speaker_id}_{sample_a.sample_id}_{sample_b.sample_id}"

    transition_token = labelling_cfg.get("transition_token", "<SIL>")
    combined_text = f"{sample_a.text.strip()} {sample_b.text.strip()}".strip()
    combined_with_token = f"{sample_a.text.strip()} {transition_token} {sample_b.text.strip()}".strip()

    speech_a_end_sample = len(audio_a) if noise_mode == "silence_passthrough" else audio_a.size
    speech_a_output_end_sample = len(audio_a) if noise_mode == "silence_passthrough" else speech_a_end
    speech_b_start_output = b_start_sample if noise_mode != "silence_passthrough" else b_start_sample + leading_silence_samples
    speech_a_output_end_sample = int(speech_a_output_end_sample)
    speech_b_start_output = int(speech_b_start_output)

    transition_start_sec = transition_start_sample / sr
    transition_end_sec = transition_end_sample / sr

    metadata = {
        "pair_id": pair_id,
        "speaker_id": sample_a.speaker_id,
        "split": sample_a.split,
        "audio": {
            "sample_rate": sr,
            "duration_sec": round(total_duration_sec, 3),
            "duration_samples": total_samples,
            "loudness": asdict(loudness_report),
        },
        "segments": {
            "utterance_a": {
                "sample_id": sample_a.sample_id,
                "duration_sec": round(duration_a_raw, 3),
                "stretch_ratio": round(ratio_a, 4),
                "output_start_sec": 0.0,
                "output_end_sec": round(speech_a_output_end_sample / sr, 3),
                "output_start_sample": 0,
                "output_end_sample": int(speech_a_output_end_sample),
            },
            "transition": {
                "type": noise_mode,
                "start_sec": round(transition_start_sec, 3),
                "end_sec": round(transition_end_sec, 3),
                "start_sample": int(transition_start_sample),
                "end_sample": int(transition_end_sample),
                "pause_sec": round(pause_sec if noise_mode != "silence_passthrough" else 0.0, 3),
                "pause_samples": pause_samples if noise_mode != "silence_passthrough" else 0,
                "crossfade_in_sec": round(crossfade_noise_samples / sr, 3),
                "crossfade_out_sec": round(crossfade_b_samples / sr, 3),
                "noise": noise_meta,
                "decision": {
                    "mode": noise_mode,
                    "long_silence": long_silence,
                },
            },
            "utterance_b": {
                "sample_id": sample_b.sample_id,
                "duration_sec": round(duration_b_raw, 3),
                "stretch_ratio": round(ratio_b, 4),
                "output_start_sec": round(speech_b_start_output / sr, 3),
                "output_end_sec": round(total_duration_sec, 3),
                "output_start_sample": int(speech_b_start_output),
                "output_end_sample": total_samples,
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
                "ratio": round(ratio_applied, 4),
            },
            "noise_mode": noise_mode,
            "snr": {
                "target_db": target_snr_db if noise_mode in {"noise", "short_noise"} else None,
                "context_before_samples": context_before_samples,
                "context_after_samples": context_after_samples,
                "bandpass": bandpass_cfg,
            },
            "silence": {
                "trailing_a_sec": sample_a.trailing_silence_sec,
                "leading_b_sec": sample_b.leading_silence_sec,
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
