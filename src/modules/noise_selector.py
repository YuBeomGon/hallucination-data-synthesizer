"""Utilities for selecting noise clips from the catalog."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class NoiseClip:
    """Metadata describing a noise segment available for augmentation."""

    audio_path: Path
    clip_start_sec: float
    clip_end_sec: float
    clip_duration_sec: float
    category_01: Optional[str]
    category_02: Optional[str]
    category_03: Optional[str]
    source_sample_rate_hz: Optional[int]
    source_channels: Optional[int]


class NoiseSelector:
    """Select deterministic noise clips based on duration and category filters."""

    def __init__(
        self,
        catalog_path: Path,
        noise_root: Path,
        rng: random.Random,
        allow_categories: Optional[List[str]] = None,
    ) -> None:
        self.catalog_path = catalog_path
        self.noise_root = noise_root
        self.rng = rng
        self.allow_categories = [c.lower() for c in allow_categories] if allow_categories else None
        self._clips = self._load_catalog()

    def sample_clip(self, desired_duration: float) -> NoiseClip:
        """Return a noise clip whose duration is at least ``desired_duration`` seconds."""

        candidates = [clip for clip in self._clips if clip.clip_duration_sec >= desired_duration]
        if not candidates:
            raise RuntimeError("No noise clips long enough for requested duration")
        return self.rng.choice(candidates)

    def _load_catalog(self) -> List[NoiseClip]:
        clips: List[NoiseClip] = []
        with self.catalog_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if self.allow_categories:
                    cat = (row.get("category_02") or "").lower()
                    if cat not in self.allow_categories:
                        continue
                audio_rel = Path(row["audio_path"])
                clip = NoiseClip(
                    audio_path=self._resolve_audio_path(audio_rel),
                    clip_start_sec=float(row.get("clip_start_sec", 0.0)),
                    clip_end_sec=float(row.get("clip_end_sec", 0.0)),
                    clip_duration_sec=float(row.get("clip_duration_sec", row.get("duration", 0.0))),
                    category_01=row.get("category_01"),
                    category_02=row.get("category_02"),
                    category_03=row.get("category_03"),
                    source_sample_rate_hz=self._parse_optional_int(row.get("source_sample_rate_hz")),
                    source_channels=self._parse_optional_int(row.get("source_channels")),
                )
                clips.append(clip)
        if not clips:
            raise RuntimeError("Noise catalog is empty or filtered out all entries")
        return clips

    def _resolve_audio_path(self, relative: Path) -> Path:
        resampled = Path("data/noise/resampled") / relative
        if resampled.exists():
            return resampled
        candidate = self.noise_root / relative
        return candidate

    @staticmethod
    def _parse_optional_int(value: Optional[str]) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(float(value))
        except ValueError:
            return None
