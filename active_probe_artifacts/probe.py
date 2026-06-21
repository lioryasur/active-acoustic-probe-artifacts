"""Probe generation utilities for the paper artifact bundle."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def sample_random_tone_sets(random_probe_config: dict, *, seed: int | None = None) -> list[dict]:
    """Sample random high-band tone sets from the configured design space."""
    rng = np.random.default_rng(random_probe_config.get("seed") if seed is None else seed)
    low_hz, high_hz = [float(value) for value in random_probe_config["frequency_range_hz"]]
    step_hz = float(random_probe_config.get("frequency_step_hz", 10.0))
    min_spacing_hz = float(random_probe_config["min_spacing_hz"])
    sets_per_probe = int(random_probe_config["sets_per_probe"])
    tones_per_set = int(random_probe_config["tones_per_set"])
    duration_s = float(random_probe_config["duration_s"])
    grid = np.arange(low_hz, high_hz + step_hz / 2.0, step_hz, dtype=np.float64)

    selected: list[float] = []
    tone_sets: list[dict] = []
    for set_index in range(sets_per_probe):
        tones: list[float] = []
        attempts = 0
        while len(tones) < tones_per_set:
            attempts += 1
            if attempts > 10000:
                raise RuntimeError("Could not sample spaced tone sets from config")
            candidate = float(rng.choice(grid))
            if any(abs(candidate - existing) < min_spacing_hz for existing in selected):
                continue
            selected.append(candidate)
            tones.append(candidate)
        tone_sets.append(
            {
                "name": chr(ord("A") + set_index),
                "frequencies_hz": sorted(int(round(tone)) if float(tone).is_integer() else tone for tone in tones),
                "duration_s": duration_s,
            }
        )
    return tone_sets


def generate_tone_set(
    frequencies_hz: Iterable[float],
    *,
    sample_rate: int,
    duration_s: float,
    amplitude: float,
    fade_s: float = 0.01,
) -> np.ndarray:
    """Generate an equal-amplitude multi-tone probe segment."""
    frequencies = [float(freq) for freq in frequencies_hz]
    sample_count = int(round(float(sample_rate) * float(duration_s)))
    times = np.arange(sample_count, dtype=np.float64) / float(sample_rate)
    signal = np.zeros(sample_count, dtype=np.float64)
    for frequency in frequencies:
        signal += np.sin(2.0 * math.pi * frequency * times)
    if frequencies:
        signal /= float(len(frequencies))
    signal *= float(amplitude)
    return apply_fade(signal, sample_rate=sample_rate, fade_s=fade_s)


def generate_timed_probe(
    tone_sets: list[dict],
    *,
    sample_rate: int,
    amplitude: float,
    gap_s: float = 0.25,
    fade_s: float = 0.01,
) -> np.ndarray:
    """Generate a probe consisting of ordered tone sets separated by silence."""
    pieces: list[np.ndarray] = []
    gap = np.zeros(int(round(float(sample_rate) * float(gap_s))), dtype=np.float64)
    for index, tone_set in enumerate(tone_sets):
        if index:
            pieces.append(gap)
        pieces.append(
            generate_tone_set(
                tone_set["frequencies_hz"],
                sample_rate=sample_rate,
                duration_s=float(tone_set.get("duration_s", 1.0)),
                amplitude=amplitude,
                fade_s=fade_s,
            )
        )
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float64)


def apply_fade(audio: np.ndarray, *, sample_rate: int, fade_s: float) -> np.ndarray:
    """Apply a short linear fade to avoid hard edges."""
    out = np.asarray(audio, dtype=np.float64).copy()
    fade_count = int(round(float(sample_rate) * float(fade_s)))
    fade_count = max(0, min(fade_count, len(out) // 2))
    if fade_count == 0:
        return out
    fade_in = np.linspace(0.0, 1.0, fade_count, endpoint=True)
    out[:fade_count] *= fade_in
    out[-fade_count:] *= fade_in[::-1]
    return out


def mix_with_limit(*signals: np.ndarray, peak_limit: float = 0.98) -> np.ndarray:
    """Sum signals and scale down only if needed to avoid clipping."""
    if not signals:
        return np.zeros(0, dtype=np.float64)
    length = max(len(signal) for signal in signals)
    mixed = np.zeros(length, dtype=np.float64)
    for signal in signals:
        signal = np.asarray(signal, dtype=np.float64)
        mixed[: len(signal)] += signal
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > float(peak_limit) and peak > 0:
        mixed *= float(peak_limit) / peak
    return mixed
