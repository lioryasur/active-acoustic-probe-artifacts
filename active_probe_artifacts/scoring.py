"""Peak-to-sideband scoring for the active acoustic probe."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class ScoringConfig:
    peak_search_hz: float
    sideband_min_hz: float
    sideband_max_hz: float
    window_s: float
    hop_s: float
    per_tone_threshold_db: float
    tones_required_per_set: int


def config_from_json(config: dict[str, Any]) -> ScoringConfig:
    detector = config["detector"]
    return ScoringConfig(
        peak_search_hz=float(detector["peak_search_hz"]),
        sideband_min_hz=float(detector["sideband_min_hz"]),
        sideband_max_hz=float(detector["sideband_max_hz"]),
        window_s=float(detector["window_s"]),
        hop_s=float(detector["hop_s"]),
        per_tone_threshold_db=float(detector["per_tone_threshold_db"]),
        tones_required_per_set=int(detector["tones_required_per_set"]),
    )


def frame_audio(audio: np.ndarray, *, sample_rate: int, window_s: float, hop_s: float) -> list[tuple[float, np.ndarray]]:
    window_count = int(round(float(sample_rate) * float(window_s)))
    hop_count = int(round(float(sample_rate) * float(hop_s)))
    if window_count <= 0 or hop_count <= 0:
        raise ValueError("window_s and hop_s must be positive")
    if len(audio) < window_count:
        padded = np.pad(np.asarray(audio, dtype=np.float64), (0, window_count - len(audio)))
        return [(0.0, padded)]
    frames: list[tuple[float, np.ndarray]] = []
    for start in range(0, len(audio) - window_count + 1, hop_count):
        frames.append((start / float(sample_rate), np.asarray(audio[start : start + window_count], dtype=np.float64)))
    return frames


def tone_score_db(
    frame: np.ndarray,
    *,
    sample_rate: int,
    frequency_hz: float,
    peak_search_hz: float,
    sideband_min_hz: float,
    sideband_max_hz: float,
) -> dict[str, float]:
    windowed = np.asarray(frame, dtype=np.float64) * np.hanning(len(frame))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / float(sample_rate))

    peak_mask = np.abs(freqs - float(frequency_hz)) <= float(peak_search_hz)
    sideband_distance = np.abs(freqs - float(frequency_hz))
    sideband_mask = (sideband_distance >= float(sideband_min_hz)) & (sideband_distance <= float(sideband_max_hz))
    if not np.any(peak_mask):
        raise ValueError(f"No FFT bins near {frequency_hz:g} Hz")
    if not np.any(sideband_mask):
        raise ValueError(f"No sideband FFT bins near {frequency_hz:g} Hz")

    peak = float(np.max(spectrum[peak_mask]))
    sideband = float(np.median(spectrum[sideband_mask]))
    score = 20.0 * math.log10((peak + EPS) / (sideband + EPS))
    peak_freq = float(freqs[peak_mask][int(np.argmax(spectrum[peak_mask]))])
    return {
        "frequency_hz": float(frequency_hz),
        "peak_frequency_hz": peak_freq,
        "peak_magnitude": peak,
        "sideband_median": sideband,
        "score_db": score,
    }


def score_tone_set(
    frames: list[tuple[float, np.ndarray]],
    *,
    sample_rate: int,
    tone_set: dict[str, Any],
    scoring: ScoringConfig,
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for start_s, frame in frames:
        tone_scores = [
            tone_score_db(
                frame,
                sample_rate=sample_rate,
                frequency_hz=float(frequency),
                peak_search_hz=scoring.peak_search_hz,
                sideband_min_hz=scoring.sideband_min_hz,
                sideband_max_hz=scoring.sideband_max_hz,
            )
            for frequency in tone_set["frequencies_hz"]
        ]
        scores = sorted((float(item["score_db"]) for item in tone_scores), reverse=True)
        aggregate = scores[scoring.tones_required_per_set - 1]
        passing = sum(float(item["score_db"]) >= scoring.per_tone_threshold_db for item in tone_scores)
        result = {
            "name": tone_set.get("name", "tone_set"),
            "start_s": start_s,
            "aggregate_score_db": aggregate,
            "passing_tones": passing,
            "pass": passing >= scoring.tones_required_per_set,
            "tones": tone_scores,
        }
        if best is None or float(result["aggregate_score_db"]) > float(best["aggregate_score_db"]):
            best = result
    if best is None:
        raise ValueError("No frames to score")
    return best


def score_timed_probe(
    audio: np.ndarray,
    *,
    sample_rate: int,
    tone_sets: list[dict[str, Any]],
    scoring: ScoringConfig,
) -> dict[str, Any]:
    frames = frame_audio(audio, sample_rate=sample_rate, window_s=scoring.window_s, hop_s=scoring.hop_s)
    set_results = [
        score_tone_set(frames, sample_rate=sample_rate, tone_set=tone_set, scoring=scoring)
        for tone_set in tone_sets
    ]
    ordered = True
    for left, right in zip(set_results, set_results[1:]):
        ordered = ordered and float(left["start_s"]) < float(right["start_s"])
    return {
        "sample_rate": int(sample_rate),
        "duration_s": len(audio) / float(sample_rate),
        "threshold_db": scoring.per_tone_threshold_db,
        "set_results": set_results,
        "timed_order_pass": ordered,
        "pass": all(bool(result["pass"]) for result in set_results) and ordered,
    }

