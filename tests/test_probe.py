from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from active_probe_artifacts.io import load_json
from active_probe_artifacts.probe import generate_timed_probe, sample_random_tone_sets
from active_probe_artifacts.scoring import config_from_json, score_timed_probe

CONFIG = load_json(Path(__file__).resolve().parents[1] / "config" / "paper_probe_config.json")


def paper_probe() -> tuple[int, list[dict], np.ndarray]:
    sample_rate = int(CONFIG["sample_rate"])
    tone_sets = sample_random_tone_sets(CONFIG["random_probe"])
    audio = generate_timed_probe(tone_sets, sample_rate=sample_rate, amplitude=float(CONFIG["amplitude"]))
    return sample_rate, tone_sets, audio


class PaperProbeTests(unittest.TestCase):
    def test_sets_are_back_to_back(self) -> None:
        # Paper: two back-to-back 1 s sets of three tones (2.0 s in total, no gap).
        sample_rate, _, audio = paper_probe()
        self.assertEqual(len(audio), 2 * sample_rate)

    def test_detector_accepts_probe_in_noise_at_paper_timing(self) -> None:
        sample_rate, tone_sets, audio = paper_probe()
        noisy = audio + np.random.default_rng(0).normal(0.0, 0.01, len(audio))
        result = score_timed_probe(noisy, sample_rate=sample_rate, tone_sets=tone_sets, scoring=config_from_json(CONFIG))
        self.assertTrue(result["pass"])
        self.assertEqual([row["start_s"] for row in result["set_results"]], [0.0, 1.0])

    def test_detector_rejects_a_challenge_that_was_not_played(self) -> None:
        sample_rate, _, audio = paper_probe()
        noisy = audio + np.random.default_rng(0).normal(0.0, 0.01, len(audio))
        wrong_sets = sample_random_tone_sets(CONFIG["random_probe"], seed=7)
        result = score_timed_probe(noisy, sample_rate=sample_rate, tone_sets=wrong_sets, scoring=config_from_json(CONFIG))
        self.assertFalse(result["pass"])


if __name__ == "__main__":
    unittest.main()
