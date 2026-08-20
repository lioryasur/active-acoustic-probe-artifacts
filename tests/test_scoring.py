from __future__ import annotations

import unittest

import numpy as np

from active_probe_artifacts.scoring import ScoringConfig, frame_audio, score_timed_probe


class TimedProbeScoringTests(unittest.TestCase):
    def test_selects_best_valid_ordered_pair(self) -> None:
        sample_rate = 8_000
        tone_sets = [
            {"name": "Set A", "frequencies_hz": [1_000, 1_200, 1_400]},
            {"name": "Set B", "frequencies_hz": [1_800, 2_000, 2_200]},
        ]
        scoring = ScoringConfig(
            peak_search_hz=2,
            sideband_min_hz=50,
            sideband_max_hz=150,
            window_s=1.0,
            hop_s=1.0,
            per_tone_threshold_db=20,
            tones_required_per_set=2,
        )

        def tone_frame(frequencies: list[int], amplitude: float) -> np.ndarray:
            time = np.arange(sample_rate, dtype=np.float64) / sample_rate
            return sum(amplitude * np.sin(2 * np.pi * frequency * time) for frequency in frequencies) / len(frequencies)

        audio = np.concatenate(
            [
                tone_frame(tone_sets[0]["frequencies_hz"], 0.20),
                tone_frame(tone_sets[1]["frequencies_hz"], 0.50),
                tone_frame(tone_sets[0]["frequencies_hz"], 0.80),
                np.zeros(sample_rate, dtype=np.float64),
            ]
        )

        result = score_timed_probe(audio, sample_rate=sample_rate, tone_sets=tone_sets, scoring=scoring)

        self.assertTrue(result["pass"])
        self.assertEqual([row["start_s"] for row in result["set_results"]], [0.0, 1.0])

    def test_frame_audio_includes_unaligned_final_window(self) -> None:
        frames = frame_audio(np.zeros(2_100), sample_rate=1_000, window_s=1.0, hop_s=0.75)
        self.assertEqual([start for start, _ in frames], [0.0, 0.75, 1.1])


if __name__ == "__main__":
    unittest.main()
