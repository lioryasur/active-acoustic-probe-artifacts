"""Generate the paper probe WAV from a JSON config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from active_probe_artifacts.io import load_json, write_json, write_wav_mono
from active_probe_artifacts.probe import generate_timed_probe, sample_random_tone_sets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "paper_probe_config.json")
    parser.add_argument("--output", type=Path, default=ROOT / "generated" / "paper_probe.wav")
    parser.add_argument("--seed", type=int, help="Override the config seed for a fresh reproducible probe.")
    parser.add_argument("--metadata-out", type=Path, default=ROOT / "generated" / "paper_probe.json")
    args = parser.parse_args()

    config = load_json(args.config)
    tone_sets = sample_random_tone_sets(config["random_probe"], seed=args.seed)
    sample_rate = int(config["sample_rate"])
    audio = generate_timed_probe(
        tone_sets,
        sample_rate=sample_rate,
        amplitude=float(config["amplitude"]),
        gap_s=float(config.get("gap_s", 0.25)),
    )
    write_wav_mono(args.output, sample_rate, audio)
    write_json(
        args.metadata_out,
        {
            "config": str(args.config),
            "seed": config["random_probe"].get("seed") if args.seed is None else args.seed,
            "tone_sets": tone_sets,
            "sample_rate": sample_rate,
            "amplitude": float(config["amplitude"]),
            "gap_s": float(config.get("gap_s", 0.25)),
        },
    )
    print(f"Wrote {args.output} ({len(audio) / sample_rate:.3f} s at {sample_rate} Hz)")
    print(f"Wrote {args.metadata_out}")
    for tone_set in tone_sets:
        print(f"{tone_set['name']}: {tone_set['frequencies_hz']}")


if __name__ == "__main__":
    main()
