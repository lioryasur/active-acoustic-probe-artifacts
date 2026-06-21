"""Score a WAV with the paper probe detector config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from active_probe_artifacts.io import load_json, read_wav_mono, write_json
from active_probe_artifacts.probe import sample_random_tone_sets
from active_probe_artifacts.scoring import config_from_json, score_timed_probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "paper_probe_config.json")
    parser.add_argument("--metadata", type=Path, help="JSON metadata written by generate_probe_wav.py or the browser prototype.")
    parser.add_argument("--seed", type=int, help="Sample tone sets from the config with this seed if no metadata is supplied.")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    config = load_json(args.config)
    metadata = load_json(args.metadata) if args.metadata else None
    if metadata and metadata.get("tone_sets"):
        tone_sets = metadata["tone_sets"]
    elif metadata and metadata.get("probe", {}).get("toneSets"):
        tone_sets = [
            {
                "name": item.get("name", f"Set {index + 1}"),
                "frequencies_hz": item.get("frequenciesHz", item.get("frequencies_hz")),
                "duration_s": item.get("durationS", item.get("duration_s", config["random_probe"]["duration_s"])),
            }
            for index, item in enumerate(metadata["probe"]["toneSets"])
        ]
    else:
        tone_sets = sample_random_tone_sets(config["random_probe"], seed=args.seed)
    scoring = config_from_json(config)
    sample_rate, audio = read_wav_mono(args.wav)
    result = score_timed_probe(
        audio,
        sample_rate=sample_rate,
        tone_sets=tone_sets,
        scoring=scoring,
    )

    print(f"wav={args.wav}")
    print(f"sample_rate={result['sample_rate']}")
    print(f"duration_s={result['duration_s']:.3f}")
    print(f"threshold_db={result['threshold_db']:.2f}")
    print(f"tone_sets={tone_sets}")
    for set_result in result["set_results"]:
        print(
            f"{set_result['name']}: pass={set_result['pass']} "
            f"start_s={set_result['start_s']:.3f} "
            f"aggregate_score_db={set_result['aggregate_score_db']:.2f} "
            f"passing_tones={set_result['passing_tones']}"
        )
    print(f"timed_order_pass={result['timed_order_pass']}")
    print(f"overall_pass={result['pass']}")

    if args.json_out:
        write_json(args.json_out, result)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
