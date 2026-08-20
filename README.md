# Active Acoustic Probe Artifacts

This is a curated release bundle for the paper **Active Acoustic Probes for AI
Voice Call Detection**. It contains the probe configuration and small
reproduction scripts needed to generate and score the paper-style high-band
active acoustic probe.

This bundle is intentionally smaller than the research workspace. It does not
include raw WAV recordings, derived CSV result tables, private provider outputs,
or exploratory scripts.

## Contents

- `config/paper_probe_config.json`: random high-band probe design and detector
  parameters.
- `active_probe_artifacts/probe.py`: tone generation helpers.
- `active_probe_artifacts/scoring.py`: peak-to-sideband detector implementation.
- `active_probe_artifacts/io.py`: WAV and JSON helpers.
- `scripts/generate_probe_wav.py`: generate the configured probe WAV.
- `scripts/score_wav_with_probe_config.py`: score a WAV against the configured
  probe.
- `browser_acoustic_test/`: minimal local browser app that plays a random probe,
  records the microphone return, scores it, and exports WAV/JSON.

The browser/WebRTC proof-of-concept is released separately as
[`active-acoustic-probe-browser`](https://github.com/lioryasur/active-acoustic-probe-browser).

## Install

```bash
python -m pip install -r requirements.txt
```

## Quick Check

```bash
python scripts/generate_probe_wav.py
python scripts/score_wav_with_probe_config.py generated/paper_probe.wav --metadata generated/paper_probe.json
```

Run the detector tests with:

```bash
python -m unittest discover -s tests
```

To run a real local speaker-to-microphone acoustic test:

```bash
python browser_acoustic_test/server.py
```

Then open `http://127.0.0.1:8766/`.

## Scope

The generator samples a fresh two-set high-band challenge from the configured
frequency range. A seed can be supplied for reproducible examples. The detector
code implements the paper scoring rule: for each target tone, it compares the
spectral peak near the expected frequency with local sidebands. It then selects
the ordered pair of windows that maximizes the weaker set's two-out-of-three
score.

This bundle is suitable for checking the detector implementation, generating
probe examples, scoring exported recordings, and running a minimal local
acoustic-loop test. It is not a full raw-data release or a benchmark of
production VoIP platforms.
