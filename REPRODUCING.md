# Reproducing The Released Artifact Checks

This bundle provides code/config checks rather than the full private recording
corpus.

## 1. Generate The Probe

```bash
python scripts/generate_probe_wav.py
```

Output:

- `generated/paper_probe.wav`
- `generated/paper_probe.json`

## 2. Score The Probe

```bash
python scripts/score_wav_with_probe_config.py generated/paper_probe.wav --metadata generated/paper_probe.json
```

The generated probe should pass the timed two-set detector.

## 3. Run A Local Browser Acoustic Test

```bash
python browser_acoustic_test/server.py
```

Open `http://127.0.0.1:8766/`, allow microphone access, and click
`Record + play probe`. The page plays a fresh random probe through the local
speaker path, records the microphone return, scores it in-browser, and exports:

- `browser_acoustic_probe.wav`
- `browser_acoustic_probe.json`

## 4. Score A Browser Export Offline

```bash
python scripts/score_wav_with_probe_config.py path/to/browser_acoustic_probe.wav --metadata path/to/browser_acoustic_probe.json
```

The offline script should agree with the browser result because the JSON records
the exact random frequencies used for that run.

## Scoring External WAVs

To score another WAV with the paper probe configuration:

```bash
python scripts/score_wav_with_probe_config.py path/to/recording.wav --metadata path/to/recording.json
```

The script reports the best scoring one-second window for each tone set, the
per-set aggregate score, the timed-order check, and the final pass/fail result.
