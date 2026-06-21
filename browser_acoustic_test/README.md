# Minimal Browser Acoustic Test

This page is a small local acoustic-loop test for the active probe detector. It
plays a fresh random high-band probe through the browser, records the local
microphone, scores the recording in-browser, and exports a WAV/JSON pair.

This is not a VoIP/network test. The separate browser VoIP proof-of-concept
repository covers the two-device WebRTC path. This page is meant to make the
artifact bundle testable with real sound using only one machine.

## Run

From the artifact root:

```bash
python browser_acoustic_test/server.py
```

Open:

```text
http://127.0.0.1:8766/
```

Use headphones cautiously. The probe is audible.

## Browser Behavior

The page requests microphone capture with:

- `echoCancellation: false`
- `noiseSuppression: false`
- `autoGainControl: false`

Browser and device behavior can still vary. The exported JSON records requested
constraints, reported track settings, sampled probe frequencies, and detector
results.

The exported JSON is authoritative for scoring because it records the exact
frequencies sampled in that browser run.
