const CONFIG = {
  sampleRate: 44100,
  frequencyRangeHz: [5500, 7000],
  frequencyStepHz: 10,
  minSpacingHz: 100,
  setsPerProbe: 2,
  tonesPerSet: 3,
  durationS: 1.0,
  gapS: 0.25,
  detector: {
    peakSearchHz: 5,
    sidebandMinHz: 100,
    sidebandMaxHz: 500,
    windowS: 1.0,
    hopS: 0.25,
    perToneThresholdDb: 10,
    tonesRequiredPerSet: 2,
  },
};

const runButton = document.getElementById("runButton");
const statusEl = document.getElementById("status");
const resultEl = document.getElementById("result");
const wavLink = document.getElementById("wavLink");
const jsonLink = document.getElementById("jsonLink");

function logStatus(message) {
  statusEl.textContent = message;
}

function mulberry32(seed) {
  let state = seed >>> 0;
  return function next() {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function sampleToneSets(seed) {
  const rng = mulberry32(seed);
  const [low, high] = CONFIG.frequencyRangeHz;
  const grid = [];
  for (let f = low; f <= high; f += CONFIG.frequencyStepHz) grid.push(f);
  const selected = [];
  const toneSets = [];
  for (let setIndex = 0; setIndex < CONFIG.setsPerProbe; setIndex += 1) {
    const tones = [];
    let attempts = 0;
    while (tones.length < CONFIG.tonesPerSet) {
      attempts += 1;
      if (attempts > 10000) throw new Error("Could not sample spaced tones");
      const candidate = grid[Math.floor(rng() * grid.length)];
      if (selected.some((existing) => Math.abs(existing - candidate) < CONFIG.minSpacingHz)) continue;
      selected.push(candidate);
      tones.push(candidate);
    }
    toneSets.push({
      name: String.fromCharCode("A".charCodeAt(0) + setIndex),
      frequencies_hz: tones.sort((a, b) => a - b),
      duration_s: CONFIG.durationS,
    });
  }
  return toneSets;
}

function createProbeBuffer(audioContext, toneSets, amplitude) {
  const sampleRate = audioContext.sampleRate;
  const totalDuration = toneSets.reduce((sum, set) => sum + set.duration_s, 0) + CONFIG.gapS * (toneSets.length - 1);
  const frameCount = Math.ceil(totalDuration * sampleRate);
  const buffer = audioContext.createBuffer(1, frameCount, sampleRate);
  const data = buffer.getChannelData(0);
  let offset = 0;
  for (let setIndex = 0; setIndex < toneSets.length; setIndex += 1) {
    if (setIndex > 0) offset += Math.round(CONFIG.gapS * sampleRate);
    const set = toneSets[setIndex];
    const count = Math.round(set.duration_s * sampleRate);
    const fadeCount = Math.min(Math.round(0.01 * sampleRate), Math.floor(count / 2));
    for (let i = 0; i < count; i += 1) {
      const t = i / sampleRate;
      let sample = 0;
      for (const frequency of set.frequencies_hz) {
        sample += Math.sin(2 * Math.PI * frequency * t);
      }
      sample = (sample / set.frequencies_hz.length) * amplitude;
      if (fadeCount > 0 && i < fadeCount) sample *= i / fadeCount;
      if (fadeCount > 0 && i >= count - fadeCount) sample *= (count - i - 1) / fadeCount;
      data[offset + i] += sample;
    }
    offset += count;
  }
  return buffer;
}

async function recordDuringPlayback(stream, playFn, durationMs) {
  const audioContext = new AudioContext();
  const source = audioContext.createMediaStreamSource(stream);
  const processor = audioContext.createScriptProcessor(4096, 1, 1);
  const chunks = [];
  processor.onaudioprocess = (event) => {
    chunks.push(new Float32Array(event.inputBuffer.getChannelData(0)));
  };
  source.connect(processor);
  processor.connect(audioContext.destination);
  await playFn();
  await new Promise((resolve) => setTimeout(resolve, durationMs));
  processor.disconnect();
  source.disconnect();
  await audioContext.close();
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const audio = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) {
    audio.set(chunk, offset);
    offset += chunk.length;
  }
  return { sampleRate: audioContext.sampleRate, audio };
}

function hann(length) {
  const out = new Float64Array(length);
  for (let i = 0; i < length; i += 1) out[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (length - 1));
  return out;
}

function goertzelMagnitude(frame, sampleRate, frequency) {
  const omega = (2 * Math.PI * frequency) / sampleRate;
  const coeff = 2 * Math.cos(omega);
  let q0 = 0;
  let q1 = 0;
  let q2 = 0;
  for (let i = 0; i < frame.length; i += 1) {
    q0 = coeff * q1 - q2 + frame[i];
    q2 = q1;
    q1 = q0;
  }
  return Math.sqrt(q1 * q1 + q2 * q2 - coeff * q1 * q2);
}

function median(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function toneScore(frame, sampleRate, frequency) {
  const detector = CONFIG.detector;
  const window = hann(frame.length);
  const windowed = new Float64Array(frame.length);
  for (let i = 0; i < frame.length; i += 1) windowed[i] = frame[i] * window[i];
  const peakCandidates = [];
  for (let f = frequency - detector.peakSearchHz; f <= frequency + detector.peakSearchHz; f += 1) {
    peakCandidates.push(goertzelMagnitude(windowed, sampleRate, f));
  }
  const sidebands = [];
  for (let delta = detector.sidebandMinHz; delta <= detector.sidebandMaxHz; delta += 25) {
    sidebands.push(goertzelMagnitude(windowed, sampleRate, frequency - delta));
    sidebands.push(goertzelMagnitude(windowed, sampleRate, frequency + delta));
  }
  const peak = Math.max(...peakCandidates);
  const sideband = median(sidebands);
  return 20 * Math.log10((peak + 1e-12) / (sideband + 1e-12));
}

function selectBestOrderedWindows(scoredSets) {
  let best = null;

  function visit(setIndex, minimumWindowIndex, selected) {
    if (setIndex === scoredSets.length) {
      const aggregateScores = selected.map((result) => result.aggregate_score_db);
      const weakestScore = Math.min(...aggregateScores);
      const scoreSum = aggregateScores.reduce((sum, score) => sum + score, 0);
      if (
        !best ||
        weakestScore > best.weakestScore ||
        (weakestScore === best.weakestScore && scoreSum > best.scoreSum)
      ) {
        best = { results: selected.slice(), weakestScore, scoreSum };
      }
      return;
    }

    for (let windowIndex = minimumWindowIndex; windowIndex < scoredSets[setIndex].length; windowIndex += 1) {
      selected.push(scoredSets[setIndex][windowIndex]);
      visit(setIndex + 1, windowIndex + 1, selected);
      selected.pop();
    }
  }

  visit(0, 0, []);
  return best;
}

function scoreRecording(audio, sampleRate, toneSets) {
  const detector = CONFIG.detector;
  const windowCount = Math.round(detector.windowS * sampleRate);
  const hopCount = Math.round(detector.hopS * sampleRate);
  const frames = [];
  const lastStart = audio.length - windowCount;
  for (let start = 0; start <= lastStart; start += hopCount) {
    frames.push({ start, startS: start / sampleRate, frame: audio.slice(start, start + windowCount) });
  }
  if (frames.length && frames[frames.length - 1].start !== lastStart) {
    frames.push({ start: lastStart, startS: lastStart / sampleRate, frame: audio.slice(lastStart) });
  }
  if (!frames.length) throw new Error("Recording is shorter than the detector window.");

  const scoredSets = toneSets.map((set) =>
    frames.map((item) => {
      const scores = set.frequencies_hz.map((frequency) => ({
        frequency_hz: frequency,
        score_db: toneScore(item.frame, sampleRate, frequency),
      }));
      const sorted = scores.map((score) => score.score_db).sort((a, b) => b - a);
      const aggregate = sorted[detector.tonesRequiredPerSet - 1];
      const passingTones = scores.filter((score) => score.score_db >= detector.perToneThresholdDb).length;
      const result = {
        name: set.name,
        start_s: item.startS,
        aggregate_score_db: aggregate,
        passing_tones: passingTones,
        pass: passingTones >= detector.tonesRequiredPerSet,
        tones: scores,
      };
      return result;
    }),
  );
  const orderedSelection = selectBestOrderedWindows(scoredSets);
  const setResults = orderedSelection
    ? orderedSelection.results
    : scoredSets.map((results) => results.reduce((best, result) =>
      (result.aggregate_score_db > best.aggregate_score_db ? result : best), results[0]));
  const timedOrderPass = Boolean(orderedSelection);
  return {
    sample_rate: sampleRate,
    duration_s: audio.length / sampleRate,
    tone_sets: toneSets,
    threshold_db: detector.perToneThresholdDb,
    set_results: setResults,
    timed_score_db: orderedSelection ? orderedSelection.weakestScore : null,
    timed_order_pass: timedOrderPass,
    pass: timedOrderPass && setResults.every((result) => result.pass),
  };
}

function wavBlob(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (const sample of samples) {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, clamped < 0 ? clamped * 32768 : clamped * 32767, true);
    offset += 2;
  }
  return new Blob([view], { type: "audio/wav" });
}

function setDownload(link, blob, filename) {
  if (link.href) URL.revokeObjectURL(link.href);
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.hidden = false;
}

function formatDb(value) {
  return `${Number(value).toFixed(1)} dB`;
}

function formatSeconds(value) {
  return `${Number(value).toFixed(2)} s`;
}

function renderResult(score) {
  const statusClass = score.pass ? "pass" : "fail";
  const statusText = score.pass ? "PASS" : "FAIL";
  const sets = score.set_results
    .map((setResult) => {
      const setClass = setResult.pass ? "pass" : "fail";
      const rows = setResult.tones
        .map((tone) => {
          const toneClass = tone.score_db >= score.threshold_db ? "pass" : "fail";
          return `
            <tr>
              <td>${tone.frequency_hz} Hz</td>
              <td class="${toneClass}">${formatDb(tone.score_db)}</td>
            </tr>
          `;
        })
        .join("");
      return `
        <article class="set-card">
          <div class="set-header">
            <span class="set-title">Set ${setResult.name}</span>
            <span class="badge ${setClass}">${setResult.pass ? "Detected" : "Missed"}</span>
          </div>
          <div class="set-body">
            <div class="set-meta">
              <div class="small-metric">
                <span>Best window</span>
                <span>${formatSeconds(setResult.start_s)}</span>
              </div>
              <div class="small-metric">
                <span>Aggregate</span>
                <span>${formatDb(setResult.aggregate_score_db)}</span>
              </div>
              <div class="small-metric">
                <span>Tones</span>
                <span>${setResult.passing_tones}/3</span>
              </div>
            </div>
            <table>
              <thead>
                <tr>
                  <th>Tone</th>
                  <th>Peak-to-sideband</th>
                </tr>
              </thead>
              <tbody>${rows}</tbody>
            </table>
          </div>
        </article>
      `;
    })
    .join("");

  resultEl.innerHTML = `
    <div class="result-summary">
      <div class="metric">
        <span class="metric-label">Detector result</span>
        <span class="metric-value ${statusClass}">${statusText}</span>
      </div>
      <div class="metric">
        <span class="metric-label">Timed order</span>
        <span class="metric-value ${score.timed_order_pass ? "pass" : "fail"}">${score.timed_order_pass ? "PASS" : "FAIL"}</span>
      </div>
      <div class="metric">
        <span class="metric-label">Threshold</span>
        <span class="metric-value">${formatDb(score.threshold_db)}</span>
      </div>
      <div class="metric">
        <span class="metric-label">Recording</span>
        <span class="metric-value">${formatSeconds(score.duration_s)}</span>
      </div>
    </div>
    <div class="set-grid">${sets}</div>
  `;
}

runButton.addEventListener("click", async () => {
  runButton.disabled = true;
  wavLink.hidden = true;
  jsonLink.hidden = true;
  try {
    const seed = Number(document.getElementById("seed").value || 42);
    const amplitude = Number(document.getElementById("amplitude").value || 0.08);
    const preRollS = Number(document.getElementById("preRoll").value || 0.4);
    const toneSets = sampleToneSets(seed);
    logStatus(`Requesting microphone...\nA: ${toneSets[0].frequencies_hz.join(", ")}\nB: ${toneSets[1].frequencies_hz.join(", ")}`);
    const constraints = {
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      },
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    const track = stream.getAudioTracks()[0];
    const playbackContext = new AudioContext();
    const probeBuffer = createProbeBuffer(playbackContext, toneSets, amplitude);
    const recordDurationMs = Math.ceil((preRollS + probeBuffer.duration + 0.8) * 1000);
    const startedAt = new Date().toISOString();
    const recording = await recordDuringPlayback(
      stream,
      async () => {
        await new Promise((resolve) => setTimeout(resolve, preRollS * 1000));
        const source = playbackContext.createBufferSource();
        source.buffer = probeBuffer;
        source.connect(playbackContext.destination);
        source.start();
        logStatus(`Playing and recording...\nA: ${toneSets[0].frequencies_hz.join(", ")}\nB: ${toneSets[1].frequencies_hz.join(", ")}`);
        await new Promise((resolve) => {
          source.onended = resolve;
        });
        await playbackContext.close();
      },
      recordDurationMs,
    );
    stream.getTracks().forEach((item) => item.stop());
    const score = scoreRecording(recording.audio, recording.sampleRate, toneSets);
    const metadata = {
      created_at: startedAt,
      user_agent: navigator.userAgent,
      requested_constraints: constraints.audio,
      track_settings: track ? track.getSettings() : null,
      config: CONFIG,
      seed,
      amplitude,
      pre_roll_s: preRollS,
      tone_sets: toneSets,
      analysis: score,
    };
    renderResult(score);
    setDownload(wavLink, wavBlob(recording.audio, recording.sampleRate), "browser_acoustic_probe.wav");
    setDownload(jsonLink, new Blob([JSON.stringify(metadata, null, 2)], { type: "application/json" }), "browser_acoustic_probe.json");
    logStatus(`Done. overall_pass=${score.pass}`);
  } catch (error) {
    logStatus(`Error: ${error.message || error}`);
    resultEl.textContent = String(error.stack || error);
  } finally {
    runButton.disabled = false;
  }
});
