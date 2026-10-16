"""
Voice Arena Q1 - mini diarization run
Runs 3 setups on the AMI test meetings and saves one RTTM per file per setup.

  A. community1          pyannote/speaker-diarization-community-1, speaker count estimated
  B. community1_oracle   same model, told the true number of speakers
  C. legacy31            pyannote/speaker-diarization-3.1, speaker count estimated

Outputs: runs/<setup>/<meeting>.rttm, runs/timings.csv, runs/run_info.json
Safe to re-run: finished files are skipped.
"""

import csv
import json
import platform
import time
from pathlib import Path

import soundfile as sf
import torch
from huggingface_hub import get_token
from pyannote.audio import Pipeline
import pyannote.audio

MEETINGS = ["ES2004a", "TS3003a", "EN2002a"]
AUDIO_DIR = Path("audio")
REF_DIR = Path("ref")
OUT_DIR = Path("runs")

SETUPS = [
    # (setup name, Hugging Face model, tell it the true speaker count?)
    ("community1", "pyannote/speaker-diarization-community-1", False),
    ("community1_oracle", "pyannote/speaker-diarization-community-1", True),
    ("legacy31", "pyannote/speaker-diarization-3.1", False),
]


def true_speaker_count(meeting):
    """Number of distinct speakers in the human reference file."""
    speakers = set()
    with open(REF_DIR / f"{meeting}.rttm", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == "SPEAKER":
                speakers.add(parts[7])
    return len(speakers)


def load_audio(meeting):
    """Load the wav with soundfile (avoids the Windows FFmpeg/torchcodec issue)."""
    data, sr = sf.read(AUDIO_DIR / f"{meeting}.wav", dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)          # shape: (channels, samples)
    if waveform.shape[0] > 1:                    # mix to mono if needed
        waveform = waveform.mean(dim=0, keepdim=True)
    return {"waveform": waveform, "sample_rate": sr}, waveform.shape[1] / sr


def main():
    OUT_DIR.mkdir(exist_ok=True)
    token = get_token()
    if not token:
        raise SystemExit("No Hugging Face token found. Run:  hf auth login")

    info = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "device": "cpu",
        "pyannote.audio": pyannote.audio.__version__,
        "torch": torch.__version__,
        "data": "AMI test set, Mix-Headset audio, only_words references (pyannote/AMI-diarization-setup)",
        "meetings": MEETINGS,
        "setups": [],
    }

    timings_path = OUT_DIR / "timings.csv"
    new_timings = not timings_path.exists()
    timings_file = open(timings_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(timings_file)
    if new_timings:
        writer.writerow(["setup", "meeting", "audio_seconds", "processing_seconds", "real_time_factor"])

    loaded = {}  # reuse one loaded pipeline per model
    for setup, model, oracle in SETUPS:
        (OUT_DIR / setup).mkdir(exist_ok=True)
        info["setups"].append({"name": setup, "model": model,
                               "speaker_count_given": oracle, "other_settings": "defaults"})
        print(f"\n=== Setup {setup} ({model}, speaker count given: {oracle}) ===")

        try:
            if model not in loaded:
                print("Loading model (first time downloads it)...")
                loaded[model] = Pipeline.from_pretrained(model, token=token)
            pipeline = loaded[model]
        except Exception as e:
            print(f"!! Could not load {model}: {e}")
            print("   Skipping this setup. Copy this error to Claude.")
            continue

        for meeting in MEETINGS:
            out_path = OUT_DIR / setup / f"{meeting}.rttm"
            if out_path.exists():
                print(f"{meeting}: already done, skipping")
                continue

            audio, seconds = load_audio(meeting)
            kwargs = {"num_speakers": true_speaker_count(meeting)} if oracle else {}
            print(f"{meeting}: {seconds/60:.1f} min of audio, running... {kwargs or ''}")

            start = time.time()
            try:
                output = pipeline(audio, **kwargs)
            except Exception as e:
                print(f"!! Failed on {meeting}: {e}")
                continue
            took = time.time() - start

            # community-1 returns an object with .speaker_diarization; 3.1 returns it directly
            annotation = getattr(output, "speaker_diarization", output)
            annotation.uri = meeting
            with open(out_path, "w", encoding="utf-8") as f:
                annotation.write_rttm(f)

            n_found = len(annotation.labels())
            print(f"   done in {took/60:.1f} min, found {n_found} speakers -> {out_path}")
            writer.writerow([setup, meeting, round(seconds, 1), round(took, 1), round(took / seconds, 3)])
            timings_file.flush()

    timings_file.close()
    with open(OUT_DIR / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print("\nAll done. Outputs are in the 'runs' folder.")


if __name__ == "__main__":
    main()
