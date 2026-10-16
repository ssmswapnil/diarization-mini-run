"""
Voice Arena Q1 - far-field add-on run
Same 3 AMI test meetings, same reference answers, same settings as run_diarization.py.
The ONLY change: audio comes from one distant table microphone (Array1-01)
instead of the mix of close-talk headset mics.

  community1_farfield   pyannote/speaker-diarization-community-1, speaker count estimated
  legacy31_farfield     pyannote/speaker-diarization-3.1,          speaker count estimated

Outputs go into runs/<setup>/ so score_diarization.py picks them up automatically.
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
AUDIO_DIR = Path("audio_far")
OUT_DIR = Path("runs")

SETUPS = [
    ("community1_farfield", "pyannote/speaker-diarization-community-1"),
    ("legacy31_farfield", "pyannote/speaker-diarization-3.1"),
]


def load_audio(meeting):
    """Load the wav with soundfile (avoids the Windows FFmpeg/torchcodec issue)."""
    data, sr = sf.read(AUDIO_DIR / f"{meeting}.wav", dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return {"waveform": waveform, "sample_rate": sr}, waveform.shape[1] / sr


def main():
    OUT_DIR.mkdir(exist_ok=True)
    token = get_token()
    if not token:
        raise SystemExit("No Hugging Face token found. Run:  hf auth login")

    for m in MEETINGS:
        if not (AUDIO_DIR / f"{m}.wav").exists():
            raise SystemExit(f"Missing {AUDIO_DIR / (m + '.wav')} - download it first.")

    timings_path = OUT_DIR / "timings.csv"
    new_timings = not timings_path.exists()
    timings_file = open(timings_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(timings_file)
    if new_timings:
        writer.writerow(["setup", "meeting", "audio_seconds", "processing_seconds", "real_time_factor"])

    for setup, model in SETUPS:
        (OUT_DIR / setup).mkdir(exist_ok=True)
        print(f"\n=== Setup {setup} ({model}, far-field mic, speaker count estimated) ===")
        try:
            pipeline = Pipeline.from_pretrained(model, token=token)
        except Exception as e:
            print(f"!! Could not load {model}: {e}")
            continue

        for meeting in MEETINGS:
            out_path = OUT_DIR / setup / f"{meeting}.rttm"
            if out_path.exists():
                print(f"{meeting}: already done, skipping")
                continue

            audio, seconds = load_audio(meeting)
            print(f"{meeting}: {seconds/60:.1f} min of audio, running...")
            start = time.time()
            try:
                output = pipeline(audio)
            except Exception as e:
                print(f"!! Failed on {meeting}: {e}")
                continue
            took = time.time() - start

            annotation = getattr(output, "speaker_diarization", output)
            annotation.uri = meeting
            with open(out_path, "w", encoding="utf-8") as f:
                annotation.write_rttm(f)

            print(f"   done in {took/60:.1f} min, found {len(annotation.labels())} speakers -> {out_path}")
            writer.writerow([setup, meeting, round(seconds, 1), round(took, 1), round(took / seconds, 3)])
            timings_file.flush()

    timings_file.close()
    with open(OUT_DIR / "run_info_farfield.json", "w", encoding="utf-8") as f:
        json.dump({
            "python": platform.python_version(),
            "os": platform.platform(),
            "device": "cpu",
            "pyannote.audio": pyannote.audio.__version__,
            "torch": torch.__version__,
            "data": "AMI test set, Array1-01 (single distant mic) audio, only_words references",
            "meetings": MEETINGS,
            "setups": [{"name": s, "model": m, "speaker_count_given": False,
                        "other_settings": "defaults"} for s, m in SETUPS],
        }, f, indent=2)
    print("\nAll done. Now run:  .venv\\Scripts\\python.exe score_diarization.py")


if __name__ == "__main__":
    main()
