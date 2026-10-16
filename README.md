# Speaker diarization mini-run (Voice Arena PM assessment, Q1)

This repo is the small diarization evaluation behind the "Mini-run" section of my Q1 leaderboard design. I ran two open-source pyannote pipelines on three AMI test meetings, recorded with both headset and far-field microphones, on my own laptop (CPU only), and scored them with `pyannote.metrics`.

All system outputs (RTTM files), timings and environment details are committed, so the result tables can be **re-scored in a few seconds without any audio or GPU** (see [Reproduce the scores](#reproduce-the-scores)).

## Results

Pooled over the 3 meetings, weighted by speech time (%). Source: `results_summary.csv`.

| Setup | DER | DER with 0.25 s collar | DER with overlap ignored | JER | DER in overlap |
|---|---|---|---|---|---|
| community-1, headset (`community1`) | 20.2 | 13.7 | 12.9 | 29.5 | 34.5 |
| community-1, headset, count given (`community1_oracle`) | 20.0 | 13.6 | 12.8 | 29.1 | 34.3 |
| pyannote 3.1, headset (`legacy31`) | 21.6 | 14.9 | 14.4 | 37.9 | 35.8 |
| community-1, far-field (`community1_farfield`) | 24.7 | 17.9 | 18.5 | 43.2 | 36.9 |
| pyannote 3.1, far-field (`legacy31_farfield`) | 27.3 | 19.7 | 19.4 | 44.0 | 42.9 |

Per-meeting numbers (missed speech, false alarm, confusion, JER, overlap share, true vs. found speakers) are in `results_per_meeting.csv`.

## What was run

| | |
|---|---|
| Meetings | AMI test set: ES2004a, TS3003a, EN2002a (4 speakers each) |
| Headset audio | `Mix-Headset` (mix of close-talk headset mics) |
| Far-field audio | `Array1-01` (one channel of a distant table microphone array, no beamforming) |
| Reference | `only_words` RTTM and UEM files from [pyannote/AMI-diarization-setup](https://github.com/pyannote/AMI-diarization-setup) |
| Systems | `pyannote/speaker-diarization-community-1` and `pyannote/speaker-diarization-3.1`, default settings |
| Variant | community-1 told the true number of speakers (`num_speakers=4`), headset only |
| Machine | Windows 11 laptop, CPU only, Python 3.11.4, PyTorch 2.14.1 (CPU), pyannote.audio 4.0.7 |
| Scoring | pyannote.metrics 4.1. Main setting: no collar, overlap scored, whole recording (UEM) |

`runs/run_info.json` and `runs/run_info_farfield.json` record the environment each run actually used. The `"os"` value `Windows-10-10.0.22631` is how Python reports Windows 11 (build 22631).

`runs/timings.csv` has the processing time for every file: 0.79–0.97× real time on the laptop CPU.

## Repo layout

```
run_diarization.py     headset runs: community1, community1_oracle, legacy31
run_farfield.py        far-field runs: community1_farfield, legacy31_farfield
score_diarization.py   scores every setup in runs/ and writes the two results CSVs
ref/                   reference RTTMs (AMI-diarization-setup, only_words)
uem/                   scoring regions (whole recording)
runs/<setup>/          system output, one RTTM per meeting
runs/timings.csv       audio length, processing time and real-time factor per file
runs/run_info*.json    Python, OS, torch and pyannote versions used
results_*.csv          scores produced by score_diarization.py
```

## Reproduce the scores

This needs only the committed RTTMs, so no audio or Hugging Face access is required:

```powershell
python -m venv .venv
.venv\Scripts\pip install pyannote.metrics==4.1 pyannote.database==6.1.1
.venv\Scripts\python.exe score_diarization.py
```

This rewrites `results_per_meeting.csv` and `results_summary.csv` and prints both tables.

## Re-run the systems from scratch

1. **Install** (Windows, PowerShell):

   ```powershell
   python -m venv .venv
   .venv\Scripts\pip install torch==2.14.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
   .venv\Scripts\pip install -r requirements.txt
   ```

2. **Hugging Face access.** Accept the user conditions on the model pages for [speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and [speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1), then run `.venv\Scripts\hf auth login`.

3. **Download the AMI audio** (about 300 MB; not committed). Files are saved as `<meeting>.wav`:

   ```powershell
   mkdir audio, audio_far
   foreach ($m in "ES2004a","TS3003a","EN2002a") {
     Invoke-WebRequest "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/$m/audio/$m.Mix-Headset.wav" -OutFile "audio\$m.wav"
     Invoke-WebRequest "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/$m/audio/$m.Array1-01.wav" -OutFile "audio_far\$m.wav"
   }
   ```

4. **Run and score:**

   ```powershell
   .venv\Scripts\python.exe run_diarization.py
   .venv\Scripts\python.exe run_farfield.py
   .venv\Scripts\python.exe score_diarization.py
   ```

   Both run scripts skip files that already have an RTTM in `runs/`. Delete a setup's folder to re-run it. On CPU each file takes roughly as long as the audio itself.

## Notes

- **Collar convention.** In pyannote.metrics `collar` is the *total* width around a boundary, while dscore and most papers quote it *per side*. The standard 0.25 s per side is therefore `collar=0.5` in `score_diarization.py`.
- **Audio loading.** Audio is loaded with `soundfile` and passed to the pipeline as a waveform, which avoids the FFmpeg/torchcodec decoding issue on Windows.
- **Pipeline output.** community-1 returns an object with a `.speaker_diarization` attribute, while 3.1 returns the annotation directly. The scripts handle both.

## Licence and data

The code is under the [MIT licence](LICENSE). The AMI Meeting Corpus audio and annotations are by the AMI Consortium and are distributed under CC BY 4.0. The reference files in `ref/` and `uem/` come from [pyannote/AMI-diarization-setup](https://github.com/pyannote/AMI-diarization-setup).
