"""
Voice Arena Q1 - score the mini run with pyannote.metrics

For every setup in runs/ and every meeting, computes:
  DER (main)        no collar, overlapping speech scored          <- primary setting
  DER (0.25 s)      0.25 s tolerance each side of a boundary, overlap scored
                    (pyannote's `collar` is the TOTAL width, so 0.25 s per side = collar=0.5)
  DER (no overlap)  no collar, overlapping speech removed from scoring (the "lenient" old style)
  Missed / False alarm / Confusion   the three parts of the main DER, as % of reference speech
  JER               Jaccard error rate, every speaker weighted equally
  DER in overlap    main DER computed only inside regions where 2+ people talk at once
  Speakers          true count vs. count the system found

Outputs: results_per_meeting.csv, results_summary.csv, and a printed table.
"""

import csv
from pathlib import Path

from pyannote.database.util import load_rttm, load_uem
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

MEETINGS = ["ES2004a", "TS3003a", "EN2002a"]
RUNS = Path("runs")


def load_one(path, meeting):
    return load_rttm(str(path))[meeting]


def pct(x):
    return round(100 * x, 1)


def main():
    setups = sorted(p.name for p in RUNS.iterdir() if p.is_dir())
    refs = {m: load_one(Path("ref") / f"{m}.rttm", m) for m in MEETINGS}
    uems = {m: load_uem(str(Path("uem") / f"{m}.uem"))[m] for m in MEETINGS}

    rows, summary = [], []
    for setup in setups:
        # metric objects accumulate across meetings -> pooled score at the end
        der_main = DiarizationErrorRate(collar=0.0, skip_overlap=False)
        der_collar = DiarizationErrorRate(collar=0.5, skip_overlap=False)
        der_noov = DiarizationErrorRate(collar=0.0, skip_overlap=True)
        der_ov = DiarizationErrorRate(collar=0.0, skip_overlap=False)
        jer = JaccardErrorRate(collar=0.0, skip_overlap=False)

        done = 0
        for m in MEETINGS:
            hyp_path = RUNS / setup / f"{m}.rttm"
            if not hyp_path.exists():
                print(f"(missing: {setup}/{m}.rttm - skipped)")
                continue
            ref, uem = refs[m], uems[m]
            hyp = load_one(hyp_path, m) if hyp_path.stat().st_size > 0 else ref.empty()

            d = der_main(ref, hyp, uem=uem, detailed=True)
            total = d["total"]
            overlap_region = ref.get_overlap().crop(uem)
            ov = der_ov(ref, hyp, uem=overlap_region) if overlap_region.duration() > 0 else float("nan")

            rows.append({
                "setup": setup,
                "meeting": m,
                "DER_main_%": pct(d["diarization error rate"]),
                "missed_%": pct(d["missed detection"] / total),
                "false_alarm_%": pct(d["false alarm"] / total),
                "confusion_%": pct(d["confusion"] / total),
                "DER_collar0.25_%": pct(der_collar(ref, hyp, uem=uem)),
                "DER_no_overlap_%": pct(der_noov(ref, hyp, uem=uem)),
                "JER_%": pct(jer(ref, hyp, uem=uem)),
                "DER_in_overlap_%": pct(ov),
                "overlap_share_of_speech_%": pct(overlap_region.duration() / total),
                "true_speakers": len(ref.labels()),
                "found_speakers": len(hyp.labels()),
            })
            done += 1

        if done:
            summary.append({
                "setup": setup,
                "meetings_scored": done,
                "DER_main_%": pct(abs(der_main)),
                "DER_collar0.25_%": pct(abs(der_collar)),
                "DER_no_overlap_%": pct(abs(der_noov)),
                "JER_%": pct(abs(jer)),
                "DER_in_overlap_%": pct(abs(der_ov)),
            })

    for name, data in [("results_per_meeting.csv", rows), ("results_summary.csv", summary)]:
        if data:
            with open(name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)

    print("\nPER MEETING (all numbers in %)")
    print(f"{'setup':<18}{'meeting':<9}{'DER':>6}{'miss':>6}{'FA':>6}{'conf':>6}"
          f"{'DER.25':>8}{'noOv':>6}{'JER':>6}{'inOv':>6}{'spk':>8}")
    for r in rows:
        print(f"{r['setup']:<18}{r['meeting']:<9}{r['DER_main_%']:>6}{r['missed_%']:>6}"
              f"{r['false_alarm_%']:>6}{r['confusion_%']:>6}{r['DER_collar0.25_%']:>8}"
              f"{r['DER_no_overlap_%']:>6}{r['JER_%']:>6}{r['DER_in_overlap_%']:>6}"
              f"{str(r['found_speakers']) + '/' + str(r['true_speakers']):>8}")

    print("\nPOOLED OVER MEETINGS (weighted by speech time, %)")
    print(f"{'setup':<18}{'n':>3}{'DER':>7}{'DER.25':>8}{'noOv':>7}{'JER':>7}{'inOv':>7}")
    for s in summary:
        print(f"{s['setup']:<18}{s['meetings_scored']:>3}{s['DER_main_%']:>7}{s['DER_collar0.25_%']:>8}"
              f"{s['DER_no_overlap_%']:>7}{s['JER_%']:>7}{s['DER_in_overlap_%']:>7}")
    print("\nSaved results_per_meeting.csv and results_summary.csv")


if __name__ == "__main__":
    main()
