# gazekit working plan — autonomous run 2026-08-01

Goal: keep improving with the frozen dataset until blocked on human data
collection. Checkboxes updated as work lands.

## Executable now (no new data needed)

- [x] 1. Fix `train-cnn` validation-session choice: newest session with
      ≥200 samples (was a 34-sample ambient session — too noisy to
      early-stop on).
- [x] 2. Blink-gate offline evaluation → `research/data/blink_gate_eval.csv`.
      Personal profile F1 0.898 / precision 1.0 / 0% false-freeze vs generic
      0.790 / 2.7% false-freeze. `collect blinks` formula tuned 0.55→0.45
      interpolation (sweep optimum, keeps hysteresis).
- [x] 3. CNN offline evaluation → `research/data/cnn_per_session.csv`.
      CNN mean 143.5px across all stored crops (in-sample caveat).
- [x] 4. Hybrid ensemble sweep → `research/data/ensemble_sweep.csv`.
      U-shaped: blend 107.6px vs ridge 147.3 / CNN 120.0 alone.
      Deployed α = 0.4 ridge / 0.6 CNN in `live --backend hybrid`.
- [ ] 5. Pursuit-data ablation: A (with) vs B (without) training in
      progress; keep the better one as data/gaze_cnn.pt, upload to HF.
- [x] 6. 3D-eyeball future-proofing: every new sample now logs raw eye
      landmarks (18 pts), 4x4 head transform, interocular px, timestamp.
- [ ] 7. Commit + push all repos; update research REPORT with new results.

## Blocked on Wilson (human data collection)

- [ ] `collect daily` × 2–3 lighting conditions (protocol:
      `research/NEXT_COLLECTION.md`) — fills days=2 and brightness=0 gaps.
- [ ] `gazekit annotate` first run — needs the context snapshots that only
      new collection sessions produce (feature landed after data freeze).
- [ ] 3D eyeball model fitting — needs sessions recorded with the raw
      landmarks from item 6.
- [ ] CNN-as-primary decision — needs ≥5 days of data variation.
