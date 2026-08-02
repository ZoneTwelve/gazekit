# Publish standard — syncing artifacts to Hugging Face

Every upload of models or dataset to the Hub — present or future — goes
through `python -m gazekit publish`, never an ad-hoc script or the web UI.
(This exists so publication gets the same gates as deployment: an
unconditionally saved model once usurped a better one; an unconditionally
uploaded one would do the same in public.)

## Repos

| artifact | repo | type |
|---|---|---|
| `data/gaze_cnn.pt`, `data/gaze_model.pkl` | `ZoneTwelve/gazekit` | model |
| `data/dataset/` (samples + eye crops) | `ZoneTwelve/gazekit` | dataset |

## The five stages

1. **Auth gate** — refuse to start without a Hub token (`hf auth login` or
   `HF_TOKEN`). Never prompt for or store a token ourselves.

2. **Privacy allowlist** — the dataset upload is allowlist-only:
   `session_*/samples.jsonl`, `session_*/crops/*.png`, `pruned.json`.
   Everything else in `data/` (context snapshots of the room, journal,
   config, ARKit streams, models) never leaves the machine via the dataset
   repo. New dataset repos are created **private**; `--public` is an
   explicit, deliberate flag.

3. **Publish gate (models)** — a model file is uploaded only if it beats
   the currently published one on the same protocol as the deploy gate:
   newest big held-out session, aligned error (`evaluate._aligned_err`);
   CNNs score mean px error on that session's crops. Byte-identical files
   are skipped. A file with no published incumbent uploads freely (first
   publication). Never `--force` past a losing gate without a written
   reason in the commit.

4. **Cards** — every repo carries a generated README (model card / dataset
   card) that states the data is personalized to one user, one camera, one
   screen. The dataset card records session count, sample count, camera
   domains, and the feature schema version.

5. **Registration** — the run returns a result dict (journaled via
   `_dispatch`): what was uploaded, what was kept back and why, and the
   gate numbers. README/AGENT.md links updated in the same session as a
   first publication.

## Shared helpers to reuse (don't reinvent)

- `evaluate.load_records`, `evaluate.clean`, `evaluate._aligned_err`,
  `evaluate._cluster_err` (gate protocol)
- `cnn.GazeNet` + `cnn.CropDataset` (CNN gate scoring)
- `journal.log_run` (automatic via `__main__._dispatch` return value)
