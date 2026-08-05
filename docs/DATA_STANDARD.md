# Dataset standard — schema v2

Every gaze sample written by GazeKit is a training artifact. This standard
defines the on-disk JSONL contract used by collection, calibration, ambient
and verification flows. It complements `COLLECTION_STANDARD.md`: that document
defines *when* a sample may be collected; this document defines *what* must be
recorded once it is accepted.

## Compatibility rule

- New writers emit `schema_version: 2` in the session metadata row.
- Readers must accept earlier rows. Missing v2 fields have conservative,
  non-destructive defaults: `quality_score = 1.0`, no quality components, and
  no additional provenance.
- Schema upgrades never rewrite or delete existing samples. Dataset cleaning
  remains the only path that can exclude a sample, via `pruned.json`.

## Session metadata row

The first row of every `samples.jsonl` has `meta: true` and contains:

- `schema_version`: integer format version.
- `session_id`: the containing session directory name.
- `created_at`: UTC ISO-8601 timestamp.
- `screen_size`: `[width, height]` in pixels.
- `camera`: source domain (`"0"` for webcam or `"phone"`).
- `feature_schema`: the raw feature layout used to train the current ridge
  model (`"ridge-raw14"` for v2).

The session directory keeps the eye crops. A full-frame context snapshot, when
captured for offline annotation, stays under `data/context/`; it is explicitly
outside the publishable dataset allowlist.

## Accepted sample row

Every accepted dwell sample records the existing target, raw ridge features,
head pose, blink score, brightness, interocular distance, eye crops, raw eye
landmarks and head transform. Schema v2 additionally records:

- `quality_score`: tracker-reliability score in `[0, 1]`.
- `quality_components`: the explainable component scores (`eyes`, `pose`,
  `distance`, `lighting`) used to produce it.
- `frame_size`: camera frame `[width, height]` in pixels when available.

The quality score measures signal reliability, not whether a pose is useful.
Head-pose and VOR samples are therefore still retained; collection's existing
frame gate is the authority on whether a sample is valid.

## Quality computation and training use

For a gated sample, quality combines the following normalized components:

- eyes: lower MediaPipe blink blendshape is better;
- pose: a more frontal face is easier to track, while still giving valid
  off-axis samples non-zero credit;
- distance: interocular distance normalized by frame width;
- lighting: even mid-range face brightness is preferred.

The components are weighted `0.40 / 0.25 / 0.20 / 0.15` respectively. Ridge
training uses quality only as a bounded soft weight:

```
quality_weight = 0.55 + 0.45 * quality_score
sample_weight = existing_trust_weight * quality_weight
```

Thus every gated sample retains at least 55% of its trust weight. Quality is
never an automatic delete or relabel signal.

## Validation

Any schema change must demonstrate all of the following before release:

1. New rows contain the required metadata and quality fields.
2. An old-style row without v2 fields still loads and receives weight `1.0`.
3. The standard self-test passes without a camera.
4. Collection modes continue to meet the five-stage collection contract.
