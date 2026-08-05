# Evaluation feedback standard

`gazekit iterate` turns evaluation results into a structured feedback artifact
after every run. This makes the next collection decision reproducible instead
of relying on terminal output or memory.

## Safety boundary

Feedback is **advisory**. It must never delete samples, alter labels, replace a
model, or start a collection run on its own. Existing cleaning and deployment
gates remain the only mechanisms allowed to prune samples or promote a model.

This is deliberate: a high residual may be an interesting camera pose, a real
user change, or a bad label. It is not enough evidence to rewrite history.

## Artifact layout

Each successful `iterate` run writes one camera-domain-specific file:

```
data/eval/feedback_webcam.json
data/eval/feedback_phone.json
```

The artifact is overwritten for the same domain, while the immutable trend
history remains in `data/eval_history.jsonl`.

Required top-level fields:

- `schema_version`: feedback format version.
- `generated_at`: UTC timestamp.
- `camera_domain`: `webcam` or `phone`.
- `summary`: key error, coverage, and quality measures from the evaluation.
- `actions`: suggested next steps, each marked `advisory: true`.
- `collection_suggestions`: weak screen regions and recommended scenarios.
- `recommendations`: human-readable explanation copied from `iterate`.

## Allowed action types

- `adjust_region_sampling`: suggested 3×3 collection weights derived from
  measured error. These guide a future collector; they do not retrain a model.
- `collect`: a named existing collection scenario (`daily`, `edges`, `vor`, or
  `posture`) that fills a measured coverage gap.
- `review_quality`: signals an elevated fraction of low-quality samples; it
  asks for collection-condition review, not automatic pruning.
- `verify`: asks for a manual path verification when the error distribution has
  a heavy tail.

Actions must contain the metric and threshold that caused them. An unknown
action type is ignored by consumers.

## Region weighting

When the evaluation has a complete 3×3 error map, each measured cell receives
a bounded suggested collection weight between `0.75` and `1.50`, proportional
to its error relative to the median measured cell. Missing cells stay at `1.0`
and are listed separately as unmeasured rather than guessed.

## Validation

Every feedback change must prove that:

1. a synthetic evaluation produces valid JSON with only advisory actions;
2. webcam and phone artifacts cannot overwrite one another;
3. writing feedback does not alter `pruned.json`, a model, or dataset rows;
4. the camera-free self-test still passes.
