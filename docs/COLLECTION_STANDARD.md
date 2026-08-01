# Collection-mode standard

Every data-collection mode in gazekit — present or future, webcam / phone /
ARKit / anything — implements this contract. A mode that skips a stage must
say why in its module docstring. (This exists because `arkit --calib` v1
shipped without gates and burned a session.)

## The five stages

1. **Pre-flight gate** — refuse to start until every input is demonstrably
   healthy, shown live on screen, held stable:
   - webcam modes: `calibrate.environment_gate` (face, distance, lighting,
     head pose; 2 s hold)
   - streamed modes: source liveness (packet rate), plus signal quality
     (face tracked, eyes open) derived from the stream itself
   - always: blink break (`say` + 2.5 s) — dry eyes drift the iris fit

2. **Sample-time gating** — each accepted sample passes per-frame checks
   (blink, pose bounds, source liveness). A dwell window that loses its
   source or collects too few samples is INVALID and the point is redone,
   never silently logged.

3. **Provenance per sample** — timestamp, tag, and enough raw signal to
   re-derive features later (raw landmarks / transforms / counts). If it
   isn't logged, a future model can't use it.

4. **Validation + verdict** — after collection, held-out probe targets the
   model never fit on measure honest error; report a verdict against the
   shared thresholds (STABLE ≤ 4.5 % of diagonal, USABLE ≤ 7.5 %). No
   collection run ends without a number.

5. **Registration** — the run's result dict reaches the journal
   (return it from `_dispatch`), artifacts land in their standard places
   (`data/dataset` / `data/arkit`), and deploy gates decide promotion —
   never unconditional overwrites.

## Shared helpers to reuse (don't reinvent)

- `calibrate.environment_gate`, `collect_point` (gating + MAD), `info_screen`
- `ui.say`, `ui.FullscreenWindow`, `ui.draw_target`, `ui.progress_bar`
- `journal.log_run` (automatic via `__main__._dispatch` return value)
- verdict thresholds: `calibrate.PASS_FRAC` / `MARGINAL_FRAC`
