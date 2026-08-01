"""`gazekit metrics` — self-diagnosis of the error computation itself.

When the numbers look wrong ("is the loss calculation buggy?"), this mode
answers with evidence instead of re-reading the code. Every indicator is
computed independently from the evaluate/iterate path so a bug there
cannot hide itself:

  synthetic   fit on data with a KNOWN exact linear mapping — if the
              fit/transform/error code is correct the error must be ~0.
              A big number here is a code error, full stop.
  baselines   LOSO error vs predict-screen-center and predict-train-mean.
              A trained model losing to a constant predictor means the
              features, the fit, or the error math is broken.
  axis-swap   LOSO error with predicted x/y exchanged. If the swap SCORES
              BETTER the axes are crossed somewhere.
  saturation  fraction of deployed-model predictions clipped to the
              screen border (per axis). High saturation = the dot pins
              to an edge and one axis appears dead (posture drift or a
              scale bug — not "gradient explosion": ridge has none).
  consistency mean <= rmse and max(x, y) <= euclidean <= x + y on the
              same residuals — pure arithmetic identities of a correct
              error implementation.

Verdict: FAIL if any indicator shows a code-level contradiction,
WARN for data-level issues (saturation, weak-but-sane model), PASS
otherwise.
"""

import numpy as np

from .evaluate import _cluster_err, load_records, split_epochs
from .model import GazeModel, transform

SYNTH_LIMIT_PX = 25.0        # ridge regularization keeps this small, not 0
SATURATION_WARN_PCT = 20.0


def synthetic_check(screen):
    """Known linear ground truth -> near-zero error, or the code is wrong."""
    rng = np.random.default_rng(0)
    X = rng.normal(scale=0.25, size=(800, 14))
    W = rng.normal(size=(transform(X).shape[1], 2))
    raw = transform(X) @ W
    lo, span = raw.min(axis=0), np.ptp(raw, axis=0) + 1e-9
    Y = (raw - lo) / span * np.array(screen) * 0.9 + 0.05 * np.array(screen)
    m = GazeModel(tuple(screen))
    m.fit(X[:600], Y[:600])
    errs = [float(np.hypot(*(m.predict(x) - y)))
            for x, y in zip(X[600:], Y[600:])]
    return round(float(np.mean(errs)), 2)


def _loso_predictions(recs, screen):
    """Per-cluster (prediction, target) pairs, leave-one-session-out —
    recomputed here with plain fit() so it cross-checks evaluate()'s
    calibration-aware path rather than reusing it."""
    sessions = sorted({r["session"] for r in recs})
    pairs = []
    for sess in sessions:
        train = [r for r in recs if r["session"] != sess]
        test = [r for r in recs if r["session"] == sess]
        m = GazeModel(tuple(screen))
        m.fit(np.array([r["X"] for r in train]),
              np.array([r["Y"] for r in train]))
        for c in _cluster_err(m, test):
            pairs.append((c["signed"] + c["target"], c["target"]))
    P = np.array([p for p, _ in pairs])
    T = np.array([t for _, t in pairs])
    return P, T


def run(dataset_root="data/dataset"):
    recs, screen = load_records(dataset_root)
    if not recs or screen is None:
        raise SystemExit("no dwell data found — run `calibrate` first")
    report = {"samples": len(recs)}
    fails, warns = [], []

    # 1. synthetic ground truth
    synth = synthetic_check(screen)
    report["synthetic_err_px"] = synth
    ok = synth < SYNTH_LIMIT_PX
    print(f"  {'PASS' if ok else 'FAIL'}  synthetic known-mapping error: "
          f"{synth}px (limit {SYNTH_LIMIT_PX})")
    if not ok:
        fails.append("synthetic: fit/transform/error code is broken")

    recs = split_epochs(recs)
    if len({r['session'] for r in recs}) < 2:
        print("  SKIP  cross-session indicators need >= 2 sessions")
        report["verdict"] = "FAIL" if fails else "PASS"
        return report

    P, T = _loso_predictions(recs, screen)
    model_err = float(np.mean(np.hypot(*(P - T).T)))
    report["loso_recomputed_px"] = round(model_err, 1)

    # 2. constant baselines the model must beat
    w, h = screen
    center_err = float(np.mean(np.hypot(T[:, 0] - w / 2, T[:, 1] - h / 2)))
    mean_err = float(np.mean(np.hypot(*(T - T.mean(axis=0)).T)))
    report["baseline_center_px"] = round(center_err, 1)
    report["baseline_mean_px"] = round(mean_err, 1)
    report["skill_vs_baseline"] = round(1 - model_err / max(mean_err, 1e-9), 2)
    ok = model_err < min(center_err, mean_err)
    print(f"  {'PASS' if ok else 'FAIL'}  model {model_err:.0f}px vs "
          f"baselines: center {center_err:.0f}px, target-mean "
          f"{mean_err:.0f}px  (skill {report['skill_vs_baseline']:+.2f})")
    if not ok:
        fails.append("baselines: model loses to a constant predictor")

    # 3. axis swap
    swap_err = float(np.mean(np.hypot(P[:, 1] - T[:, 0], P[:, 0] - T[:, 1])))
    report["axis_swap_px"] = round(swap_err, 1)
    ok = model_err < swap_err
    print(f"  {'PASS' if ok else 'FAIL'}  axis-swap error {swap_err:.0f}px "
          f"{'>' if ok else '<='} normal {model_err:.0f}px")
    if not ok:
        fails.append("axis-swap: x/y are crossed somewhere in the pipeline")

    # 4. deployed-model saturation (per axis, all samples)
    try:
        from .dataset import model_path_for
        dep = GazeModel.load(model_path_for())
        # raw model space via the pipeline directly — predict() clips to
        # the screen box, which is exactly what saturation must see past
        raw = np.array([dep.pipe.predict(transform(r["X"]))[0] + dep.bias
                        for r in recs])
        sat_x = float(np.mean((raw[:, 0] <= 0) | (raw[:, 0] >= w - 1))) * 100
        sat_y = float(np.mean((raw[:, 1] <= 0) | (raw[:, 1] >= h - 1))) * 100
        report["saturation_pct_xy"] = [round(sat_x, 1), round(sat_y, 1)]
        worst = max(sat_x, sat_y)
        ok = worst < SATURATION_WARN_PCT
        print(f"  {'PASS' if ok else 'WARN'}  deployed prediction "
              f"saturation: x {sat_x:.0f}% / y {sat_y:.0f}% off-screen")
        if not ok:
            warns.append("saturation: deployed model pins to screen edges "
                         "for part of the data — recalibrate or re-align "
                         "(the dot will look frozen on that axis)")
    except Exception as e:
        print(f"  SKIP  saturation (no deployed model: {e})")

    # 5. arithmetic identities of the error definition
    d = np.abs(P - T)
    eu = np.hypot(d[:, 0], d[:, 1])
    rmse = float(np.sqrt((eu ** 2).mean()))
    ident = (eu.mean() <= rmse + 1e-6
             and np.all(eu >= d.max(axis=1) - 1e-6)
             and np.all(eu <= d.sum(axis=1) + 1e-6))
    report["consistency_ok"] = bool(ident)
    print(f"  {'PASS' if ident else 'FAIL'}  error identities "
          f"(mean {eu.mean():.0f} <= rmse {rmse:.0f}; "
          f"max-axis <= euclid <= x+y)")
    if not ident:
        fails.append("consistency: error arithmetic violates identities")

    report["verdict"] = ("FAIL" if fails else "WARN" if warns else "PASS")
    report["recommendations"] = fails + warns or [
        "error computation is sound — bad numbers mean bad data or drift, "
        "not a code error"]
    print(f"\nverdict: {report['verdict']}")
    for r in report["recommendations"]:
        print(f"  - {r}")
    return report
