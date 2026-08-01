"""`gazekit iterate` — the dataset lifecycle: clean -> train -> validate ->
evaluate -> update. Run it after every data collection.

  clean     per-cluster MAD pruning (glitch frames) + residual-based label
            noise detection for low-trust tags (click/ambient). Pruned ids
            go to data/dataset/pruned.json — files are never deleted, and
            every loader (ridge refits, CNN training) skips them.
  train     candidate ridge on all cleaned dwell data.
  validate  leave-one-SESSION-out: how well does the model generalize to a
            sitting it never saw (drift robustness)? Plus leave-one-target-
            out on pooled data (interpolation quality).
  evaluate  per-tag, per-axis and 3x3 screen-region error breakdown;
            appended to data/eval_history.jsonl so trends are visible.
  update    deployed model replaced only if the candidate beats it on the
            newest session held out; otherwise kept.
"""

import json
import time
from pathlib import Path

import numpy as np

from .dataset import DWELL_TAGS, load_pruned
from .model import GazeModel

LOW_TRUST_TAGS = {"click", "ambient", "mouse"}
# vor/posture clusters have deliberately large feature spread (head moves
# on purpose) — MAD pruning would delete exactly the valuable samples
HIGH_VARIANCE_TAGS = {"vor", "posture"}
MAD_LIMIT = 5.0  # max robust-z over 8 dims inflates scores; 4.0 over-pruned
RESIDUAL_LIMIT_PX = 320.0


def load_records(root: str | Path):
    """All dwell samples with provenance (ignores existing prune list — the
    clean stage re-decides from scratch each run)."""
    recs, screen = [], None
    for sess in sorted(Path(root).glob("session_*")):
        jl = sess / "samples.jsonl"
        if not jl.exists():
            continue
        for line in open(jl):
            rec = json.loads(line)
            if rec.get("meta"):
                screen = rec["screen_size"]
                continue
            if rec.get("tag") not in DWELL_TAGS:
                continue
            recs.append({"session": sess.name, "i": rec["i"],
                         "tag": rec["tag"],
                         "X": np.array(rec["features"]),
                         "Y": np.array(rec["target"], dtype=float)})
    return recs, screen


ALIGN_ANCHORS = ((0.5, 0.5), (0.22, 0.25), (0.78, 0.75))


def _fit(model, recs):
    """Calibration-aware fit on a record list (sessions known here)."""
    return model.fit_calaware(
        np.array([r["X"] for r in recs]),
        np.array([r["Y"] for r in recs]),
        [r["session"] for r in recs],
        sample_weight=_weights(recs))


def _aligned_err(model, recs, sw, sh):
    """Session error under the live protocol: per-axis affine fit on the 3
    targets nearest the quick-align anchors, error on the rest."""
    meds = {}
    clusters = {}
    for r in recs:
        clusters.setdefault(tuple(r["Y"]), []).append(r)
    for t, cl in clusters.items():
        meds[t] = np.median([model.predict(r["X"]) for r in cl], axis=0)
    keys = list(meds)
    anchors = list(dict.fromkeys(
        min(keys, key=lambda t: (t[0] - fx * sw) ** 2 + (t[1] - fy * sh) ** 2)
        for fx, fy in ALIGN_ANCHORS))
    P = np.array([meds[t] for t in anchors])
    T = np.array(anchors, dtype=float)
    coef = []
    for ax in (0, 1):
        var = P[:, ax].var()
        if len(anchors) >= 2 and var > 1e-6:
            a = float(np.clip(np.cov(P[:, ax], T[:, ax], bias=True)[0, 1]
                              / var, 0.5, 1.8))
        else:
            a = 1.0
        coef.append((a, float(T[:, ax].mean() - a * P[:, ax].mean())))
    rest = [t for t in keys if t not in anchors]
    if not rest:
        return None
    return float(np.mean([np.hypot(coef[0][0] * meds[t][0] + coef[0][1] - t[0],
                                   coef[1][0] * meds[t][1] + coef[1][1] - t[1])
                          for t in rest]))


def _cluster_err(model, recs):
    """Median-prediction error per (session, target) cluster."""
    out = []
    clusters = {}
    for r in recs:
        clusters.setdefault((r["session"], tuple(r["Y"])), []).append(r)
    for (sess, tgt), rs in clusters.items():
        pred = np.median([model.predict(r["X"]) for r in rs], axis=0)
        out.append({"session": sess, "target": np.array(tgt),
                    "tag": rs[0]["tag"], "n": len(rs),
                    "err": float(np.hypot(*(pred - tgt))),
                    "err_xy": np.abs(pred - np.array(tgt)),
                    "signed": pred - np.array(tgt)})
    return out


def clean(recs, screen):
    """Returns (kept_records, pruned {session: [ids]}, stats)."""
    pruned = {}

    def prune(r, reason):
        pruned.setdefault(r["session"], []).append(int(r["i"]))
        stats[reason] = stats.get(reason, 0) + 1

    stats = {}
    # stage 0: frozen frames — a stalled camera repeats identical features;
    # duplicates add no information and overweight one instant
    prev_key = None
    fresh = []
    for r in sorted(recs, key=lambda r: (r["session"], r["i"])):
        key = (r["session"], tuple(np.round(r["X"][:8], 6)))
        if key == prev_key:
            prune(r, "frozen-frame")
        else:
            fresh.append(r)
        prev_key = key
    recs = fresh

    # stage 1: within-VISIT MAD on iris features (glitch frames). A cluster
    # (session, target) can contain several dwell visits — e.g. calibration
    # round 1 and round 2 at different postures — so split on gaps in the
    # sequential sample ids and judge each visit against itself only.
    clusters = {}
    for r in recs:
        clusters.setdefault((r["session"], tuple(r["Y"])), []).append(r)
    kept = []
    for rs in clusters.values():
        rs.sort(key=lambda r: r["i"])
        visits, cur = [], [rs[0]]
        for prev, r in zip(rs, rs[1:]):
            if r["i"] - prev["i"] > 20:
                visits.append(cur)
                cur = []
            cur.append(r)
        visits.append(cur)
        for vs in visits:
            if len(vs) < 8 or vs[0]["tag"] in HIGH_VARIANCE_TAGS:
                kept.extend(vs)  # judged by residuals below
                continue
            F = np.array([r["X"][:8] for r in vs])
            med = np.median(F, axis=0)
            mad = np.median(np.abs(F - med), axis=0) + 1e-9
            dev = np.max(np.abs(F - med) / mad, axis=1)
            for r, d in zip(vs, dev):
                if d > MAD_LIMIT:
                    prune(r, "mad")
                else:
                    kept.append(r)

    # stage 2: residual-based label noise, leave-one-session-out so a
    # sample never judges itself
    sessions = sorted({r["session"] for r in kept})
    kept2 = []
    if len(sessions) >= 2:
        for sess in sessions:
            train = [r for r in kept if r["session"] != sess]
            test = [r for r in kept if r["session"] == sess]
            m = GazeModel(tuple(screen))
            m.fit(np.array([r["X"] for r in train]),
                  np.array([r["Y"] for r in train]))
            for r in test:
                err = float(np.hypot(*(m.predict(r["X"]) - r["Y"])))
                if r["tag"] in LOW_TRUST_TAGS and err > RESIDUAL_LIMIT_PX:
                    prune(r, "label-noise")
                else:
                    kept2.append(r)
    else:
        kept2 = kept
    return kept2, pruned, stats


def _weights(recs):
    """Recency + trust sample weights (newer sessions matter more)."""
    from .dataset import LOW_TRUST_WEIGHT, RECENCY_DECAY
    sessions = sorted({r["session"] for r in recs})
    age = {s: len(sessions) - 1 - i for i, s in enumerate(sessions)}
    return np.array([
        (LOW_TRUST_WEIGHT if r["tag"] in LOW_TRUST_TAGS else 1.0)
        * RECENCY_DECAY ** age[r["session"]] for r in recs])


def split_epochs(recs, jump=0.30, min_run=40):
    """Split sessions into camera-pose epochs. A deliberate camera move
    shows as a sustained jump in the head pose/translation baseline
    (features 8:14 are relative to the camera). Each epoch then gets its
    own affine in calibration-aware training and its own alignment unit,
    so intentionally moving the camera enriches the data instead of
    poisoning the mapping. Labels were never at risk — targets are ground
    truth regardless of where the camera sits."""
    out = []
    by_sess = {}
    for r in recs:
        by_sess.setdefault(r["session"], []).append(r)
    for sess, rs in by_sess.items():
        rs.sort(key=lambda r: r["i"])
        base = None
        epoch, run = 0, 0
        for r in rs:
            v = np.asarray(r["X"][8:14], dtype=float)
            if base is None:
                base = v.copy()
            if np.linalg.norm(v - base) > jump:
                run += 1
                if run >= min_run:      # sustained -> the camera moved
                    epoch += 1
                    base = v.copy()
                    run = 0
            else:
                run = 0
                base = 0.98 * base + 0.02 * v
            r = dict(r)
            r["session"] = f"{sess}" if epoch == 0 else f"{sess}#e{epoch}"
            out.append(r)
    n_ep = len({r["session"] for r in out}) - len(by_sess)
    if n_ep:
        print(f"  camera-pose epochs: {n_ep} split(s) detected")
    return out


def evaluate(recs, screen):
    """LOSO + leave-target-out metrics on cleaned records."""
    recs = split_epochs(recs)
    sessions = sorted({r["session"] for r in recs})
    report = {"sessions": len(sessions), "samples": len(recs)}

    loso_clusters, aligned = [], []
    if len(sessions) >= 2:
        for sess in sessions:
            train = [r for r in recs if r["session"] != sess]
            test = [r for r in recs if r["session"] == sess]
            m = GazeModel(tuple(screen))
            _fit(m, train)
            loso_clusters.extend(_cluster_err(m, test))
            a = _aligned_err(m, test, *screen)
            if a is not None:
                aligned.append(a)
        if aligned:
            report["loso_aligned_px"] = round(float(np.mean(aligned)), 1)
        errs = np.array([c["err"] for c in loso_clusters])
        report["loso_px"] = round(float(errs.mean()), 1)
        report["loso_rmse_px"] = round(float(np.sqrt((errs ** 2).mean())), 1)
        report["loso_p50_px"] = round(float(np.percentile(errs, 50)), 1)
        report["loso_p90_px"] = round(float(np.percentile(errs, 90)), 1)
        report["loso_p95_px"] = round(float(np.percentile(errs, 95)), 1)
        report["loso_max_px"] = round(float(errs.max()), 1)
        # bias-variance split: how much of the error is a fixable offset?
        signed = np.array([c["signed"] for c in loso_clusters])
        report["bias_xy_px"] = [round(float(signed[:, 0].mean()), 1),
                                round(float(signed[:, 1].mean()), 1)]
        report["residual_sd_xy_px"] = [round(float(signed[:, 0].std()), 1),
                                       round(float(signed[:, 1].std()), 1)]
        report["loso_x_px"] = round(float(np.mean(
            [c["err_xy"][0] for c in loso_clusters])), 1)
        report["loso_y_px"] = round(float(np.mean(
            [c["err_xy"][1] for c in loso_clusters])), 1)
        by_tag = {}
        for c in loso_clusters:
            by_tag.setdefault(c["tag"], []).append(c["err"])
        report["per_tag_px"] = {t: round(float(np.mean(v)), 1)
                                for t, v in sorted(by_tag.items())}
        # 3x3 screen-region map
        w, h = screen
        grid = [[[] for _ in range(3)] for _ in range(3)]
        for c in loso_clusters:
            gx = min(int(c["target"][0] / w * 3), 2)
            gy = min(int(c["target"][1] / h * 3), 2)
            grid[gy][gx].append(c["err"])
        report["region_map_px"] = [
            [round(float(np.mean(cell)), 0) if cell else None
             for cell in row] for row in grid]

    # coverage: which edge cases the dataset HAS vs still needs — cleaning
    # protects them (vor/posture exempt from MAD; extreme-pose samples are
    # never pruned for being extreme), this reports whether they exist
    pose = np.degrees(np.array([r["X"][8:10] for r in recs]))
    dev = np.zeros(len(recs))
    for s in sessions:
        mask = np.array([r["session"] == s for r in recs])
        dev[mask] = np.linalg.norm(
            pose[mask] - np.median(pose[mask], axis=0), axis=1)
    w, h = screen
    Yv = np.array([r["Y"] for r in recs])
    edge_band = ((Yv[:, 0] < 0.1 * w) | (Yv[:, 0] > 0.9 * w)
                 | (Yv[:, 1] < 0.1 * h) | (Yv[:, 1] > 0.9 * h))
    days = {r["session"].split("_")[1][:8] for r in recs}
    report["coverage"] = {
        "days": len(days),
        "pose_gt10deg_pct": round(100 * float((dev > 10).mean()), 1),
        "screen_edge_pct": round(100 * float(edge_band.mean()), 1),
        "tags": {t: sum(1 for r in recs if r["tag"] == t)
                 for t in sorted({r["tag"] for r in recs})},
    }

    # VLM condition slicing: once Florence-2 annotations exist, break the
    # LOSO error down by environment condition — this is how the VLM feeds
    # optimization (condition labels as context for slicing/weighting/the
    # bandit), while the reward signal itself stays prediction error
    ann_path = Path("data/context/annotations.jsonl")
    if ann_path.exists() and loso_clusters:
        flags = {}
        for line in open(ann_path):
            a = json.loads(line)
            flags[a["session"]] = [k for k in ("glasses", "dark",
                                               "lamp_or_window") if a.get(k)]
        by_cond = {}
        for c in loso_clusters:
            for f in flags.get(c["session"], ["unlabeled"]) or ["plain"]:
                by_cond.setdefault(f, []).append(c["err"])
        if by_cond:
            report["per_condition_px"] = {
                k: round(float(np.mean(v)), 1)
                for k, v in sorted(by_cond.items())}

    # calibration-aware fit on all data (deployed candidate)
    m = GazeModel(tuple(screen))
    loto = _fit(m, recs)
    report["loto_px"] = round(loto, 1)
    return report, m


def run(dataset_root="data/dataset", model_out="data/gaze_model.pkl",
        do_clean=True, do_update=True, train_cnn=False):
    root = Path(dataset_root)
    recs, screen = load_records(root)
    if not recs or screen is None:
        raise SystemExit("no dwell data found — run `calibrate` first")
    print(f"loaded {len(recs)} dwell samples from "
          f"{len({r['session'] for r in recs})} sessions")

    if do_clean:
        kept, pruned, stats = clean(recs, screen)
        n_pruned = sum(len(v) for v in pruned.values())
        with open(root / "pruned.json", "w") as f:
            json.dump(pruned, f)
        print(f"clean: pruned {n_pruned} samples "
              f"({', '.join(f'{k}={v}' for k, v in stats.items()) or 'none'})"
              f" -> {root / 'pruned.json'}")
    else:
        kept = recs

    report, candidate = evaluate(kept, screen)
    print(f"\nvalidate/evaluate ({report['samples']} samples, "
          f"{report['sessions']} sessions):")
    if "loso_px" in report:
        if "loso_aligned_px" in report:
            print(f"  cross-session ALIGNED (live protocol): "
                  f"{report['loso_aligned_px']}px")
        print(f"  cross-session (LOSO): {report['loso_px']}px  "
              f"(x {report['loso_x_px']} / y {report['loso_y_px']})  "
              f"p50/p90/p95: {report['loso_p50_px']}/"
              f"{report['loso_p90_px']}/{report['loso_p95_px']}")
        bx, by = report["bias_xy_px"]
        sx, sy = report["residual_sd_xy_px"]
        print(f"  bias/variance: signed bias ({bx:+.0f},{by:+.0f})px, "
              f"residual SD ({sx:.0f},{sy:.0f})px "
              f"{'-> mostly offset-correctable' if abs(by) > sy * 0.5 else ''}")
        print(f"  per tag: " + "  ".join(
            f"{t}={e}px" for t, e in report["per_tag_px"].items()))
        print("  region map (px):")
        for row in report["region_map_px"]:
            print("    " + " ".join(f"{int(c):>5}" if c is not None else
                                    "    -" for c in row))
    else:
        print("  only 1 session — cross-session metrics need >= 2; "
              "collect more and re-run")
    print(f"  interpolation (leave-target-out): {report['loto_px']}px")
    if "per_condition_px" in report:
        print("  per condition (VLM labels): " + "  ".join(
            f"{k}={v}px" for k, v in report["per_condition_px"].items()))
    cov = report["coverage"]
    print(f"  coverage: {cov['days']} day(s), "
          f"{cov['pose_gt10deg_pct']}% samples with head >10° off, "
          f"{cov['screen_edge_pct']}% near screen edges")
    missing = []
    if cov["days"] < 3:
        missing.append("more DAYS (different lighting)")
    if cov["pose_gt10deg_pct"] < 10:
        missing.append("head-pose extremes (collect vor)")
    if cov["screen_edge_pct"] < 8:
        missing.append("screen edges (collect edges)")
    if missing:
        print(f"  -> generalization gaps: {'; '.join(missing)}")

    # update gate: newest session held out, candidate vs deployed
    updated = False
    if do_update:
        sessions = sorted({r["session"] for r in kept})
        # gate needs a session with enough distinct targets to align + test
        big = [s for s in sessions
               if len({tuple(r["Y"]) for r in kept if r["session"] == s}) >= 6]
        if len(sessions) >= 2 and big:
            newest = big[-1]
            train = [r for r in kept if r["session"] != newest]
            test = [r for r in kept if r["session"] == newest]
            cand = GazeModel(tuple(screen))
            _fit(cand, train)
            # gate on the ALIGNED protocol — live always quick-aligns, so
            # raw offset differences shouldn't decide deployment
            cand_err = _aligned_err(cand, test, *screen) or float("inf")
            try:
                deployed = GazeModel.load(model_out)
                depl_err = (_aligned_err(deployed, test, *screen)
                            or float("inf"))
                # sanity: alignment can rescue a badly-scaled model, so an
                # incumbent whose RAW error is wildly worse is corrupt and
                # forfeits (a 6-click live refit once defended itself here)
                raw_d = float(np.mean([c["err"]
                                       for c in _cluster_err(deployed, test)]))
                raw_c = float(np.mean([c["err"]
                                       for c in _cluster_err(cand, test)]))
                if raw_d > 3 * raw_c:
                    print(f"  deployed model looks corrupt (raw {raw_d:.0f}px "
                          f"vs candidate {raw_c:.0f}px) — forfeits")
                    depl_err = float("inf")
            except Exception:
                # includes feature-dimension mismatch after a transform
                # upgrade — the old pickle can't score the new features
                depl_err = float("inf")
            print(f"\nupdate gate on newest session ({newest}, aligned): "
                  f"candidate {cand_err:.0f}px vs deployed {depl_err:.0f}px")
            if cand_err <= depl_err * 1.05:
                candidate.save(model_out, {"refit_from": "iterate", **report})
                updated = True
                print(f"  -> deployed updated (trained on all "
                      f"{len(kept)} cleaned samples)")
            else:
                print("  -> deployed model kept (candidate not better; "
                      "note: deployed may have trained on this session)")
        else:
            candidate.save(model_out, {"refit_from": "iterate", **report})
            updated = True
            print("\nsingle session — model saved")

    # 3D eyeball prototype: report alongside ridge once raw-landmark
    # sessions exist (harmless no-op until then)
    try:
        from .eyeball import evaluate as eyeball_eval
        eb = eyeball_eval(dataset_root)
        if eb is not None:
            report["eyeball_loso_px"] = round(float(eb), 1)
    except Exception as e:
        print(f"  (eyeball eval skipped: {e})")

    # VLM environment annotation: run automatically whenever collection
    # sessions have produced snapshots that aren't annotated yet
    try:
        from .annotate import CONTEXT_DIR, OUT, run as annotate_run
        snaps = {p.stem for p in CONTEXT_DIR.glob("session_*.jpg")}
        done = ({json.loads(l)["session"] for l in open(OUT)}
                if OUT.exists() else set())
        if snaps - done:
            print(f"\nannotating {len(snaps - done)} new context "
                  "snapshot(s) with Florence-2...")
            annotate_run()
    except Exception as e:
        print(f"  (annotation skipped: {e})")

    # advisor: turn the numbers into "what to improve next" so nobody has
    # to interpret the tables — recommendations derived from this project's
    # own research findings
    rec = []
    cov = report.get("coverage", {})
    if cov.get("days", 9) < 3:
        rec.append("collect on more days/lighting: run `collect daily` "
                   "(biggest measured gap)")
    if report.get("loso_y_px", 0) > 1.8 * max(report.get("loso_x_px", 1), 1):
        rec.append("vertical axis is the weak one: `collect vor` + trust the "
                   "hybrid backend (CNN helps vertical most)")
    if report.get("loso_p95_px", 0) > 2.2 * max(report.get("loso_p50_px", 1), 1):
        rm = report.get("region_map_px") or []
        worst = max(((v, (i, j)) for i, row in enumerate(rm)
                     for j, v in enumerate(row) if v), default=None)
        where = (f" (worst region row{worst[1][0]} col{worst[1][1]}, "
                 f"{worst[0]:.0f}px)" if worst else "")
        rec.append("heavy error tail" + where + ": ambient bandit will "
                   "target it; a `verify --mode path` run localizes it")
    tag_px = report.get("per_tag_px", {})
    if tag_px.get("posture", 0) > 2 * report.get("loso_px", 1e9):
        rec.append("posture extrapolation is weak: one more `collect "
                   "posture` at a genuinely different sitting position")
    cond = report.get("per_condition_px", {})
    if len(cond) >= 2 and max(cond.values()) > 1.5 * min(cond.values()):
        worst_c = max(cond, key=cond.get)
        rec.append(f"condition '{worst_c}' is much worse — collect more "
                   "sessions under it")
    if report.get("eyeball_loso_px") and report.get("loso_px") and \
            report["eyeball_loso_px"] < report["loso_px"] * 0.9:
        rec.append("3D eyeball model is beating ridge — consider promoting it")
    if not rec:
        rec.append("no glaring gap — keep ambient running and re-check the "
                   "trend after the next collection")
    report["recommendations"] = rec
    print("\nadvice:")
    for r in rec:
        print(f"  - {r}")

    entry = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), **report,
             "pruned": sum(len(v) for v in load_pruned(root).values()),
             "updated": updated}
    hist_path = Path("data/eval_history.jsonl")
    with open(hist_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    hist = [json.loads(l) for l in open(hist_path)]
    if len(hist) > 1:
        trend = [h.get("loso_px") or h.get("loto_px") for h in hist[-6:]]
        print(f"\ntrend (last {len(trend)} runs): "
              + " -> ".join(f"{v:.0f}px" for v in trend if v is not None))

    if train_cnn:
        from .cnn import train
        print("\ntraining CNN on the cleaned dataset...")
        train(dataset_root=dataset_root)
    return report
