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

LOW_TRUST_TAGS = {"click", "ambient"}
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
                    "err_xy": np.abs(pred - np.array(tgt))})
    return out


def clean(recs, screen):
    """Returns (kept_records, pruned {session: [ids]}, stats)."""
    pruned = {}

    def prune(r, reason):
        pruned.setdefault(r["session"], []).append(int(r["i"]))
        stats[reason] = stats.get(reason, 0) + 1

    stats = {}
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


def evaluate(recs, screen):
    """LOSO + leave-target-out metrics on cleaned records."""
    sessions = sorted({r["session"] for r in recs})
    report = {"sessions": len(sessions), "samples": len(recs)}

    loso_clusters = []
    if len(sessions) >= 2:
        for sess in sessions:
            train = [r for r in recs if r["session"] != sess]
            test = [r for r in recs if r["session"] == sess]
            m = GazeModel(tuple(screen))
            m.fit(np.array([r["X"] for r in train]),
                  np.array([r["Y"] for r in train]))
            loso_clusters.extend(_cluster_err(m, test))
        report["loso_px"] = round(float(np.mean(
            [c["err"] for c in loso_clusters])), 1)
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

    # leave-one-target-out on the pooled data (interpolation quality)
    m = GazeModel(tuple(screen))
    loto = m.fit(np.array([r["X"] for r in recs]),
                 np.array([r["Y"] for r in recs]))
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
        print(f"  cross-session (LOSO): {report['loso_px']}px  "
              f"(x {report['loso_x_px']} / y {report['loso_y_px']})")
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

    # update gate: newest session held out, candidate vs deployed
    updated = False
    if do_update:
        sessions = sorted({r["session"] for r in kept})
        if len(sessions) >= 2:
            newest = sessions[-1]
            train = [r for r in kept if r["session"] != newest]
            test = [r for r in kept if r["session"] == newest]
            cand = GazeModel(tuple(screen))
            cand.fit(np.array([r["X"] for r in train]),
                     np.array([r["Y"] for r in train]))
            cand_err = float(np.mean([c["err"]
                                      for c in _cluster_err(cand, test)]))
            try:
                deployed = GazeModel.load(model_out)
                depl_err = float(np.mean([c["err"] for c in
                                          _cluster_err(deployed, test)]))
            except Exception:
                depl_err = float("inf")
            print(f"\nupdate gate on newest session ({newest}): "
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
