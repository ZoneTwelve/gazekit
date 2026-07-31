"""`gazekit auto` — one command that runs the whole pipeline.

Inspects current state (blink profile? which scenario tags exist in the
dataset?), builds a plan of only the missing steps, guides you through each
with voice + on-screen prompts, then trains/validates/evaluates via
`iterate`, optionally trains the CNN, and finishes in ambient mode so the
model keeps adapting to your environment.

Every step is logged to data/auto_log.jsonl:
    {"run": ..., "t": ..., "elapsed_s": ..., "step": ..., "status":
     "start|done|aborted|failed|skipped", ...detail}
"""

import json
import time
import traceback
from pathlib import Path

from .ui import say

CNN_MIN_SAMPLES = 1500


class AutoLog:
    def __init__(self, path="data/auto_log.jsonl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(path, "a")
        self._t0 = time.time()
        self.run_id = time.strftime("%Y%m%d_%H%M%S")

    def event(self, step, status, **detail):
        rec = {"run": self.run_id, "t": time.strftime("%Y-%m-%d %H:%M:%S"),
               "elapsed_s": round(time.time() - self._t0, 1),
               "step": step, "status": status}
        rec.update(detail)
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()
        extra = f"  {detail}" if detail else ""
        print(f"[auto {time.strftime('%H:%M:%S')}] {step}: {status}{extra}")

    def close(self):
        self._f.close()


def dataset_state(root="data/dataset"):
    tags, n, days = set(), 0, set()
    for jl in Path(root).glob("session_*/samples.jsonl"):
        days.add(jl.parent.name.split("_")[1][:8])
        for line in open(jl):
            rec = json.loads(line)
            if rec.get("meta"):
                continue
            tags.add(rec.get("tag"))
            n += 1
    return tags, n, days


def build_plan(full=False):
    tags, n_samples, days = dataset_state()
    have_blinks = Path("data/blink_profile.json").exists()
    plan = []
    if full or not have_blinks:
        plan.append(("blinks", "blink calibration (~25 s, voice-guided)"))
    today = time.strftime("%Y%m%d")
    if days and today not in days and not full:
        # first run of a new day: the 2-min daily probe (new lighting/day
        # coverage) replaces a full recalibration
        plan.append(("daily", "daily probe (~2 min, new day detected)"))
        plan.append(("iterate", "clean + train + validate + evaluate + update"))
        return plan, n_samples
    plan.append(("calibrate", "gaze calibration (~2 min)"))
    for tag, desc in (("vor", "head-movement training (~1 min)"),
                      ("posture", "3-posture grid (~2 min)"),
                      ("edges", "screen edges & corners (~40 s)")):
        if full or tag not in tags:
            plan.append((tag, desc))
    if full:
        plan.append(("pursuit", "smooth-pursuit sweep (~45 s, CNN data)"))
    plan.append(("iterate", "clean + train + validate + evaluate + update"))
    return plan, n_samples


def run(camera_index=0, full=False, cnn="auto", ambient_after=True):
    log = AutoLog()
    plan, n_samples = build_plan(full)
    log.event("plan", "start", steps=[s for s, _ in plan],
              existing_samples=n_samples, full=full)
    print("\nplan:")
    for s, desc in plan:
        print(f"  - {s:10s} {desc}")
    print()
    say("Starting the training pipeline.")

    def step(name, fn):
        log.event(name, "start")
        t0 = time.monotonic()
        try:
            out = fn()
            dur = round(time.monotonic() - t0, 1)
            if out is False or out is None and name in ("calibrate",):
                log.event(name, "aborted", duration_s=dur)
                return "aborted", None
            log.event(name, "done", duration_s=dur)
            return "done", out
        except KeyboardInterrupt:
            log.event(name, "aborted", duration_s=round(
                time.monotonic() - t0, 1))
            raise
        except SystemExit as e:
            log.event(name, "failed", error=str(e))
            return "failed", None
        except Exception as e:
            log.event(name, "failed", error=repr(e),
                      trace=traceback.format_exc()[-1500:])
            print(traceback.format_exc())
            return "failed", None

    report = None
    try:
        for name, desc in plan:
            say(f"Next: {desc.split('(')[0].strip()}")
            time.sleep(1.2)
            if name == "blinks":
                from .collect import run as collect_run
                status, _ = step(name, lambda: collect_run(
                    "blinks", camera_index=camera_index))
            elif name == "calibrate":
                from .calibrate import run as calibrate_run
                status, rep = step(name, lambda: calibrate_run(
                    camera_index=camera_index))
                if status == "done" and rep:
                    log.event(name, "result",
                              verdict=rep.get("verdict"),
                              mean_error_px=rep.get("mean_error_px"))
            elif name in ("vor", "posture", "edges", "pursuit", "daily"):
                from .collect import run as collect_run
                status, _ = step(name, lambda n=name: collect_run(
                    n, camera_index=camera_index))
            elif name == "iterate":
                from .evaluate import run as iterate_run
                status, report = step(name, iterate_run)
                if status == "done" and report:
                    log.event(name, "result",
                              loso_px=report.get("loso_px"),
                              loto_px=report.get("loto_px"),
                              samples=report.get("samples"))
            if status == "aborted":
                say("Stopping. Run gazekit auto again to resume.")
                print("\naborted — run `gazekit auto` again to resume; "
                      "finished steps are detected and skipped")
                return

        # CNN: explicit yes, or auto once there is enough data
        _, n_samples, _ = dataset_state()
        want_cnn = cnn == "yes" or (cnn == "auto"
                                    and n_samples >= CNN_MIN_SAMPLES)
        if want_cnn:
            say("Training the neural network. This takes a few minutes.")
            from .cnn import train as cnn_train
            step("train-cnn", lambda: cnn_train())
        else:
            log.event("train-cnn", "skipped",
                      reason=f"samples={n_samples} < {CNN_MIN_SAMPLES}"
                      if cnn == "auto" else "disabled")

        log.event("pipeline", "done")
        say("Pipeline complete.")
        if report and report.get("loso_px"):
            print(f"\npipeline complete — cross-session error "
                  f"{report['loso_px']}px on {report['samples']} samples")

        if ambient_after:
            say("Ambient mode is now watching. Dots will appear while "
                "you work.")
            log.event("ambient", "start")
            from .ambient import run as ambient_run
            try:
                ambient_run(camera_index=camera_index)
            finally:
                log.event("ambient", "done")
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        log.close()
