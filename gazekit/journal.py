"""Unified run journal: one structured line per CLI invocation.

data/journal.jsonl is the single place to answer "what was run, how many
times, with what results, and what should improve next" — written for both
the human and the assistant who picks this project up later. Command
modules stay journal-unaware; __main__ wraps every dispatch.
"""

import json
import time
from pathlib import Path

PATH = Path("data/journal.jsonl")


def log_run(cmd: str, argv: list, status: str, duration_s: float,
            result: dict | None = None):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    n_prev = 0
    if PATH.exists():
        n_prev = sum(1 for l in open(PATH) if json.loads(l)["cmd"] == cmd)
    entry = {
        "t": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cmd": cmd,
        "run_no": n_prev + 1,          # how many times this command has run
        "argv": argv,
        "status": status,               # done | aborted | failed
        "duration_s": round(duration_s, 1),
    }
    if result:
        entry["result"] = result
    with open(PATH, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def summary(last: int = 15) -> str:
    """Human/assistant-readable tail of the journal."""
    if not PATH.exists():
        return "journal is empty — no runs recorded yet"
    lines = [json.loads(l) for l in open(PATH)][-last:]
    out = []
    for e in lines:
        res = e.get("result") or {}
        keys = ("verdict", "mean_error_px", "loso_px", "loso_aligned_px",
                "samples", "recommendations")
        brief = ", ".join(f"{k}={res[k]}" for k in keys if k in res)
        out.append(f"{e['t']}  {e['cmd']}#{e['run_no']} "
                   f"[{e['status']} {e['duration_s']}s]  {brief}")
    return "\n".join(out)
