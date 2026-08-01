# Release checklist — run before tagging a beta

Two tiers: **automated** (`python -m gazekit selftest`, no camera needed)
and **manual** (needs you, a camera, and a screen). A beta ships only when
every automated check passes and the manual list is walked once per camera
source you support.

## Automated — `python -m gazekit selftest`

Covers: every module imports; CLI parses all subcommands; camera-source
resolution picks the right model file; feature transform shape; model
save/load round-trip; dataset write/read round-trip; blink profile loads;
deploy gate rejects a deliberately corrupt model; phone protocol
end-to-end against a simulated phone (connect → session_start → frames →
session_stop); journal writes.

## Manual — per camera source

Set the source first: `gazekit camera cam` (webcam) or `gazekit camera app`
(iPhone). Each source has its own model file — they are different domains.

| # | Check | Pass when |
|---|---|---|
| 1 | `gazekit cameras` | lists your camera |
| 2 | `gazekit doctor` | SETUP CHECK reaches all-green and exits cleanly |
| 3 | `gazekit calibrate` | finishes with a verdict; STABLE or USABLE |
| 4 | `gazekit live` | dot follows your gaze; `q` quits; blink freezes it |
| 5 | `gazekit live --backend hybrid` | starts (CNN present) and tracks |
| 6 | `gazekit verify --mode path` | track colours in; prints a summary |
| 7 | `gazekit collect vor` | runs; new session appears |
| 8 | `gazekit ambient --test` | three rings shrink onto dots |
| 9 | `gazekit iterate` | prints LOSO + advice; deploy gate reports |
| 10 | `gazekit journal` | shows the runs you just did |

Phone source only:

| # | Check | Pass when |
|---|---|---|
| P1 | app launch (no taps) | `link: connected` within ~5 s |
| P2 | `gazekit camera status` | reports connected + frame/gaze rates |
| P3 | `gazekit camera start/stop` | phone session toggles remotely |
| P4 | kill a running collection mid-way | phone returns to `retry:` then reconnects on the next run |
| P5 | `gazekit arkit --calib` | pre-flight gate passes, verdict printed |
| P6 | `gazekit arkit --fit` | teacher mapping saved |

## Beta gate

- automated selftest: all pass
- manual: webcam column complete (this is the baseline everyone has)
- `iterate` LOSO no worse than the last tagged release
- README quickstart matches reality on a clean clone
