# GazeTeacher (in-repo Xcode project)

The iPhone as gazekit's **camera AND gaze teacher** at the same time.
Streams ARKit gaze (UDP :5577) + camera frames (TCP :5578), with an
on-device preview (ARKit owns the camera exclusively — this preview is
the only way to see what it sees).

Build: `open ios/GazeTeacher.xcodeproj` → Run on a Face ID iPhone
(team pre-filled; first run: confirm device registration, trust the
developer on the phone).

## The one-device workflow (Mac mini + iPhone)

```sh
# phone: open GazeTeacher, IP = your Mac's LAN address, tap Start
python -m gazekit arkit --monitor     # optional: see what the phone sees
python -m gazekit calibrate --camera phone   # normal calibration, phone as camera
python -m gazekit arkit --fit         # fit the ARKit->screen teacher mapping
```

`--camera phone` works for every command (calibrate / collect / live /
ambient / verify / auto). While it runs, the app's gaze stream is recorded
in-process with the same clock — every collection doubles as
teacher-pairing data. `arkit --calib` remains as the camera-free fallback.
