# GazeTeacher

iPhone TrueDepth gaze teacher for [gazekit](https://github.com/ZoneTwelve/gazekit).
Streams ARKit `lookAtPoint`, eye/face transforms and blink blendshapes over
UDP JSON (~60 Hz) to the gazekit receiver on your Mac — real gaze labels
for distillation.

## Build & run (once)

1. Open `GazeTeacher.xcodeproj` in Xcode (16+).
2. Signing & Capabilities → select your team (automatic signing).
3. Run on an iPhone with Face ID (TrueDepth required).

## Use

```sh
# on the Mac — prints the IP to enter in the app, records the stream
python -m gazekit arkit
```

Put the phone near your Mac screen facing you, enter the IP, tap **Start**.
Leave it streaming while you run any gazekit collection (calibrate /
collect / ambient). Then:

```sh
python -m gazekit arkit --fit   # pair + fit the ARKit->screen mapping
```

Packet format: `{"t", "look":[x,y,z], "face":[16], "leye":[16],
"reye":[16], "blinkL", "blinkR"}` — column-major 4×4 transforms, face
coordinate space for `look`.
