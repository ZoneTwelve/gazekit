// GazeTeacher — streams ARKit TrueDepth gaze data to gazekit on your Mac.
//
// Build: Xcode > New Project > iOS App (SwiftUI), name "GazeTeacher",
// replace ContentView.swift/App file with these two files, add
// NSCameraUsageDescription ("gaze teacher streaming") to Info, set your
// team for signing, run on an iPhone with Face ID (TrueDepth).
//
// Use: put the phone near your Mac screen facing you, enter the Mac's IP
// (shown by `python -m gazekit arkit`), tap Start.

import SwiftUI

@main
struct GazeTeacherApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @StateObject private var tracker = FaceStreamer()
    @AppStorage("host") private var host = "192.168.1.10"

    var body: some View {
        VStack(spacing: 16) {
            Text("GazeTeacher").font(.title2).bold()
            TextField("Mac IP", text: $host)
                .textFieldStyle(.roundedBorder)
                .keyboardType(.decimalPad)
            Button(tracker.running ? "Stop" : "Start") {
                tracker.running ? tracker.stop()
                                : tracker.start(host: host)
            }
            .buttonStyle(.borderedProminent)
            Text(tracker.status).font(.footnote).monospaced()
            Text("\(tracker.framesSent) frames sent")
                .font(.footnote).foregroundColor(.secondary)
        }
        .padding()
    }
}
