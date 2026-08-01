// GazeTeacher — the iPhone as gazekit's camera AND gaze teacher.
// Streams ARKit gaze (UDP :5577) + camera frames (TCP :5578) to the Mac,
// with an on-device preview (ARKit owns the camera exclusively, so this
// preview is the only way to see what it sees).

import SwiftUI

@main
struct GazeTeacherApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @StateObject private var tracker = FaceStreamer()
    @AppStorage("host") private var host = "192.168.3.59"

    var body: some View {
        VStack(spacing: 14) {
            Text("GazeTeacher").font(.title2).bold()

            ZStack {
                RoundedRectangle(cornerRadius: 12)
                    .fill(Color.black.opacity(0.85))
                if let img = tracker.preview {
                    Image(uiImage: img)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                } else {
                    Text(tracker.running ? "waiting for camera…"
                                         : "preview appears after Start")
                        .foregroundColor(.gray).font(.footnote)
                }
                VStack {
                    HStack {
                        Circle()
                            .fill(tracker.faceTracked ? .green : .red)
                            .frame(width: 10, height: 10)
                        Text(tracker.faceTracked ? "tracking" : "no face")
                            .font(.caption2).foregroundColor(.white)
                        Spacer()
                    }.padding(8)
                    Spacer()
                }
            }
            .frame(height: 300)

            TextField("Mac IP", text: $host)
                .textFieldStyle(.roundedBorder)
                .keyboardType(.decimalPad)
            Button(tracker.running ? "Stop" : "Start") {
                tracker.running ? tracker.stop()
                                : tracker.start(host: host)
            }
            .buttonStyle(.borderedProminent)

            Text(tracker.status).font(.footnote).monospaced()
            HStack(spacing: 16) {
                Text("gaze \(tracker.gazeSent)")
                Text("frames \(tracker.framesSent)")
                Text(String(format: "look %.2f %.2f",
                            tracker.look.x, tracker.look.y))
            }
            .font(.caption.monospaced())
            .foregroundColor(.secondary)
        }
        .padding()
    }
}
