// GazeTeacher — the iPhone as gazekit's camera AND gaze teacher.
// The phone is the ACTIVE side: it arms itself on launch, reconnects
// forever, and the Mac remote-controls the session (see gazekit
// docs/PHONE_PROTOCOL.md). Manual Start/Stop remains as an override.

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
                if let img = tracker.preview, tracker.running {
                    Image(uiImage: img)
                        .resizable()
                        .scaledToFit()
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                } else {
                    Text(tracker.running ? "waiting for camera…"
                         : "idle — the Mac can start me remotely")
                        .foregroundColor(.gray).font(.footnote)
                }
                VStack {
                    HStack {
                        Circle()
                            .fill(tracker.running
                                  ? (tracker.faceTracked ? .green : .orange)
                                  : .gray)
                            .frame(width: 10, height: 10)
                        Text(tracker.running
                             ? (tracker.faceTracked ? "tracking" : "no face")
                             : "session off")
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
                .onSubmit { tracker.rearm(host: host) }
            Button(tracker.running ? "Stop" : "Start") {
                tracker.running ? tracker.stopSession()
                                : tracker.startSession()
            }
            .buttonStyle(.borderedProminent)

            Text("link: \(tracker.linkState)").font(.footnote).monospaced()
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
        .onAppear { tracker.arm(host: host) }
    }
}
