// GazeTeacher network core — the PHONE is the active side (see
// docs/PHONE_PROTOCOL.md in the gazekit repo):
//   - auto-arms on launch: control TCP (:5578) connects and retries every
//     2 s forever; the user never has to babysit the app
//   - Mac remote-controls via length-prefixed JSON on that socket:
//     session_start / session_stop / stream_on / stream_off / ping
//   - gaze goes UDP :5577 fire-and-forget; frames go on the TCP channel
//     at <=15 fps while streaming is enabled

import ARKit
import CoreImage
import Network
import UIKit
import simd

final class FaceStreamer: NSObject, ObservableObject, ARSessionDelegate {
    @Published var running = false          // ARKit session active
    @Published var linkState = "arming…"    // control-channel state
    @Published var framesSent = 0
    @Published var gazeSent = 0
    @Published var faceTracked = false
    @Published var look = SIMD3<Float>(0, 0, 0)
    @Published var preview: UIImage?

    private let session = ARSession()
    private var gazeConn: NWConnection?
    private var ctrl: NWConnection?
    private var host = "192.168.3.59"
    private var armed = false
    private var framesEnabled = true
    private var lastFrameAt: TimeInterval = 0
    private var lastPreviewAt: TimeInterval = 0
    private let ciContext = CIContext()
    private let net = DispatchQueue(label: "net", qos: .userInitiated)

    // MARK: arming / reconnect (phone is the active side)

    func arm(host: String) {
        self.host = host
        guard !armed else { return }
        armed = true
        gazeConn = NWConnection(host: .init(host), port: 5577, using: .udp)
        gazeConn?.start(queue: net)
        connectControl()
    }

    func rearm(host: String) {   // called when the user edits the IP
        armed = false
        ctrl?.cancel(); ctrl = nil
        gazeConn?.cancel(); gazeConn = nil
        arm(host: host)
    }

    private func connectControl() {
        guard armed else { return }
        let c = NWConnection(host: .init(host), port: 5578, using: .tcp)
        ctrl = c
        c.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                DispatchQueue.main.async { self.linkState = "connected" }
                self.receiveLoop(c)
            case .failed, .cancelled:
                DispatchQueue.main.async {
                    self.linkState = "retrying… (\(self.host))"
                }
                self.net.asyncAfter(deadline: .now() + 2) {
                    if self.ctrl === c || self.ctrl == nil {
                        self.connectControl()
                    }
                }
            case .waiting:
                DispatchQueue.main.async { self.linkState = "waiting… " }
            default: break
            }
        }
        c.start(queue: net)
    }

    // MARK: control messages from the Mac

    private func receiveLoop(_ c: NWConnection) {
        c.receive(minimumIncompleteLength: 4, maximumLength: 4) {
            [weak self] head, _, _, err in
            guard let self, err == nil, let head, head.count == 4 else {
                c.cancel(); return
            }
            let n = Int(UInt32(bigEndian: head.withUnsafeBytes {
                $0.load(as: UInt32.self) }))
            guard n > 0, n < 1_000_000 else { c.cancel(); return }
            c.receive(minimumIncompleteLength: n, maximumLength: n) {
                body, _, _, err2 in
                guard err2 == nil, let body,
                      let msg = try? JSONSerialization
                          .jsonObject(with: body) as? [String: Any]
                else { c.cancel(); return }
                self.handle(cmd: msg["cmd"] as? String ?? "")
                self.receiveLoop(c)
            }
        }
    }

    private func handle(cmd: String) {
        DispatchQueue.main.async {
            switch cmd {
            case "session_start": self.startSession()
            case "session_stop": self.stopSession()
            case "stream_on":
                self.framesEnabled = true
                if self.running { self.linkState = "connected" }
            case "stream_off":
                self.framesEnabled = false
                self.linkState = "paused by server"
            default: break
            }
        }
    }

    // MARK: ARKit session

    func startSession() {
        guard !running,
              ARFaceTrackingConfiguration.isSupported else { return }
        let cfg = ARFaceTrackingConfiguration()
        cfg.isLightEstimationEnabled = false
        session.delegate = self
        session.run(cfg, options: [.resetTracking])
        UIApplication.shared.isIdleTimerDisabled = true
        running = true
    }

    func stopSession() {
        guard running else { return }
        session.pause()
        UIApplication.shared.isIdleTimerDisabled = false
        running = false
        faceTracked = false
    }

    // MARK: outgoing data

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        guard let face = anchors.compactMap({ $0 as? ARFaceAnchor }).first
        else { return }
        DispatchQueue.main.async {
            self.faceTracked = face.isTracked
            self.look = face.lookAtPoint
        }
        guard face.isTracked else { return }
        let m: (simd_float4x4) -> [Float] = { t in
            (0..<4).flatMap { c in (0..<4).map { r in t[c][r] } }
        }
        let pkt: [String: Any] = [
            "t": Date().timeIntervalSince1970,
            "look": [face.lookAtPoint.x, face.lookAtPoint.y,
                     face.lookAtPoint.z],
            "face": m(face.transform),
            "leye": m(face.leftEyeTransform),
            "reye": m(face.rightEyeTransform),
            "blinkL": face.blendShapes[.eyeBlinkLeft]?.floatValue ?? 0,
            "blinkR": face.blendShapes[.eyeBlinkRight]?.floatValue ?? 0,
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: pkt)
        else { return }
        gazeConn?.send(content: data, completion: .contentProcessed { _ in })
        DispatchQueue.main.async { self.gazeSent += 1 }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let now = Date().timeIntervalSince1970
        guard now - lastFrameAt > 1.0 / 15.0 else { return }
        lastFrameAt = now
        let buffer = frame.capturedImage
        let sendable = framesEnabled && ctrl?.state == .ready
        let wantPreview = now - lastPreviewAt > 1.0 / 10.0
        guard sendable || wantPreview else { return }
        net.async { [weak self] in
            guard let self else { return }
            var image = CIImage(cvPixelBuffer: buffer)
            let scale = 640.0 / image.extent.width
            image = image.transformed(by: .init(scaleX: scale, y: scale))
            guard let jpg = self.ciContext.jpegRepresentation(
                of: image, colorSpace: CGColorSpaceCreateDeviceRGB(),
                options: [kCGImageDestinationLossyCompressionQuality
                          as CIImageRepresentationOption: 0.55])
            else { return }
            if wantPreview {
                self.lastPreviewAt = now
                let upright = image.oriented(.right)
                if let pjpg = self.ciContext.jpegRepresentation(
                    of: upright, colorSpace: CGColorSpaceCreateDeviceRGB(),
                    options: [kCGImageDestinationLossyCompressionQuality
                              as CIImageRepresentationOption: 0.55]) {
                    let ui = UIImage(data: pjpg)
                    DispatchQueue.main.async { self.preview = ui }
                }
            }
            guard sendable else { return }
            let msg: [String: Any] = ["type": "frame", "t": now,
                                      "jpg": jpg.base64EncodedString()]
            guard let body = try? JSONSerialization.data(withJSONObject: msg)
            else { return }
            var len = UInt32(body.count).bigEndian
            var out = Data(bytes: &len, count: 4)
            out.append(body)
            self.ctrl?.send(content: out,
                            completion: .contentProcessed { _ in })
            DispatchQueue.main.async { self.framesSent += 1 }
        }
    }
}
