// ARKit face tracking -> Mac, two streams + on-device preview.
//   UDP :5577  gaze packets every ARKit update (~60 Hz), JSON:
//     {"t", "look":[x,y,z], "face":[16], "leye":[16], "reye":[16],
//      "blinkL", "blinkR"}
//   TCP :5578  camera frames (~15 Hz), 4-byte big-endian length prefix +
//     JSON {"type":"frame", "t": <unix>, "jpg": "<base64 JPEG>"}
//
// One iPhone therefore serves as BOTH the Mac's camera and the gaze
// teacher — gazekit consumes the frames via `--camera phone` and pairs
// the gaze stream by receive time. The in-app preview works even with no
// Mac connected (ARKit exclusively owns the camera, so this preview is
// the only way to see what it sees).

import ARKit
import CoreImage
import Network
import UIKit
import simd

final class FaceStreamer: NSObject, ObservableObject, ARSessionDelegate {
    @Published var running = false
    @Published var status = "idle"
    @Published var framesSent = 0
    @Published var gazeSent = 0
    @Published var faceTracked = false
    @Published var look = SIMD3<Float>(0, 0, 0)
    @Published var preview: UIImage?

    private let session = ARSession()
    private var gazeConn: NWConnection?
    private var frameConn: NWConnection?
    private var lastFrameAt: TimeInterval = 0
    private var lastPreviewAt: TimeInterval = 0
    private let ciContext = CIContext()
    private let jpegQueue = DispatchQueue(label: "jpeg", qos: .userInitiated)

    func start(host: String) {
        guard ARFaceTrackingConfiguration.isSupported else {
            status = "no TrueDepth on this device"; return
        }
        gazeConn = NWConnection(host: .init(host), port: 5577, using: .udp)
        gazeConn?.start(queue: .global(qos: .userInitiated))
        frameConn = NWConnection(host: .init(host), port: 5578, using: .tcp)
        frameConn?.start(queue: jpegQueue)
        let cfg = ARFaceTrackingConfiguration()
        cfg.isLightEstimationEnabled = false
        session.delegate = self
        session.run(cfg, options: [.resetTracking])
        UIApplication.shared.isIdleTimerDisabled = true
        running = true
        status = "gaze→udp:5577  frames→tcp:5578  @\(host)"
    }

    func stop() {
        session.pause()
        gazeConn?.cancel(); gazeConn = nil
        frameConn?.cancel(); frameConn = nil
        UIApplication.shared.isIdleTimerDisabled = false
        running = false
        faceTracked = false
        status = "stopped"
    }

    // gaze packets: every anchor update
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

    // camera frames: JPEG once, used for both preview and network
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let now = Date().timeIntervalSince1970
        guard now - lastFrameAt > 1.0 / 15.0 else { return }
        lastFrameAt = now
        let buffer = frame.capturedImage
        let sendable = frameConn?.state == .ready
        let wantPreview = now - lastPreviewAt > 1.0 / 10.0
        guard sendable || wantPreview else { return }
        jpegQueue.async { [weak self] in
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
                // sensor delivers landscape; rotate for a portrait preview
                // (network frames stay raw — the Mac auto-orients them).
                // If your preview appears upside down, use .left instead.
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
            self.frameConn?.send(content: out,
                                 completion: .contentProcessed { _ in })
            DispatchQueue.main.async { self.framesSent += 1 }
        }
    }
}
