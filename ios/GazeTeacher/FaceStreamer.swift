// ARKit face tracking -> UDP JSON lines. One packet per frame (~60 Hz):
// {"t": <unix seconds>, "look": [x,y,z],            // lookAtPoint, face space
//  "face": [16 floats],                             // faceAnchor.transform
//  "leye": [16], "reye": [16],                      // eye transforms
//  "blinkL": f, "blinkR": f}

import ARKit
import Network
import simd

final class FaceStreamer: NSObject, ObservableObject, ARSessionDelegate {
    @Published var running = false
    @Published var status = "idle"
    @Published var framesSent = 0

    private let session = ARSession()
    private var conn: NWConnection?

    func start(host: String) {
        guard ARFaceTrackingConfiguration.isSupported else {
            status = "no TrueDepth on this device"; return
        }
        conn = NWConnection(host: .init(host), port: 5577, using: .udp)
        conn?.start(queue: .global(qos: .userInitiated))
        let cfg = ARFaceTrackingConfiguration()
        cfg.isLightEstimationEnabled = false
        session.delegate = self
        session.run(cfg, options: [.resetTracking])
        running = true
        status = "streaming to \(host):5577"
    }

    func stop() {
        session.pause()
        conn?.cancel(); conn = nil
        running = false
        status = "stopped"
    }

    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        guard let face = anchors.compactMap({ $0 as? ARFaceAnchor }).first,
              face.isTracked else { return }
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
        conn?.send(content: data, completion: .contentProcessed { _ in })
        DispatchQueue.main.async { self.framesSent += 1 }
    }
}
