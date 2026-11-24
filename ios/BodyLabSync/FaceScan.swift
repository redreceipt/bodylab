import ARKit
import SwiftUI

/// Face scan with the TrueDepth (Face ID) camera: look straight at the phone, then slowly turn your head left and right;
/// a row of dots shows which turns are captured and where your head is now. At each 10° of turn it keeps one frame with ARKit's 3D face mesh (1,220 points, metres) and where each point lands in
/// that frame; BodyLab shapes your avatar's face from the mesh and takes its face photo from the straightest frame.
struct FaceScanView: View {
    let done: (Data) -> Void
    @Environment(\.dismiss) private var dismiss
    @StateObject private var scan = FaceScan()

    var body: some View {
        ZStack(alignment: .bottom) {
            FaceScanCamera(scan: scan).ignoresSafeArea()
            VStack(spacing: 10) {
                Text(scan.message).font(.headline).multilineTextAlignment(.center)
                AngleGuide(captured: scan.bins, yaw: scan.yaw)
                Button("Cancel", role: .cancel) { dismiss() }
            }
            .padding().background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14)).padding()
        }
        .onChange(of: scan.result) { _, body in
            if let body { done(body); dismiss() }
        }
    }
}

/// The turns to capture, your left to your right as in the mirror: filled once captured; the ring is your head now.
struct AngleGuide: View {
    let captured: Set<Int>, yaw: Float

    var body: some View {
        HStack(spacing: 8) {
            Text("left").font(.caption).foregroundStyle(.secondary)
            ForEach(FaceScan.angles, id: \.self) { a in
                Circle().fill(captured.contains(a) ? Color.green : Color.white.opacity(0.25))
                    .frame(width: a == 0 ? 22 : 18, height: a == 0 ? 22 : 18)
                    .overlay(Circle().stroke(Color.white, lineWidth: abs(Float(a) - yaw) < 5 ? 3 : 0))
            }
            Text("right").font(.caption).foregroundStyle(.secondary)
        }
        .accessibilityElement()
        .accessibilityLabel("\(captured.count) of \(FaceScan.angles.count) angles captured")
    }
}

final class FaceScan: NSObject, ObservableObject, ARSessionDelegate {
    static let angles = Array(stride(from: -30, through: 30, by: 10))  // degrees of head turn to capture (−: your left)
    @Published var message = "Look straight at the phone, face neutral, mouth closed"
    @Published var bins: Set<Int> = []
    @Published var yaw: Float = 0
    @Published var result: Data?
    private var frames: [[String: Any]] = []
    private var meshes: [[simd_float3]] = []
    private var triangles: [Int] = []
    private let context = CIContext()
    private var busy = false

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard result == nil, !busy, let face = frame.anchors.compactMap({ $0 as? ARFaceAnchor }).first, face.isTracked else { return }
        // The face's turn and nod: its forward axis seen from the camera, whose x axis runs down the phone's long side
        // and y toward its right edge (ARKit's camera space is the sensor's, landscape), so held upright: turn is y, nod x.
        let rel = simd_mul(frame.camera.transform.inverse, face.transform), z = rel.columns.2
        let yaw = atan2(z.y, z.z) * 180 / .pi, pitch = atan2(z.x, z.z) * 180 / .pi
        self.yaw = yaw
        if abs(pitch) > 15 { message = "Keep your chin level"; return }
        message = hint(yaw)
        let bin = Self.angles.min(by: { abs(Float($0) - yaw) < abs(Float($1) - yaw) })!
        guard abs(Float(bin) - yaw) < 3, !bins.contains(bin), bins.contains(0) || bin == 0 else { return }
        busy = true
        let size = frame.camera.imageResolution
        let px = face.geometry.vertices.map { v -> [Double] in
            let p = frame.camera.projectPoint(simd_make_float3(simd_mul(face.transform, simd_float4(v, 1))), orientation: .landscapeRight, viewportSize: size)
            return [Double(p.x), Double(p.y)]
        }
        let mesh = face.geometry.vertices
        if triangles.isEmpty { triangles = face.geometry.triangleIndices.map { Int($0) } }
        let image = CIImage(cvPixelBuffer: frame.capturedImage)
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            let jpeg = context.jpegRepresentation(of: image, colorSpace: CGColorSpaceCreateDeviceRGB(), options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.9])
            DispatchQueue.main.async { [self] in
                busy = false
                guard let jpeg else { return }
                frames.append(["yaw": Double(yaw), "pitch": Double(pitch), "jpg": jpeg.base64EncodedString(), "px": px, "w": Double(size.width), "h": Double(size.height)])
                meshes.append(mesh)
                bins.insert(bin)
                message = hint(yaw)
                if bins.count == Self.angles.count { finish() }
            }
        }
    }

    /// What to do next: look straight first, then turn toward the nearest angle still missing.
    private func hint(_ yaw: Float) -> String {
        guard bins.contains(0) else { return "Look straight at the phone, face neutral, mouth closed" }
        let missing = Self.angles.filter { !bins.contains($0) }
        guard let next = missing.min(by: { abs(Float($0) - yaw) < abs(Float($1) - yaw) }) else { return "Got it — sending to BodyLab…" }
        let way = next < 0 ? "left" : "right"
        return Float(next) * yaw > 0 && abs(yaw) > 5 ? "A little further to your \(way)" : "Slowly turn your head to your \(way)"
    }

    private func finish() {
        // The face's shape: the mean of every frame's mesh (each is ARKit's fit to your face at that moment).
        let n = Float(meshes.count)
        let mean = (0..<meshes[0].count).map { i in meshes.reduce(simd_float3()) { $0 + $1[i] } / n }
        let body: [String: Any] = ["vertices": mean.map { [Double($0.x), Double($0.y), Double($0.z)] }, "triangles": triangles, "frames": frames]
        result = try? JSONSerialization.data(withJSONObject: body)
    }
}

struct FaceScanCamera: UIViewRepresentable {
    let scan: FaceScan

    func makeUIView(context: Context) -> ARSCNView {
        let view = ARSCNView()
        view.session.delegate = scan
        if ARFaceTrackingConfiguration.isSupported {
            view.session.run(ARFaceTrackingConfiguration())
        } else {
            scan.message = "This iPhone has no Face ID camera"
        }
        return view
    }

    func updateUIView(_ view: ARSCNView, context: Context) {}

    static func dismantleUIView(_ view: ARSCNView, coordinator: ()) { view.session.pause() }
}
