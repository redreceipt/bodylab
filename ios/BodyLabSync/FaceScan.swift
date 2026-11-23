import ARKit
import SwiftUI

/// Face scan with the TrueDepth (Face ID) camera: look straight at the phone, then slowly turn your head left and right.
/// At each 10° of turn it keeps one frame with ARKit's 3D face mesh (1,220 points, metres) and where each point lands in
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
                ProgressView(value: Double(scan.bins.count), total: Double(FaceScan.angles.count))
                Button("Cancel", role: .cancel) { dismiss() }
            }
            .padding().background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14)).padding()
        }
        .onChange(of: scan.result) { _, body in
            if let body { done(body); dismiss() }
        }
    }
}

final class FaceScan: NSObject, ObservableObject, ARSessionDelegate {
    static let angles = Array(stride(from: -30, through: 30, by: 10))  // degrees of head turn to capture
    @Published var message = "Look straight at the phone, face neutral, mouth closed"
    @Published var bins: Set<Int> = []
    @Published var result: Data?
    private var frames: [[String: Any]] = []
    private var meshes: [[simd_float3]] = []
    private var triangles: [Int] = []
    private let context = CIContext()
    private var busy = false

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard result == nil, !busy, let face = frame.anchors.compactMap({ $0 as? ARFaceAnchor }).first, face.isTracked else { return }
        // The face's turn: its forward axis seen from the camera.
        let rel = simd_mul(frame.camera.transform.inverse, face.transform)
        let yaw = atan2(rel.columns.2.x, rel.columns.2.z) * 180 / .pi
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
                frames.append(["yaw": Double(yaw), "jpg": jpeg.base64EncodedString(), "px": px, "w": Double(size.width), "h": Double(size.height)])
                meshes.append(mesh)
                bins.insert(bin)
                message = bins.count < Self.angles.count ? "Now slowly turn your head left and right" : "Got it — sending to BodyLab…"
                if bins.count == Self.angles.count { finish() }
            }
        }
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
