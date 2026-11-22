import SwiftUI

@main
struct BodyLabSyncApp: App {
    @StateObject private var health = HealthSync()
    @Environment(\.scenePhase) private var phase

    var body: some Scene {
        WindowGroup {
            ContentView().environmentObject(health)
                .onOpenURL { url in Task { await health.pair(url) } }
        }
        .onChange(of: phase) { _, now in
            if now == .active { Task { await health.sync() } }
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var health: HealthSync
    @AppStorage("server") private var server = ""
    @AppStorage("token") private var token = ""
    @State private var camera = false

    var body: some View {
        NavigationStack {
            Form {
                if token.isEmpty {
                    Section("1 · Pair") {
                        Text("Open BodyLab on your Mac → Profile and scan the QR code with the iPhone Camera.")
                    }
                } else {
                    Section("Paired") { Text(URL(string: server)?.host() ?? server).foregroundStyle(.secondary) }
                }
                Section {
                    Button("Connect Apple Health") { Task { await health.connect() } }
                    Button("Take face photo") { camera = true }
                    Button("Sync now") { Task { await health.sync() } }
                } header: {
                    Text(token.isEmpty ? "2 · Then" : "Health")
                } footer: {
                    Text("Health fills in your height, sex and age, then sends 90 days of body, nutrition, activity, heart and sleep data plus workouts — whenever you open the app or Health gets new data. The face photo (straight on, neutral, mouth closed) makes your 3D body look like you; it stays on your BodyLab server.")
                }
                Section("Status") { Text(health.status) }
                DisclosureGroup("Pair manually") {
                    TextField("https://your-mac.your-tailnet.ts.net:8443", text: $server)
                        .keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                    SecureField("Sync token", text: $token)
                }
            }
            .navigationTitle("BodyLab Sync")
            .sheet(isPresented: $camera) {
                CameraPicker { image in Task { await health.uploadFace(image) } }.ignoresSafeArea()
            }
        }
    }
}

/// Front camera (photo library in the simulator, which has no camera).
struct CameraPicker: UIViewControllerRepresentable {
    let done: (UIImage) -> Void
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        if UIImagePickerController.isSourceTypeAvailable(.camera) {
            picker.sourceType = .camera
            picker.cameraDevice = .front
        }
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ controller: UIImagePickerController, context: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraPicker
        init(_ parent: CameraPicker) { self.parent = parent }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let image = info[.originalImage] as? UIImage { parent.done(image) }
            parent.dismiss()
        }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) { parent.dismiss() }
    }
}
