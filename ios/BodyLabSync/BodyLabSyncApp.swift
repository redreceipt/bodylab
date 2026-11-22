import SwiftUI
import WebKit

@main
struct BodyLabSyncApp: App {
    @StateObject private var health = HealthSync()
    @Environment(\.scenePhase) private var phase

    var body: some Scene {
        WindowGroup {
            RootView().environmentObject(health)
                .preferredColorScheme(.dark)
                .tint(Color(red: 0.24, green: 0.94, blue: 1))  // the web app's accent
                .onOpenURL { url in Task { await health.pair(url) } }
        }
        .onChange(of: phase) { _, now in
            if now == .active { Task { await health.sync() } }
        }
    }
}

struct RootView: View {
    @AppStorage("token") private var token = ""

    var body: some View {
        if token.isEmpty {
            PairView()
        } else {
            TabView {
                BodyView().tabItem { Label("Body", systemImage: "figure.stand") }
                LogView().tabItem { Label("Log", systemImage: "plus.circle") }
                MeView().tabItem { Label("Me", systemImage: "person.crop.circle") }
            }
        }
    }
}

struct PairView: View {
    @EnvironmentObject var health: HealthSync
    @AppStorage("server") private var server = ""
    @AppStorage("token") private var token = ""
    @State private var manualToken = ""

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Text("Open BodyLab on your Mac → Settings and scan the QR code with the iPhone Camera.")
                }
                DisclosureGroup("Pair manually") {
                    TextField("https://your-mac.your-tailnet.ts.net:8443", text: $server)
                        .keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                    SecureField("Sync token", text: $manualToken)
                    Button("Pair") { token = manualToken; Task { await health.connect() } }.disabled(manualToken.isEmpty || server.isEmpty)
                }
                if !health.status.isEmpty { Text(health.status).foregroundStyle(.secondary) }
            }
            .navigationTitle("BodyLab")
        }
    }
}

/// What BodyLab has today and what's still missing.
/// The BodyLab web app itself (same UI everywhere), over your tailnet.
struct BodyView: View {
    @AppStorage("server") private var server = ""

    var body: some View {
        WebView(url: URL(string: server)?.appending(queryItems: [.init(name: "app", value: "1")])).ignoresSafeArea(edges: .top).background(Color.black)
    }
}

struct WebView: UIViewRepresentable {
    let url: URL?

    func makeUIView(context: Context) -> WKWebView {
        let view = WKWebView()
        view.isOpaque = false
        view.backgroundColor = .black
        view.scrollView.backgroundColor = .black
        view.scrollView.bounces = false
        if let url { view.load(URLRequest(url: url)) }
        return view
    }

    func updateUIView(_ view: WKWebView, context: Context) {}
}

/// Food, weight and workouts, saved to Apple Health (BodyLab reads them back).
struct LogView: View {
    @EnvironmentObject var health: HealthSync
    @State private var kcal = ""
    @State private var protein = ""
    @State private var weight = ""
    @State private var workoutType = "Strength"
    @State private var minutes = "45"
    @State private var effort = 7.0
    @State private var focus: Set<String> = []
    private let types = ["Strength", "Cardio", "HIIT", "Sports", "Mobility"]
    private let areas = ["full body", "chest", "back", "shoulders", "arms", "core", "legs", "glutes", "cardio"]

    var body: some View {
        NavigationStack {
            Form {
                Section("Food (adds to Apple Health)") {
                    TextField("Calories (kcal)", text: $kcal).keyboardType(.numberPad)
                    TextField("Protein (g)", text: $protein).keyboardType(.decimalPad)
                    Button("Add food") {
                        Task { await health.logFood(kcal: Double(kcal), protein: Double(protein)); kcal = ""; protein = "" }
                    }.disabled(Double(kcal) == nil && Double(protein) == nil)
                }
                Section("Weight") {
                    TextField(health.metric ? "Weight (kg)" : "Weight (lb)", text: $weight).keyboardType(.decimalPad)
                    Button("Save weight") { Task { await health.logWeight(Double(weight)!); weight = "" } }.disabled(Double(weight) == nil)
                }
                Section("Workout") {
                    Picker("Type", selection: $workoutType) { ForEach(types, id: \.self) { Text($0) } }
                    TextField("Minutes", text: $minutes).keyboardType(.numberPad)
                    VStack(alignment: .leading) { Text("Effort \(Int(effort))/10"); Slider(value: $effort, in: 1...10, step: 1) }
                    ForEach(areas, id: \.self) { area in
                        Toggle(area.capitalized, isOn: Binding(get: { focus.contains(area) }, set: { on in if on { focus.insert(area) } else { focus.remove(area) } }))
                    }
                    Button("Save workout") {
                        Task { await health.logWorkout(type: workoutType, minutes: Double(minutes) ?? 0, effort: effort, focus: Array(focus)); focus = [] }
                    }.disabled((Double(minutes) ?? 0) <= 0)
                }
                Section { Text(health.status).font(.footnote).foregroundStyle(.secondary) }
            }
            .navigationTitle("Log")
        }
    }
}

struct MeView: View {
    @EnvironmentObject var health: HealthSync
    @AppStorage("server") private var server = ""
    @AppStorage("token") private var token = ""
    @State private var camera = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Button("Connect Apple Health") { Task { await health.connect() } }
                    Button("Take face photo") { camera = true }
                } footer: {
                    Text("Height, sex, birth date and units come from Apple Health and this phone. The face photo (straight on, neutral, mouth closed) makes your 3D body look like you; it stays on your BodyLab server.")
                }
                Section("Paired with") {
                    Text(URL(string: server)?.host() ?? server).foregroundStyle(.secondary)
                    Button("Unpair", role: .destructive) { token = "" }
                }
            }
            .navigationTitle("Me")
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
