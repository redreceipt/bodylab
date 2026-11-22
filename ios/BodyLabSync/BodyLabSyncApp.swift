import SwiftUI

@main
struct BodyLabSyncApp: App {
    @StateObject private var health = HealthSync()
    @Environment(\.scenePhase) private var phase

    var body: some Scene {
        WindowGroup {
            ContentView().environmentObject(health)
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

    var body: some View {
        NavigationStack {
            Form {
                Section("BodyLab server") {
                    TextField("https://your-mac.your-tailnet.ts.net:8443", text: $server)
                        .keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                    SecureField("Sync token (BodyLab → Profile)", text: $token)
                }
                Section {
                    Button("Connect Apple Health") { Task { await health.connect() } }
                    Button("Sync now") { Task { await health.sync() } }
                } footer: {
                    Text("Sends the last 60 days of weight, calories, protein, sleep, active energy and workouts to your BodyLab, whenever you open the app or Health gets new data.")
                }
                Section("Status") { Text(health.status) }
            }
            .navigationTitle("BodyLab Sync")
        }
    }
}
