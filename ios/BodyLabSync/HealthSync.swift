import HealthKit
import SwiftUI

/// Reads Apple Health and pushes daily totals + workouts to BodyLab (`POST /api/sync`).
/// The server upserts by day / workout ID, so every sync can resend the whole window.
@MainActor
final class HealthSync: ObservableObject {
    @Published var status = "Not synced yet"

    private let store = HKHealthStore()
    private let windowDays = 60
    private var observing = false
    // Health type, unit, BodyLab key, how samples combine within a day
    private let quantities: [(HKQuantityTypeIdentifier, HKUnit, String, HKStatisticsOptions)] = [
        (.bodyMass, .gramUnit(with: .kilo), "weight", .discreteAverage),
        (.dietaryEnergyConsumed, .kilocalorie(), "kcal", .cumulativeSum),
        (.dietaryProtein, .gram(), "protein", .cumulativeSum),
        (.activeEnergyBurned, .kilocalorie(), "active", .cumulativeSum),
    ]

    init() {
        if !(UserDefaults.standard.string(forKey: "token") ?? "").isEmpty { observe() }
    }

    func connect() async {
        let read: Set<HKObjectType> = Set(quantities.map { HKQuantityType($0.0) } as [HKObjectType])
            .union([HKQuantityType(.heartRate), HKCategoryType(.sleepAnalysis), HKObjectType.workoutType()])
        do {
            try await store.requestAuthorization(toShare: [], read: read)
            observe()
            await sync()
        } catch {
            status = "Health access failed: \(error.localizedDescription)"
        }
    }

    /// Wake the app (even in the background) when new weight, food or workouts land in Health.
    private func observe() {
        guard !observing else { return }
        observing = true
        let types: [HKSampleType] = [HKQuantityType(.bodyMass), HKQuantityType(.dietaryEnergyConsumed), HKObjectType.workoutType()]
        for type in types {
            store.execute(HKObserverQuery(sampleType: type, predicate: nil) { [weak self] _, done, _ in
                Task { @MainActor in
                    await self?.sync()
                    done()
                }
            })
            store.enableBackgroundDelivery(for: type, frequency: .hourly) { _, _ in }
        }
    }

    func sync() async {
        let defaults = UserDefaults.standard
        guard let server = defaults.string(forKey: "server"), let token = defaults.string(forKey: "token"), !token.isEmpty,
              let url = URL(string: server)?.appending(path: "api/sync") else {
            status = "Enter the server and token first"
            return
        }
        status = "Syncing…"
        do {
            let start = Calendar.current.date(byAdding: .day, value: -windowDays, to: Calendar.current.startOfDay(for: .now))!
            let since = HKQuery.predicateForSamples(withStart: start, end: nil)
            var daily: [String: [String: Any]] = [:]
            for (id, unit, key, option) in quantities {
                let query = HKStatisticsCollectionQueryDescriptor(
                    predicate: .quantitySample(type: HKQuantityType(id), predicate: since),
                    options: option, anchorDate: start, intervalComponents: DateComponents(day: 1))
                try await query.result(for: store).enumerateStatistics(from: start, to: .now) { stats, _ in
                    if let q = option == .cumulativeSum ? stats.sumQuantity() : stats.averageQuantity() {
                        daily[Self.day(stats.startDate), default: [:]][key] = q.doubleValue(for: unit)
                    }
                }
            }
            for (day, hours) in try await sleepHours(since) { daily[day, default: [:]]["sleep"] = hours }
            let body: [String: Any] = [
                "days": daily.map { day, values in values.merging(["date": day]) { a, _ in a } },
                "workouts": try await workouts(since),
            ]

            var request = URLRequest(url: url)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
            let (data, response) = try await URLSession.shared.data(for: request)
            let reply = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            status = code == 200
                ? "Synced \(reply["saved"] ?? 0) records at \(Date.now.formatted(date: .omitted, time: .shortened))"
                : "BodyLab said \(code): \(reply["error"] ?? "unknown error")"
        } catch {
            status = "Sync failed: \(error.localizedDescription)"
        }
    }

    /// Hours asleep per night. Overlapping samples (Watch + phone) are merged so they aren't counted twice.
    private func sleepHours(_ since: NSPredicate) async throws -> [String: Double] {
        let asleep = Set(HKCategoryValueSleepAnalysis.allAsleepValues.map(\.rawValue))
        let samples = try await HKSampleQueryDescriptor(
            predicates: [.categorySample(type: HKCategoryType(.sleepAnalysis), predicate: since)],
            sortDescriptors: [SortDescriptor(\.startDate)]).result(for: store)
        var merged: [(start: Date, end: Date)] = []
        for s in samples where asleep.contains(s.value) {
            if let last = merged.last, s.startDate <= last.end {
                merged[merged.count - 1].end = max(last.end, s.endDate)
            } else {
                merged.append((s.startDate, s.endDate))
            }
        }
        // +6 h puts segments that end just before midnight on the morning they belong to.
        return merged.reduce(into: [:]) { hours, seg in
            hours[Self.day(seg.end.addingTimeInterval(6 * 3600)), default: 0] += seg.end.timeIntervalSince(seg.start) / 3600
        }
    }

    private func workouts(_ since: NSPredicate) async throws -> [[String: Any]] {
        let bpm = HKUnit.count().unitDivided(by: .minute())
        return try await HKSampleQueryDescriptor(predicates: [.workout(since)], sortDescriptors: [SortDescriptor(\.startDate)])
            .result(for: store)
            .filter { $0.duration >= 60 }
            .map { w in
                let (type, focus) = Self.kind(w.workoutActivityType)
                var out: [String: Any] = ["id": w.uuid.uuidString, "start": w.startDate.timeIntervalSince1970,
                                          "minutes": min(w.duration / 60, 600), "type": type, "focus": focus]
                if let hr = w.statistics(for: HKQuantityType(.heartRate))?.averageQuantity()?.doubleValue(for: bpm) {
                    // shortcut: effort from % of an assumed 185 bpm max HR; use age-based max HR if this proves too coarse
                    out["rpe"] = min(10, max(1, (hr / 185 - 0.5) / 0.05))
                }
                return out
            }
    }

    private static func kind(_ type: HKWorkoutActivityType) -> (String, [String]) {
        switch type {
        case .traditionalStrengthTraining, .functionalStrengthTraining, .crossTraining: ("Strength", ["full body"])
        case .coreTraining: ("Strength", ["core"])
        case .highIntensityIntervalTraining: ("HIIT", ["full body"])
        case .yoga, .pilates, .flexibility, .mindAndBody, .cooldown: ("Mobility", [])
        case .running, .walking, .cycling, .swimming, .rowing, .elliptical, .stairClimbing, .hiking, .mixedCardio,
             .jumpRope, .stepTraining: ("Cardio", ["cardio"])
        default: ("Sports", [])
        }
    }

    private static func day(_ date: Date) -> String {
        date.formatted(Date.ISO8601FormatStyle(timeZone: .current).year().month().day())
    }
}
