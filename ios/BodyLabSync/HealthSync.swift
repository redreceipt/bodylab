import HealthKit
import SwiftUI

/// Reads Apple Health and pushes it to BodyLab (`POST /api/sync`): profile (height, sex, birth date),
/// daily body/nutrition/activity/vitals/sleep, and workouts. The server upserts by day / workout ID,
/// so every sync can resend the whole window.
@MainActor
final class HealthSync: ObservableObject {
    @Published var status = ""
    @Published var summary: Summary?
    @Published var today: [Check] = []
    @Published var female = false
    let metric = Locale.current.measurementSystem == .metric

    struct Summary { let bodyFat: Double?, weight: String? }
    struct Check { let label: String, done: Bool, hint: String }
    struct Field { let key: String, label: String }
    var tapeFields: [Field] {
        [.init(key: "tape_neck", label: "Neck"), .init(key: female ? "tape_bust" : "tape_chest", label: female ? "Bust" : "Chest"),
         .init(key: "tape_waist", label: "Waist"), .init(key: "tape_hips", label: "Hips"), .init(key: "tape_arm", label: "Upper arm"),
         .init(key: "tape_thigh", label: "Thigh"), .init(key: "tape_calf", label: "Calf")]
            + (female ? [.init(key: "tape_underbust", label: "Underbust")]
                      : [.init(key: "penis_len_soft", label: "Penis length, resting"), .init(key: "penis_girth_soft", label: "Penis girth, resting"),
                         .init(key: "penis_len_erect", label: "Penis length, erect"), .init(key: "penis_girth_erect", label: "Penis girth, erect")])
    }

    private let store = HKHealthStore()
    private let windowDays = 90
    private var observing = false
    private static let bpm = HKUnit.count().unitDivided(by: .minute())

    private struct Metric {
        let id: HKQuantityTypeIdentifier, unit: HKUnit, key: String, option: HKStatisticsOptions
        var scale = 1.0
    }
    private let metrics: [Metric] = [
        // body
        .init(id: .bodyMass, unit: .gramUnit(with: .kilo), key: "weight", option: .discreteAverage),
        .init(id: .bodyFatPercentage, unit: .percent(), key: "scale_bf", option: .discreteAverage, scale: 100),
        .init(id: .leanBodyMass, unit: .gramUnit(with: .kilo), key: "scale_lean", option: .discreteAverage),
        .init(id: .waistCircumference, unit: .meterUnit(with: .centi), key: "tape_waist", option: .discreteAverage),
        // nutrition
        .init(id: .dietaryEnergyConsumed, unit: .kilocalorie(), key: "kcal", option: .cumulativeSum),
        .init(id: .dietaryProtein, unit: .gram(), key: "protein", option: .cumulativeSum),
        .init(id: .dietaryCarbohydrates, unit: .gram(), key: "carbs", option: .cumulativeSum),
        .init(id: .dietaryFatTotal, unit: .gram(), key: "fat", option: .cumulativeSum),
        .init(id: .dietaryFiber, unit: .gram(), key: "fiber", option: .cumulativeSum),
        .init(id: .dietarySugar, unit: .gram(), key: "sugar", option: .cumulativeSum),
        .init(id: .dietaryWater, unit: .liter(), key: "water", option: .cumulativeSum),
        .init(id: .numberOfAlcoholicBeverages, unit: .count(), key: "alcohol", option: .cumulativeSum),
        // activity
        .init(id: .activeEnergyBurned, unit: .kilocalorie(), key: "active", option: .cumulativeSum),
        .init(id: .basalEnergyBurned, unit: .kilocalorie(), key: "basal", option: .cumulativeSum),
        .init(id: .stepCount, unit: .count(), key: "steps", option: .cumulativeSum),
        .init(id: .appleExerciseTime, unit: .minute(), key: "exercise", option: .cumulativeSum),
        // recovery & fitness
        .init(id: .restingHeartRate, unit: bpm, key: "rhr", option: .discreteAverage),
        .init(id: .heartRateVariabilitySDNN, unit: .secondUnit(with: .milli), key: "hrv", option: .discreteAverage),
        .init(id: .vo2Max, unit: HKUnit(from: "ml/kg*min"), key: "vo2max", option: .discreteAverage),
        .init(id: .walkingHeartRateAverage, unit: bpm, key: "walk_hr", option: .discreteAverage),
        .init(id: .respiratoryRate, unit: bpm, key: "resp", option: .discreteAverage),
    ]

    init() {
        if !(UserDefaults.standard.string(forKey: "token") ?? "").isEmpty { observe() }
    }

    /// `bodylab://pair?server=…&token=…` from the QR code in BodyLab → Profile.
    func pair(_ url: URL) async {
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        guard url.host == "pair", let server = items.first(where: { $0.name == "server" })?.value,
              let token = items.first(where: { $0.name == "token" })?.value else { return }
        UserDefaults.standard.set(server, forKey: "server")
        UserDefaults.standard.set(token, forKey: "token")
        await connect()
    }

    func connect() async {
        var read: Set<HKObjectType> = Set(metrics.map { HKQuantityType($0.id) })
        read.formUnion([HKQuantityType(.height), HKQuantityType(.heartRate), HKCategoryType(.sleepAnalysis),
                        HKObjectType.workoutType(), HKCharacteristicType(.biologicalSex), HKCharacteristicType(.dateOfBirth)])
        let write: Set<HKSampleType> = [HKQuantityType(.dietaryEnergyConsumed), HKQuantityType(.dietaryProtein), HKQuantityType(.bodyMass),
                                        HKQuantityType(.waistCircumference), HKObjectType.workoutType()]
        do {
            try await store.requestAuthorization(toShare: write, read: read)
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
        status = "Syncing…"
        do {
            let start = Calendar.current.date(byAdding: .day, value: -windowDays, to: Calendar.current.startOfDay(for: .now))!
            let since = HKQuery.predicateForSamples(withStart: start, end: nil)
            var daily: [String: [String: Any]] = [:]
            for m in metrics {
                let query = HKStatisticsCollectionQueryDescriptor(
                    predicate: .quantitySample(type: HKQuantityType(m.id), predicate: since),
                    options: m.option, anchorDate: start, intervalComponents: DateComponents(day: 1))
                try await query.result(for: store).enumerateStatistics(from: start, to: .now) { stats, _ in
                    if let q = m.option == .cumulativeSum ? stats.sumQuantity() : stats.averageQuantity() {
                        daily[Self.day(stats.startDate), default: [:]][m.key] = q.doubleValue(for: m.unit) * m.scale
                    }
                }
            }
            for (key, hours) in try await sleep(since) {
                for (day, h) in hours { daily[day, default: [:]][key] = h }
            }
            let profile = await profile()
            let body: [String: Any] = [
                "profile": profile,
                "days": daily.map { day, values in values.merging(["date": day]) { a, _ in a } },
                "workouts": try await workouts(since, age: profile["age"] as? Int),
            ]
            let reply = try await post("api/sync", JSONSerialization.data(withJSONObject: body), type: "application/json")
            status = "Synced \(reply["saved"] ?? 0) records at \(Date.now.formatted(date: .omitted, time: .shortened))"
            await refreshToday()
        } catch {
            status = "Sync failed: \(error.localizedDescription)"
        }
    }

    /// Today's checklist and latest numbers, from BodyLab.
    func refreshToday() async {
        guard let state = try? await request("api/state", method: "GET") else { return }
        let profile = state["profile"] as? [String: Any] ?? [:]
        female = profile["sex"] as? String == "female"
        let insights = state["insights"] as? [String: Any] ?? [:]
        let day = (insights["days"] as? [[String: Any]])?.last ?? [:]
        let usual = insights["usual_kcal"] as? Double, kcal = day["kcal"] as? Double, views = day["views"] as? Int ?? 0
        today = [
            .init(label: "Body scan", done: day["scan"] as? Bool ?? false, hint: views > 0 ? "\(views) views so far — keep walking past" : "Walk past the camera; turning helps"),
            .init(label: "Weigh-in", done: day["weight"] as? Bool ?? false, hint: "Step on the scale, or log it here"),
            .init(label: "Food", done: day["food"] as? Bool ?? false,
                  hint: kcal.map { "\(Int($0)) \(usual.map { "of ~\(Int($0)) " } ?? "")kcal logged" } ?? "Log meals here or in any food app"),
            .init(label: "Sleep", done: day["sleep"] as? Bool ?? false, hint: "Wear your Watch to bed"),
            .init(label: "Activity", done: day["activity"] as? Bool ?? false, hint: "Wear your Watch"),
        ]
        let m = (state["scans"] as? [[String: Any]])?.last?["m"] as? [String: Any] ?? [:]
        let kg = (state["logs"] as? [[String: Any]])?.last(where: { $0["weight"] != nil })?["weight"] as? Double
        summary = .init(bodyFat: m["bf"] as? Double, weight: kg.map { metric ? String(format: "%.1f kg", $0) : String(format: "%.1f lb", $0 * 2.20462) })
    }

    // MARK: logging (Apple Health is the record; BodyLab reads it back on sync)

    private func save(_ id: HKQuantityTypeIdentifier, _ value: Double, _ unit: HKUnit) async throws {
        try await store.save(HKQuantitySample(type: HKQuantityType(id), quantity: HKQuantity(unit: unit, doubleValue: value), start: .now, end: .now))
    }

    func logFood(kcal: Double?, protein: Double?) async {
        do {
            if let kcal { try await save(.dietaryEnergyConsumed, kcal, .kilocalorie()) }
            if let protein { try await save(.dietaryProtein, protein, .gram()) }
            await sync()
        } catch { status = "Couldn't save to Apple Health: \(error.localizedDescription)" }
    }

    func logWeight(_ value: Double) async {
        do {
            try await save(.bodyMass, value, metric ? .gramUnit(with: .kilo) : .pound())
            await sync()
        } catch { status = "Couldn't save to Apple Health: \(error.localizedDescription)" }
    }

    func logWorkout(type: String, minutes: Double, effort: Double, focus: [String]) async {
        let config = HKWorkoutConfiguration()
        config.activityType = ["Strength": .traditionalStrengthTraining, "Cardio": .mixedCardio, "HIIT": .highIntensityIntervalTraining,
                               "Sports": .other, "Mobility": .flexibility][type] ?? .other
        let builder = HKWorkoutBuilder(healthStore: store, configuration: config, device: .local())
        let end = Date.now, start = end.addingTimeInterval(-minutes * 60)
        do {
            try await builder.beginCollection(at: start)
            try await builder.addMetadata(["BodyLabFocus": focus.joined(separator: ","), "BodyLabEffort": effort])
            try await builder.endCollection(at: end)
            _ = try await builder.finishWorkout()
            await sync()
        } catch { status = "Couldn't save the workout: \(error.localizedDescription)" }
    }

    /// Waist goes to Apple Health; the rest (other body parts, genital and bust measurements) to BodyLab directly.
    func logTape(_ values: [String: Double]) async {
        let cm = values.mapValues { metric ? $0 : $0 * 2.54 }
        do {
            if let waist = cm["tape_waist"] { try await save(.waistCircumference, waist, .meterUnit(with: .centi)) }
            let rest = cm.filter { $0.key != "tape_waist" }
            if !rest.isEmpty {
                var body: [String: Any] = rest
                body["kind"] = "measure"
                _ = try await post("api/logs", JSONSerialization.data(withJSONObject: body), type: "application/json")
            }
            await sync()
        } catch { status = "Couldn't save measurements: \(error.localizedDescription)" }
    }

    /// Face photo for the realistic avatar. Stays on your BodyLab server.
    func uploadFace(_ image: UIImage) async {
        status = "Uploading face photo…"
        let side = 1600.0, scale = min(1, side / max(image.size.width, image.size.height))
        let size = CGSize(width: image.size.width * scale, height: image.size.height * scale)
        // Redrawing also bakes in the photo's orientation.
        let jpeg = UIGraphicsImageRenderer(size: size).jpegData(withCompressionQuality: 0.9) { _ in image.draw(in: CGRect(origin: .zero, size: size)) }
        do {
            _ = try await post("api/face", jpeg, type: "image/jpeg")
            status = "Face photo saved ✓"
        } catch {
            status = "Face upload failed: \(error.localizedDescription)"
        }
    }

    private func post(_ path: String, _ body: Data, type: String) async throws -> [String: Any] {
        try await request(path, method: "POST", body: body, type: type)
    }

    private func request(_ path: String, method: String, body: Data? = nil, type: String? = nil) async throws -> [String: Any] {
        let defaults = UserDefaults.standard
        guard let token = defaults.string(forKey: "token"), !token.isEmpty,
              let url = URL(string: defaults.string(forKey: "server") ?? "")?.appending(path: path) else {
            throw URLError(.userAuthenticationRequired, userInfo: [NSLocalizedDescriptionKey: "Pair with BodyLab first (scan the QR code)"])
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        if let type { request.setValue(type, forHTTPHeaderField: "Content-Type") }
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.httpBody = body
        let (data, response) = try await URLSession.shared.data(for: request)
        let reply = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        let code = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else {
            throw URLError(.badServerResponse, userInfo: [NSLocalizedDescriptionKey: "BodyLab said \(code): \(reply["error"] ?? "unknown error")"])
        }
        return reply
    }

    private func profile() async -> [String: Any] {
        var p: [String: Any] = ["units": metric ? "metric" : "imperial"]
        let latest = HKSampleQueryDescriptor(predicates: [.quantitySample(type: HKQuantityType(.height))],
                                             sortDescriptors: [SortDescriptor(\.endDate, order: .reverse)], limit: 1)
        if let height = try? await latest.result(for: store).first {
            p["height_cm"] = height.quantity.doubleValue(for: .meterUnit(with: .centi))
        }
        switch (try? store.biologicalSex())?.biologicalSex {
        case .male: p["sex"] = "male"
        case .female: p["sex"] = "female"
        default: break
        }
        if let dob = try? store.dateOfBirthComponents(), let y = dob.year, let m = dob.month, let d = dob.day {
            p["birth_date"] = String(format: "%04d-%02d-%02d", y, m, d)
            p["age"] = Calendar.current.dateComponents([.year], from: Calendar.current.date(from: dob) ?? .now, to: .now).year
        }
        return p
    }

    /// Hours per night: total asleep, deep, REM. Overlapping samples (Watch + phone) are merged, not double counted.
    private func sleep(_ since: NSPredicate) async throws -> [String: [String: Double]] {
        let samples = try await HKSampleQueryDescriptor(
            predicates: [.categorySample(type: HKCategoryType(.sleepAnalysis), predicate: since)],
            sortDescriptors: [SortDescriptor(\.startDate)]).result(for: store)
        let stages: [String: Set<Int>] = [
            "sleep": Set(HKCategoryValueSleepAnalysis.allAsleepValues.map(\.rawValue)),
            "sleep_deep": [HKCategoryValueSleepAnalysis.asleepDeep.rawValue],
            "sleep_rem": [HKCategoryValueSleepAnalysis.asleepREM.rawValue],
        ]
        return stages.mapValues { values in
            var merged: [(start: Date, end: Date)] = []
            for s in samples where values.contains(s.value) {
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
    }

    private func workouts(_ since: NSPredicate, age: Int?) async throws -> [[String: Any]] {
        let maxHR = age.map { 208 - 0.7 * Double($0) } ?? 185  // Tanaka formula; 185 if age is unknown
        return try await HKSampleQueryDescriptor(predicates: [.workout(since)], sortDescriptors: [SortDescriptor(\.startDate)])
            .result(for: store)
            .filter { $0.duration >= 60 }
            .map { w in
                let (type, focus) = Self.kind(w.workoutActivityType)
                var out: [String: Any] = ["id": w.uuid.uuidString, "start": w.startDate.timeIntervalSince1970,
                                          "minutes": min(w.duration / 60, 600), "type": type, "focus": focus]
                if let hr = w.statistics(for: HKQuantityType(.heartRate))?.averageQuantity()?.doubleValue(for: Self.bpm) {
                    out["hr"] = hr
                    out["rpe"] = min(10, max(1, (hr / maxHR - 0.5) / 0.05))  // effort from % of max heart rate
                }
                // Logged in this app: which areas you trained and how hard, kept in the workout's Health metadata.
                if let areas = w.metadata?["BodyLabFocus"] as? String, !areas.isEmpty { out["focus"] = areas.split(separator: ",").map(String.init) }
                if let effort = w.metadata?["BodyLabEffort"] as? Double { out["rpe"] = effort }
                if let kcal = w.statistics(for: HKQuantityType(.activeEnergyBurned))?.sumQuantity()?.doubleValue(for: .kilocalorie()) {
                    out["kcal"] = kcal
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
