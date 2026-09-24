import SwiftUI
import UIKit
import OnnxRuntimeBindings

@main
struct LayaBenchApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

@MainActor
final class BenchModel: ObservableObject {
    @Published var results: [RunResult] = []
    @Published var status = "Tap Run. Keep the phone unlocked and on the charger."
    @Published var running = false
    @Published var useCoreML = true

    // Results are saved after every run, so a run that gets the app killed (iOS ends apps that
    // use too much memory) still leaves the earlier results and names the run that died.
    // Files > On My iPhone > LayaBench, or Finder > iPhone > Files, shows laya_report.json.
    nonisolated private static let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    nonisolated private static let resultsURL = docs.appendingPathComponent("laya_results.json")
    nonisolated private static let reportURL = docs.appendingPathComponent("laya_report.json")
    nonisolated private static let runningURL = docs.appendingPathComponent("laya_running.txt")

    init() {
        if let data = try? Data(contentsOf: Self.resultsURL),
           let saved = try? JSONDecoder().decode([RunResult].self, from: data) {
            results = saved
        }
        if let died = try? String(contentsOf: Self.runningURL, encoding: .utf8) {
            let parts = died.split(separator: "/").map(String.init)
            var r = RunResult(variant: parts.first ?? "?", provider: parts.last ?? "?", sizeMB: 0)
            r.error = "The app was closed during this run, most likely because iOS ran out of memory for it."
            results.append(r)
            try? FileManager.default.removeItem(at: Self.runningURL)
            save()
        }
        if !results.isEmpty { status = "Results from the last session. Tap Run to start again." }
    }

    private func save() {
        if let data = try? JSONEncoder().encode(results) { try? data.write(to: Self.resultsURL) }
        try? report.data(using: .utf8)?.write(to: Self.reportURL)
    }

    private func record(_ r: RunResult) {
        results.append(r)
        try? FileManager.default.removeItem(at: Self.runningURL)
        save()
        if let data = try? JSONEncoder().encode(r) { print("LAYA_RESULT " + String(decoding: data, as: UTF8.self)) }
    }

    func start() {
        running = true
        results = []
        save()
        let coreML = useCoreML
        Task.detached(priority: .userInitiated) {
            do {
                let inputs = try Bench.loadInputs()
                let env = try ORTEnv(loggingLevel: .warning)
                // Smallest variant first, and every CPU run before any CoreML run: CoreML
                // compiles the model and needs the most memory, so it is the likeliest to be killed.
                let variants = inputs.variants.sorted { Bench.sizeMB(Bench.modelsURL!.appendingPathComponent($0)) <
                                                        Bench.sizeMB(Bench.modelsURL!.appendingPathComponent($1)) }
                let runs = variants.map { ($0, false) } + (coreML ? variants.map { ($0, true) } : [])
                for (variant, ml) in runs {
                    let name = "\(variant)/\(ml ? "CoreML" : "CPU")"
                    try? name.write(to: Self.runningURL, atomically: true, encoding: .utf8)
                    await MainActor.run { self.status = "Running \(variant) on \(ml ? "CoreML" : "CPU")..." }
                    // Each run builds and drops its own sessions, so variants never share memory.
                    let r = autoreleasepool { Bench.run(variant: variant, coreML: ml, inputs: inputs, env: env) }
                    await MainActor.run { self.record(r) }
                }
                await MainActor.run { self.status = "Done on \(deviceModel()). Share the report, or take it from Files." }
            } catch {
                await MainActor.run { self.status = error.localizedDescription }
            }
            await MainActor.run { self.running = false }
        }
    }

    var report: String {
        let payload: [String: Any] = [
            "device": deviceModel(),
            "ios": UIDevice.current.systemVersion,
            "thermal": ProcessInfo.processInfo.thermalState.rawValue,
            "results": (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(results))) ?? [],
        ]
        let data = (try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])) ?? Data()
        return String(decoding: data, as: UTF8.self)
    }
}

struct ContentView: View {
    @StateObject private var model = BenchModel()

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Toggle("Also try CoreML (Neural Engine)", isOn: $model.useCoreML).disabled(model.running)
                    Button(model.running ? "Running..." : "Run benchmark") { model.start() }.disabled(model.running)
                    Text(model.status).font(.footnote).foregroundStyle(.secondary)
                }
                ForEach(model.results) { r in
                    Section("\(r.variant) · \(r.provider) · \(String(format: "%.0f MB", r.sizeMB))") {
                        if let e = r.error {
                            Text(e).foregroundStyle(.red).font(.footnote)
                        } else {
                            row("Short text (<256 tok), median", String(format: "%.0f ms", r.shortMedianMs))
                            row("Long text (~600 tok), median", String(format: "%.0f ms", r.longMedianMs))
                            row("Load", String(format: "%.1f s", r.loadMs / 1000))
                            row("Memory footprint", String(format: "%.0f MB", r.footprintMB))
                            row("Same answers as desktop", r.allSame ? "yes" : "NO",
                                color: r.allSame ? .green : .red)
                            row("Max prob. difference", String(format: "%.4f", r.worstDiff))
                            DisclosureGroup("Per question") {
                                ForEach(r.items, id: \.name) { it in
                                    VStack(alignment: .leading) {
                                        Text(it.name).font(.caption.bold())
                                        Text(String(format: "%@ · %ld tok · %.0f ms · Δ%.4f",
                                                    it.answer, it.tokens, it.medianMs, it.maxAbsDiff))
                                            .font(.caption.monospaced())
                                    }
                                }
                            }
                        }
                    }
                }
                if !model.results.isEmpty && !model.running {
                    Section {
                        Button("Copy report (JSON)") { UIPasteboard.general.string = model.report }
                        ShareLink(item: model.report) { Text("Share report") }
                    }
                }
            }
            .navigationTitle("Laya on-device")
        }
    }

    private func row(_ title: String, _ value: String, color: Color = .primary) -> some View {
        HStack {
            Text(title)
            Spacer()
            Text(value).monospacedDigit().foregroundStyle(color)
        }
    }
}
