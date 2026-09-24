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

    func start() {
        running = true
        results = []
        let coreML = useCoreML
        Task.detached(priority: .userInitiated) {
            do {
                let inputs = try Bench.loadInputs()
                let env = try ORTEnv(loggingLevel: .warning)
                let providers = coreML ? [false, true] : [false]
                for variant in inputs.variants {
                    for ml in providers {
                        await MainActor.run { self.status = "Running \(variant) on \(ml ? "CoreML" : "CPU")..." }
                        // Each run builds and drops its own sessions, so variants never share memory.
                        let r = autoreleasepool { Bench.run(variant: variant, coreML: ml, inputs: inputs, env: env) }
                        await MainActor.run { self.results.append(r) }
                    }
                }
                await MainActor.run { self.status = "Done on \(deviceModel()). Copy the report and share it." }
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
