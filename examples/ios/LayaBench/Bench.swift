import Foundation
import OnnxRuntimeBindings

/// One pre-tokenized question from Models/bench_inputs.json (written by prepare_models.py).
struct BenchItem: Decodable {
    let name: String
    let type: String
    let labels: [String]
    let input_ids: [Int64]
    let marker_pos: [Int64]
    let qtype: Int64
    let expected: [String: [Double]]
}

struct BenchInputs: Decodable {
    let repo: String
    let variants: [String]
    let items: [BenchItem]
}

struct ItemResult: Codable {
    let name: String
    let tokens: Int
    let medianMs: Double
    let answer: String
    let maxAbsDiff: Double
    let sameAnswer: Bool
}

struct RunResult: Codable, Identifiable {
    var id: String { "\(variant)/\(provider)" }
    let variant: String
    let provider: String
    let sizeMB: Double
    var loadMs: Double = 0
    var footprintMB: Double = 0
    var items: [ItemResult] = []
    var error: String?

    var shortMedianMs: Double { median(items.filter { $0.tokens < 256 }.map(\.medianMs)) }
    var longMedianMs: Double { median(items.filter { $0.tokens >= 256 }.map(\.medianMs)) }
    var allSame: Bool { items.allSatisfy(\.sameAnswer) }
    var worstDiff: Double { items.map(\.maxAbsDiff).max() ?? 0 }
}

func median(_ xs: [Double]) -> Double {
    guard !xs.isEmpty else { return .nan }
    let s = xs.sorted()
    return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
}

/// Physical memory footprint of this process (what iOS uses to decide on a jetsam kill).
func footprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : .nan
}

private func tensor<T>(_ values: [T], _ type: ORTTensorElementDataType, _ shape: [Int]) throws -> ORTValue {
    let data = values.withUnsafeBufferPointer { NSMutableData(bytes: $0.baseAddress, length: $0.count * MemoryLayout<T>.stride) }
    return try ORTValue(tensorData: data, elementType: type, shape: shape.map { NSNumber(value: $0) })
}

/// encoder.onnx + head.onnx for one variant, on one execution provider.
final class LayaRunner {
    private let encoder: ORTSession
    private let head: ORTSession

    init(env: ORTEnv, dir: URL, coreML: Bool) throws {
        let options = try ORTSessionOptions()
        try options.setGraphOptimizationLevel(.all)
        if coreML {
            let ml = ORTCoreMLExecutionProviderOptions()
            ml.enableOnSubgraphs = true
            try options.appendCoreMLExecutionProvider(with: ml)
        }
        encoder = try ORTSession(env: env, modelPath: dir.appendingPathComponent("encoder.onnx").path, sessionOptions: options)
        head = try ORTSession(env: env, modelPath: dir.appendingPathComponent("head.onnx").path, sessionOptions: options)
    }

    /// Softmax over the item's marker logits (temperature 1, as prepare_models.py computes `expected`).
    func probabilities(_ item: BenchItem) throws -> [Double] {
        let n = item.input_ids.count
        let k = item.marker_pos.count
        let mask = [Int64](repeating: 1, count: n)
        let hidden = try encoder.run(
            withInputs: ["input_ids": try tensor(item.input_ids, .int64, [1, n]),
                         "attention_mask": try tensor(mask, .int64, [1, n])],
            outputNames: ["last_hidden_state"], runOptions: nil)["last_hidden_state"]!
        let out = try head.run(
            withInputs: ["hidden_states": hidden,
                         "marker_pos": try tensor(item.marker_pos, .int64, [1, k]),
                         "marker_mask_u8": try tensor([UInt8](repeating: 1, count: k), .uInt8, [1, k]),
                         "qtype": try tensor([item.qtype], .int64, [1, 1]),
                         "attention_mask": try tensor(mask, .int64, [1, n])],
            outputNames: ["logits"], runOptions: nil)["logits"]!
        let data = try out.tensorData() as Data
        let logits: [Double] = data.withUnsafeBytes { raw in
            Array(raw.bindMemory(to: Float.self).prefix(k)).map(Double.init)
        }
        let top = logits.max() ?? 0
        let e = logits.map { exp($0 - top) }
        let sum = e.reduce(0, +)
        return e.map { $0 / sum }
    }
}

func argmax(_ xs: [Double]) -> Int { xs.indices.max { xs[$0] < xs[$1] } ?? 0 }

enum Bench {
    static let modelsURL = Bundle.main.url(forResource: "Models", withExtension: nil)

    static func loadInputs() throws -> BenchInputs {
        guard let dir = modelsURL,
              let data = try? Data(contentsOf: dir.appendingPathComponent("bench_inputs.json")) else {
            throw NSError(domain: "LayaBench", code: 1, userInfo: [NSLocalizedDescriptionKey:
                "Models/bench_inputs.json is missing. Run: python examples/ios/prepare_models.py"])
        }
        return try JSONDecoder().decode(BenchInputs.self, from: data)
    }

    static func sizeMB(_ dir: URL) -> Double {
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        let bytes = files.compactMap { try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize }.reduce(0, +)
        return Double(bytes) / 1_000_000
    }

    /// Loads one variant, runs every item `runs` times after `warmup` runs, and reports medians.
    static func run(variant: String, coreML: Bool, inputs: BenchInputs, env: ORTEnv,
                    warmup: Int = 2, runs: Int = 10) -> RunResult {
        let dir = modelsURL!.appendingPathComponent(variant)
        var result = RunResult(variant: variant, provider: coreML ? "CoreML" : "CPU", sizeMB: sizeMB(dir))
        do {
            let t0 = Date()
            let runner = try LayaRunner(env: env, dir: dir, coreML: coreML)
            result.loadMs = Date().timeIntervalSince(t0) * 1000
            for item in inputs.items {
                var p: [Double] = []
                for _ in 0..<warmup { p = try runner.probabilities(item) }
                var times: [Double] = []
                for _ in 0..<runs {
                    let t = DispatchTime.now().uptimeNanoseconds
                    p = try runner.probabilities(item)
                    times.append(Double(DispatchTime.now().uptimeNanoseconds - t) / 1_000_000)
                }
                let expected = item.expected[variant] ?? item.expected["fp32"] ?? p
                let diff = zip(p, expected).map { abs($0 - $1) }.max() ?? 0
                result.items.append(ItemResult(
                    name: item.name, tokens: item.input_ids.count, medianMs: median(times),
                    answer: item.labels.indices.contains(argmax(p)) ? item.labels[argmax(p)] : "?",
                    maxAbsDiff: diff, sameAnswer: argmax(p) == argmax(expected)))
            }
            result.footprintMB = footprintMB()
        } catch {
            result.error = error.localizedDescription
        }
        return result
    }
}

func deviceModel() -> String {
    var sys = utsname()
    uname(&sys)
    return withUnsafeBytes(of: &sys.machine) { String(cString: $0.bindMemory(to: CChar.self).baseAddress!) }
}
