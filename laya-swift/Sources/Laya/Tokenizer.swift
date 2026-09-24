import Foundation

public enum LayaError: Error, CustomStringConvertible {
    case invalidTokenizer(String)

    public var description: String {
        switch self {
        case .invalidTokenizer(let why): return "invalid tokenizer.json: \(why)"
        }
    }
}

/// Encoder for the Laya checkpoints' `tokenizer.json`: added tokens, a `Replace(" " -> "▁")`
/// normalizer, a `Metaspace(prepend_scheme: always, split: true)` pre-tokenizer and a BPE model
/// with byte fallback. `encode(_:)` returns what `tok(text, add_special_tokens=False)["input_ids"]`
/// returns in Python, which is the only way `build_sequence` calls the tokenizer.
///
/// Text is handled as Unicode scalars, not Swift `Character`s: the Rust tokenizer splits and
/// looks up `char`s, and a grapheme cluster such as a flag emoji is several of those.
public final class LayaTokenizer {
    public let vocabSize: Int
    public let padId: Int
    public let unkId: Int
    public let bosId: Int
    public let eosId: Int
    public let maskId: Int

    /// Keyed by UTF-8 bytes, not String: Swift compares strings by canonical equivalence, and
    /// the vocab has distinct tokens that are canonically equivalent (61 in laya-multilingual).
    private let vocab: [[UInt8]: Int32]
    /// (left id << 32 | right id) -> (merge rank, merged id)
    private let merges: [UInt64: (rank: Int32, id: Int32)]
    private let byteIds: [Int32]?
    private let fuseUnk: Bool
    private let addedByFirst: [Unicode.Scalar: [AddedToken]]

    private struct AddedToken {
        let scalars: [Unicode.Scalar]
        let id: Int32
        let lstrip: Bool
        let rstrip: Bool
    }

    private static let space = Unicode.Scalar(0x20)!
    private static let metaspace = Unicode.Scalar(0x2581)!

    public convenience init(contentsOf url: URL) throws {
        try self.init(data: try Data(contentsOf: url))
    }

    public init(data: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = root["model"] as? [String: Any],
              model["type"] as? String == "BPE",
              let rawMerges = model["merges"] as? [Any] else {
            throw LayaError.invalidTokenizer("expected a BPE model with vocab and merges")
        }
        try Self.checkPipeline(root)
        // Not model["vocab"]: bridging it to [String: Any] merges canonically equivalent keys.
        let vocab = try RawJSON.vocab(in: data)
        var merges = [UInt64: (rank: Int32, id: Int32)](minimumCapacity: rawMerges.count)
        for (rank, entry) in rawMerges.enumerated() {
            let pair: [String]
            if let p = entry as? [String], p.count == 2 {
                pair = p
            } else if let s = entry as? String, let space = s.firstIndex(of: " ") {
                pair = [String(s[..<space]), String(s[s.index(after: space)...])]
            } else {
                throw LayaError.invalidTokenizer("merge \(rank) is not a pair")
            }
            guard let a = vocab[Array(pair[0].utf8)], let b = vocab[Array(pair[1].utf8)],
                  let m = vocab[Array(pair[0].utf8) + Array(pair[1].utf8)] else {
                throw LayaError.invalidTokenizer("merge \(rank) refers to tokens missing from the vocab")
            }
            let key = UInt64(UInt32(bitPattern: a)) << 32 | UInt64(UInt32(bitPattern: b))
            if merges[key] == nil { merges[key] = (Int32(rank), m) }
        }

        var byFirst = [Unicode.Scalar: [AddedToken]]()
        for case let t as [String: Any] in (root["added_tokens"] as? [Any] ?? []) {
            guard let content = t["content"] as? String, let id = (t["id"] as? NSNumber)?.int32Value,
                  let first = content.unicodeScalars.first else { continue }
            if t["normalized"] as? Bool == true {
                throw LayaError.invalidTokenizer("normalized added tokens are not supported (\(content))")
            }
            if t["single_word"] as? Bool == true {
                throw LayaError.invalidTokenizer("single_word added tokens are not supported (\(content))")
            }
            byFirst[first, default: []].append(AddedToken(
                scalars: Array(content.unicodeScalars), id: id,
                lstrip: t["lstrip"] as? Bool ?? false, rstrip: t["rstrip"] as? Bool ?? false))
        }
        for k in byFirst.keys { byFirst[k]!.sort { $0.scalars.count > $1.scalars.count } }

        func id(_ token: String) throws -> Int {
            guard let n = vocab[Array(token.utf8)] else { throw LayaError.invalidTokenizer("\(token) missing from vocab") }
            return Int(n)
        }
        let unk = (model["unk_token"] as? String) ?? "<unk>"
        self.unkId = try id(unk)
        self.padId = try id("<pad>")
        self.bosId = try id("<bos>")
        self.eosId = try id("<eos>")
        self.maskId = try id("<mask>")
        self.vocab = vocab
        self.merges = merges
        self.vocabSize = vocab.count
        self.fuseUnk = model["fuse_unk"] as? Bool ?? false
        self.addedByFirst = byFirst
        if model["byte_fallback"] as? Bool == true {
            // One entry per byte value; -1 where the vocab lacks <0xNN> (this checkpoint has no <0x09>).
            self.byteIds = (0..<256).map { vocab[Array(String(format: "<0x%02X>", $0).utf8)] ?? -1 }
        } else {
            self.byteIds = nil
        }
    }

    /// Refuse a tokenizer.json whose pipeline this encoder does not reproduce, instead of
    /// producing different ids from Python without saying so.
    private static func checkPipeline(_ root: [String: Any]) throws {
        let norm = root["normalizer"] as? [String: Any]
        let pattern = (norm?["pattern"] as? [String: Any])?["String"] as? String
        guard norm?["type"] as? String == "Replace", pattern == " ", norm?["content"] as? String == "\u{2581}" else {
            throw LayaError.invalidTokenizer("normalizer must be Replace(\" \" -> \"▁\")")
        }
        let pre = root["pre_tokenizer"] as? [String: Any]
        guard pre?["type"] as? String == "Metaspace", pre?["replacement"] as? String == "\u{2581}",
              pre?["prepend_scheme"] as? String == "always", pre?["split"] as? Bool == true else {
            throw LayaError.invalidTokenizer("pre_tokenizer must be Metaspace(▁, prepend_scheme: always, split: true)")
        }
        let model = root["model"] as? [String: Any]
        if let d = model?["dropout"], !(d is NSNull) { throw LayaError.invalidTokenizer("BPE dropout is not supported") }
        if model?["ignore_merges"] as? Bool == true { throw LayaError.invalidTokenizer("ignore_merges is not supported") }
        for key in ["continuing_subword_prefix", "end_of_word_suffix"] {
            if let v = model?[key] as? String, !v.isEmpty { throw LayaError.invalidTokenizer("\(key) is not supported") }
        }
    }

    /// Token ids for `text` without special tokens.
    public func encode(_ text: String) -> [Int] {
        var out: [Int] = []
        let scalars = Array(text.unicodeScalars)
        var segmentStart = 0
        var i = 0
        while i < scalars.count {
            guard let (token, end) = addedMatch(scalars, at: i) else { i += 1; continue }
            var start = i
            if token.lstrip { while start > segmentStart && scalars[start - 1].properties.isWhitespace { start -= 1 } }
            var stop = end
            if token.rstrip { while stop < scalars.count && scalars[stop].properties.isWhitespace { stop += 1 } }
            encodeSegment(scalars[segmentStart..<start], into: &out)
            out.append(Int(token.id))
            segmentStart = stop
            i = stop
        }
        encodeSegment(scalars[segmentStart..<scalars.count], into: &out)
        return out
    }

    /// Longest added token starting at `i` (the scan is left to right, so the first hit is the
    /// leftmost match, as with the Rust tokenizer's leftmost-longest Aho-Corasick).
    private func addedMatch(_ s: [Unicode.Scalar], at i: Int) -> (AddedToken, Int)? {
        guard let candidates = addedByFirst[s[i]] else { return nil }
        for t in candidates where i + t.scalars.count <= s.count {
            var ok = true
            for k in 1..<t.scalars.count where s[i + k] != t.scalars[k] { ok = false; break }
            if ok { return (t, i + t.scalars.count) }
        }
        return nil
    }

    private func encodeSegment(_ segment: ArraySlice<Unicode.Scalar>, into out: inout [Int]) {
        if segment.isEmpty { return }
        var s = segment.map { $0 == Self.space ? Self.metaspace : $0 }
        if s.first != Self.metaspace { s.insert(Self.metaspace, at: 0) }
        var start = 0
        for k in 1...s.count where k == s.count || s[k] == Self.metaspace {
            bpe(s[start..<k], into: &out)
            start = k
        }
    }

    private func bpe(_ word: ArraySlice<Unicode.Scalar>, into out: inout [Int]) {
        var symbols: [Int32] = []
        symbols.reserveCapacity(word.count)
        var lastWasUnk = false
        for scalar in word {
            if let id = vocab[Array(String(scalar).utf8)] {
                symbols.append(id)
                lastWasUnk = false
                continue
            }
            if let byteIds {
                let bytes = Array(String(scalar).utf8)
                if bytes.allSatisfy({ byteIds[Int($0)] >= 0 }) {
                    symbols.append(contentsOf: bytes.map { byteIds[Int($0)] })
                    lastWasUnk = false
                    continue
                }
            }
            if !(fuseUnk && lastWasUnk) { symbols.append(Int32(unkId)) }
            lastWasUnk = true
        }
        // Merge the lowest-rank adjacent pair, leftmost first, until none applies.
        while symbols.count > 1 {
            var best: (rank: Int32, at: Int, id: Int32)?
            for k in 0..<(symbols.count - 1) {
                let key = UInt64(UInt32(bitPattern: symbols[k])) << 32 | UInt64(UInt32(bitPattern: symbols[k + 1]))
                if let m = merges[key], best == nil || m.rank < best!.rank { best = (m.rank, k, m.id) }
            }
            guard let b = best else { break }
            symbols[b.at] = b.id
            symbols.remove(at: b.at + 1)
        }
        out.append(contentsOf: symbols.map(Int.init))
    }
}

/// Reads the "vocab" object of tokenizer.json as raw UTF-8 keys.
enum RawJSON {
    static func vocab(in data: Data) throws -> [[UInt8]: Int32] {
        try data.withUnsafeBytes { (buf: UnsafeRawBufferPointer) -> [[UInt8]: Int32] in
            let b = buf.bindMemory(to: UInt8.self)
            var i = try objectStart(ofKey: Array("\"vocab\"".utf8), in: b)
            var out = [[UInt8]: Int32](minimumCapacity: 1 << 18)
            func skipSpace() { while i < b.count, b[i] == 0x20 || b[i] == 0x0A || b[i] == 0x0D || b[i] == 0x09 { i += 1 } }
            func fail(_ why: String) -> LayaError { .invalidTokenizer("vocab at byte \(i): \(why)") }
            skipSpace()
            if i < b.count, b[i] == UInt8(ascii: "}") { return out }
            while true {
                skipSpace()
                guard i < b.count, b[i] == UInt8(ascii: "\"") else { throw fail("expected a key") }
                let key = try string(b, &i)
                skipSpace()
                guard i < b.count, b[i] == UInt8(ascii: ":") else { throw fail("expected ':'") }
                i += 1
                skipSpace()
                var n: Int64 = 0
                let digits = i
                while i < b.count, b[i] >= 0x30, b[i] <= 0x39 { n = n * 10 + Int64(b[i] - 0x30); i += 1 }
                guard i > digits, n <= Int64(Int32.max) else { throw fail("expected a token id") }
                if out.updateValue(Int32(n), forKey: key) != nil { throw fail("duplicate key") }
                skipSpace()
                guard i < b.count else { throw fail("unterminated object") }
                if b[i] == UInt8(ascii: ",") { i += 1; continue }
                if b[i] == UInt8(ascii: "}") { return out }
                throw fail("expected ',' or '}'")
            }
        }
    }

    /// Index just past the '{' of the first `key: {`.
    private static func objectStart(ofKey key: [UInt8], in b: UnsafeBufferPointer<UInt8>) throws -> Int {
        var i = 0
        while i + key.count < b.count {
            if b[i] == key[0], zip(key.indices, key).allSatisfy({ b[i + $0] == $1 }) {
                var j = i + key.count
                while j < b.count, b[j] == 0x20 || b[j] == 0x0A || b[j] == 0x0D || b[j] == 0x09 { j += 1 }
                if j < b.count, b[j] == UInt8(ascii: ":") {
                    j += 1
                    while j < b.count, b[j] == 0x20 || b[j] == 0x0A || b[j] == 0x0D || b[j] == 0x09 { j += 1 }
                    if j < b.count, b[j] == UInt8(ascii: "{") { return j + 1 }
                }
            }
            i += 1
        }
        throw LayaError.invalidTokenizer("no \"vocab\" object")
    }

    /// Decodes the JSON string starting at `b[i] == '"'` into UTF-8 bytes; leaves `i` past the closing quote.
    private static func string(_ b: UnsafeBufferPointer<UInt8>, _ i: inout Int) throws -> [UInt8] {
        var out: [UInt8] = []
        i += 1
        func hex4() throws -> UInt32 {
            guard i + 4 <= b.count else { throw LayaError.invalidTokenizer("truncated \\u escape") }
            var v: UInt32 = 0
            for _ in 0..<4 {
                let c = b[i]
                let d: UInt32
                switch c {
                case 0x30...0x39: d = UInt32(c - 0x30)
                case 0x41...0x46: d = UInt32(c - 0x41 + 10)
                case 0x61...0x66: d = UInt32(c - 0x61 + 10)
                default: throw LayaError.invalidTokenizer("bad \\u escape")
                }
                v = v << 4 | d
                i += 1
            }
            return v
        }
        while i < b.count {
            let c = b[i]
            if c == UInt8(ascii: "\"") { i += 1; return out }
            if c != UInt8(ascii: "\\") { out.append(c); i += 1; continue }
            i += 1
            guard i < b.count else { break }
            let e = b[i]
            i += 1
            switch e {
            case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"): out.append(e)
            case UInt8(ascii: "b"): out.append(0x08)
            case UInt8(ascii: "f"): out.append(0x0C)
            case UInt8(ascii: "n"): out.append(0x0A)
            case UInt8(ascii: "r"): out.append(0x0D)
            case UInt8(ascii: "t"): out.append(0x09)
            case UInt8(ascii: "u"):
                var v = try hex4()
                if v >= 0xD800, v < 0xDC00, i + 1 < b.count, b[i] == UInt8(ascii: "\\"), b[i + 1] == UInt8(ascii: "u") {
                    i += 2
                    let lo = try hex4()
                    v = 0x10000 + ((v - 0xD800) << 10) + (lo - 0xDC00)
                }
                guard let scalar = Unicode.Scalar(v) else { throw LayaError.invalidTokenizer("lone surrogate in vocab") }
                out.append(contentsOf: Array(String(scalar).utf8))
            default: throw LayaError.invalidTokenizer("bad escape")
            }
        }
        throw LayaError.invalidTokenizer("unterminated string")
    }
}
