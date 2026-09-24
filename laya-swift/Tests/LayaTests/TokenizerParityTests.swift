import Foundation
import XCTest
@testable import Laya

/// Compares LayaTokenizer with the Python tokenizer on Fixtures/tokenizer_cases.json
/// (written by scripts/make_fixtures.py). Needs the checkpoint's tokenizer.json, which is too
/// large for git: set LAYA_TOKENIZER_JSON to its path, or the test is skipped.
final class TokenizerParityTests: XCTestCase {
    private struct Fixture: Decodable {
        struct Case: Decodable { let text: String; let ids: [Int] }
        let cases: [Case]
    }

    private static var tokenizer: LayaTokenizer?

    private func loadTokenizer() throws -> LayaTokenizer {
        if let t = Self.tokenizer { return t }
        guard let path = ProcessInfo.processInfo.environment["LAYA_TOKENIZER_JSON"] else {
            throw XCTSkip("set LAYA_TOKENIZER_JSON to the checkpoint's tokenizer.json")
        }
        let t = try LayaTokenizer(contentsOf: URL(fileURLWithPath: path))
        Self.tokenizer = t
        return t
    }

    func testMatchesPython() throws {
        let tok = try loadTokenizer()
        let url = try XCTUnwrap(Bundle.module.url(forResource: "tokenizer_cases", withExtension: "json",
                                                  subdirectory: "Fixtures"))
        let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: url))
        var failures = 0
        for c in fixture.cases {
            let got = tok.encode(c.text)
            if got != c.ids {
                failures += 1
                if failures <= 10 {
                    XCTFail("\(c.text.debugDescription.prefix(80)): got \(got.prefix(20)), want \(c.ids.prefix(20))")
                }
            }
        }
        XCTAssertEqual(failures, 0, "\(failures) of \(fixture.cases.count) cases differ from Python")
    }

    func testSpecialIds() throws {
        let tok = try loadTokenizer()
        XCTAssertEqual([tok.padId, tok.eosId, tok.bosId, tok.unkId, tok.maskId], [0, 1, 2, 3, 4])
        // Every id survives loading; bridging through [String: Any] loses 61 canonically equivalent keys.
        XCTAssertEqual(tok.vocabSize, 256_000)
    }
}
