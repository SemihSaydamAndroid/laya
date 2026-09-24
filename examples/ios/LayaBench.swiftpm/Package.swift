// swift-tools-version: 5.9

// Open this folder in Xcode (File > Open... > LayaBench.swiftpm), pick your iPhone and press Run.
// Models/ is filled by examples/ios/prepare_models.py; see examples/ios/README.md.

import PackageDescription
import AppleProductTypes

let package = Package(
    name: "LayaBench",
    platforms: [
        .iOS("16.0")
    ],
    products: [
        .iOSApplication(
            name: "LayaBench",
            targets: ["AppModule"],
            bundleIdentifier: "io.github.laya.LayaBench",
            displayVersion: "1.0",
            bundleVersion: "1",
            appIcon: .placeholder(icon: .bolt),
            accentColor: .presetColor(.blue),
            supportedDeviceFamilies: [.pad, .phone],
            supportedInterfaceOrientations: [.portrait]
        )
    ],
    dependencies: [
        // ORT 1.24.2 (main at 2026-02-25). The repo's newest tag, v1.19.2, predates the
        // 8-bit MatMulNBits kernel the w8e8 variant needs.
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager",
                 revision: "b7fb7f7dea8a2469e6335d95a61b8f36d0dc83b2")
    ],
    targets: [
        .executableTarget(
            name: "AppModule",
            dependencies: [
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager")
            ],
            path: ".",
            resources: [
                .copy("Models")
            ]
        )
    ]
)
