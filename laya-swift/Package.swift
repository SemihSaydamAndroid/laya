// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "Laya",
    platforms: [.iOS(.v16), .macOS(.v13)],
    products: [
        .library(name: "Laya", targets: ["Laya"]),
    ],
    targets: [
        .target(name: "Laya"),
        .testTarget(
            name: "LayaTests",
            dependencies: ["Laya"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
