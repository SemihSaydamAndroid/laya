# Laya on iPhone: LayaBench

A small iOS app that runs `laya-multilingual` on the phone with ONNX Runtime and reports
latency, memory, and whether the phone's answers match the desktop's. Everything runs
on the device; nothing is sent anywhere.

It is a benchmark, not an SDK: inputs are tokenized ahead of time by `prepare_models.py`,
so the app needs no tokenizer.

## Model variants

Measured on 1,000 MASSIVE intent cases (10 languages x 100, 20 options each, the
`research/eval/laya_eval.py` suite), CPU, ONNX Runtime 1.30. The reference is the fp32 ONNX
export, which matches PyTorch within 2e-6.

| variant | files (encoder + head) | accuracy | same answer as fp32 | mean abs change in p(gold) |
|---|---|---|---|---|
| fp32 | 1,288 MB | 0.520 | 100% | 0 |
| fp16 | 676 MB | 0.520 | 99.2% | 0.004 |
| **w8e8** | **384 MB** | **0.517** | **97.6%** | **0.012** |
| int8, `quantize_dynamic` (not offered) | 369 MB | 0.470 | 61.5% | 0.164 |

- **w8e8** is 8-bit weight-only `MatMulNBits` (activations stay float) plus a per-row int8
  embedding table. The 256k-token embedding is about 60% of the parameters.
- Plain dynamic int8 also quantizes activations. On this encoder that changes more than a
  third of the answers, so the script does not produce it.
- On a 4-core x86 CPU, w8e8 was slower than fp32 for one request (424 ms vs 287 ms) because
  its weights are unpacked on every call. Phone speed is what this app measures.

## Run it

You need a Mac with Xcode 15 or newer, Python 3.10 or newer, and an iPhone on iOS 16 or newer.

```bash
git clone https://github.com/NandhaKishorM/laya && cd laya
python -m pip install -e . -r examples/ios/requirements.txt
python examples/ios/prepare_models.py            # fp16 + w8e8; add --variants fp32,fp16,w8e8 for fp32
open examples/ios/LayaBench.swiftpm
```

`prepare_models.py` downloads the checkpoint (about 650 MB), exports it with
`laya-ts/scripts/export_onnx.py`, and writes the variants and `bench_inputs.json` into
`LayaBench.swiftpm/Models/`, which git ignores. It takes about five minutes.

Then, in Xcode:

1. Wait for the `onnxruntime` package to resolve.
2. Select the LayaBench target. Under **Signing & Capabilities**, choose your team. A free
   Apple ID ("Personal Team") works.
3. Connect the iPhone, select it as the run destination, and press **Run**.
   - The first time, turn on **Settings > Privacy & Security > Developer Mode** on the phone.
   - Then trust the developer under **Settings > General > VPN & Device Management**.
4. Tap **Run benchmark**, keep the phone unlocked, and use **Share report** to send the JSON.

The report lists the device model, the iOS version, and, for each variant on CPU and on
CoreML:

- load time
- memory footprint
- median latency for short (<256 tokens) and long (~600 tokens) inputs
- the largest probability difference from the desktop run

## Notes

- The head's `marker_mask` input is `uint8` in these files (`marker_mask_u8`), because the
  ONNX Runtime Objective-C API has no bool tensor type.
- The CoreML execution provider does not run `MatMulNBits`, so w8e8 on CoreML runs partly on
  the CPU. The report shows whether that helps or hurts.
- fp32 needs about 1.3 GB of memory for its weights alone and may be killed by iOS on phones
  with 4 GB of RAM.
