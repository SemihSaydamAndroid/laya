"""Build the ONNX variants and inputs the LayaBench iOS app measures.

Run on the Mac that builds the app (the files are too large for git):

    pip install -r examples/ios/requirements.txt
    python examples/ios/prepare_models.py                     # fp16 + w8e8
    python examples/ios/prepare_models.py --variants fp32,fp16,w8e8

Writes ``examples/ios/LayaBench.swiftpm/Models/<variant>/{encoder,head}.onnx``
plus ``Models/bench_inputs.json``: pre-tokenized cases (so the app needs no
tokenizer) with the probabilities each variant gives on this machine, which the
app compares against what the phone computes.

Variants:
  fp32  the exported graph as is (~1.29 GB; may not fit in memory on older phones)
  fp16  fp16 encoder weights and compute, fp32 head (~676 MB)
  w8e8  8-bit weight-only MatMul (MatMulNBits, compute stays float) and a
        per-row int8 embedding table, fp32 head (~384 MB)

Plain dynamic int8 (quantize_dynamic) is deliberately not offered: it also
quantizes activations and changed 38% of answers on a 10-language MASSIVE run.
"""
import argparse
import copy
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUT = os.path.join(HERE, "LayaBench.swiftpm", "Models")
DEFAULT_WORK = os.path.join(HERE, ".work")

LONG_TR = (
    "Merhaba, geçen hafta sitenizden bir çamaşır makinesi sipariş ettim ve ürün iki gün gecikmeyle "
    "teslim edildi. Kurulum için gelen teknik servis cihazın kapağında bir çatlak olduğunu, bu yüzden "
    "kurulumu yapamayacağını söyledi. Müşteri hizmetlerini üç kez aradım, her seferinde farklı bir "
    "kişiyle görüştüm ve her biri bana başka bir şey söyledi. Biri değişim yapılacağını, diğeri önce "
    "fotoğraf göndermem gerektiğini, üçüncüsü ise iade sürecinin on beş iş günü süreceğini söyledi. "
    "Fotoğrafları e-posta ile gönderdim ama hiçbir yanıt alamadım. Şu anda evde çalışmayan, kutusu "
    "açılmış bir makine duruyor ve çamaşırlarımı yıkayamıyorum. Kredi kartımdan tutarın tamamı çekildi. "
) * 3

CASES = [
    ("tr_scam_sms",
     "Sayın müşterimiz, kargonuz adres eksikliği nedeniyle teslim edilemedi. 24 saat içinde "
     "http://kargo-teslimat.co adresinden 12,90 TL ödeme yapmazsanız paketiniz iade edilecektir.",
     {"scam": {"type": "noul", "instructions": "Is this message a scam or phishing attempt?"},
      "urgency": {"type": "score", "instructions": "How much pressure does the message put on the reader to act now?",
                  "criteria": ["none", "some", "strong"]}}),
    ("tr_support",
     "Merhaba, mart ayında iki kez ücret alındı. Fazla ödemeyi bugün iade etmezseniz aboneliğimizi iptal edeceğiz.",
     {"department": {"type": "choice", "instructions": "Which department should handle this?",
                     "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                                  "other": "everything else"}},
      "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"}}),
    ("en_support",
     "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
     {"department": {"type": "choice", "instructions": "Which department should handle this?",
                     "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                                  "other": "everything else"}}}),
    ("es_bug",
     "La aplicación se cierra cada vez que abro la configuración.",
     {"department": {"type": "choice", "instructions": "Which department should handle this?",
                     "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, system errors",
                                  "other": "everything else"}}}),
    ("tr_long_complaint", LONG_TR,
     {"intent": {"type": "choice", "instructions": "What does the customer want?",
                 "criteria": {"refund": "money back", "replacement": "a new or exchanged product",
                              "information": "an answer or status update", "other": "anything else"}},
      "anger": {"type": "score", "instructions": "How angry is the customer?",
                "criteria": ["calm", "annoyed", "angry", "furious"]}}),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="convaiinnovations/laya-multilingual")
    p.add_argument("--variants", default="fp16,w8e8", help="comma list of fp32, fp16, w8e8 (default fp16,w8e8)")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--work", default=DEFAULT_WORK, help="scratch dir for the fp32 export (kept for re-runs)")
    return p.parse_args(argv)


def export_fp32(repo, work):
    d = os.path.join(work, "fp32")
    if not os.path.exists(os.path.join(d, "encoder.onnx")):
        subprocess.check_call([sys.executable, os.path.join(ROOT, "laya-ts", "scripts", "export_onnx.py"),
                               "--repo", repo, "--out-dir", d])
    return d


def patch_head(head):
    """marker_mask becomes a uint8 input: the ONNX Runtime Objective-C API has no bool tensor type."""
    from onnx import TensorProto, helper
    g = head.graph
    inp = next(i for i in g.input if i.name == "marker_mask")
    inp.name = "marker_mask_u8"
    inp.type.tensor_type.elem_type = TensorProto.UINT8
    g.node.insert(0, helper.make_node("Cast", ["marker_mask_u8"], ["marker_mask"], to=TensorProto.BOOL))
    return head


def to_fp16(enc):
    from onnx import TensorProto
    from onnxconverter_common import float16
    # The converter edits its input in place; later variants still need the fp32 graph.
    m = float16.convert_float_to_float16(copy.deepcopy(enc), keep_io_types=True, disable_shape_infer=True)
    # The dynamo export carries Cast(to=float) nodes whose outputs the converter retyped to
    # fp16; ORT then refuses the graph. Point those casts at fp16.
    types = {v.name: v.type.tensor_type.elem_type for v in list(m.graph.value_info) + list(m.graph.output)}
    for node in m.graph.node:
        if node.op_type != "Cast":
            continue
        for a in node.attribute:
            if a.name == "to" and a.i == TensorProto.FLOAT and types.get(node.output[0]) == TensorProto.FLOAT16:
                a.i = TensorProto.FLOAT16
    return m


def to_w8e8(enc):
    from onnx import TensorProto, helper, numpy_helper
    from onnxruntime.quantization import QuantFormat
    from onnxruntime.quantization import matmul_nbits_quantizer as mq

    cfg = mq.DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=8, quant_format=QuantFormat.QOperator,
                                          op_types_to_quantize=("MatMul",), quant_axes=(("MatMul", 0),))
    q = mq.MatMulNBitsQuantizer(copy.deepcopy(enc), bits=8, block_size=32, is_symmetric=True, algo_config=cfg,
                                op_types_to_quantize=("MatMul",), quant_axes=(("MatMul", 0),))
    q.process()
    m = q.model.model
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    node = next(n for n in g.node if n.op_type == "Gather" and n.input[0] in inits
                and len(inits[n.input[0]].dims) == 2 and inits[n.input[0]].dims[0] > 100000)
    table = inits[node.input[0]]
    emb = numpy_helper.to_array(table).astype(np.float32)
    scale = np.maximum(np.abs(emb).max(1, keepdims=True), 1e-8) / 127.0
    qemb = np.clip(np.round(emb / scale), -127, 127).astype(np.int8)
    g.initializer.remove(table)
    g.initializer.extend([numpy_helper.from_array(qemb, "emb_q8"), numpy_helper.from_array(scale.astype(np.float32), "emb_s8")])
    ids, out = node.input[1], node.output[0]
    at = list(g.node).index(node)
    g.node.remove(node)
    for k, n in enumerate([
        helper.make_node("Gather", ["emb_q8", ids], ["emb_q8_g"], axis=0),
        helper.make_node("Cast", ["emb_q8_g"], ["emb_q8_f"], to=TensorProto.FLOAT),
        helper.make_node("Gather", ["emb_s8", ids], ["emb_s8_g"], axis=0),
        helper.make_node("Mul", ["emb_q8_f", "emb_s8_g"], [out]),
    ]):
        g.node.insert(at + k, n)
    return m


def build_items(repo):
    from laya import Agent
    from laya.common import QTYPES, build_sequence, render_options

    agent = Agent(repo, device="cpu")
    max_len, head_max_len = agent.cfg.get("max_len", 512), agent.cfg.get("head_max_len", 192)
    items = []
    for name, state, questions in CASES:
        answers = agent.predict(state, questions)["answers"]
        for qid, qdef in questions.items():
            q = {"t": qdef["type"], "ins": qdef["instructions"], "crit": qdef.get("criteria")}
            ids, markers = build_sequence(agent.tok, state, q, max_len, head_max_len)
            assert len(markers) == len(render_options(q))
            crit = qdef.get("criteria")
            labels = list(crit) if isinstance(crit, dict) else (list(crit) if crit else ["no", "yes"])
            items.append({"name": "%s/%s" % (name, qid), "text": state[:120], "type": qdef["type"],
                          "labels": labels, "input_ids": [int(x) for x in ids],
                          "marker_pos": [int(x) for x in markers], "qtype": int(QTYPES[q["t"]]),
                          "torch_answer": answers[qid]})
    return items


def run_variant(d, items):
    import onnxruntime as ort
    enc = ort.InferenceSession(os.path.join(d, "encoder.onnx"), providers=["CPUExecutionProvider"])
    head = ort.InferenceSession(os.path.join(d, "head.onnx"), providers=["CPUExecutionProvider"])
    out = []
    for it in items:
        ids = np.array([it["input_ids"]], dtype=np.int64)
        att = np.ones_like(ids)
        (h,) = enc.run(None, {"input_ids": ids, "attention_mask": att})
        k = len(it["marker_pos"])
        logits, _ = head.run(None, {"hidden_states": h.astype(np.float32),
                                    "marker_pos": np.array([it["marker_pos"]], dtype=np.int64),
                                    "marker_mask_u8": np.ones((1, k), dtype=np.uint8),
                                    "qtype": np.array([[it["qtype"]]], dtype=np.int64),
                                    "attention_mask": att})
        z = logits[0, :k].astype(np.float64)
        p = np.exp(z - z.max())
        out.append([round(float(x), 6) for x in p / p.sum()])
    return out


def main(argv=None):
    args = parse_args(argv)
    import onnx

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    bad = set(variants) - {"fp32", "fp16", "w8e8"}
    if bad:
        raise SystemExit("unknown variant(s): %s" % ", ".join(sorted(bad)))
    src = export_fp32(args.repo, args.work)
    head = patch_head(onnx.load(os.path.join(src, "head.onnx")))
    enc = onnx.load(os.path.join(src, "encoder.onnx"))

    items = build_items(args.repo)
    for v in variants:
        d = os.path.join(args.out, v)
        os.makedirs(d, exist_ok=True)
        m = enc if v == "fp32" else to_fp16(enc) if v == "fp16" else to_w8e8(enc)
        onnx.save(m, os.path.join(d, "encoder.onnx"))
        onnx.save(head, os.path.join(d, "head.onnx"))
        probs = run_variant(d, items)
        for it, p in zip(items, probs):
            it.setdefault("expected", {})[v] = p
        size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
        print("%-5s %7.1f MB  %s" % (v, size / 1e6, d), flush=True)

    with open(os.path.join(args.out, "bench_inputs.json"), "w", encoding="utf-8") as f:
        json.dump({"repo": args.repo, "variants": variants, "items": items}, f, ensure_ascii=False)
    print("wrote %d items to %s" % (len(items), os.path.join(args.out, "bench_inputs.json")))


if __name__ == "__main__":
    main()
