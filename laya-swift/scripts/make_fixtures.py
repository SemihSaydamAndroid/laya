"""Write the Python reference outputs the Swift parity tests compare against.

    python laya-swift/scripts/make_fixtures.py [--repo convaiinnovations/laya-multilingual]

Writes laya-swift/Tests/LayaTests/Fixtures/tokenizer_cases.json: texts and the ids that the
checkpoint's tokenizer gives them with add_special_tokens=False, exactly as build_sequence
calls it. Texts: MASSIVE utterances from every language (needs `datasets`), the strings
build_sequence builds from questions and options, and edge cases for whitespace, added
tokens, byte fallback and scripts without spaces.
"""
import argparse
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "Tests", "LayaTests", "Fixtures", "tokenizer_cases.json")

EDGE = [
    "", " ", "  ", "\n", "\t", "a\tb", "a\t\tb", "a\nb", "a\n\n\nb", "a \nb", "a\r\nb", " leading", "trailing ",
    "double  space", "   three", "Hello world", "x <mask> y", "x<mask>y", "<bos>hi<unused3> x", "<eos><eos>",
    "<start_of_turn>user", "[@BOS@] a", "▁already▁metaspace", "▁▁▁", "𠜎𠜎 x", "\x00", "\x07bell", "tab\t\tx",
    "İstanbul'da çay içtik.", "ŞşĞğİıÖöÜüÇç", "日本語のテキストです。", "中文文本没有空格", "한국어 텍스트",
    "สวัสดีครับ", "مرحبا بالعالم", "שלום עולם", "नमस्ते दुनिया", "👍🏽 thumbs", "🇹🇷 bayrak", "👨‍👩‍👧 family",
    "é combining", "ﬁ ligature", "Ａｂｃ fullwidth", "​zero​width", "a nbsp", "12,90 TL",
    "http://kargo-teslimat.co/abc?x=1&y=2", "{\"utterance\": \"wake me up at 5am\"}", "emoji🙂inside", "x" * 300,
    "ab" * 200, "日本" * 150,
]

QUESTION_TEXTS = [
    "choice question: Which department should handle this?",
    "score question: How urgent is this?",
    "noul question: Does the user threaten to cancel or leave?",
    "noul question: Is this message a scam or phishing attempt?",
    " billing: invoices, payments, refunds", " technical: bugs, outages, system errors", " other: everything else",
    " level 0: not urgent", " level 1: soon", " level 2: blocking",
    " false: no, the statement does not hold", " true: yes, the statement holds",
    "choice question: What is the user asking for in `utterance`?", " alarm: set", " iot: hue lightoff",
]


def massive_texts(per_lang):
    from datasets import get_dataset_config_names, load_dataset
    rng = random.Random(7)
    out = []
    for lang in sorted(n for n in get_dataset_config_names("mteb/amazon_massive_intent") if n != "default"):
        rows = load_dataset("mteb/amazon_massive_intent", lang, split="test")
        idx = rng.sample(range(len(rows)), per_lang)
        out += [rows[i]["text"] for i in idx]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default="convaiinnovations/laya-multilingual")
    p.add_argument("--per-lang", type=int, default=20)
    args = p.parse_args()

    from laya import Agent
    tok = Agent(args.repo, device="cpu").tok
    texts = EDGE + QUESTION_TEXTS + massive_texts(args.per_lang)
    cases = [{"text": t, "ids": tok(t, add_special_tokens=False)["input_ids"]} for t in texts]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"repo": args.repo, "cases": cases}, f, ensure_ascii=False, indent=0)
    print("wrote %d cases to %s" % (len(cases), os.path.normpath(OUT)))


if __name__ == "__main__":
    main()
