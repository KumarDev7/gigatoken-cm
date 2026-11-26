"""coder-mini tokenizer: gigatoken (coder_mini scheme) vs HuggingFace tokenizers, ids must be identical.

    python parity.py <tokenizer.json> <docs.jsonl> [adversarial]

docs.jsonl: one {"text": ...} per line (real documents decoded from the coder-mini training stream). Prints the
mismatch count (with the first few diffs) and the speed of both engines on the same documents.
"""
import json
import sys
import time

import gigatoken as gt
from tokenizers import Tokenizer

tok_path, docs_path = sys.argv[1], sys.argv[2]
hf = Tokenizer.from_file(tok_path)
fast = gt.Tokenizer(tok_path)
docs = [json.loads(l)["text"] for l in open(docs_path)]
ADV = [
    "", " ", "\n", "\r\n\r\n", "\t\t x", "a\u00a0b", "x\u2028y", "1234567890", "٣٤٥", "½Ⅷ",
    "क्\u200dष", "क्\u200cष परीक्षा", " \u200dabc", "!\u200d", "\u200d", "\u200c\u200c", "a\u200d́b",
    "देवनागरी में परीक्षण", "अंग्रेज़ी", "हिन्दी 123 abc", "it's IT'S they'LL we'Ve", "'ſ", "it'ſ",
    "<|file|>src/a.py\nx = 1\n", "<|repo|>org/repo\n<|file|>a.md\n# t", "<|endoftext|>", "a<|endoftext|>b",
    "def f(x):\n    return x  # ok\n\n\n", "    \n\t  \n", "emoji 🚀🧪 é (e + ́) العربية", "x" * 5000,
    "日本語のテキスト 中文 한국어", "\x00\x01\x7f", "\ufeffbom", "é́", "!!́x", "́́two",
]
docs = docs + ADV * (1 if len(sys.argv) < 4 else 50)

t0 = time.time()
ref = [e.ids for e in hf.encode_batch(docs, add_special_tokens=False)]
t_hf = time.time() - t0
t0 = time.time()
got = [list(map(int, x)) for x in fast.encode_batch_list(docs)]
t_gt = time.time() - t0

bad = [i for i, (a, b) in enumerate(zip(ref, got)) if a != b]
ntok = sum(map(len, ref))
nbytes = sum(len(d.encode()) for d in docs)
print(json.dumps({"docs": len(docs), "tokens": ntok, "MB": round(nbytes / 1e6, 1), "mismatches": len(bad),
                  "hf_s": round(t_hf, 2), "gigatoken_s": round(t_gt, 2), "speedup": round(t_hf / max(t_gt, 1e-9), 1),
                  "hf_Mtok_s": round(ntok / t_hf / 1e6, 2), "gt_Mtok_s": round(ntok / max(t_gt, 1e-9) / 1e6, 2)}))
for i in bad[:5]:
    a, b = ref[i], got[i]
    k = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
    print("MISMATCH doc", i, "at token", k, "hf:", [hf.id_to_token(x) for x in a[max(0, k - 3):k + 4]],
          "gt:", [hf.id_to_token(x) for x in b[max(0, k - 3):k + 4]], "text:", repr(docs[i][:200]))
sys.exit(1 if bad else 0)
