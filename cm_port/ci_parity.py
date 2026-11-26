"""CI check of the coder_mini scheme without the real vocabulary: train a small byte-level BPE with the same
pre_tokenizer (the coder-mini Split regex + ByteLevel), then require gigatoken to give exactly HuggingFace's ids on a
mixed corpus (code, English, numbers, Devanagari with ZWJ/ZWNJ, marks, other scripts, special tokens)."""
import json
import os
import random
import sys
import tempfile

import gigatoken as gt
from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

ZW = "\u200c\u200d"
REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
         rf"|[^\r\n\p{{L}}\p{{M}}\p{{N}}{ZW}]?[\p{{L}}\p{{M}}{ZW}]+"
         r"|\p{N}"
         rf"| ?[^\s\p{{L}}\p{{M}}\p{{N}}{ZW}]+[\r\n]*"
         r"|\s*[\r\n]+"
         r"|\s+(?!\S)"
         r"|\s+")
SPECIAL = ["<|endoftext|>", "<|file|>", "<|repo|>", "<|user|>", "<|assistant|>"]

rng = random.Random(0)
words = ["def", "return", "self", "import", "numpy", "x", "value", "the", "model", "data", "क्\u200dष", "परीक्षा",
         "हिन्दी", "अंग्रेज़ी", "café", "naïve", "日本語", "한국어", "العربية", "é", "क्\u200cष", "it's", "THEY'LL"]
seps = [" ", "  ", "\n", "\n\n", "\t", "\r\n", ", ", ". ", "(", ")", " = ", "!\u200d", " 123 ", "4567", "\u00a0", "    "]


def doc():
    return "".join(rng.choice(words) + rng.choice(seps) for _ in range(rng.randrange(5, 400)))


train = [doc() for _ in range(3000)]
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(REGEX), behavior="isolated"),
                                             pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
tok.decoder = decoders.ByteLevel()
tok.train_from_iterator(train, trainers.BpeTrainer(vocab_size=2000, special_tokens=SPECIAL,
                                                   initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
path = os.path.join(tempfile.mkdtemp(), "tokenizer.json")
tok.save(path)

fast = gt.Tokenizer(path)
test = [doc() for _ in range(5000)] + ["", "\u200d", "\u200c\u200c", " \u200dabc", "<|file|>a.py\nx=1\n",
                                       "a<|endoftext|>b", "\ufeffbom", "x" * 3000, "é́", "!!́x"]
ref = [e.ids for e in tok.encode_batch(test, add_special_tokens=False)]
got = [list(map(int, x)) for x in fast.encode_batch_list(test)]
bad = [i for i in range(len(test)) if ref[i] != got[i]]
print(json.dumps({"docs": len(test), "tokens": sum(map(len, ref)), "mismatches": len(bad)}))
for i in bad[:5]:
    print("MISMATCH", repr(test[i][:200]))
sys.exit(1 if bad else 0)
