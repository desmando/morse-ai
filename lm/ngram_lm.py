"""Character n-gram language model for CTC beam search rescoring.

Trained on the QSO corpus (contest/contact exchanges, prosigns, Q-codes) plus
real ARRL W1AW transcripts. Uppercase-only to match the acoustic model's vocab.

The LM scores how plausible each next character is given the last (order-1)
characters of the decoded prefix so far. Combined with CTC beam search this
lets the decoder prefer sequences that look like real ham radio text
(valid callsign prefixes, known Q-codes, ARRL section abbreviations, RST
values) over acoustically similar but linguistically implausible alternatives.

Usage:
  python lm/ngram_lm.py --build              # train from local data, save
  python lm/ngram_lm.py --test "CQ TEST DE"  # show next-char probabilities
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataprep.build_manifest import clean_transcript
from paths import DATA_ROOT

DEFAULT_LM_PATH = DATA_ROOT / "models" / "ham_char_lm.json"
ORDER = 4           # 4-gram: conditions on last 3 characters
SMOOTHING = 0.1     # add-k smoothing; small k keeps counts dominant

# Characters the acoustic model can produce — restrict LM to the same set
# so we never score characters the decoder can't actually emit.
CW_CHARS = set(" ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?/=-'\"()<>")


def _preprocess(text: str) -> str:
    return "".join(c for c in text.upper() if c in CW_CHARS)


def _context(prefix: str, order: int) -> str:
    pad = "\x00" * (order - 1)
    full = pad + prefix
    return full[-(order - 1):]


class CharNgramLM:
    """Smoothed character n-gram language model.

    Stores raw counts during training, computes log-probabilities on demand.
    Serializes to JSON so it only needs to be built once."""

    def __init__(self, order: int = ORDER, smoothing: float = SMOOTHING):
        self.order = order
        self.smoothing = smoothing
        self._counts: dict[str, dict[str, float]] = {}
        self._vocab: set[str] = set()

    def train(self, texts):
        for text in texts:
            text = _preprocess(text)
            if not text:
                continue
            padded = "\x00" * (self.order - 1) + text
            for i, char in enumerate(text):
                ctx = padded[i: i + self.order - 1]
                self._vocab.add(char)
                if ctx not in self._counts:
                    self._counts[ctx] = {}
                self._counts[ctx][char] = self._counts[ctx].get(char, 0) + 1

    def log_prob(self, prefix: str, char: str) -> float:
        """Log probability of char given the tail of prefix."""
        if char not in CW_CHARS:
            return math.log(self.smoothing / (self.smoothing * len(CW_CHARS) + 1))
        ctx = _context(prefix, self.order)
        counts = self._counts.get(ctx, {})
        total = sum(counts.values()) + self.smoothing * len(self._vocab)
        char_count = counts.get(char, 0) + self.smoothing
        return math.log(char_count / total)

    def top_next(self, prefix: str, k: int = 10) -> list[tuple[str, float]]:
        """Return the k most probable next characters given prefix."""
        scored = [(c, self.log_prob(prefix, c)) for c in CW_CHARS if c != "\x00"]
        return sorted(scored, key=lambda x: -x[1])[:k]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "order": self.order,
            "smoothing": self.smoothing,
            "vocab": sorted(self._vocab),
            "counts": {ctx: counts for ctx, counts in self._counts.items()},
        }
        path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        print(f"saved LM to {path} ({len(self._counts)} contexts, {len(self._vocab)} chars)")

    @classmethod
    def load(cls, path=None) -> "CharNgramLM":
        path = Path(path or DEFAULT_LM_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"LM not found at {path} — run: python lm/ngram_lm.py --build")
        data = json.loads(path.read_text(encoding="utf-8"))
        lm = cls(order=data["order"], smoothing=data["smoothing"])
        lm._vocab = set(data["vocab"])
        lm._counts = data["counts"]
        return lm


def _iter_training_texts():
    """Yields text strings from all available local training sources."""
    # QSO corpus — primary source: contest/contact exchanges, prosigns, Q-codes
    corpus_path = DATA_ROOT / "text_corpus" / "qso_corpus.txt"
    if corpus_path.exists():
        n = 0
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                yield line
                n += 1
        print(f"  QSO corpus: {n} lines")
    else:
        print(f"  warning: QSO corpus not found at {corpus_path}")

    # ARRL W1AW transcripts — real on-air CW text (bulletins, code practice)
    arrl_dir = DATA_ROOT / "raw" / "arrl"
    if arrl_dir.exists():
        n = 0
        for txt_path in sorted(arrl_dir.rglob("*.txt")):
            try:
                raw = txt_path.read_text(encoding="utf-8", errors="replace")
                text = clean_transcript(raw)
                if text:
                    yield text
                    n += 1
            except Exception:
                pass
        print(f"  ARRL transcripts: {n} files")
    else:
        print(f"  warning: ARRL transcripts not found at {arrl_dir}")


def build(out_path=None, order=ORDER):
    print(f"building {order}-gram character LM from local training data ...")
    lm = CharNgramLM(order=order)
    lm.train(_iter_training_texts())
    lm.save(out_path or DEFAULT_LM_PATH)
    return lm


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", action="store_true", help="train from local data and save")
    parser.add_argument("--out", default=str(DEFAULT_LM_PATH))
    parser.add_argument("--order", type=int, default=ORDER)
    parser.add_argument("--test", default=None, metavar="PREFIX",
                         help="show top next-character predictions given this prefix")
    args = parser.parse_args()

    if args.build:
        lm = build(args.out, args.order)
    else:
        lm = CharNgramLM.load(args.out)

    if args.test is not None:
        prefix = args.test
        print(f"\ntop next chars after {prefix!r}:")
        for char, lp in lm.top_next(prefix, k=15):
            print(f"  {char!r}: {math.exp(lp):.4f}")


if __name__ == "__main__":
    main()
