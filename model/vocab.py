"""Character vocabulary for CTC: index 0 is reserved for the CTC blank token."""
from pathlib import Path

BLANK_IDX = 0

DEFAULT_CHARS = list(" ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,?/=-'\"()<>")


class Vocab:
    def __init__(self, chars: list[str]):
        self.chars = chars
        self.char_to_idx = {c: i + 1 for i, c in enumerate(chars)}  # +1: 0 is blank
        self.idx_to_char = {i + 1: c for i, c in enumerate(chars)}

    @classmethod
    def from_file(cls, path: str | Path) -> "Vocab":
        path = Path(path)
        if not path.exists():
            return cls(DEFAULT_CHARS)
        chars = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.rstrip("\n")]
        # vocab.txt strips trailing newline chars; the literal space line survives as "" after rstrip("\n")
        # but splitlines() already drops it if it was just "\n" with no content -> re-add space explicitly if missing
        if " " not in chars:
            chars = [" "] + chars
        return cls(chars)

    def __len__(self) -> int:
        return len(self.chars) + 1  # + blank

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices) -> str:
        return "".join(self.idx_to_char.get(int(i), "") for i in indices)
