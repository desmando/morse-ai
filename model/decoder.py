"""CNN-LSTM-CTC acoustic model: spectrogram-band features -> character sequence.

Architecture follows the approach validated by prior CW-decoder work (e.g.
AG1LE's real-time Morse decoder): a small CNN front end picks up local
dot/dash/space patterns in the time-frequency image, a BiLSTM models the
sequence, and CTC loss avoids needing exact per-character alignment.
"""
import csv
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import Dataset

from model.features import extract_features, N_FREQ_BINS
from model.vocab import Vocab
from paths import DATA_ROOT


def load_manifest_rows(manifest_path: str) -> list[dict]:
    with open(manifest_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def split_rows_by_source(rows: list[dict], val_fraction: float, seed: int = 0):
    """Hold out whole source recordings (not individual rows) for validation,
    so augmented variants of the same clip never end up split across train/val."""
    if not val_fraction:
        return rows, []
    sources = sorted(set(r["source"] for r in rows))
    rng = random.Random(seed)
    rng.shuffle(sources)
    n_val = max(1, int(len(sources) * val_fraction))
    val_sources = set(sources[:n_val])
    train_rows = [r for r in rows if r["source"] not in val_sources]
    val_rows = [r for r in rows if r["source"] in val_sources]
    return train_rows, val_rows


class MorseClipDataset(Dataset):
    def __init__(self, rows: list[dict], vocab: Vocab):
        self.vocab = vocab
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        # manifests may have been generated on Windows (backslash separators);
        # forward slashes are valid on both Windows and POSIX, so normalize.
        clip_path = row["clip_path"].replace("\\", "/")
        audio, sr = sf.read(DATA_ROOT / clip_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        features = extract_features(audio, sr)  # (T, F)
        target = self.vocab.encode(row["label"])
        return torch.from_numpy(features), torch.tensor(target, dtype=torch.long)


def collate_batch(batch):
    features, targets = zip(*batch)
    input_lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)

    max_t = int(input_lengths.max())
    feat_dim = features[0].shape[1]
    padded = torch.zeros(len(features), max_t, feat_dim, dtype=torch.float32)
    for i, f in enumerate(features):
        padded[i, : f.shape[0]] = f

    targets_concat = torch.cat(targets) if len(targets) else torch.tensor([], dtype=torch.long)
    return padded, targets_concat, input_lengths, target_lengths


class CWDecoder(nn.Module):
    def __init__(self, n_freq_bins: int = N_FREQ_BINS, vocab_size: int = 64,
                 cnn_channels: int = 32, lstm_hidden: int = 128, lstm_layers: int = 2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # downsample frequency axis only
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        cnn_out_freq = n_freq_bins // 4
        lstm_input_size = cnn_channels * cnn_out_freq

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(lstm_hidden * 2, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, F)
        x = x.unsqueeze(1)  # (batch, 1, T, F)
        x = self.cnn(x)  # (batch, C, T, F')
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)  # (batch, T, C*F')
        x, _ = self.lstm(x)
        logits = self.fc(x)  # (batch, T, vocab_size)
        return logits.log_softmax(dim=-1)


def _log_sum_exp(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def ctc_beam_decode(log_probs: torch.Tensor, vocab: Vocab, lm=None,
                    lm_weight: float = 0.3, beam_width: int = 20) -> str:
    """CTC prefix beam search with optional ham-domain character LM scoring.

    Keeps beam_width candidate prefix sequences alive across all T frames,
    scoring each with acoustic probability + (lm_weight * LM log-probability).
    Handles CTC's same-consecutive-character semantics exactly: emitting the
    same character twice in the output requires a blank between them in the
    CTC path, so extending a prefix with its own last character only advances
    the prefix via a blank path, while a non-blank path just continues keying
    the same character without adding it to the output again.

    log_probs: (T, vocab_size) for ONE sequence (not batched).
    lm: CharNgramLM instance, or None to run as pure acoustic beam search.
    Returns the best-scoring decoded string."""
    import math as _math
    NEG_INF = float("-inf")
    lp = log_probs.float().cpu()
    T = lp.shape[0]

    # beams: {prefix_str: [log_Pb, log_Pnb]}
    # Pb  = log-prob of all CTC paths producing prefix AND ending with blank
    # Pnb = log-prob of all CTC paths producing prefix AND ending with non-blank
    beams: dict[str, list[float]] = {"": [0.0, NEG_INF]}

    for t in range(T):
        lp_t = lp[t]
        new_beams: dict[str, list[float]] = {}

        def _ensure(prefix):
            if prefix not in new_beams:
                new_beams[prefix] = [NEG_INF, NEG_INF]

        for prefix, (log_Pb, log_Pnb) in beams.items():
            log_P = _log_sum_exp(log_Pb, log_Pnb)
            last = prefix[-1] if prefix else None

            # extend with blank — prefix stays the same
            _ensure(prefix)
            new_beams[prefix][0] = _log_sum_exp(
                new_beams[prefix][0], log_P + float(lp_t[0]))

            # extend with each non-blank character
            for c in range(1, lp_t.shape[0]):
                char = vocab.idx_to_char.get(c)
                if not char:
                    continue
                lp_char = float(lp_t[c])
                lm_score = lm.log_prob(prefix, char) * lm_weight if lm is not None else 0.0

                if char == last:
                    # same char: non-blank path stays at current prefix (CTC collapse)
                    _ensure(prefix)
                    new_beams[prefix][1] = _log_sum_exp(
                        new_beams[prefix][1], log_Pnb + lp_char)
                    # blank path does extend the prefix (blank separated the repeat)
                    ext = prefix + char
                    _ensure(ext)
                    new_beams[ext][1] = _log_sum_exp(
                        new_beams[ext][1], log_Pb + lp_char + lm_score)
                else:
                    ext = prefix + char
                    _ensure(ext)
                    new_beams[ext][1] = _log_sum_exp(
                        new_beams[ext][1], log_P + lp_char + lm_score)

        # prune to beam_width
        beams = dict(sorted(new_beams.items(),
                            key=lambda kv: -_log_sum_exp(kv[1][0], kv[1][1]))[:beam_width])

    best = max(beams.items(), key=lambda kv: _log_sum_exp(kv[1][0], kv[1][1]))
    return best[0]


def ctc_greedy_decode(log_probs: torch.Tensor, vocab: Vocab) -> list[str]:
    """log_probs: (batch, T, vocab_size) -> collapse repeats + drop blanks."""
    pred_ids = log_probs.argmax(dim=-1)  # (batch, T)
    results = []
    for seq in pred_ids:
        chars = []
        prev = None
        for idx in seq.tolist():
            if idx != prev and idx != 0:
                chars.append(vocab.idx_to_char.get(idx, ""))
            prev = idx
        results.append("".join(chars))
    return results


def ctc_greedy_decode_with_times(log_probs: torch.Tensor, vocab: Vocab,
                                  hop_seconds: float) -> list[list[tuple[str, float]]]:
    """Like ctc_greedy_decode, but pairs each decoded character with its
    approximate start time (frame index * hop_seconds) within whatever audio
    window log_probs came from. Needed to stitch decodes from overlapping
    sliding windows without dropping or duplicating characters at the seams
    - see decode_stream()."""
    pred_ids = log_probs.argmax(dim=-1)  # (batch, T)
    results = []
    for seq in pred_ids:
        chars = []
        prev = None
        for frame_idx, idx in enumerate(seq.tolist()):
            if idx != prev and idx != 0:
                chars.append((vocab.idx_to_char.get(idx, ""), frame_idx * hop_seconds))
            prev = idx
        results.append(chars)
    return results


def ctc_forced_align(log_probs: torch.Tensor, target_indices: list[int], blank: int = 0):
    """Given one recording's log_probs (T, vocab_size) and its KNOWN correct
    target character sequence (vocab indices, no blanks), finds the
    highest-probability path through the CTC lattice constrained to produce
    exactly that sequence - forced alignment, not free decoding. Returns a
    list of (first_frame, last_frame) per target character (None, None for
    a character forced alignment assigned zero frames to).

    This is what lets dataprep/build_manifest.py's crude proportional
    chars-per-second guess (the source of the mid-word label corruption
    found in the real-ARRL eval) be replaced with exact, model-derived
    timing - the same role dataprep/synthesize_morse_audio.py's char_spans
    play for synthetic audio, just recovered from the model instead of
    known directly from synthesis.

    Standard CTC Viterbi forced alignment: the target is expanded with
    blanks (blank, c1, blank, c2, ..., cL, blank, length 2L+1), and the best
    path through T frames is found via the same transition structure
    CTCLoss uses internally for its forward-backward algorithm, but taking
    the max (Viterbi) instead of the sum, with backpointers to recover the
    actual path instead of just its probability.
    """
    T = log_probs.shape[0]
    L = len(target_indices)
    states = [blank]
    for c in target_indices:
        states.append(c)
        states.append(blank)
    n_states = len(states)

    neg_inf = float("-inf")
    log_probs_list = log_probs.tolist()

    v_prev = [neg_inf] * n_states
    v_prev[0] = log_probs_list[0][states[0]]
    if n_states > 1:
        v_prev[1] = log_probs_list[0][states[1]]

    back = [[0] * n_states for _ in range(T)]

    for t in range(1, T):
        row = log_probs_list[t]
        v_cur = [neg_inf] * n_states
        for s in range(n_states):
            best, choice = v_prev[s], 0
            if s >= 1 and v_prev[s - 1] > best:
                best, choice = v_prev[s - 1], 1
            if s >= 2 and states[s] != states[s - 2] and v_prev[s - 2] > best:
                best, choice = v_prev[s - 2], 2
            back[t][s] = choice
            if best != neg_inf:
                v_cur[s] = best + row[states[s]]
        v_prev = v_cur

    s = (n_states - 1) if v_prev[n_states - 1] >= v_prev[n_states - 2] else (n_states - 2)
    path_states = [0] * T
    for t in range(T - 1, -1, -1):
        path_states[t] = s
        s = max(0, s - back[t][s])

    spans = []
    for i in range(L):
        s_idx = 2 * i + 1
        frames = [t for t in range(T) if path_states[t] == s_idx]
        spans.append((min(frames), max(frames)) if frames else (None, None))
    return spans


def decode_window_core(window_audio, window_abs_start: float, core_start: float, core_end: float,
                        model, vocab: Vocab, device: str, sr: int,
                        lm=None, lm_weight: float = 0.3, beam_width: int = 20) -> str:
    """Decodes one audio window and keeps only the characters whose absolute
    start time (window_abs_start + their offset within the window) falls in
    [core_start, core_end) - the stretch of the timeline this window "owns"
    in an overlapping sliding-window scheme. See decode_stream().

    When lm is provided, uses CTC beam search scored by the ham-domain
    character LM instead of greedy decoding. Timing for the core-region
    filter is recovered via forced alignment of the beam search result back
    to frame positions, so the overlap-trim-stitch logic works identically
    regardless of decode strategy."""
    from model.features import HOP_SAMPLES, extract_features

    hop_seconds = HOP_SAMPLES / sr
    features = extract_features(window_audio, sr)
    with torch.no_grad():
        x = torch.from_numpy(features).unsqueeze(0).to(device)
        log_probs = model(x)
    log_probs_cpu = log_probs[0].cpu()  # (T, vocab_size)

    if lm is not None:
        # Beam search for best text, forced-align back to frames for trimming
        text = ctc_beam_decode(log_probs_cpu, vocab, lm=lm,
                                lm_weight=lm_weight, beam_width=beam_width)
        if not text:
            return ""
        target_indices = [vocab.char_to_idx[c] for c in text if c in vocab.char_to_idx]
        if not target_indices:
            return ""
        spans = ctc_forced_align(log_probs_cpu, target_indices)
        chars_with_time = []
        char_idx = 0
        for c in text:
            if c not in vocab.char_to_idx:
                continue
            f0, f1 = spans[char_idx]
            t = f0 * hop_seconds if f0 is not None else 0.0
            chars_with_time.append((c, t))
            char_idx += 1
    else:
        chars_with_time = ctc_greedy_decode_with_times(
            log_probs_cpu.unsqueeze(0), vocab, hop_seconds)[0]

    return "".join(ch for ch, t in chars_with_time
                    if core_start <= window_abs_start + t < core_end)


def decode_stream(audio, sr: int, model, vocab: Vocab, device: str,
                   window_seconds: float = 8.0, stride_seconds: float = 4.0,
                   lm=None, lm_weight: float = 0.3, beam_width: int = 20) -> str:
    """Decodes a long, continuous recording as one piece of text, without the
    boundary-chopping bug fixed in dataprep/synthesize_morse_audio.py and
    inference/realtime_decode.py's original non-overlapping-window design:
    independently decoding fixed windows corrupts whatever character straddles
    each window edge, since the model has no context on the other side of the
    cut.

    Instead, overlapping windows are decoded and only each window's "core"
    region - the middle stretch not near either edge - is kept; consecutive
    windows' core regions tile exactly with stride_seconds = window_seconds/2,
    so every moment of audio is covered by exactly one window's core (except
    the very start/end of the whole recording, which only one window can ever
    see at all and so is taken in full from the first/last window). This
    needs the model to have decent context on both sides of a character to
    decode it well in the first place - a guarantee a non-overlapping chunked
    design never had.
    """
    n_samples = len(audio)
    window_samples = int(window_seconds * sr)
    stride_samples = int(stride_seconds * sr)
    guard_seconds = (window_seconds - stride_seconds) / 2
    total_seconds = n_samples / sr

    pieces = []
    window_start_sample = 0
    while window_start_sample < n_samples:
        window_end_sample = min(window_start_sample + window_samples, n_samples)
        is_first = window_start_sample == 0
        is_last = window_end_sample >= n_samples
        window_abs_start = window_start_sample / sr

        core_start = 0.0 if is_first else window_abs_start + guard_seconds
        core_end = total_seconds if is_last else window_abs_start + guard_seconds + stride_seconds

        pieces.append(decode_window_core(audio[window_start_sample:window_end_sample], window_abs_start,
                                          core_start, core_end, model, vocab, device, sr,
                                          lm=lm, lm_weight=lm_weight, beam_width=beam_width))

        if is_last:
            break
        window_start_sample += stride_samples

    return "".join(pieces)


def edit_distance(a, b) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def cer(pred: str, ref: str) -> float:
    if not ref:
        return 0.0 if not pred else 1.0
    return edit_distance(list(pred), list(ref)) / len(ref)


def wer(pred: str, ref: str) -> float:
    ref_words = ref.split()
    if not ref_words:
        return 0.0 if not pred.split() else 1.0
    return edit_distance(pred.split(), ref_words) / len(ref_words)
