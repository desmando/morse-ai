"""CNN-LSTM-CTC acoustic model: spectrogram-band features -> character sequence.

Architecture follows the approach validated by prior CW-decoder work (e.g.
AG1LE's real-time Morse decoder): a small CNN front end picks up local
dot/dash/space patterns in the time-frequency image, a BiLSTM models the
sequence, and CTC loss avoids needing exact per-character alignment.

The model's input contract (feature hop, CNN time-downsampling factor,
layer sizes, dropout) is recorded in every checkpoint as "model_config" and
must be reconstructed from there at load time - use load_checkpoint_model().
Checkpoints from before this was recorded get LEGACY_MODEL_CONFIG (hop 112,
no time-downsampling, no dropout), which matches how they were trained.
"""
import csv
import math
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset

from dataprep.augment_hf_channel import augment_clip
from model.features import (DEFAULT_HOP_SAMPLES, LEGACY_HOP_SAMPLES, N_FREQ_BINS, SAMPLE_RATE,
                             ToneTracker, cw_activity_db, detect_tone_freq, extract_features,
                             resample_to_model_rate)
from model.vocab import Vocab
from paths import DATA_ROOT

# Everything an inference/eval script needs to rebuild the exact model+feature
# pipeline a checkpoint was trained with. Saved into checkpoints by train.py.
LEGACY_MODEL_CONFIG = dict(n_freq_bins=N_FREQ_BINS, cnn_channels=32, lstm_hidden=128,
                            lstm_layers=2, cnn_groups=8, dropout=0.0,
                            time_stride=1, hop_samples=LEGACY_HOP_SAMPLES)
DEFAULT_MODEL_CONFIG = dict(n_freq_bins=N_FREQ_BINS, cnn_channels=32, lstm_hidden=128,
                             lstm_layers=2, cnn_groups=8, dropout=0.2,
                             time_stride=2, hop_samples=DEFAULT_HOP_SAMPLES)


def model_config_from_checkpoint(ckpt: dict) -> dict:
    """Config recorded in the checkpoint, else the legacy recipe (which is
    what every checkpoint from before configs were recorded was trained as)."""
    return {**LEGACY_MODEL_CONFIG, **(ckpt.get("model_config") or {})}


def load_checkpoint_model(checkpoint_path, device: str, vocab_path=None):
    """Build a CWDecoder matching the checkpoint's recorded config and load
    its weights. Returns (model, vocab, ckpt_dict); config is on model.config.

    The vocab comes from the checkpoint's own recorded vocab_chars (every
    checkpoint ever produced by train.py records them), NOT from whatever
    vocab.txt happens to be on this machine - so extending the vocab file for
    a new training run (e.g. adding '<'/'>' for prosigns) can never silently
    remap or break an older checkpoint's character indices. vocab_path is
    only the fallback for checkpoints missing vocab_chars."""
    # weights_only=False explicitly: checkpoints carry optimizer/scheduler
    # state beyond plain tensors (PyTorch >= 2.6 flips the default), and
    # they're this project's own files
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if ckpt.get("vocab_chars"):
        vocab = Vocab(list(ckpt["vocab_chars"]))
    elif vocab_path is not None:
        vocab = Vocab.from_file(vocab_path)
    else:
        raise ValueError(f"{checkpoint_path} has no recorded vocab_chars and no vocab_path given")
    config = model_config_from_checkpoint(ckpt)
    model = CWDecoder(vocab_size=len(vocab), **config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab, ckpt


def load_manifest_rows(manifest_path: str) -> list[dict]:
    # utf-8-sig: transparently strips a UTF-8 BOM if present (Windows tools
    # like PowerShell's Set-Content add one, which would otherwise corrupt
    # the first column name into '﻿clip_path'); no-op for plain UTF-8
    with open(manifest_path, newline="", encoding="utf-8-sig") as f:
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
    """Clip audio -> (features, target) pairs.

    augment=None:     clean clips, exactly as stored (legacy behavior).
    augment="random": fresh random HF impairments every access - the model
                      never sees the same noise twice, and set_augmentation()
                      lets train.py anneal strength/SNR smoothly per epoch
                      (a continuous curriculum, replacing the pre-rendered
                      noise-ramp manifests whose abrupt ratio jumps caused
                      repeated training collapses - see CLOUD_TRAINING.md).
    augment="fixed":  deterministic per-clip impairments (seeded by clip
                      index) at full strength - a stationary noisy set, so
                      val_loss stays comparable across epochs while the
                      training distribution moves underneath it.

    If a manifest row has a tone_hz column (synthetic data does), feature
    extraction centers on that known frequency - corrected by whatever drift
    augmentation applied, plus a small random offset emulating realistic
    detector error - instead of re-detecting the tone from noisy audio, where
    a detector mistake (e.g. locking onto QRM) silently mislabels the clip.
    """

    def __init__(self, rows: list[dict], vocab: Vocab, hop_samples: int = DEFAULT_HOP_SAMPLES,
                 augment: str | None = None, snr_db_range=(-3.0, 20.0),
                 aug_strength: float = 1.0, aug_seed: int = 0):
        assert augment in (None, "random", "fixed")
        self.vocab = vocab
        self.rows = rows
        self.hop_samples = hop_samples
        self.augment = augment
        self.snr_db_range = tuple(snr_db_range)
        self.aug_strength = aug_strength
        self.aug_seed = aug_seed
        self._worker_rng = None

    def set_augmentation(self, strength: float, snr_db_range):
        """Called by train.py between epochs to anneal the curriculum. Takes
        effect for the next epoch's DataLoader workers (they snapshot the
        dataset when the epoch's iterator is created)."""
        self.aug_strength = strength
        self.snr_db_range = tuple(snr_db_range)

    def _rng_for(self, idx: int) -> np.random.Generator:
        if self.augment == "fixed":
            return np.random.default_rng((self.aug_seed, idx))
        if self._worker_rng is None:
            info = torch.utils.data.get_worker_info()
            worker_id = info.id if info is not None else 0
            # entropy included so re-created workers don't repeat last epoch's noise
            self._worker_rng = np.random.default_rng(
                np.random.SeedSequence([self.aug_seed, worker_id,
                                        int.from_bytes(os.urandom(4), "little")]))
        return self._worker_rng

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
        # The feature spectrogram's NFFT/hop are in samples: any sample rate
        # other than SAMPLE_RATE silently changes time/frequency resolution
        # (native-rate real recordings used to hit exactly that).
        audio = resample_to_model_rate(audio, sr)

        tone_freq = float(row["tone_hz"]) if row.get("tone_hz") else None
        if self.augment:
            rng = self._rng_for(idx)
            # random leading/trailing silence: an inference window rarely
            # starts exactly at a keyed element, so teach the model clips
            # that begin/end in dead air (the noise below covers it too)
            pad_lo = np.zeros(int(rng.uniform(0.0, 0.5) * SAMPLE_RATE))
            pad_hi = np.zeros(int(rng.uniform(0.0, 0.5) * SAMPLE_RATE))
            audio = np.concatenate([pad_lo, audio, pad_hi])
            audio, info = augment_clip(audio, SAMPLE_RATE, rng, self.snr_db_range,
                                        strength=self.aug_strength)
            if tone_freq is not None:
                tone_freq += info["freq_shift_hz"]
                tone_freq += rng.uniform(-25.0, 25.0)  # emulate detector error

        features = extract_features(audio, SAMPLE_RATE, hop_samples=self.hop_samples,
                                     tone_freq=tone_freq)  # (T, F)
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


class MaskedGroupNorm(nn.GroupNorm):
    """GroupNorm whose statistics ignore zero-padded time frames.

    Plain GroupNorm computes each sample's mean/var over ALL of (C, T, F) -
    including the zero-padded tail of shorter sequences in a batch - so a
    clip's normalization silently depended on how much padding its batch
    happened to have, making padded-batch training numerically different
    from single-clip inference (measured ~1e-2 divergence in output
    log-probs). With lengths given, stats come from valid frames only and
    padded frames are forced back to exact zeros afterward, so every
    downstream conv/pool sees the same implicit zero padding a solo forward
    would - batch and solo outputs then match to float precision.

    Subclasses nn.GroupNorm (same parameters), so checkpoints are
    interchangeable, and behaves identically to it when lengths is None
    (the single-sequence inference path)."""

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is None:
            return super().forward(x)
        b, c, t, f = x.shape
        g = self.num_groups
        mask = (torch.arange(t, device=x.device)[None, :] < lengths.to(x.device)[:, None])
        mask_g = mask[:, None, None, :, None].to(x.dtype)  # (B,1,1,T,1)
        xg = x.view(b, g, c // g, t, f)
        n_valid = mask_g.sum(dim=(2, 3, 4), keepdim=True) * (c // g) * f
        mean = (xg * mask_g).sum(dim=(2, 3, 4), keepdim=True) / n_valid
        var = ((xg - mean) ** 2 * mask_g).sum(dim=(2, 3, 4), keepdim=True) / n_valid
        xg = (xg - mean) / torch.sqrt(var + self.eps)
        x = xg.view(b, c, t, f)
        x = x * self.weight.view(1, c, 1, 1) + self.bias.view(1, c, 1, 1)
        return x * mask[:, None, :, None].to(x.dtype)


class CWDecoder(nn.Module):
    def __init__(self, n_freq_bins: int = N_FREQ_BINS, vocab_size: int = 48,
                 cnn_channels: int = 32, lstm_hidden: int = 128, lstm_layers: int = 2,
                 cnn_groups: int = 8, dropout: float = 0.2, time_stride: int = 2,
                 hop_samples: int = DEFAULT_HOP_SAMPLES):
        super().__init__()
        self.n_freq_bins = n_freq_bins
        self.cnn_channels = cnn_channels
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.cnn_groups = cnn_groups
        self.dropout = dropout
        self.time_stride = time_stride
        self.hop_samples = hop_samples

        # MaskedGroupNorm instead of BatchNorm2d: no running statistics, so its
        # normalisation behaviour can't drift during long runs at a near-zero
        # LR where BatchNorm statistics kept updating even when the optimizer
        # wasn't making meaningful weight updates - and unlike plain GroupNorm,
        # its statistics exclude each sample's zero-padded frames, so
        # padded-batch training matches single-clip inference exactly.
        #
        # The first MaxPool also downsamples time by time_stride: paired with a
        # correspondingly finer feature hop, the CNN sees high-resolution keying
        # edges while the LSTM runs at the summarized (cheaper) frame rate.
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, kernel_size=3, padding=1),
            MaskedGroupNorm(num_groups=min(cnn_groups, cnn_channels), num_channels=cnn_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(time_stride, 2)),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            MaskedGroupNorm(num_groups=min(cnn_groups, cnn_channels), num_channels=cnn_channels),
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
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(lstm_hidden * 2, vocab_size)

    @property
    def config(self) -> dict:
        """The reconstruction recipe saved into checkpoints (everything except
        vocab_size, which comes from the vocab file)."""
        return dict(n_freq_bins=self.n_freq_bins, cnn_channels=self.cnn_channels,
                    lstm_hidden=self.lstm_hidden, lstm_layers=self.lstm_layers,
                    cnn_groups=self.cnn_groups, dropout=self.dropout,
                    time_stride=self.time_stride, hop_samples=self.hop_samples)

    def downsampled_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Valid output frames per sequence after the CNN's time pooling -
        this is what CTC input_lengths must be, not the raw feature length."""
        return torch.clamp(lengths // self.time_stride, min=1)

    def frame_seconds(self, sr: int = SAMPLE_RATE) -> float:
        """Duration of one model output frame."""
        return self.hop_samples * self.time_stride / sr

    @staticmethod
    def _zero_pad_frames(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        t = x.shape[2]
        mask = torch.arange(t, device=x.device)[None, :] < lengths.to(x.device)[:, None]
        return x * mask[:, None, :, None].to(x.dtype)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, T, F); lengths: raw feature lengths (pre-CNN) or None.
        # Pass lengths for padded batches - with them, each sequence's valid
        # output is bit-identical to running it alone: norm statistics exclude
        # pad frames (MaskedGroupNorm), pad frames re-zero after time pooling
        # (a boundary-straddling pooled frame would otherwise leak non-zeros
        # into the next conv's receptive field), and the LSTM is packed so its
        # backward direction never carries pad state into real frames. Output
        # frames beyond a sequence's valid length are meaningless - callers
        # must mask/trim by downsampled_lengths (CTC does via input_lengths).
        x = x.unsqueeze(1)  # (batch, 1, T, F)
        if lengths is not None:
            # don't trust callers to have zero-padded: convs read past each
            # sequence's end, so anything non-zero there leaks into the output
            x = self._zero_pad_frames(x, lengths)
        cur_lengths = lengths
        for layer in self.cnn:
            if isinstance(layer, MaskedGroupNorm):
                x = layer(x, cur_lengths)
            else:
                x = layer(x)
                if isinstance(layer, nn.MaxPool2d) and cur_lengths is not None:
                    k_t = layer.kernel_size[0]
                    if k_t > 1:
                        cur_lengths = torch.clamp(cur_lengths // k_t, min=1)
                        x = self._zero_pad_frames(x, cur_lengths)
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)  # (batch, T', C*F')
        if lengths is not None:
            valid = torch.clamp(self.downsampled_lengths(lengths), max=t).cpu()
            packed = pack_padded_sequence(x, valid, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            x, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=t)
        else:
            x, _ = self.lstm(x)
        logits = self.fc(x)  # (batch, T', vocab_size)
        return logits.log_softmax(dim=-1)


def _log_sum_exp(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _max_repeat_run(text: str, max_period: int = 2) -> int:
    """Longest streak of a period-1 or period-2 repetition ('AAAA' or
    'AEAEAE') - the runaway-collapse signature, measured in repeated chars."""
    best = 0
    for p in range(1, max_period + 1):
        run = 0
        for i in range(p, len(text)):
            if text[i] == text[i - p]:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
    return best


def ctc_beam_decode(log_probs: torch.Tensor, vocab: Vocab, lm=None,
                    lm_weight: float = 0.3, beam_width: int = 20,
                    top_k: int = 15, final_rescore=None,
                    length_bonus: float = 0.0, repeat_penalty: float = 0.0) -> str:
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
    final_rescore: optional callable(text) -> log-score bonus, applied once
    to the finished beams before picking the winner (e.g. FCC callsign
    verification boosting beams whose callsigns are real licensed calls).
    length_bonus adds +bonus per output character (counteracts CTC/LM bias
    toward short outputs); repeat_penalty subtracts penalty * the longest
    period-1/2 repetition run (targets the runaway-repetition collapse mode
    specifically). Both default 0.0 = no effect; sweep them with
    evaluate.py --sweep before trusting nonzero values.
    Returns the best-scoring decoded string."""
    import numpy as _np
    NEG_INF = float("-inf")
    lp = log_probs.float().cpu().numpy()
    T, V = lp.shape

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

        # Only expand the top-K non-blank characters by acoustic probability.
        # CTC posteriors are heavily peaked — the bottom V-top_k characters
        # contribute near-zero probability mass and expanding them costs O(beam_width)
        # dict operations each for negligible accuracy gain.
        k = min(top_k, V - 1)
        top_chars = _np.argpartition(lp_t[1:], -k)[-k:] + 1  # +1: skip blank at 0

        for prefix, (log_Pb, log_Pnb) in beams.items():
            log_P = _log_sum_exp(log_Pb, log_Pnb)
            last = prefix[-1] if prefix else None

            # extend with blank — prefix stays the same
            _ensure(prefix)
            new_beams[prefix][0] = _log_sum_exp(
                new_beams[prefix][0], log_P + float(lp_t[0]))

            # extend with each top-K non-blank character
            for c in top_chars:
                char = vocab.idx_to_char.get(int(c))
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

    def _final_score(kv):
        score = _log_sum_exp(kv[1][0], kv[1][1])
        if final_rescore is not None:
            score += final_rescore(kv[0])
        if length_bonus:
            score += length_bonus * len(kv[0])
        if repeat_penalty:
            score -= repeat_penalty * _max_repeat_run(kv[0])
        return score

    best = max(beams.items(), key=_final_score)
    return best[0]


def ctc_greedy_decode(log_probs: torch.Tensor, vocab: Vocab,
                      lengths: torch.Tensor | None = None) -> list[str]:
    """log_probs: (batch, T, vocab_size) -> collapse repeats + drop blanks.

    lengths: valid output frames per sequence (model.downsampled_lengths of
    the raw feature lengths). Required for padded batches - the packed LSTM
    leaves pad frames as zero vectors whose argmax is arbitrary junk, so
    decoding past a sequence's valid length appends garbage characters."""
    pred_ids = log_probs.argmax(dim=-1)  # (batch, T)
    results = []
    for i, seq in enumerate(pred_ids):
        ids = seq.tolist()
        if lengths is not None:
            ids = ids[: int(lengths[i])]
        chars = []
        prev = None
        for idx in ids:
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


def ctc_forced_align(log_probs, target_indices: list[int], blank: int = 0):
    """Given one recording's log_probs (T, vocab_size) and its KNOWN correct
    target character sequence (vocab indices, no blanks), finds the
    highest-probability path through the CTC lattice constrained to produce
    exactly that sequence - forced alignment, not free decoding. Returns a
    list of (first_frame, last_frame) per target character (None, None for
    a character forced alignment assigned zero frames to).

    This is what lets dataprep/build_manifest.py's crude proportional
    chars-per-second guess (the source of the mid-word label corruption
    found in the real-ARRL eval) be replaced with exact, model-derived
    timing - dataprep/realign_arrl_labels.py runs it over whole real
    recordings, so the DP is vectorized over the state axis with numpy
    (the original pure-Python version was fine for clips but far too slow
    for T in the tens of thousands).

    Standard CTC Viterbi forced alignment: the target is expanded with
    blanks (blank, c1, blank, c2, ..., cL, blank, length 2L+1), and the best
    path through T frames is found via the same transition structure
    CTCLoss uses internally for its forward-backward algorithm, but taking
    the max (Viterbi) instead of the sum, with backpointers to recover the
    actual path instead of just its probability.
    """
    if isinstance(log_probs, torch.Tensor):
        lp = log_probs.detach().float().cpu().numpy()
    else:
        lp = np.asarray(log_probs, dtype=np.float32)
    T = lp.shape[0]
    L = len(target_indices)

    states = np.empty(2 * L + 1, dtype=np.int64)
    states[0::2] = blank
    states[1::2] = target_indices
    n_states = len(states)

    neg_inf = np.float32(-np.inf)
    # skip transition (s-2 -> s) is only legal when it doesn't jump over a
    # required distinct state: allowed iff states[s] != states[s-2]
    skip_allowed = np.zeros(n_states, dtype=bool)
    if n_states > 2:
        skip_allowed[2:] = states[2:] != states[:-2]

    v = np.full(n_states, neg_inf, dtype=np.float32)
    v[0] = lp[0, states[0]]
    if n_states > 1:
        v[1] = lp[0, states[1]]

    back = np.zeros((T, n_states), dtype=np.int8)
    emit = lp[:, states]  # (T, n_states)

    for t in range(1, T):
        stay = v
        diag = np.concatenate(([neg_inf], v[:-1]))
        skip = np.concatenate(([neg_inf, neg_inf], v[:-2]))
        skip = np.where(skip_allowed, skip, neg_inf)

        choice = np.zeros(n_states, dtype=np.int8)
        best = stay.copy()
        take_diag = diag > best
        best = np.where(take_diag, diag, best)
        choice[take_diag] = 1
        take_skip = skip > best
        best = np.where(take_skip, skip, best)
        choice[take_skip] = 2

        back[t] = choice
        v = np.where(best == neg_inf, neg_inf, best + emit[t])

    if n_states == 1:
        s = 0
    else:
        s = (n_states - 1) if v[n_states - 1] >= v[n_states - 2] else (n_states - 2)
    path_states = np.zeros(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path_states[t] = s
        s = max(0, s - int(back[t, s]))

    spans = []
    for i in range(L):
        s_idx = 2 * i + 1
        frames = np.nonzero(path_states == s_idx)[0]
        spans.append((int(frames[0]), int(frames[-1])) if len(frames) else (None, None))
    return spans


def decode_window_core(window_audio, window_abs_start: float, core_start: float, core_end: float,
                        model, vocab: Vocab, device: str, sr: int,
                        lm=None, lm_weight: float = 0.3, beam_width: int = 20,
                        top_k: int = 15, tone_freq: float | None = None,
                        final_rescore=None, squelch_db: float = 0.0,
                        length_bonus: float = 0.0, repeat_penalty: float = 0.0) -> str:
    """Decodes one audio window and keeps only the characters whose absolute
    start time (window_abs_start + their offset within the window) falls in
    [core_start, core_end) - the stretch of the timeline this window "owns"
    in an overlapping sliding-window scheme. See decode_stream().

    tone_freq: known/tracked CW tone frequency to center the feature band on
    (e.g. a ToneTracker's smoothed estimate); None = detect per window.

    squelch_db > 0 gates the window on CW-likeness first (see
    features.cw_activity_db): silence, static, and steady carriers return ""
    instead of being hallucinated into characters. 0 = no gating (the right
    default when the audio is known to contain CW, e.g. clip evaluation).

    When lm is provided, uses CTC beam search scored by the ham-domain
    character LM instead of greedy decoding. Timing for the core-region
    filter is recovered via forced alignment of the beam search result back
    to frame positions, so the overlap-trim-stitch logic works identically
    regardless of decode strategy."""
    if squelch_db > 0 and cw_activity_db(window_audio, sr, hop_samples=model.hop_samples,
                                          tone_freq=tone_freq) < squelch_db:
        return ""
    hop_seconds = model.frame_seconds(sr)
    features = extract_features(window_audio, sr, hop_samples=model.hop_samples,
                                 tone_freq=tone_freq)
    with torch.no_grad():
        x = torch.from_numpy(features).unsqueeze(0).to(device)
        log_probs = model(x)
    log_probs_cpu = log_probs[0].cpu()  # (T, vocab_size)

    if lm is not None:
        # Beam search for best text, forced-align back to frames for trimming
        text = ctc_beam_decode(log_probs_cpu, vocab, lm=lm,
                                lm_weight=lm_weight, beam_width=beam_width, top_k=top_k,
                                final_rescore=final_rescore,
                                length_bonus=length_bonus, repeat_penalty=repeat_penalty)
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
                   lm=None, lm_weight: float = 0.3, beam_width: int = 20,
                   top_k: int = 15, final_rescore=None, squelch_db: float = 0.0,
                   length_bonus: float = 0.0, repeat_penalty: float = 0.0) -> str:
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

    Audio at any sample rate is accepted and resampled to the model rate.
    The CW tone is tracked across windows with the same ToneTracker policy
    the live StreamDecoder uses (EMA + outlier rejection), so offline
    evaluation exercises the same front end as the radio - a QRM burst in
    one window can't yank the feature band off the signal here either.
    """
    audio = resample_to_model_rate(audio, sr)
    sr = SAMPLE_RATE
    n_samples = len(audio)
    window_samples = int(window_seconds * sr)
    stride_samples = int(stride_seconds * sr)
    guard_seconds = (window_seconds - stride_seconds) / 2
    total_seconds = n_samples / sr
    tone_tracker = ToneTracker()

    pieces = []
    window_start_sample = 0
    while window_start_sample < n_samples:
        window_end_sample = min(window_start_sample + window_samples, n_samples)
        is_first = window_start_sample == 0
        is_last = window_end_sample >= n_samples
        window_abs_start = window_start_sample / sr

        core_start = 0.0 if is_first else window_abs_start + guard_seconds
        core_end = total_seconds if is_last else window_abs_start + guard_seconds + stride_seconds

        window_audio = audio[window_start_sample:window_end_sample]
        tone = tone_tracker.update(detect_tone_freq(window_audio, sr,
                                                     hop_samples=model.hop_samples))
        pieces.append(decode_window_core(window_audio, window_abs_start,
                                          core_start, core_end, model, vocab, device, sr,
                                          lm=lm, lm_weight=lm_weight, beam_width=beam_width,
                                          top_k=top_k, final_rescore=final_rescore,
                                          tone_freq=tone, squelch_db=squelch_db,
                                          length_bonus=length_bonus, repeat_penalty=repeat_penalty))

        if is_last:
            break
        window_start_sample += stride_samples

    return "".join(pieces)


def edit_distance(a, b) -> int:
    """Levenshtein distance with the inner DP loop vectorized in numpy -
    evaluate_streaming.py runs this on multi-thousand-character whole-recording
    transcripts, where the pure-Python O(n^2) version dominated eval time.

    The left-neighbor recurrence cur[j] = min(t[j-1], cur[j-1] + 1) is a
    running minimum with +1 per step, which closes to
    cur[j] = j + min(i, min_{k<=j}(t[k-1] - k)) - one minimum.accumulate."""
    if len(a) < len(b):
        a, b = b, a
    if not len(b):
        return len(a)
    # map tokens (chars or whole words) to ints so comparisons are vectorized
    ids: dict = {}
    a_ids = np.fromiter((ids.setdefault(x, len(ids)) for x in a), dtype=np.int32, count=len(a))
    b_ids = np.fromiter((ids.setdefault(x, len(ids)) for x in b), dtype=np.int32, count=len(b))
    m = len(b_ids)
    jrange = np.arange(1, m + 1, dtype=np.int32)
    prev = np.arange(m + 1, dtype=np.int32)
    for i, ca in enumerate(a_ids, 1):
        t = np.minimum(prev[:-1] + (b_ids != ca), prev[1:] + 1)
        cur = np.empty(m + 1, dtype=np.int32)
        cur[0] = i
        cur[1:] = jrange + np.minimum(np.minimum.accumulate(t - jrange), np.int32(i))
        prev = cur
    return int(prev[-1])


def cer(pred: str, ref: str) -> float:
    if not ref:
        return 0.0 if not pred else 1.0
    return edit_distance(list(pred), list(ref)) / len(ref)


def wer(pred: str, ref: str) -> float:
    ref_words = ref.split()
    if not ref_words:
        return 0.0 if not pred.split() else 1.0
    return edit_distance(pred.split(), ref_words) / len(ref_words)
