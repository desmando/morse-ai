"""CNN-LSTM-CTC acoustic model: spectrogram-band features -> character sequence.

Architecture follows the approach validated by prior CW-decoder work (e.g.
AG1LE's real-time Morse decoder): a small CNN front end picks up local
dot/dash/space patterns in the time-frequency image, a BiLSTM models the
sequence, and CTC loss avoids needing exact per-character alignment.
"""
import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import Dataset

from model.features import extract_features, N_FREQ_BINS
from model.vocab import Vocab

REPO_ROOT = Path(__file__).resolve().parent.parent


class MorseClipDataset(Dataset):
    def __init__(self, manifest_path: str, vocab: Vocab):
        self.vocab = vocab
        with open(manifest_path, newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        audio, sr = sf.read(REPO_ROOT / row["clip_path"])
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
