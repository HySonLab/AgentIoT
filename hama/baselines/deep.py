"""Supervised deep-learning baselines (R1-Q11, R3-5).

* MLPClassifier      - for i.i.d. tabular data (Boiler: rows are independent
                       simulated operating points, so sequence models are
                       methodologically inapplicable there; the manuscript
                       states this explicitly).
* SequenceClassifier - LSTM or Transformer encoder over sliding windows for
                       genuine time series (Wind SCADA).

All models train with class-weighted BCE (the honest wind task is ~2-4%
positive), early-stop on validation loss, and never see test data.
Decision thresholds are calibrated on validation scores like every other
system in the study.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_windows(X: np.ndarray, y: np.ndarray, seq_len: int):
    """Sliding windows over a time-sorted series; label = last timestep."""
    Xw = np.lib.stride_tricks.sliding_window_view(X, (seq_len, X.shape[1]))
    Xw = Xw.reshape(-1, seq_len, X.shape[1])
    return Xw.copy(), y[seq_len - 1 :].copy()


class MLPClassifier(nn.Module):
    def __init__(self, n_features: int, hidden=(64, 32), dropout: float = 0.2):
        super().__init__()
        layers, d = [], n_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SequenceClassifier(nn.Module):
    def __init__(self, n_features: int, arch: str = "lstm",
                 hidden: int = 64, n_layers: int = 2, n_heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.arch = arch
        if arch == "lstm":
            self.encoder = nn.LSTM(n_features, hidden, num_layers=2,
                                   batch_first=True, dropout=dropout)
            d_out = hidden
        elif arch == "transformer":
            self.proj = nn.Linear(n_features, hidden)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads, dim_feedforward=4 * hidden,
                dropout=dropout, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            d_out = hidden
        else:
            raise ValueError(arch)
        self.head = nn.Sequential(nn.Linear(d_out, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        if self.arch == "lstm":
            out, _ = self.encoder(x)
            z = out[:, -1]
        else:
            z = self.encoder(self.proj(x)).mean(dim=1)
        return self.head(z).squeeze(-1)


def train_classifier(model: nn.Module, X_train, y_train, X_val, y_val,
                     epochs: int = 50, batch_size: int = 256, lr: float = 1e-3,
                     patience: int = 5, seed: int = 42, verbose: bool = False):
    torch.manual_seed(seed)
    dev = torch.device("cpu")
    model = model.to(dev)

    Xt = torch.as_tensor(np.asarray(X_train), dtype=torch.float32)
    yt = torch.as_tensor(np.asarray(y_train), dtype=torch.float32)
    Xv = torch.as_tensor(np.asarray(X_val), dtype=torch.float32)
    yv = torch.as_tensor(np.asarray(y_val), dtype=torch.float32)

    pos = float(yt.mean().clamp(min=1e-6))
    pos_weight = torch.tensor((1 - pos) / pos)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))

    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(crit(model(Xv), yv))
        if verbose:
            print(f"    epoch {epoch+1}: val_loss={val_loss:.4f}")
        if val_loss < best_val - 1e-5:
            best_val, best_state, bad = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_scores(model: nn.Module, X, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    out = []
    for i in range(0, len(X), batch_size):
        out.append(torch.sigmoid(model(X[i : i + batch_size])).numpy())
    return np.concatenate(out)
