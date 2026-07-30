"""LSTM RUL regressor for C-MAPSS with real degradation labels (R3-2).

Standard literature protocol:
  * per-unit sliding windows (seq_len=30) over normalized sensors;
  * train on train-split units, early-stop on val-split units;
  * headline metric: RMSE over the 100 test units at their LAST cycle,
    against the official RUL_FD001.txt targets (directly comparable to
    published FD001 results), plus MAE and the NASA scoring function.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2,
                            batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def unit_windows(pack: dict, seq_len: int = 30, last_only: bool = False):
    """Sliding windows that never cross unit boundaries."""
    X = pack["X"].to_numpy()
    rul = pack["RUL"]
    units = pack["unit"]
    Xw, yw = [], []
    for u in np.unique(units):
        idx = np.flatnonzero(units == u)
        Xu, ru = X[idx], rul[idx]
        if len(idx) < seq_len:
            # left-pad short test units by repeating the first row
            pad = np.repeat(Xu[[0]], seq_len - len(idx), axis=0)
            Xu = np.vstack([pad, Xu])
            ru = np.concatenate([np.full(seq_len - len(idx), ru[0]), ru])
        if last_only:
            Xw.append(Xu[-seq_len:])
            yw.append(ru[-1])
        else:
            for i in range(seq_len, len(Xu) + 1):
                Xw.append(Xu[i - seq_len : i])
                yw.append(ru[i - 1])
    return np.asarray(Xw, dtype=np.float32), np.asarray(yw, dtype=np.float32)


def train_rul(data: dict, seq_len: int = 30, epochs: int = 30,
              batch_size: int = 256, lr: float = 1e-3, patience: int = 4,
              seed: int = 42, verbose: bool = False):
    torch.manual_seed(seed)
    Xtr, ytr = unit_windows(data["train"], seq_len)
    Xva, yva = unit_windows(data["val"], seq_len)

    model = LSTMRegressor(Xtr.shape[2])
    crit = nn.L1Loss()  # MAE, per manuscript Eq. (RUL loss)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)

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
            val_mae = float(crit(model(Xva_t), yva_t))
        if verbose:
            print(f"    epoch {epoch+1}: val MAE={val_mae:.2f}")
        if val_mae < best_val - 1e-4:
            best_val, best_state, bad = val_mae, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_rul(model, pack: dict, seq_len: int = 30, last_only: bool = True):
    Xw, yw = unit_windows(pack, seq_len, last_only=last_only)
    model.eval()
    preds = []
    for i in range(0, len(Xw), 512):
        preds.append(model(torch.from_numpy(Xw[i : i + 512])).numpy())
    return np.concatenate(preds), yw
