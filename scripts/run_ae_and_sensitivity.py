"""Two audit-driven reruns:

A) Unsupervised autoencoder baselines on the CORRECTED data pipeline
   (the manuscript's current AE numbers came from the old, mislabeled data):
   * Boiler: dense autoencoder (rows are i.i.d. -> sequence AE inapplicable),
   * Wind:   LSTM autoencoder over 12-step windows (genuine time series).
   Both train on normal-labeled training rows only; threshold = best-F1 on
   validation reconstruction error (same protocol as every other system).

B) Real hyperparameter sensitivity sweep (replaces an unverified legacy
   claim of "+-20% stability"): vary w1 in {0.4,0.5,0.6} and tau in
   {0.8, 1.0, 1.2} x calibrated value on the Boiler stack (seed 42, 123, 456;
   no PPO), report the F1 range on test.
"""

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from semas.baselines.deep import make_windows
from semas.data import load_boiler, load_wind
from semas.evaluation import calibrate_threshold, classification_metrics
from semas.seeding import set_seeds
from semas.agents.fog_node import FogNode

results = {}

# ---------------------------------------------------------------- A) AEs
class DenseAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, 6), nn.ReLU(), nn.Linear(6, 3))
        self.dec = nn.Sequential(nn.Linear(3, 6), nn.ReLU(), nn.Linear(6, d))

    def forward(self, x):
        return self.dec(self.enc(x))


class LSTMAE(nn.Module):
    def __init__(self, d, hidden=32):
        super().__init__()
        self.enc = nn.LSTM(d, hidden, batch_first=True)
        self.dec = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, d)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        y, _ = self.dec(z)
        return self.out(y)


def train_ae(model, X, epochs=30, batch=256, lr=1e-3, seed=42):
    torch.manual_seed(seed)
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    crit, opt = nn.MSELoss(), torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(X), batch_size=batch, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    for _ in range(epochs):
        model.train()
        for (xb,) in loader:
            opt.zero_grad()
            loss = crit(model(xb), xb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def recon_error(model, X, batch=1024):
    model.eval()
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    errs = []
    for i in range(0, len(X), batch):
        xb = X[i:i + batch]
        e = ((model(xb) - xb) ** 2).mean(dim=tuple(range(1, xb.dim())))
        errs.append(e.numpy())
    return np.concatenate(errs)


print("A1) Boiler dense autoencoder (corrected pipeline)")
set_seeds(42)
b = load_boiler(seed=42)
normal_tr = b.X_train.values[b.y_train == 0]
ae = train_ae(DenseAE(b.X_train.shape[1]), normal_tr, seed=42)
s_val = recon_error(ae, b.X_val.values)
s_test = recon_error(ae, b.X_test.values)
tau = calibrate_threshold(b.y_val, s_val)
m = classification_metrics(b.y_test, s_test, tau)
results["boiler_dense_ae"] = m
print(f"   F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} AUC={m['roc_auc']:.3f}")

print("A2) Wind LSTM autoencoder (12-step windows, corrected pipeline)")
w = load_wind()
SEQ = 12
Xtr, ytr = make_windows(w.X_train.values, w.y_train, SEQ)
Xva, yva = make_windows(w.X_val.values, w.y_val, SEQ)
Xte, yte = make_windows(w.X_test.values, w.y_test, SEQ)
lstm_ae = train_ae(LSTMAE(Xtr.shape[2]), Xtr[ytr == 0], epochs=15, seed=42)
s_val = recon_error(lstm_ae, Xva)
s_test = recon_error(lstm_ae, Xte)
tau = calibrate_threshold(yva, s_val)
m = classification_metrics(yte, s_test, tau)
results["wind_lstm_ae"] = m
print(f"   F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} AUC={m['roc_auc']:.3f}")

# ------------------------------------------------ B) sensitivity sweep
print("B) Hyperparameter sensitivity (w1, tau) on Boiler stack, 3 seeds")
sweep = []
for seed in [42, 123, 456]:
    set_seeds(seed)
    ds = load_boiler(seed=seed)
    node = FogNode(node_id=0, seed=seed).fit(ds.X_train.values)
    s_val_b1 = node.b1.scores(ds.X_val.values)
    s_val_b2 = node.b2.scores(ds.X_val.values)
    s_te_b1 = node.b1.scores(ds.X_test.values)
    s_te_b2 = node.b2.scores(ds.X_test.values)
    for w1 in [0.4, 0.5, 0.6]:
        sv = w1 * s_val_b1 + (1 - w1) * s_val_b2
        st = w1 * s_te_b1 + (1 - w1) * s_te_b2
        tau_star = calibrate_threshold(ds.y_val, sv)
        for mult in [0.8, 1.0, 1.2]:
            m = classification_metrics(ds.y_test, st, tau_star * mult)
            sweep.append({"seed": seed, "w1": w1, "tau_mult": mult, "f1": m["f1"]})
            print(f"   seed={seed} w1={w1} tau_x={mult}: F1={m['f1']:.4f}")

f1s = np.array([s["f1"] for s in sweep])
nominal = np.mean([s["f1"] for s in sweep if s["w1"] == 0.5 and s["tau_mult"] == 1.0])
results["sensitivity"] = {
    "nominal_f1": float(nominal),
    "min_f1": float(f1s.min()), "max_f1": float(f1s.max()),
    "max_abs_dev_from_nominal": float(np.max(np.abs(f1s - nominal))),
    "grid": sweep,
}
print(f"\n   nominal F1={nominal:.4f}, range [{f1s.min():.4f}, {f1s.max():.4f}], "
      f"max |dev|={np.max(np.abs(f1s - nominal)):.4f}")

out = Path("results/ae_and_sensitivity.json")
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved -> {out}")
