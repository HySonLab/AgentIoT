"""Phase 3 verification run:
  A) supervised MLP on Boiler,
  B) supervised LSTM + Transformer classifiers on Wind (windowed),
  C) LSTM RUL regressor on CMAPSS FD001 with real labels.
Thresholds calibrated on validation only. Results saved to results/phase3.json.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from hama.baselines.deep import (
    MLPClassifier, SequenceClassifier, make_windows, predict_scores, train_classifier,
)
from hama.baselines.rul import predict_rul, train_rul
from hama.data import load_boiler, load_cmapss, load_wind
from hama.evaluation import calibrate_threshold, classification_metrics, rul_metrics
from hama.seeding import set_seeds

SEED = 42
set_seeds(SEED)
results = {}

print("=" * 70)
print("A) Boiler — supervised MLP classifier")
b = load_boiler(seed=SEED)
t0 = time.perf_counter()
mlp = MLPClassifier(len(b.feature_names))
mlp = train_classifier(mlp, b.X_train.values, b.y_train, b.X_val.values, b.y_val, seed=SEED)
s_val = predict_scores(mlp, b.X_val.values)
s_test = predict_scores(mlp, b.X_test.values)
tau = calibrate_threshold(b.y_val, s_val)
m = classification_metrics(b.y_test, s_test, tau)
m["train_time_s"] = time.perf_counter() - t0
results["boiler_mlp_supervised"] = m
print(f"   F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
      f"AUC={m['roc_auc']:.3f}  ({m['train_time_s']:.0f}s)")

print("\nB) Wind — supervised sequence classifiers (seq_len=12 = 2h)")
w = load_wind()
SEQ = 12
Xtr, ytr = make_windows(w.X_train.values, w.y_train, SEQ)
Xva, yva = make_windows(w.X_val.values, w.y_val, SEQ)
Xte, yte = make_windows(w.X_test.values, w.y_test, SEQ)
print(f"   windows: train={len(ytr)} ({ytr.mean():.1%} pos), "
      f"val={len(yva)} ({yva.mean():.1%}), test={len(yte)} ({yte.mean():.1%})")

for arch in ["lstm", "transformer"]:
    t0 = time.perf_counter()
    net = SequenceClassifier(Xtr.shape[2], arch=arch)
    net = train_classifier(net, Xtr, ytr, Xva, yva, seed=SEED, batch_size=256)
    s_val = predict_scores(net, Xva)
    s_test = predict_scores(net, Xte)
    tau = calibrate_threshold(yva, s_val)
    m = classification_metrics(yte, s_test, tau)
    m["train_time_s"] = time.perf_counter() - t0
    results[f"wind_{arch}_supervised"] = m
    print(f"   {arch:12s} F1={m['f1']:.3f} P={m['precision']:.3f} "
          f"R={m['recall']:.3f} AUC={m['roc_auc']:.3f}  ({m['train_time_s']:.0f}s)")

print("\nC) CMAPSS FD001 — LSTM RUL with real labels")
c = load_cmapss(seed=SEED)
t0 = time.perf_counter()
rul_model = train_rul(c, seed=SEED, verbose=True)
pred_last, true_last = predict_rul(rul_model, c["test"], last_only=True)
m_last = rul_metrics(true_last, pred_last)
pred_all, true_all = predict_rul(rul_model, c["test"], last_only=False)
m_all = rul_metrics(true_all, pred_all)
m_last["train_time_s"] = time.perf_counter() - t0
results["cmapss_lstm_rul_lastcycle"] = m_last
results["cmapss_lstm_rul_allcycles"] = m_all
print(f"   last-cycle (literature-comparable): MAE={m_last['mae']:.2f} "
      f"RMSE={m_last['rmse']:.2f} NASA={m_last['nasa_score']:.0f}")
print(f"   all-cycles: MAE={m_all['mae']:.2f} RMSE={m_all['rmse']:.2f}")

out = Path("results/phase3.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved -> {out}")
