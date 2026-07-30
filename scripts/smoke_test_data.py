"""Smoke test: load all three datasets, print honest summaries, and run a
quick Isolation Forest sanity check with validation-calibrated thresholds."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.ensemble import IsolationForest

from semas.data import load_boiler, load_cmapss, load_wind
from semas.evaluation import calibrate_threshold, classification_metrics
from semas.seeding import set_seeds

set_seeds(42)

print("=" * 70)
print("BOILER (stratified)")
b = load_boiler(seed=42)
print(b.summary())
print("features:", b.feature_names)

print("\nBOILER (severity drift split)")
bd = load_boiler(seed=42, split="severity_drift")
print(bd.summary())

print("\n" + "=" * 70)
print("WIND TURBINE (chronological)")
w = load_wind()
print(w.summary())
print("train period:", w.meta["train_period"])
print("test period :", w.meta["test_period"])
print("n features:", len(w.feature_names))

print("\n" + "=" * 70)
print("CMAPSS FD001")
c = load_cmapss()
for split in ["train", "val", "test"]:
    d = c[split]
    print(f"{split}: n={len(d['RUL'])}, units={len(np.unique(d['unit']))}, "
          f"RUL mean={d['RUL'].mean():.1f}")
print("features:", c["feature_names"])

print("\n" + "=" * 70)
print("SANITY: IsolationForest, threshold calibrated on VAL only")
for ds in [b, w]:
    clf = IsolationForest(n_estimators=200, random_state=42, contamination="auto")
    clf.fit(ds.X_train)
    s_val = -clf.score_samples(ds.X_val)   # higher = more anomalous
    s_test = -clf.score_samples(ds.X_test)
    tau = calibrate_threshold(ds.y_val, s_val)
    m = classification_metrics(ds.y_test, s_test, tau)
    print(f"{ds.name:8s} F1={m['f1']:.3f} P={m['precision']:.3f} "
          f"R={m['recall']:.3f} AUC={m['roc_auc']:.3f} (tau from val)")
