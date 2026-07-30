"""Robustness experiments (R1-Q12): sensor noise, missing data, class
imbalance. (Operational drift is covered by the boiler severity-drift
protocol in run_experiments.py.)

Corruptions are applied to the TEST set only (deployment-time degradation);
systems are trained and calibrated on clean data. SEMAS (static mode, no
extra adaptation) vs Baseline1, 3 seeds.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from semas.baselines.systems import Baseline1Static
from semas.data import load_boiler
from semas.evaluation import calibrate_threshold, classification_metrics
from semas.agents.fog_node import FogPolicy
from semas.seeding import set_seeds
from semas.system import SemasSystem

SEEDS = [42, 123, 456, 789, 1024]


def corruptions(X, rng):
    X = np.asarray(X)
    out = {"clean": X}
    for sigma in (0.1, 0.3):
        out[f"noise_{sigma}"] = X + rng.normal(0, sigma, X.shape)
    for frac in (0.1, 0.3):
        Xm = X.copy()
        mask = rng.random(X.shape) < frac
        Xm[mask] = 0.0  # zero-fill = scaled mean imputation
        out[f"missing_{frac}"] = Xm
    return out


def imbalance_subsets(y, rng, rates=(0.05, 0.15)):
    """Index subsets with reduced anomaly prevalence."""
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = {}
    for r in rates:
        n_pos = int(len(neg) * r / (1 - r))
        out[f"prevalence_{r}"] = np.sort(
            np.concatenate([neg, rng.choice(pos, min(n_pos, len(pos)), replace=False)])
        )
    return out


results = []
for seed in SEEDS:
    set_seeds(seed)
    rng = np.random.default_rng(seed)
    ds = load_boiler(seed=seed)
    Xtr, Xva, yva = ds.X_train.values, ds.X_val.values, ds.y_val
    Xte, yte = ds.X_test.values, ds.y_test

    systems = {}
    sem = SemasSystem(k_nodes=3, seed=seed).fit(Xtr)
    sem.edge.tune(Xva, yva)
    tau_s = calibrate_threshold(yva, sem.scores(Xva))
    p = sem.global_policy()
    sem.set_global_policy(FogPolicy(p.w1, p.contamination, tau_s))
    systems["semas"] = (sem.scores, tau_s)
    b1 = Baseline1Static(seed=seed).fit(Xtr, Xva, yva)
    systems["baseline1"] = (b1.scores, b1.tau)

    for cname, Xc in corruptions(Xte, rng).items():
        for sname, (scores_fn, tau) in systems.items():
            m = classification_metrics(yte, scores_fn(Xc), tau)
            results.append({"seed": seed, "condition": cname, "system": sname, **m})
            print(f"seed={seed} {cname:15s} {sname:10s} F1={m['f1']:.3f} "
                  f"AUC={m['roc_auc']:.3f}", flush=True)

    for cname, idx in imbalance_subsets(yte, rng).items():
        for sname, (scores_fn, tau) in systems.items():
            m = classification_metrics(yte[idx], scores_fn(Xte[idx]), tau)
            results.append({"seed": seed, "condition": cname, "system": sname, **m})
            print(f"seed={seed} {cname:15s} {sname:10s} F1={m['f1']:.3f} "
                  f"AUC={m['roc_auc']:.3f}", flush=True)

out = Path("results/robustness.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved -> {out}")
