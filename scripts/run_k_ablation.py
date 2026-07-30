"""Controlled K-ablation: isolate the multi-node effect from data volume.

The original ablation's K=1 vs K=3 comparison was confounded: K=1 gives one
node the FULL training set, while each K=3 node sees one third, but K=3 also
instantiates three independently seeded ensembles (extra diversity).

Three volume-controlled configurations, 5 seeds each (Boiler static):

  A) K=3, partitioned    : 3 nodes, disjoint thirds of the training data
                           (the paper's SEMAS configuration).
  B) K=1, full data      : 1 node, all training data
                           (the original 'no_federated' arm).
  C) K=1, third data     : 1 node, a random third of the training data
                           (volume-matched to one K=3 node).

Interpretation:
  A vs C  -> effect of adding nodes + aggregation at FIXED per-node volume.
  B vs C  -> effect of data volume alone at fixed K=1.
  A vs B  -> the confounded comparison reported in the main ablation.

No PPO (it measured zero effect in this setting); validation-calibrated
thresholds only, so the comparison is purely about the detection topology.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from semas.agents.fog_node import FogPolicy
from semas.baselines.systems import Baseline1Static
from semas.data import load_boiler
from semas.evaluation import calibrate_threshold, classification_metrics
from semas.seeding import set_seeds
from semas.system import SemasSystem

SEEDS = [42, 123, 456, 789, 1024]


def run_config(config: str, seed: int, ds) -> dict:
    set_seeds(seed)
    rng = np.random.default_rng(seed)
    Xtr = ds.X_train.values
    Xva, yva = ds.X_val.values, ds.y_val
    Xte, yte = ds.X_test.values, ds.y_test

    if config == "K3_partitioned":
        sys_ = SemasSystem(k_nodes=3, seed=seed).fit(Xtr)
        sys_.edge.tune(Xva, yva)
        tau = calibrate_threshold(yva, sys_.scores(Xva))
        p = sys_.global_policy()
        sys_.set_global_policy(FogPolicy(p.w1, p.contamination, tau))
        scores = sys_.scores(Xte)
    elif config == "K1_full":
        b = Baseline1Static(seed=seed).fit(Xtr, Xva, yva)
        tau = b.tau
        scores = b.scores(Xte)
    elif config == "K1_third":
        idx = rng.choice(len(Xtr), len(Xtr) // 3, replace=False)
        b = Baseline1Static(seed=seed).fit(Xtr[idx], Xva, yva)
        tau = b.tau
        scores = b.scores(Xte)
    else:
        raise ValueError(config)

    return classification_metrics(yte, scores, tau)


results = {}
for config in ["K3_partitioned", "K1_full", "K1_third"]:
    rows = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        ds = load_boiler(seed=seed)
        m = run_config(config, seed, ds)
        rows.append(m)
        print(f"{config:16s} seed={seed:5d} F1={m['f1']:.4f} "
              f"AUC={m['roc_auc']:.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)
    f1s = [r["f1"] for r in rows]
    aucs = [r["roc_auc"] for r in rows]
    results[config] = {
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "seeds": rows,
    }

print("\n" + "=" * 64)
for config, r in results.items():
    print(f"{config:16s} F1={r['f1_mean']:.4f}+-{r['f1_std']:.4f}  "
          f"AUC={r['auc_mean']:.4f}+-{r['auc_std']:.4f}")

from scipy import stats
a = [r["f1"] for r in results["K3_partitioned"]["seeds"]]
b = [r["f1"] for r in results["K1_full"]["seeds"]]
c = [r["f1"] for r in results["K1_third"]["seeds"]]
for name, x, y in [("A(K3) vs C(K1,1/3 data)", a, c),
                   ("B(K1,full) vs C(K1,1/3 data)", b, c),
                   ("A(K3) vs B(K1,full)", a, b)]:
    t, p = stats.ttest_ind(x, y, equal_var=False)
    print(f"{name:30s} dF1={np.mean(x)-np.mean(y):+.4f} t={t:.2f} p={p:.4f}")

out = Path("results/k_ablation.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved -> {out}")
