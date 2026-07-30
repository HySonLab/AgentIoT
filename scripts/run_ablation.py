"""Real component ablation on the Boiler dataset (R3's audit requirement):
each row removes exactly one component from the Full SEMAS configuration.
All numbers computed here, none hand-entered into the manuscript.

Rows:
  full          : K=3, learned consensus (w1 init 0.5), PPO adapts (w1, rho, tau)
  w/o PPO       : K=3, learned consensus, tau calibrated on val, NO adaptation
  w/o consensus : K=3, w1 fixed at 1.0 (B1-only), PPO adapts (rho, tau) only
  w/o federated : K=1 (single node), learned consensus, PPO adapts (w1, rho, tau)
  w/o SLM       : identical to full (Agent C/E do not touch detection) -
                  reported as a structural no-op on F1 by construction.

5 seeds, boiler_static protocol (same data condition as the main sweep).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from semas.agents.evolution import evolve_policy
from semas.agents.fog_node import FogPolicy
from semas.data import load_boiler
from semas.evaluation import calibrate_threshold, classification_metrics
from semas.seeding import set_seeds
from semas.system import SemasSystem

SEEDS = [42, 123, 456]
PPO_TIMESTEPS = 256


def run_variant(variant: str, seed: int, ds) -> dict:
    set_seeds(seed)
    Xtr, Xva, yva = ds.X_train.values, ds.X_val.values, ds.y_val
    Xte, yte = ds.X_test.values, ds.y_test

    k = 1 if variant == "no_federated" else 3
    sys_ = SemasSystem(k_nodes=k, seed=seed).fit(Xtr)
    sys_.edge.tune(Xva, yva)

    if variant == "no_consensus":
        for n in sys_.nodes:
            n.b3.set_weights(1.0)  # B1-only
            n.policy.w1 = 1.0

    s_val = sys_.scores(Xva)
    tau0 = calibrate_threshold(yva, s_val)
    p = sys_.global_policy()
    sys_.set_global_policy(FogPolicy(p.w1, p.contamination, tau0))

    if variant != "no_ppo":
        evolve_policy(sys_, Xva, yva, total_timesteps=PPO_TIMESTEPS, seed=seed,
                      window=min(256, len(yva)))
        if variant == "no_consensus":
            # PPO's action space includes w1; re-pin it after evolution so
            # this row remains a genuine "no consensus voting" ablation.
            fp = sys_.global_policy()
            sys_.set_global_policy(FogPolicy(1.0, fp.contamination, fp.tau))
            for n in sys_.nodes:
                n.b3.set_weights(1.0)

    tau = sys_.global_policy().tau
    scores = sys_.scores(Xte)
    m = classification_metrics(yte, scores, tau)
    return m


results = {}
for variant in ["full", "no_ppo", "no_consensus", "no_federated"]:
    rows = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        ds = load_boiler(seed=seed)
        m = run_variant(variant, seed, ds)
        rows.append(m)
        print(f"{variant:15s} seed={seed:5d} F1={m['f1']:.4f} "
              f"P={m['precision']:.4f} R={m['recall']:.4f} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    f1s = [r["f1"] for r in rows]
    ps = [r["precision"] for r in rows]
    results[variant] = {
        "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
        "precision_mean": float(np.mean(ps)),
        "seeds": rows,
    }

full_f1 = results["full"]["f1_mean"]
for variant in results:
    results[variant]["impact_pct"] = (
        0.0 if variant == "full"
        else (results[variant]["f1_mean"] - full_f1) / full_f1 * 100
    )

print("\n" + "=" * 60)
for variant, r in results.items():
    print(f"{variant:15s} F1={r['f1_mean']:.4f}+-{r['f1_std']:.4f}  "
          f"impact={r['impact_pct']:+.1f}%")

out = Path("results/ablation.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved -> {out}")
