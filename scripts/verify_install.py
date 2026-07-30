# -*- coding: utf-8 -*-
"""Zero-data installation check and end-to-end self-test.

Purpose: let anyone - in particular a reviewer who does not yet have the
datasets - confirm in one command that this repository actually installs,
imports, and RUNS the full HAMA pipeline end to end. It builds a small
synthetic dataset in memory, then exercises every architectural component
the manuscript claims: the Edge pre-filter, K=3 Fog nodes with the
5-model ensemble, weighted consensus, validation-only threshold
calibration, real PPO policy adaptation (stable-baselines3), real SHAP
attribution, federated-style parameter aggregation, and (optionally, if
Ollama is running) the locally hosted SLM.

This does NOT reproduce the paper's numbers - synthetic data cannot do
that. It answers the prior question: "does this code exist and work?"
Run scripts/run_experiments.py on the real datasets for the paper's
numbers (see README, Tier 2).

Runtime measured on the paper's hardware (8-core CPU, no GPU): about
3-4 minutes with --skip-slm, or ~5 minutes including the optional Ollama
SLM check. Most of that is the PPO step; the fog ensemble fit (OC-SVM and
LOF are superlinear in sample count) accounts for most of the rest.

Note on budgets: the PPO step here uses a deliberately tiny training
budget. The goal is to prove the real stable-baselines3 learner runs and
updates the policy - not to converge it. The paper's runs use far larger
budgets (see scripts/run_experiments.py).

Exit code 0 = all required checks passed.
"""
import importlib
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PASS, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"
failures = []
warnings_ = []


def report(status, label, detail=""):
    line = f"{status} {label}"
    if detail:
        line += f" - {detail}"
    print(line, flush=True)


# ----------------------------------------------------------------- 1. Python
print("=" * 68)
print("HAMA installation self-test")
print("=" * 68)
print("\n1. Interpreter")
v = sys.version_info
if v >= (3, 10):
    report(PASS, f"Python {v.major}.{v.minor}.{v.micro}")
else:
    report(FAIL, f"Python {v.major}.{v.minor}", "3.10+ required (uses PEP 604 syntax)")
    failures.append("python-version")

# ----------------------------------------------------------- 2. Dependencies
print("\n2. Dependencies")
EXPECTED = {
    "numpy": "2.1.3", "pandas": "2.3.3", "sklearn": "1.8.0", "scipy": "1.17.0",
    "torch": "2.12.1", "stable_baselines3": "2.9.0", "gymnasium": "1.3.0",
    "shap": "0.52.0", "psutil": "5.9.0", "matplotlib": "3.11.1",
}
for mod, expected in EXPECTED.items():
    try:
        m = importlib.import_module(mod)
        got = getattr(m, "__version__", "?").split("+")[0]
        if got == expected:
            report(PASS, f"{mod} {got}")
        else:
            report(WARN, f"{mod} {got}", f"paper used {expected}; results may differ slightly")
            warnings_.append(f"{mod}!={expected}")
    except Exception as e:
        report(FAIL, mod, f"import failed: {e}")
        failures.append(f"import-{mod}")

# ------------------------------------------------------------ 3. Package imports
print("\n3. HAMA package")
for name in ["hama.seeding", "hama.evaluation", "hama.system",
             "hama.aggregation", "hama.agents.edge", "hama.agents.detectors",
             "hama.agents.fog_node", "hama.agents.evolution",
             "hama.agents.meta", "hama.baselines.systems", "hama.data"]:
    try:
        importlib.import_module(name)
        report(PASS, name)
    except Exception as e:
        report(FAIL, name, str(e))
        failures.append(f"import-{name}")

if failures:
    print("\nAborting: fix the failures above before continuing.")
    sys.exit(1)

# ----------------------------------------------------------------- 4. Datasets
print("\n4. Datasets (optional for this self-test)")
root = Path(__file__).resolve().parents[1]
checks = {
    "Boiler": root / "dataset" / "Boiler_emulator_dataset.csv",
    "Wind SCADA": root / "dataset" / "iiot-data-of-wind-turbine" / "scada_data.csv",
    "C-MAPSS FD001": root / "dataset" / "cmapss" / "train_FD001.txt",
}
present = []
for label, p in checks.items():
    if p.exists():
        report(PASS, label, "found")
        present.append(label)
    else:
        report(WARN, label, f"not found at {p.relative_to(root)}")
if not present:
    print("      -> no datasets present; running synthetic self-test only.")
    print("      -> see README 'Getting the data' to reproduce paper numbers.")

# --------------------------------------------------- 5. End-to-end synthetic run
print("\n5. End-to-end pipeline on synthetic data")
t_start = time.perf_counter()
try:
    import numpy as np
    import pandas as pd

    from hama.seeding import set_seeds
    from hama.system import HamaSystem
    from hama.agents.fog_node import FogPolicy
    from hama.evaluation import calibrate_threshold, classification_metrics, measure_latency

    set_seeds(42)
    rng = np.random.default_rng(42)

    # Synthetic 6-feature process: normals ~ N(0,1); anomalies shifted+scaled.
    def make(n, anom_rate):
        n_a = int(n * anom_rate)
        n_n = n - n_a
        X = np.vstack([rng.normal(0, 1, (n_n, 6)),
                       rng.normal(1.8, 1.5, (n_a, 6))])
        y = np.concatenate([np.zeros(n_n, int), np.ones(n_a, int)])
        idx = rng.permutation(n)
        cols = [f"sensor_{i}" for i in range(6)]
        return pd.DataFrame(X[idx], columns=cols), y[idx]

    Xtr, ytr = make(500, 0.30)
    Xva, yva = make(200, 0.30)
    Xte, yte = make(200, 0.30)
    report(PASS, "synthetic data", f"train={len(ytr)} val={len(yva)} test={len(yte)}")

    sysm = HamaSystem(k_nodes=3, seed=42).fit(Xtr.values)
    report(PASS, "Fog tier fitted", f"K={len(sysm.nodes)} nodes, B1+B2 5-model ensemble")

    z = sysm.edge.tune(Xva.values, yva)
    report(PASS, "Edge pre-filter tuned", f"z_cut={z:.2f}")

    s_val = sysm.scores(Xva.values)
    tau = calibrate_threshold(yva, s_val)
    p = sysm.global_policy()
    sysm.set_global_policy(FogPolicy(p.w1, p.contamination, tau))
    report(PASS, "threshold calibrated on VALIDATION only", f"tau={tau:.3f}")

    m0 = classification_metrics(yte, sysm.scores(Xte.values), tau)
    report(PASS, "detection (pre-PPO)",
           f"F1={m0['f1']:.3f} P={m0['precision']:.3f} R={m0['recall']:.3f}")

    agg = sysm.global_policy()
    report(PASS, "federated-style aggregation",
           f"theta_global=(w1={agg.w1:.3f}, rho={agg.contamination:.3f}, tau={agg.tau:.3f})")

    from hama.agents.evolution import evolve_policy
    _, best = evolve_policy(sysm, Xva.values, yva, total_timesteps=32,
                            seed=42, window=64)
    report(PASS, "PPO policy adaptation (stable-baselines3)",
           f"w1={best.w1:.3f} rho={best.contamination:.3f} tau={best.tau:.3f}")

    m1 = classification_metrics(yte, sysm.scores(Xte.values), best.tau)
    report(PASS, "detection (post-PPO)", f"F1={m1['f1']:.3f}")

    lat = measure_latency(lambda X: sysm.predict(X, tau=best.tau), Xte.values,
                          boundary="edge filter + routed fog node, CPU")
    budget = "within" if lat.per_sample_ms < 100 else "EXCEEDS"
    report(PASS, "end-to-end latency",
           f"{lat.per_sample_ms:.3f} ms/sample ({budget} 100 ms budget)")

    from hama.agents.meta import AgentE
    scores_te = sysm.scores(Xte.values)
    alerts = np.flatnonzero(scores_te >= best.tau)
    if len(alerts):
        e = AgentE(sysm.nodes[0], list(Xte.columns), Xtr.values)
        attr = e.explain(Xte.values[[alerts[int(np.argmax(scores_te[alerts]))]]])[0]
        report(PASS, "SHAP attribution (real TreeExplainer)",
               ", ".join(f"{f}={v:+.2f}" for f, v in attr))
    else:
        report(WARN, "SHAP attribution", "no alerts fired on synthetic data")

    from hama.baselines.systems import Baseline1Static, Baseline2RuleBased
    b1 = Baseline1Static(seed=42).fit(Xtr.values, Xva.values, yva)
    b2 = Baseline2RuleBased(seed=42).fit(Xtr.values, Xva.values, yva)
    mb1 = classification_metrics(yte, b1.scores(Xte.values), b1.tau)
    mb2 = classification_metrics(yte, b2.scores(Xte.values), b2.tau)
    report(PASS, "baselines run",
           f"BL1 F1={mb1['f1']:.3f}, BL2 F1={mb2['f1']:.3f}, HAMA F1={m1['f1']:.3f}")

except Exception:
    report(FAIL, "pipeline", "exception (traceback below)")
    traceback.print_exc()
    failures.append("pipeline")

elapsed = time.perf_counter() - t_start

# ------------------------------------------------------------------- 6. Ollama
print("\n6. Optional: locally hosted SLM (Agent C)")
try:
    if "--skip-slm" in sys.argv:
        raise RuntimeError("skipped by flag")
    from hama.agents.response import AgentC, model_memory_mb
    rec = AgentC().generate(0.9, ["sensor_0", "sensor_3"], "synthetic test asset")
    mem = model_memory_mb()  # query AFTER generation, while model is resident
    mem_s = f"{mem:.0f} MB resident" if mem else "resident size unavailable"
    report(PASS, "Ollama SLM reachable",
           f"{rec['model']}, {rec['latency_s']:.1f}s, {mem_s}")
except Exception as e:
    report(WARN, "Ollama not reachable",
           f"{type(e).__name__} - install ollama.com + 'ollama pull llama3.2:1b' "
           "to exercise Agent C (all other components verified above)")
    warnings_.append("ollama")

# ------------------------------------------------------------------- verdict
print("\n" + "=" * 68)
if failures:
    print(f"RESULT: FAILED ({len(failures)} problem(s)): {', '.join(failures)}")
    sys.exit(1)
print(f"RESULT: PASS - full HAMA pipeline ran end to end in {elapsed:.0f}s.")
if warnings_:
    print(f"        {len(warnings_)} non-blocking warning(s): {', '.join(warnings_)}")
print("\nThis verified that the code RUNS. To reproduce the paper's NUMBERS,")
print("obtain the datasets (README -> 'Getting the data') and run:")
print("    python scripts/run_experiments.py --seeds 5")
print("    python scripts/analyze_stats.py")
print("=" * 68)
sys.exit(0)
