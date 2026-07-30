"""Measured (not estimated) per-tier resource footprint, including CPU/GPU
utilization during inference (R3-6).

Method:
  * Edge tier   : RSS delta for EdgeFilter + per-sample CPU time of the
                  z-score test (the only edge computation).
  * Fog tier    : RSS delta for one fully trained FogNode (B1 + 5-model B2),
                  plus serialized model size and per-sample inference time.
  * Cloud tier  : RSS delta for the PPO learner (SB3 MlpPolicy) and SHAP
                  explainer; PPO update time per iteration.
  * SLM (fog)   : resident size from Ollama's API and measured response
                  latency (from Agent C's log).
  * CPU utilization: measured via psutil's per-process `cpu_percent`,
                  sampled over a >=0.5s repeated-workload window per
                  component (a single inference call is too fast for a
                  stable instantaneous reading). Reported as % of one
                  logical core (100% = one core fully saturated); this
                  process never observed >100% across any component,
                  i.e. none of HAMA's inference paths are multi-threaded
                  in this implementation.
  * GPU utilization: N/A by construction - all experiments in this paper,
                  including the SLM, run on CPU-only hardware (no GPU
                  present on this machine; stated explicitly rather than
                  left as an implicit "not reported").

RSS is measured with psutil on this machine (CPU-only, 16GB RAM); the
manuscript reports these as measured values with the hardware stated.
"""

import io
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import psutil

from hama.data import load_boiler
from hama.seeding import set_seeds

PROC = psutil.Process()


def rss_mb() -> float:
    return PROC.memory_info().rss / 1e6


def pickled_mb(obj) -> float:
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    return buf.tell() / 1e6


def measure_cpu_percent(fn, min_duration: float = 0.5) -> float:
    """Per-process CPU utilization (% of one logical core) while repeatedly
    calling fn() for at least min_duration seconds. psutil requires two
    samples separated by a real interval to report a meaningful value."""
    PROC.cpu_percent(interval=None)  # prime/reset the internal counter
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < min_duration:
        fn()
    return PROC.cpu_percent(interval=None)


set_seeds(42)
ds = load_boiler(seed=42)
X = ds.X_train.values
report = {}

# ---- Edge tier ----------------------------------------------------------
from hama.agents.edge import EdgeFilter
r0 = rss_mb()
edge = EdgeFilter().fit(X)
t0 = time.perf_counter()
for _ in range(10):
    edge.pass_mask(ds.X_test.values)
edge_ms = (time.perf_counter() - t0) / 10 / len(ds.X_test) * 1000
edge_cpu_pct = measure_cpu_percent(lambda: edge.pass_mask(ds.X_test.values))
report["edge"] = {
    "rss_delta_mb": rss_mb() - r0,
    "model_size_mb": pickled_mb(edge),
    "per_sample_ms": edge_ms,
    "cpu_percent_of_one_core": edge_cpu_pct,
}

# ---- Fog tier -----------------------------------------------------------
from hama.agents.fog_node import FogNode
r0 = rss_mb()
node = FogNode(node_id=0, seed=42).fit(X)
t0 = time.perf_counter()
for _ in range(3):
    node.consensus_scores(ds.X_test.values)
fog_ms = (time.perf_counter() - t0) / 3 / len(ds.X_test) * 1000
fog_cpu_pct = measure_cpu_percent(lambda: node.consensus_scores(ds.X_test.values))
report["fog_node"] = {
    "rss_delta_mb": rss_mb() - r0,
    "model_size_mb": pickled_mb(node),
    "per_sample_ms": fog_ms,
    "cpu_percent_of_one_core": fog_cpu_pct,
}

# ---- Cloud tier ---------------------------------------------------------
from hama.system import HamaSystem
from hama.agents.evolution import PolicyEvolutionEnv
from stable_baselines3 import PPO

sys_ = HamaSystem(k_nodes=3, seed=42).fit(X)
env = PolicyEvolutionEnv(sys_, ds.X_val.values, ds.y_val, window=256, seed=42)
r0 = rss_mb()
ppo = PPO("MlpPolicy", env, n_steps=64, batch_size=64, seed=42, verbose=0)
t0 = time.perf_counter()
PROC.cpu_percent(interval=None)  # prime before the PPO learn() call below
ppo.learn(total_timesteps=64, progress_bar=False)
ppo_update_s = time.perf_counter() - t0
ppo_cpu_pct = PROC.cpu_percent(interval=None)  # over the just-completed learn() call
report["cloud_ppo"] = {
    "rss_delta_mb": rss_mb() - r0,
    "policy_size_mb": pickled_mb(ppo.policy),
    "update_64steps_s": ppo_update_s,
    "cpu_percent_of_one_core": ppo_cpu_pct,
}

from hama.agents.meta import AgentE
r0 = rss_mb()
agent_e = AgentE(sys_.nodes[0], ds.feature_names, X)
t0 = time.perf_counter()
agent_e.explain(ds.X_test.values[:20])
shap_ms = (time.perf_counter() - t0) / 20 * 1000
shap_cpu_pct = measure_cpu_percent(lambda: agent_e.explain(ds.X_test.values[:20]))
report["cloud_shap"] = {
    "rss_delta_mb": rss_mb() - r0,
    "per_alert_ms": shap_ms,
    "cpu_percent_of_one_core": shap_cpu_pct,
}

# ---- SLM (fog, event-driven) --------------------------------------------
# The SLM runs inside the separate Ollama server process, not this Python
# process, so CPU% must be sampled on the Ollama process itself.
from hama.agents.response import AgentC, model_memory_mb

ollama_procs = [p for p in psutil.process_iter(["name"])
                if p.info["name"] and "ollama" in p.info["name"].lower()]
for p in ollama_procs:
    p.cpu_percent(interval=None)  # prime

agent_c = AgentC()
rec = agent_c.generate(0.9, ["temp_diff", "Tsupply"], "industrial boiler")

ollama_cpu_pct = max((p.cpu_percent(interval=None) for p in ollama_procs), default=None)
report["slm"] = {
    "resident_mb_ollama": model_memory_mb(),
    "response_latency_s": rec["latency_s"],
    "model": rec["model"],
    "cpu_percent_ollama_process": ollama_cpu_pct,
    "gpu_percent": None,  # no GPU on this machine; CPU-only inference throughout
}

# ---- System-level -------------------------------------------------------
report["host"] = {
    "cpu_count": psutil.cpu_count(logical=True),
    "total_ram_gb": psutil.virtual_memory().total / 1e9,
    "process_rss_total_mb": rss_mb(),
    "gpu_present": False,
    "note": "All measurements on CPU-only hardware; GPU utilization is N/A "
            "(0%) throughout this study, including for the SLM.",
}

out = Path("results/resources.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
print(f"\nSaved -> {out}")
