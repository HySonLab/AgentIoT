"""End-to-end HAMA smoke test on the Boiler dataset.

Pipeline: K=3 fog nodes -> validation threshold -> PPO evolution ->
aggregation -> test evaluation -> SHAP -> SLM response, with honest
latency at every stage.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from hama.data import load_boiler
from hama.evaluation import calibrate_threshold, classification_metrics, measure_latency
from hama.seeding import set_seeds
from hama.system import HamaSystem
from hama.agents.fog_node import FogPolicy

SEED = 42
set_seeds(SEED)

print("1) Data")
ds = load_boiler(seed=SEED)
print("  ", ds.summary())

print("2) Fit HAMA (K=3 fog nodes)")
t0 = time.perf_counter()
sys_ = HamaSystem(k_nodes=3, seed=SEED).fit(ds.X_train.values)
print(f"   fit time: {time.perf_counter()-t0:.1f}s")

print("3) Tune edge filter + calibrate initial tau on VALIDATION")
z = sys_.edge.tune(ds.X_val.values, ds.y_val)
print(f"   tuned z_cut={z:.2f}")
s_val = sys_.scores(ds.X_val.values)
tau0 = calibrate_threshold(ds.y_val, s_val)
p = sys_.global_policy()
sys_.set_global_policy(FogPolicy(w1=p.w1, contamination=p.contamination, tau=tau0))
print(f"   tau0={tau0:.4f}, edge filter rate={sys_.last_edge_filter_rate:.1%}")

m_val = classification_metrics(ds.y_val, s_val, tau0)
print(f"   val  F1={m_val['f1']:.3f} P={m_val['precision']:.3f} R={m_val['recall']:.3f}")

s_test = sys_.scores(ds.X_test.values)
m0 = classification_metrics(ds.y_test, s_test, tau0)
print(f"   test (pre-PPO)  F1={m0['f1']:.3f} P={m0['precision']:.3f} "
      f"R={m0['recall']:.3f} AUC={m0['roc_auc']:.3f}")

print("4) PPO policy evolution on validation stream")
from hama.agents.evolution import evolve_policy
t0 = time.perf_counter()
model, best_policy = evolve_policy(sys_, ds.X_val.values, ds.y_val,
                                   total_timesteps=512, seed=SEED, window=256)
print(f"   PPO time: {time.perf_counter()-t0:.1f}s")
print(f"   evolved policy: w1={best_policy.w1:.3f}, rho={best_policy.contamination:.3f}, "
      f"tau={best_policy.tau:.3f}")

s_test2 = sys_.scores(ds.X_test.values)
m1 = classification_metrics(ds.y_test, s_test2, best_policy.tau)
print(f"   test (post-PPO) F1={m1['f1']:.3f} P={m1['precision']:.3f} "
      f"R={m1['recall']:.3f} AUC={m1['roc_auc']:.3f}")

print("5) Honest latency (feature vector in -> decision out)")
lat = measure_latency(lambda X: sys_.predict(X, tau=best_policy.tau),
                      ds.X_test.values,
                      boundary="edge filter + routed fog node ensemble, CPU")
print(f"   per-sample: {lat.per_sample_ms:.3f} ms | batch({lat.n_samples}): {lat.batch_ms:.0f} ms")

print("6) SHAP attribution for top alert")
from hama.agents.meta import AgentE
alerts = np.flatnonzero(s_test2 >= best_policy.tau)
agent_e = AgentE(sys_.nodes[0], ds.feature_names, ds.X_train.values)
top_alert = alerts[np.argmax(s_test2[alerts])]
attribution = agent_e.explain(ds.X_test.values[[top_alert]])[0]
print(f"   alert sample #{top_alert}, severity={s_test2[top_alert]:.2f}")
for feat, val in attribution:
    print(f"     {feat}: {val:+.3f}")

print("7) SLM response (llama3.2:1b via Ollama)")
from hama.agents.response import AgentC, model_memory_mb
agent_c = AgentC()
rec = agent_c.generate(
    severity=float(s_test2[top_alert]),
    top_features=[f for f, _ in attribution],
    context="industrial boiler; supply/return temperature and fuel/water flow sensors",
)
print(f"   latency: {rec['latency_s']:.1f}s, tokens: {rec['eval_tokens']}")
print(f"   memory (ollama ps): {model_memory_mb()} MB")
print("   response:")
print("   " + rec["response"].replace("\n", "\n   "))
