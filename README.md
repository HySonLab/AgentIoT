# HAMA — Hierarchical Adaptive Multi-Agent Architecture for IIoT Predictive Maintenance

Reference implementation for *"HAMA: A Hierarchical Adaptive Multi-Agent Architecture for Industrial
IoT Predictive Maintenance"*
(IEEE Access, manuscript **Access-2026-28815**, under review).

Every number in the manuscript is produced by the scripts here and written to
`results/` as machine-readable artifacts. Nothing in the paper's results
section is hand-entered.

---

## Quick start for reviewers

**You do not need any dataset to check that this code works.** One command
runs the entire HAMA pipeline end to end on synthetic data — Edge filter,
K=3 Fog nodes, 5-model ensemble, consensus, validation-only threshold
calibration, real PPO adaptation, real SHAP attribution, federated-style
aggregation, and both baselines:

```bash
pip install -r requirements.txt
python scripts/verify_install.py --skip-slm
```

Expected: a checklist ending in `RESULT: PASS`. Takes **3–4 minutes** on a
laptop CPU (most of it the PPO step). Drop `--skip-slm` to also exercise the
locally hosted language model, which adds ~1 minute and needs Ollama.

This proves the code *runs*. Reproducing the paper's *numbers* needs the
datasets — see below.

---

## Reproduction tiers

| Tier | Needs | Time | Reproduces |
|---|---|---|---|
| **0** | nothing | 3–4 min | Pipeline runs end to end (`verify_install.py`) |
| **1** | nothing (Boiler is bundled) | ~15 min | Boiler detection results, e.g. F1 0.531 / ROC-AUC 0.647 |
| **2** | C-MAPSS (auto-download) | ~20 min | RUL benchmark: MAE 11.20 / RMSE 15.95 |
| **3** | + Wind SCADA (manual, Kaggle) | several hours | Every table in the paper |

Tiers 0–2 need **no account and no manual download**. Only the Wind SCADA
tables require fetching a dataset by hand.

### Tier 1 — the bundled dataset

The Boiler Emulator data ships with this repository, so this runs on a fresh
clone with nothing else fetched:

```bash
python scripts/run_experiments.py --datasets boiler_static boiler_drift --seeds 5
```

### Tier 2 — the externally comparable result

C-MAPSS is public and downloads automatically. This is the paper's only
result directly comparable to published literature, so it is the most
useful single check:

```bash
python scripts/get_data.py        # downloads NASA C-MAPSS (~12 MB)
python scripts/run_phase3.py      # RUL + supervised deep baselines
```

Expect `MAE=11.20, RMSE=15.95` cycles (last-cycle protocol), against
RMSE 16.14 for the comparable published LSTM we cite.

### Tier 2 — everything

```bash
python scripts/run_experiments.py --seeds 5   # main sweep (resumable)
python scripts/run_ablation.py
python scripts/run_k_ablation.py              # volume-controlled K ablation
python scripts/run_robustness.py              # noise / missing / prevalence
python scripts/run_ae_and_sensitivity.py
python scripts/measure_resources.py           # measured per-tier memory + CPU
python scripts/analyze_stats.py               # THE statistics table
```

`run_experiments.py` is resumable: it appends one JSON line per completed run
to `results/experiments.jsonl` and skips runs already present, so you can
interrupt and restart it.

---

## Getting the data

Run `python scripts/get_data.py --check` at any time for status.

| Dataset | How to obtain | Redistributed here? |
|---|---|---|
| **Boiler Emulator** | Already in `dataset/` — nothing to do | **Yes** — open access under CC BY |
| **NASA C-MAPSS** | `python scripts/get_data.py` (automatic) | No — downloaded from NASA |
| **Wind Turbine SCADA** | Kaggle; URL printed by `get_data.py` | No — third-party terms |

The Boiler Emulator dataset is bundled because it is open access under CC BY
(IEEE DataPort, [doi:10.21227/awav-bn36](https://dx.doi.org/10.21227/awav-bn36))
and only 1.3 MB, so the Boiler results reproduce with no account and no
download. **Attribution is a licence condition** — if you use it, cite
Shohet, Kandil & McArthur (2019); see
[`dataset/BOILER_DATASET_LICENSE.md`](dataset/BOILER_DATASET_LICENSE.md).

Beyond that we do not redistribute datasets we do not own. Every loader fails
with an explicit message telling you where the file should go and how to get
it, rather than a bare `FileNotFoundError`.

### Verifying your inputs

```bash
python scripts/get_data.py --verify
```

`dataset/CHECKSUMS.sha256` records the SHA-256 of every file behind the
published numbers, so you can confirm your copies are byte-identical instead
of assuming it. A mismatch means the numbers will not reproduce exactly —
better to find that out before a multi-hour run than after.

---

## What this system is

- **Edge** (`hama/agents/edge.py`) — O(d) z-score pre-filter; cutoff tuned on
  validation data under an anomaly-pass-rate constraint.
- **Fog**, K=3 nodes on disjoint training partitions
  (`hama/agents/fog_node.py`, `detectors.py`) — Agent B1 (Isolation Forest),
  Agent B2 (5-model ensemble: IF, OC-SVM, LOF, Elliptic Envelope, second IF;
  soft voting), Agent B3 (weighted consensus). Agent C (`response.py`)
  generates operator-facing text with a locally hosted SLM (Llama-3.2-1B via
  Ollama), grounded in SHAP attributions.
- **Cloud** (`hama/agents/evolution.py`) — Agent D, real PPO via
  stable-baselines3 over `(w1, ρ, τ)` with the manuscript's reward;
  `meta.py` — Agent E, real SHAP TreeExplainer attributions;
  `hama/aggregation.py` — data-proportional federated-style aggregation
  across the K fog nodes.

## Evaluation invariants (enforced in code, not just claimed)

1. Thresholds are calibrated on the **validation split only** — never on test
   labels (`hama/evaluation.py`).
2. Latency is measured end-to-end, feature vector in → decision out, with the
   boundary stated (`measure_latency`) — never from a sub-timer.
3. Every seed flows through `hama/seeding.py`, so "N seeds" means N genuinely
   different runs.
4. All statistics come from `scripts/analyze_stats.py`, computed directly from
   per-seed logs.
5. RUL is evaluated only on C-MAPSS, which has real run-to-failure labels.
   No synthetic RUL targets anywhere.

## Headline findings — reported as-is, including the null result

- PPO-based adaptation is **statistically indistinguishable** from both a
  static and a rule-based adaptive baseline on detection F1 (all pairwise
  Welch's t-tests p > 0.5, 5 seeds, every condition). We report this directly.
- All systems meet the 100 ms real-time budget (0.27–3.27 ms measured, CPU).
  HAMA is **not** uniformly faster than the simpler baselines.
- The volume-controlled K-ablation shows **no** accuracy effect from the
  multi-node Fog tier; its value is architectural, not accuracy.
- C-MAPSS FD001 RUL: MAE 11.20 / RMSE 15.95 cycles, comparable to published
  results.
- A supervised MLP beats every unsupervised system on labelled Boiler data
  (F1 0.906). HAMA's scope is label-scarce, explainability-first deployment,
  and the paper says so.

## Environment

Python **3.10+** required (PEP 604 annotations); developed and measured on
3.13.5. `requirements.txt` pins exact versions;
`scripts/verify_install.py` reports any drift from them.

All measured figures in the paper come from: 8 logical CPU cores, 17 GB RAM,
**no GPU**, Windows 11; SLM served by Ollama 0.32 (llama3.2:1b, 1.52 GB
resident, 35–82 s per response on CPU).

## Layout

```
hama/            # the system (agents, data loaders, evaluation, experiment protocol)
scripts/          # everything runnable; each writes to results/
results/          # machine-readable artifacts backing every number in the paper
HAMA_rebuilt.ipynb   # narrative walkthrough with outputs already executed
```
