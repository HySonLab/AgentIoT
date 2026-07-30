"""The ONE statistics table (R3-1).

Reads results/experiments.jsonl and produces, from actual computed values:
  * per system x dataset: mean +/- std and 95% CI of F1/precision/recall/AUC
    across independent seed runs;
  * Welch t-tests (SEMAS vs each baseline) on seed-level F1, with Cohen's d;
  * drift-mode per-segment F1 trajectories.

Every statistical number in the manuscript must come from this script's
output (results/stats_table.md). Nothing is hand-entered.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import stats

rows = []
for line in Path("results/experiments.jsonl").read_text().splitlines():
    r = json.loads(line)
    rows.append({
        "system": r["system"], "dataset": r["dataset_key"], "seed": r["seed"],
        "f1": r["pooled"]["f1"], "precision": r["pooled"]["precision"],
        "recall": r["pooled"]["recall"], "roc_auc": r["pooled"]["roc_auc"],
        "delta_f1": r["delta_f1"], "per_sample_ms": r["per_sample_ms"],
        "seg_f1": [s["f1"] for s in r["segments"]],
    })
df = pd.DataFrame(rows)

lines = ["# Consolidated statistics (auto-generated — do not hand-edit)\n"]

lines.append("## Per-system results (mean ± std over seeds; n = #seeds)\n")
summary = (
    df.groupby(["dataset", "system"])
    .agg(n=("f1", "size"),
         f1_mean=("f1", "mean"), f1_std=("f1", "std"),
         prec_mean=("precision", "mean"), prec_std=("precision", "std"),
         rec_mean=("recall", "mean"), rec_std=("recall", "std"),
         auc_mean=("roc_auc", "mean"), auc_std=("roc_auc", "std"),
         dF1_mean=("delta_f1", "mean"),
         lat_ms=("per_sample_ms", "mean"))
    .round(4)
)
lines.append(summary.to_markdown() + "\n")

lines.append("## Welch t-tests on seed-level F1 (SEMAS vs baselines)\n")
lines.append("| Dataset | Comparison | ΔF1 | t | p | Cohen's d | Significant (α=0.05) |")
lines.append("|---|---|---|---|---|---|---|")
for dataset in sorted(df["dataset"].unique()):
    d = df[df["dataset"] == dataset]
    a = d[d["system"] == "semas"]["f1"].to_numpy()
    for base in ["baseline1", "baseline2"]:
        b = d[d["system"] == base]["f1"].to_numpy()
        if len(a) < 2 or len(b) < 2:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False)
        pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        cohen_d = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else float("nan")
        lines.append(
            f"| {dataset} | SEMAS vs {base} | {a.mean()-b.mean():+.4f} "
            f"| {t:.2f} | {p:.4f} | {cohen_d:.2f} "
            f"| {'yes' if p < 0.05 else 'no'} |"
        )
lines.append("")

drift = df[df["dataset"].str.contains("drift")]
if len(drift):
    lines.append("## Drift-mode per-segment F1 (mean over seeds)\n")
    for system in sorted(drift["system"].unique()):
        segs = np.mean(np.stack(drift[drift["system"] == system]["seg_f1"]), axis=0)
        traj = " -> ".join(f"{v:.3f}" for v in segs)
        lines.append(f"* {system}: {traj}")
    lines.append("")

out = Path("results/stats_table.md")
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
print(f"\nSaved -> {out}")
