"""Phase 4 main sweep: systems x datasets x seeds.

Usage:
    python scripts/run_experiments.py --seeds 5 --ppo-timesteps 512

Writes one JSON line per run to results/experiments.jsonl (append; a run
already present is skipped, so the sweep is resumable).
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from semas.data import load_boiler, load_wind
from semas.experiment import run_system

SEEDS_BASE = [42, 123, 456, 789, 1024, 2048, 3141, 4096, 5150, 6174]


def dataset_configs(seed: int):
    return {
        "boiler_static": dict(loader=lambda: load_boiler(seed=seed), mode="static",
                              contiguous=False),
        "boiler_drift": dict(loader=lambda: load_boiler(seed=seed, split="severity_drift"),
                             mode="drift", contiguous=False),
        "wind_static": dict(loader=lambda: load_wind(), mode="static", contiguous=True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ppo-timesteps", type=int, default=512)
    ap.add_argument("--systems", nargs="+",
                    default=["baseline1", "baseline2", "semas"])
    ap.add_argument("--datasets", nargs="+",
                    default=["boiler_static", "boiler_drift", "wind_static"])
    args = ap.parse_args()

    out = Path("results/experiments.jsonl")
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["system"], r["dataset_key"], r["seed"], r["mode"]))

    seeds = SEEDS_BASE[: args.seeds]
    total = len(seeds) * len(args.systems) * len(args.datasets)
    i = 0
    for seed in seeds:
        cfgs = dataset_configs(seed)
        for ds_key in args.datasets:
            cfg = cfgs[ds_key]
            ds = None
            for system in args.systems:
                i += 1
                key = (system, ds_key, seed, cfg["mode"])
                if key in done:
                    print(f"[{i}/{total}] skip {key}")
                    continue
                if ds is None:
                    ds = cfg["loader"]()
                print(f"[{i}/{total}] run  {key} ...", flush=True)
                t0 = time.perf_counter()
                try:
                    r = run_system(system, ds, seed, mode=cfg["mode"],
                                   ppo_timesteps=args.ppo_timesteps,
                                   contiguous=cfg["contiguous"])
                    r["dataset_key"] = ds_key
                    r["wall_s"] = time.perf_counter() - t0
                    with out.open("a") as f:
                        f.write(json.dumps(r) + "\n")
                    print(f"          F1={r['pooled']['f1']:.3f} "
                          f"AUC={r['pooled']['roc_auc']:.3f} "
                          f"dF1={r['delta_f1']:+.3f} "
                          f"({r['wall_s']:.0f}s)", flush=True)
                except Exception:
                    print(f"          FAILED:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
