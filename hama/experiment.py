"""Experiment protocols for the comparative study (Phase 4).

Two evaluation modes:

* static:  fit -> calibrate on val -> single pass over the test set.
* drift:   the test stream is processed in `n_segments` sequential segments
           (for the boiler severity-drift split, segments are ordered by
           increasing fault severity distance from training). After each
           segment, adaptive systems receive that segment's labels and may
           adapt BEFORE seeing the next segment. Metrics are reported
           per-segment and pooled. No system is ever tuned on labels of a
           segment it has not yet finished processing.

All latencies: wall-clock per-sample, feature vector in -> decision out.
"""

import time

import numpy as np

from .agents.evolution import evolve_policy
from .agents.fog_node import FogPolicy
from .baselines.systems import Baseline1Static, Baseline2RuleBased
from .evaluation import calibrate_threshold, classification_metrics
from .seeding import set_seeds
from .system import HamaSystem


def _segments(n: int, k: int):
    edges = np.linspace(0, n, k + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(k)]


def _order_test(ds, n_segments: int, seed: int = 0):
    """Drift mode: build a stratified curriculum with escalating fault
    severity, WITHOUT collapsing to a degenerate class split.

    Sorting the full test set by raw severity is unsound here: normal
    samples carry severity 0 by construction, so a naive sort clusters
    every normal at the front and every anomaly at the back, leaving
    early segments with zero positives (uninformative for any adaptation
    rule, and pathological for PPO's reward). Instead: anomalies are
    binned by ascending severity (the actual drift signal); normals are
    shuffled and split evenly across segments (their severity value is
    not meaningful). This keeps class balance roughly constant across
    segments while typical fault severity increases.
    """
    sev = ds.meta.get("severity_test")
    y = ds.y_test
    if sev is None:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    anom_idx = np.flatnonzero(y == 1)
    anom_idx = anom_idx[np.argsort(sev[anom_idx], kind="stable")]
    norm_idx = np.flatnonzero(y == 0)
    rng.shuffle(norm_idx)

    anom_bins = np.array_split(anom_idx, n_segments)
    norm_bins = np.array_split(norm_idx, n_segments)
    order = []
    for a, n in zip(anom_bins, norm_bins):
        seg = np.concatenate([a, n])
        rng.shuffle(seg)
        order.append(seg)
    return np.concatenate(order), [len(s) for s in order]


def run_system(system_name: str, ds, seed: int, mode: str = "static",
               n_segments: int = 3, ppo_timesteps: int = 512,
               k_nodes: int = 3, contiguous: bool = False) -> dict:
    set_seeds(seed)
    Xtr = ds.X_train.values
    Xva, yva = ds.X_val.values, ds.y_val
    Xte, yte = ds.X_test.values, ds.y_test

    t_fit0 = time.perf_counter()
    if system_name == "hama":
        sys_ = HamaSystem(k_nodes=k_nodes, seed=seed,
                           contiguous_partitions=contiguous).fit(Xtr)
        sys_.edge.tune(Xva, yva)
        s_val = sys_.scores(Xva)
        tau0 = calibrate_threshold(yva, s_val)
        p = sys_.global_policy()
        sys_.set_global_policy(FogPolicy(p.w1, p.contamination, tau0))
        evolve_policy(sys_, Xva, yva, total_timesteps=ppo_timesteps, seed=seed,
                      window=min(256, len(yva)))
        scores_fn = sys_.scores
        get_tau = lambda: sys_.global_policy().tau

        def adapt_fn(X_seen, y_seen):
            evolve_policy(sys_, X_seen, y_seen,
                          total_timesteps=max(256, ppo_timesteps // 2),
                          seed=seed, window=min(256, len(y_seen)))
    elif system_name == "baseline1":
        b = Baseline1Static(seed=seed).fit(Xtr, Xva, yva)
        scores_fn, get_tau, adapt_fn = b.scores, lambda: b.tau, b.adapt
    elif system_name == "baseline2":
        b = Baseline2RuleBased(seed=seed).fit(Xtr, Xva, yva)
        scores_fn, get_tau, adapt_fn = b.scores, lambda: b.tau, b.adapt
    else:
        raise ValueError(system_name)
    fit_s = time.perf_counter() - t_fit0

    if mode == "drift":
        order, seg_sizes = _order_test(ds, n_segments, seed=seed)
        bounds = np.cumsum([0] + seg_sizes)
        segs = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_segments)]
    else:
        order = np.arange(len(yte))
        segs = _segments(len(yte), 1)
    Xte, yte = Xte[order], yte[order]

    seg_metrics, all_scores, all_taus, adapt_s = [], [], [], 0.0
    for i, seg in enumerate(segs):
        tau = get_tau()
        t0 = time.perf_counter()
        s = scores_fn(Xte[seg])
        infer_ms = (time.perf_counter() - t0) / len(seg) * 1000.0
        m = classification_metrics(yte[seg], s, tau)
        m.update(segment=i, per_sample_ms=infer_ms, tau_used=tau)
        seg_metrics.append(m)
        all_scores.append(s)
        all_taus.append(np.full(len(seg), tau))
        if mode == "drift" and i < len(segs) - 1:
            t0 = time.perf_counter()
            adapt_fn(Xte[seg], yte[seg])
            adapt_s += time.perf_counter() - t0

    pooled_scores = np.concatenate(all_scores)
    pooled_pred = (pooled_scores >= np.concatenate(all_taus)).astype(int)
    pooled = classification_metrics(yte, pooled_scores, 0.5)  # AUC from scores
    pooled.update(
        f1=float(_f1(yte, pooled_pred)),
        precision=float(_prec(yte, pooled_pred)),
        recall=float(_rec(yte, pooled_pred)),
    )

    return {
        "system": system_name, "dataset": ds.name, "seed": seed, "mode": mode,
        "pooled": pooled, "segments": seg_metrics,
        "fit_time_s": fit_s, "adapt_time_s": adapt_s,
        "delta_f1": seg_metrics[-1]["f1"] - seg_metrics[0]["f1"]
        if len(seg_metrics) > 1 else 0.0,
        "per_sample_ms": float(np.mean([m["per_sample_ms"] for m in seg_metrics])),
    }


def _f1(y, p):
    from sklearn.metrics import f1_score
    return f1_score(y, p, zero_division=0)


def _prec(y, p):
    from sklearn.metrics import precision_score
    return precision_score(y, p, zero_division=0)


def _rec(y, p):
    from sklearn.metrics import recall_score
    return recall_score(y, p, zero_division=0)
