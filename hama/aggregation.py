"""Data-proportional parameter aggregation across K fog nodes (Eq. 14).

theta_global = sum_k (n_k * theta_k) / sum_k n_k

Aggregated parameters are the fog policies (consensus weight w1,
contamination rho, alert threshold tau) — not deep-network weights — as the
manuscript's appendix states. With K>1 nodes trained on disjoint partitions
this is a real (if lightweight) federated-style mechanism; the manuscript
must still call it "federated-style parameter aggregation", not federated
learning (R3-3).
"""

import numpy as np

from .agents.fog_node import FogNode, FogPolicy


def aggregate_policies(nodes: list[FogNode]) -> FogPolicy:
    weights = np.array([n.n_samples for n in nodes], dtype=float)
    weights = weights / weights.sum()
    stacked = np.stack([n.policy.as_vector() for n in nodes])
    return FogPolicy.from_vector(weights @ stacked)


def broadcast_policy(nodes: list[FogNode], policy: FogPolicy,
                     retrain_contamination: bool = False) -> None:
    for n in nodes:
        n.apply_policy(policy, retrain_contamination=retrain_contamination)
