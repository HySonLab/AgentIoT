"""HAMA system orchestrator: Edge filter -> K fog nodes -> consensus alerts.

Deployment model made explicit for the manuscript:
  * Each incoming sample is handled by exactly ONE fog node (round-robin
    routing here, standing in for sensor-to-nearest-node locality).
  * Nodes are trained on disjoint partitions of the training stream.
  * Policies evolve per node and are periodically unified by
    data-proportional aggregation (hama/aggregation.py).

Latency boundary (quoted in the paper): scaled feature vector in ->
binary decision out, including the edge filter and the routed fog node's
ensemble, measured on this machine's CPU. LLM response generation happens
asynchronously after an alert and is reported as a separate latency.
"""

import numpy as np

from .agents.edge import EdgeFilter
from .agents.fog_node import FogNode, FogPolicy
from .aggregation import aggregate_policies, broadcast_policy


class HamaSystem:
    def __init__(self, k_nodes: int = 3, seed: int = 42, z_cut: float = 2.0,
                 contiguous_partitions: bool = False):
        self.k = k_nodes
        self.seed = seed
        self.z_cut = z_cut
        self.contiguous = contiguous_partitions  # True for time-series data

    def fit(self, X_train):
        X = np.asarray(X_train)
        self.edge = EdgeFilter(z_cut=self.z_cut).fit(X)

        if self.contiguous:
            parts = np.array_split(np.arange(len(X)), self.k)
        else:
            rng = np.random.default_rng(self.seed)
            idx = rng.permutation(len(X))
            parts = np.array_split(idx, self.k)

        self.nodes = [
            FogNode(node_id=i, seed=self.seed).fit(X[p]) for i, p in enumerate(parts)
        ]
        return self

    # ---- policy plumbing -------------------------------------------------
    def global_policy(self) -> FogPolicy:
        return aggregate_policies(self.nodes)

    def set_global_policy(self, policy: FogPolicy, retrain_contamination: bool = False):
        broadcast_policy(self.nodes, policy, retrain_contamination=retrain_contamination)

    # ---- inference -------------------------------------------------------
    def _route(self, n: int) -> np.ndarray:
        return np.arange(n) % self.k

    def scores(self, X) -> np.ndarray:
        """Consensus anomaly scores; edge-filtered samples get score 0."""
        X = np.asarray(X)
        out = np.zeros(len(X))
        mask = self.edge.pass_mask(X)
        routes = self._route(len(X))
        for k in range(self.k):
            sel = mask & (routes == k)
            if sel.any():
                out[sel] = self.nodes[k].consensus_scores(X[sel])
        self.last_edge_filter_rate = 1.0 - float(mask.mean())
        return out

    def predict(self, X, tau: float | None = None) -> np.ndarray:
        tau = tau if tau is not None else self.global_policy().tau
        return (self.scores(X) >= tau).astype(int)
