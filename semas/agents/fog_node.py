"""A fog node: B1 + B2 detectors and B3 consensus over a data partition.

K nodes are instantiated over disjoint partitions of the training stream,
which is what makes the parameter-aggregation mechanism (semas/aggregation.py)
meaningful (K>1), addressing R1-Q6 / R3-3.
"""

from dataclasses import dataclass, field

import numpy as np

from .detectors import AgentB1, AgentB2Ensemble, AgentB3Consensus


@dataclass
class FogPolicy:
    """The locally adaptable policy parameters of one fog node."""
    w1: float = 0.5
    contamination: float = 0.3
    tau: float = 0.5  # alert threshold on the consensus score

    def as_vector(self) -> np.ndarray:
        return np.array([self.w1, self.contamination, self.tau], dtype=float)

    @classmethod
    def from_vector(cls, v) -> "FogPolicy":
        return cls(w1=float(v[0]), contamination=float(v[1]), tau=float(v[2]))


class FogNode:
    def __init__(self, node_id: int, seed: int = 42, policy: FogPolicy | None = None):
        self.node_id = node_id
        self.seed = seed
        self.policy = policy or FogPolicy()
        self.n_samples = 0  # data volume for data-proportional aggregation

    def fit(self, X_partition):
        self.n_samples = len(X_partition)
        self.b1 = AgentB1(
            contamination=self.policy.contamination, seed=self.seed + self.node_id
        ).fit(X_partition)
        self.b2 = AgentB2Ensemble(
            contamination=self.policy.contamination, seed=self.seed + 100 + self.node_id
        ).fit(X_partition)
        self.b3 = AgentB3Consensus(w1=self.policy.w1)
        return self

    def apply_policy(self, policy: FogPolicy, retrain_contamination: bool = False):
        self.policy = policy
        self.b3.set_weights(policy.w1)
        if retrain_contamination:
            self.b1.set_contamination(policy.contamination)

    def consensus_scores(self, X) -> np.ndarray:
        return self.b3.fuse(self.b1.scores(X), self.b2.scores(X))

    def predict(self, X) -> np.ndarray:
        return (self.consensus_scores(X) >= self.policy.tau).astype(int)
