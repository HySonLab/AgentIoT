"""Comparison systems, re-implemented honestly in the new framework.

To isolate the contribution of *adaptation and coordination* (rather than
detector choice), all three systems share the same detector stack
(B1 Isolation Forest + B2 5-model ensemble + weighted consensus):

  * Baseline1 (static):     single fog node (K=1), threshold calibrated
                            once on validation, never adapted.
  * Baseline2 (rule-based): single fog node; between evaluation segments it
                            applies the manuscript's heuristic rules
                            (contamination +/-0.02 by F1 band; threshold
                            step by precision-recall imbalance).
  * HAMA (PPO, K=3):       hama.system.HamaSystem + PPO evolution.

Adaptation protocol (drift experiments): the evaluation stream is processed
in sequential segments; after each segment the adaptive systems receive that
segment's labels (simulating delayed operator feedback / fault confirmation
- stated as an assumption in the manuscript) and may update their policy
before the next segment. Test labels are never used to tune anything applied
to the SAME segment.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from ..agents.fog_node import FogNode, FogPolicy
from ..evaluation import calibrate_threshold, classification_metrics


class Baseline1Static:
    """Single-node detector stack, calibrated once, never adapted."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def fit(self, X_train, X_val, y_val):
        self.node = FogNode(node_id=0, seed=self.seed).fit(np.asarray(X_train))
        s_val = self.node.consensus_scores(np.asarray(X_val))
        self.tau = calibrate_threshold(y_val, s_val)
        self.node.policy.tau = self.tau
        return self

    def scores(self, X) -> np.ndarray:
        return self.node.consensus_scores(np.asarray(X))

    def adapt(self, X_seen, y_seen):
        pass  # static by definition


class Baseline2RuleBased(Baseline1Static):
    """Adds the manuscript's three rule-based policy updates between segments:
    contamination band, threshold step, and ensemble-weight redistribution.

    Note (contamination): with a fixed random_state, IsolationForest's
    score_samples ranking is provably invariant to the contamination
    parameter - it only shifts sklearn's internal .predict() cutoff, which
    this system does not use (see hama/agents/detectors.py). The
    contamination rule is retained for fidelity to the manuscript's
    described mechanism, but it has NO effect on detection scores here;
    this is stated explicitly in the manuscript rather than left to look
    like an oscillating lever that quietly does nothing.
    """

    def adapt(self, X_seen, y_seen):
        s = self.scores(X_seen)
        m = classification_metrics(y_seen, s, self.tau)

        # Rule 1: contamination band (see docstring: a no-op on ranking here)
        rho = self.node.policy.contamination
        if m["f1"] < 0.6:
            rho += 0.02
        elif m["f1"] > 0.7:
            rho -= 0.02
        rho = float(np.clip(rho, 0.05, 0.5))
        if abs(rho - self.node.policy.contamination) > 1e-6:
            self.node.policy.contamination = rho
            self.node.b1.set_contamination(rho)

        # Rule 2: threshold step from precision-recall imbalance
        self.tau = float(np.clip(
            self.tau - 0.05 * (m["precision"] - m["recall"]), 0.05, 0.95
        ))
        self.node.policy.tau = self.tau

        # Rule 3: performance-based ensemble weight redistribution
        # (the manuscript's Baseline2 mechanism this class previously omitted)
        s_b1 = self.node.b1.scores(np.asarray(X_seen))
        s_b2 = self.node.b2.scores(np.asarray(X_seen))
        if len(np.unique(y_seen)) > 1:
            auc_b1 = roc_auc_score(y_seen, s_b1)
            auc_b2 = roc_auc_score(y_seen, s_b2)
            w1_new = self.node.policy.w1 + 0.1 * np.sign(auc_b1 - auc_b2) * abs(auc_b1 - auc_b2)
            self.node.policy.w1 = float(np.clip(w1_new, 0.1, 0.9))
            self.node.b3.set_weights(self.node.policy.w1)
