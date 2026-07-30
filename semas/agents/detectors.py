"""Fog-tier detection agents.

Agent B1: Isolation Forest (statistical detector).
Agent B2: 5-model heterogeneous ensemble — exactly the configuration the
          manuscript appendix specifies (IF-200, One-Class SVM RBF nu=0.25,
          LOF k=20, Elliptic Envelope, secondary IF with a different seed).
          The previous implementation was a Transformer fit on all-zero
          labels; it is replaced entirely.
Agent B3: weighted consensus over B1/B2 scores with adaptable weights.

All detectors expose `scores(X)` where HIGHER = more anomalous, min-max
normalized to [0,1] using statistics from the *training* score distribution
(never from the evaluation batch, which would leak batch composition).
"""

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


class _ScoreNormalizer:
    """Min-max normalization frozen on training scores."""

    def fit(self, train_scores: np.ndarray):
        self.lo = float(np.min(train_scores))
        self.hi = float(np.max(train_scores))
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return np.clip((scores - self.lo) / (self.hi - self.lo + 1e-12), 0.0, 1.0)


class AgentB1:
    """Isolation Forest statistical detector."""

    def __init__(self, contamination: float = 0.3, seed: int = 42, n_estimators: int = 200):
        self.contamination = contamination
        self.seed = seed
        self.n_estimators = n_estimators
        self._fit_X = None

    def fit(self, X):
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=min(256, len(X)),
            random_state=self.seed,
        ).fit(X)
        self._fit_X = X
        self.norm = _ScoreNormalizer().fit(-self.model.score_samples(X))
        return self

    def set_contamination(self, contamination: float):
        """Policy-driven retrain with a new contamination value."""
        contamination = float(np.clip(contamination, 0.01, 0.5))
        if abs(contamination - self.contamination) > 1e-3 and self._fit_X is not None:
            self.contamination = contamination
            self.fit(self._fit_X)

    def scores(self, X) -> np.ndarray:
        return self.norm.transform(-self.model.score_samples(X))


@dataclass
class _Member:
    name: str
    model: object
    norm: _ScoreNormalizer


class AgentB2Ensemble:
    """5-model heterogeneous ensemble with soft (mean-score) voting.

    Soft voting over normalized scores is used rather than hard majority
    voting so that B3's weighted consensus receives a continuous signal;
    the manuscript should state this explicitly.
    """

    def __init__(self, contamination: float = 0.3, seed: int = 42, max_fit: int = 20000):
        self.contamination = contamination
        self.seed = seed
        self.max_fit = max_fit  # OCSVM/LOF are O(n^2); subsample fit set

    def fit(self, X):
        X = np.asarray(X)
        rng = np.random.default_rng(self.seed)
        fit_idx = (
            rng.choice(len(X), self.max_fit, replace=False)
            if len(X) > self.max_fit
            else np.arange(len(X))
        )
        Xf = X[fit_idx]
        cont = float(np.clip(self.contamination, 0.01, 0.5))

        defs = [
            ("if_200", IsolationForest(n_estimators=200, contamination=cont,
                                       max_samples=min(256, len(Xf)), random_state=self.seed)),
            ("ocsvm", OneClassSVM(kernel="rbf", nu=0.25, gamma="scale")),
            ("lof", LocalOutlierFactor(n_neighbors=20, novelty=True)),
            ("ee", EllipticEnvelope(contamination=cont, support_fraction=0.9,
                                    random_state=self.seed)),
            ("if_2", IsolationForest(n_estimators=100, contamination=cont,
                                     max_samples=min(512, len(Xf)),
                                     random_state=self.seed + 1)),
        ]
        self.members = []
        for name, model in defs:
            model.fit(Xf)
            raw = self._raw_scores(name, model, Xf)
            self.members.append(_Member(name, model, _ScoreNormalizer().fit(raw)))
        return self

    @staticmethod
    def _raw_scores(name: str, model, X) -> np.ndarray:
        # decision_function: higher = more normal -> negate
        return -model.decision_function(X)

    def member_scores(self, X) -> dict:
        X = np.asarray(X)
        return {
            m.name: m.norm.transform(self._raw_scores(m.name, m.model, X))
            for m in self.members
        }

    def scores(self, X) -> np.ndarray:
        per = self.member_scores(X)
        return np.mean(list(per.values()), axis=0)


class AgentB3Consensus:
    """Weighted consensus: a_fog = w1 * a_B1 + w2 * a_B2, w1 + w2 = 1."""

    def __init__(self, w1: float = 0.5):
        self.set_weights(w1)

    def set_weights(self, w1: float):
        self.w1 = float(np.clip(w1, 0.0, 1.0))
        self.w2 = 1.0 - self.w1

    def fuse(self, s_b1: np.ndarray, s_b2: np.ndarray) -> np.ndarray:
        return self.w1 * s_b1 + self.w2 * s_b2
