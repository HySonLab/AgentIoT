"""Edge-tier agent: cheap pre-filtering.

The manuscript claims Edge agents pre-filter a majority of clearly-normal
samples before Fog inference. This implements that claim measurably: a
z-score envelope test (O(d) per sample) marks samples whose every feature
lies within `z_cut` standard deviations of the training mean as
"clearly normal"; only the remainder is sent to the Fog ensemble.

The filter's false-negative behavior is part of the evaluation: filtered
samples receive fog score 0 and prediction 0, so any anomaly wrongly
filtered at the edge shows up as a miss in the reported metrics.
"""

import numpy as np


class EdgeFilter:
    def __init__(self, z_cut: float = 2.0):
        self.z_cut = z_cut

    def fit(self, X_train):
        X = np.asarray(X_train)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-8
        return self

    def pass_mask(self, X) -> np.ndarray:
        """True = forward to fog; False = filtered as clearly normal."""
        z = np.abs((np.asarray(X) - self.mu) / self.sd)
        return (z > self.z_cut).any(axis=1)

    def tune(self, X_val, y_val, min_anomaly_pass: float = 0.99,
             grid: np.ndarray | None = None) -> float:
        """Pick the most aggressive z_cut whose validation anomaly
        pass-rate stays >= min_anomaly_pass. Uses validation labels only;
        this trades filter savings against a bounded miss risk, and the
        chosen operating point is reported in the manuscript."""
        grid = grid if grid is not None else np.arange(0.25, 4.01, 0.25)
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)
        anom = y_val == 1
        best = float(grid[0])
        for z in sorted(grid):
            self.z_cut = float(z)
            mask = self.pass_mask(X_val)
            pass_rate = mask[anom].mean() if anom.any() else 1.0
            if pass_rate >= min_anomaly_pass:
                best = float(z)  # larger z = more filtering, still safe
            else:
                break
        self.z_cut = best
        return best
