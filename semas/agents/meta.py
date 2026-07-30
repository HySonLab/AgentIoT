"""Agent E: SHAP-based oversight (Cloud tier).

Computes real SHAP values (TreeExplainer over Agent B1's Isolation Forest)
for alerted samples — the previous implementation imported shap and never
called it. Attributions feed two consumers:
  * Agent C's prompts (top contributing sensors), grounding the SLM's
    explanation in the actual detector evidence;
  * the policy audit trail (per-alert attribution logged with the alert).
"""

import numpy as np
import shap


class AgentE:
    def __init__(self, fog_node, feature_names: list[str], background, max_background: int = 200):
        bg = np.asarray(background)
        if len(bg) > max_background:
            idx = np.random.default_rng(0).choice(len(bg), max_background, replace=False)
            bg = bg[idx]
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(fog_node.b1.model, data=bg)

    def explain(self, X_alert, top_k: int = 3):
        """Per-sample top-k contributing features for alerted samples."""
        X_alert = np.asarray(X_alert)
        sv = self.explainer.shap_values(X_alert, check_additivity=False)
        out = []
        for row in np.atleast_2d(sv):
            order = np.argsort(-np.abs(row))[:top_k]
            out.append([
                (self.feature_names[i], float(row[i])) for i in order
            ])
        return out
