"""Agent D: PPO-based policy evolution (Cloud tier).

This is actual Proximal Policy Optimization via stable-baselines3 — clipped
surrogate objective, GAE, learned actor-critic — replacing the previous
implementation's if/else heuristics that were mislabeled as PPO.

MDP design (documented for the manuscript):
  State  s_t = [F1, precision, recall, FPR, w1, contamination, tau]
         from the most recent evaluation window of the VALIDATION stream.
  Action a_t = deltas on (w1, contamination, tau), each clipped to +/-0.05
         (trust-region-style bounded updates at the environment level, in
         addition to PPO's ratio clipping).
  Reward r_t = 0.4*F1 + 0.3*Precision - 0.2*FPR - 0.1*latency_norm   (Eq. 6)
         with latency_norm = per-sample inference ms / 100ms budget.

Episodes iterate over random contiguous windows of the validation stream so
the policy cannot overfit one fixed batch. The test set is never touched
during evolution.
"""

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..agents.fog_node import FogPolicy
from ..evaluation import classification_metrics

REWARD_WEIGHTS = dict(f1=0.4, precision=0.3, fpr=0.2, latency=0.1)
LATENCY_BUDGET_MS = 100.0


class PolicyEvolutionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, system, X_val, y_val, window: int = 512,
                 episode_len: int = 16, seed: int = 42,
                 retrain_contamination: bool = False):
        super().__init__()
        self.system = system
        self.X_val = np.asarray(X_val)
        self.y_val = np.asarray(y_val)
        self.window = min(window, len(self.y_val))
        self.episode_len = episode_len
        self.retrain_contamination = retrain_contamination
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
        self.action_scale = np.array([0.05, 0.02, 0.05])  # w1, contamination, tau
        self.bounds_lo = np.array([0.05, 0.05, 0.05])
        self.bounds_hi = np.array([0.95, 0.50, 0.95])

    # ------------------------------------------------------------------
    def _eval_window(self) -> tuple[dict, float]:
        start = self.rng.integers(0, len(self.y_val) - self.window + 1)
        Xw = self.X_val[start : start + self.window]
        yw = self.y_val[start : start + self.window]
        t0 = time.perf_counter()
        scores = self.system.scores(Xw)
        per_sample_ms = (time.perf_counter() - t0) / len(Xw) * 1000.0
        tau = self.system.global_policy().tau
        m = classification_metrics(yw, scores, tau)
        y_pred = (scores >= tau).astype(int)
        neg = yw == 0
        m["fpr"] = float(y_pred[neg].mean()) if neg.any() else 0.0
        return m, per_sample_ms

    def _obs(self, m: dict) -> np.ndarray:
        p = self.system.global_policy()
        return np.array(
            [m["f1"], m["precision"], m["recall"], m["fpr"],
             p.w1, p.contamination, p.tau],
            dtype=np.float32,
        )

    @staticmethod
    def _reward(m: dict, per_sample_ms: float) -> float:
        w = REWARD_WEIGHTS
        lat = min(per_sample_ms / LATENCY_BUDGET_MS, 1.0)
        return w["f1"] * m["f1"] + w["precision"] * m["precision"] \
            - w["fpr"] * m["fpr"] - w["latency"] * lat

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        m, _ = self._eval_window()
        return self._obs(m), {}

    def step(self, action):
        v = self.system.global_policy().as_vector()
        v = np.clip(v + np.asarray(action) * self.action_scale,
                    self.bounds_lo, self.bounds_hi)
        self.system.set_global_policy(
            FogPolicy.from_vector(v),
            retrain_contamination=self.retrain_contamination,
        )
        m, per_sample_ms = self._eval_window()
        reward = self._reward(m, per_sample_ms)
        self.steps += 1
        terminated = False
        truncated = self.steps >= self.episode_len
        return self._obs(m), reward, terminated, truncated, {"metrics": m}


def evolve_policy(system, X_val, y_val, total_timesteps: int = 1024,
                  seed: int = 42, window: int = 512,
                  retrain_contamination: bool = False, verbose: int = 0):
    """Train PPO on the validation stream; leave the system at the best
    policy found (greedy final rollout)."""
    from stable_baselines3 import PPO

    env = PolicyEvolutionEnv(system, X_val, y_val, window=window, seed=seed,
                             retrain_contamination=retrain_contamination)
    model = PPO(
        "MlpPolicy", env,
        n_steps=64, batch_size=64, learning_rate=3e-4,
        gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        seed=seed, verbose=verbose,
    )
    initial_policy = system.global_policy()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    # Greedy rollout collects candidate policies; each candidate (plus the
    # initial policy) is then scored on the FULL validation stream, and the
    # best full-validation reward wins. Selecting on single random windows
    # is too noisy and can end worse than the starting point.
    candidates = [initial_policy]
    obs, _ = env.reset()
    for _ in range(env.episode_len):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, truncated, _ = env.step(action)
        candidates.append(system.global_policy())
        if truncated:
            break

    def full_val_reward(policy: FogPolicy) -> float:
        system.set_global_policy(policy, retrain_contamination=retrain_contamination)
        t0 = time.perf_counter()
        scores = system.scores(env.X_val)
        per_ms = (time.perf_counter() - t0) / len(env.X_val) * 1000.0
        m = classification_metrics(env.y_val, scores, policy.tau)
        y_pred = (scores >= policy.tau).astype(int)
        neg = env.y_val == 0
        m["fpr"] = float(y_pred[neg].mean()) if neg.any() else 0.0
        return PolicyEvolutionEnv._reward(m, per_ms)

    rewards = [full_val_reward(p) for p in candidates]
    best_policy = candidates[int(np.argmax(rewards))]
    system.set_global_policy(best_policy, retrain_contamination=retrain_contamination)
    return model, best_policy
