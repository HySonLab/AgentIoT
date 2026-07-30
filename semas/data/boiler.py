"""Boiler Emulator dataset loader with honest task construction.

Raw data: 27,280 rows = 31 conditions x 880 operating points.
  Classes: Nominal (air ratio 0.1 / 'Nominal'), Lean (0.05),
  ExcessAir (0.15-0.50), Fouling (F=0.01-0.46), Scaling (S=0.01-0.46).

Task construction (documented for the manuscript):
  * normal  = Class == 'Nominal'            (1,760 rows)
  * anomaly = any fault class, subsampled stratified-by-condition to a
              configurable prevalence (default 30%).

The previous implementation label-encoded Class alphabetically and treated
class==1 (Fouling only) as "anomaly", which mislabeled Lean/ExcessAir/Scaling
faults as normal. That is corrected here.

Rows are independent simulated operating points, NOT a time series, so a
time-based split is not applicable (this answers R1-Q4 for this dataset:
we use a stratified split here and a *severity-drift* split for the
robustness study; the genuinely temporal datasets - Wind SCADA, CMAPSS -
get time/unit-based splits).

Splits: 60/20/20 train/val/test, stratified by (Class, Condition).
Scaler is fit on train only. Thresholds must be calibrated on val only.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_FEATURES = ["Fuel_Mdot", "Tair", "Treturn", "Tsupply", "Water_Mdot"]


def _require(path, label, howto):
    """Fail with actionable guidance instead of a bare FileNotFoundError."""
    from pathlib import Path as _P
    p = _P(path)
    if not p.exists():
        raise FileNotFoundError(
            f"\n\n  Missing dataset: {label}\n"
            f"  Expected at: {p}\n"
            f"  {howto}\n"
            f"  Run 'python scripts/get_data.py' for status and download help.\n"
            f"  (No dataset is needed for 'python scripts/verify_install.py',\n"
            f"   which exercises the full pipeline on synthetic data.)\n"
        )
    return p


@dataclass
class DatasetSplit:
    name: str
    X_train: pd.DataFrame
    y_train: np.ndarray
    X_val: pd.DataFrame
    y_val: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    feature_names: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        parts = []
        for split, y in [("train", self.y_train), ("val", self.y_val), ("test", self.y_test)]:
            parts.append(f"{split}: n={len(y)}, anomaly={y.mean():.1%}")
        return f"[{self.name}] " + " | ".join(parts)


def _parse_condition(cond: str):
    """Return (mechanism, severity) from the Condition string."""
    cond = cond.strip()
    if cond == "Nominal":
        return "nominal", 0.0
    if cond.startswith("%="):
        return "air_ratio", float(cond[2:])
    if cond.startswith("F"):
        return "fouling", float(cond.split("=")[1])
    if cond.startswith("S"):
        return "scaling", float(cond.split("=")[1])
    raise ValueError(f"Unrecognized condition: {cond!r}")


def _engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Physics-motivated features. No rolling statistics: rows are
    independent operating points, so temporal aggregates would be invalid."""
    X = X.copy()
    X["temp_diff"] = X["Tsupply"] - X["Treturn"]
    X["temp_rise_per_fuel"] = X["temp_diff"] / (X["Fuel_Mdot"] + 1e-8)
    X["water_fuel_ratio"] = X["Water_Mdot"] / (X["Fuel_Mdot"] + 1e-8)
    X["supply_air_delta"] = X["Tsupply"] - X["Tair"]
    return X


def load_boiler(
    csv_path: str | Path = "dataset/Boiler_emulator_dataset.csv",
    seed: int = 42,
    anomaly_prevalence: float = 0.30,
    split: str = "stratified",  # or "severity_drift"
    drift_severity_quantile: float = 0.5,
) -> DatasetSplit:
    _require(csv_path, "Boiler Emulator",
             "Not redistributed in this repository; request from the "
             "corresponding author (see README).")
    df = pd.read_csv(csv_path)
    parsed = df["Condition"].map(_parse_condition)
    df["mechanism"] = parsed.map(lambda t: t[0])
    df["severity"] = parsed.map(lambda t: t[1])
    df["y"] = (df["Class"] != "Nominal").astype(int)

    rng = np.random.default_rng(seed)

    normal = df[df["y"] == 0]
    faults = df[df["y"] == 1]

    # Subsample faults stratified by Condition to reach target prevalence.
    n_anom_target = int(round(len(normal) * anomaly_prevalence / (1 - anomaly_prevalence)))
    per_cond = max(1, n_anom_target // faults["Condition"].nunique())
    sampled = faults.groupby("Condition", group_keys=False).sample(
        n=per_cond, random_state=seed
    )
    data = pd.concat([normal, sampled]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    X = _engineer_features(data[RAW_FEATURES])
    y = data["y"].to_numpy()
    strata = data["Class"].astype(str) + "|" + data["Condition"].astype(str)

    if split == "stratified":
        idx_train, idx_val, idx_test = _stratified_60_20_20(strata, rng)
    elif split == "severity_drift":
        # Train/val on mild faults (+ half the normals), test on severe faults
        # (+ the other half). Tests adaptation under unseen fault magnitudes.
        sev = data["severity"].to_numpy()
        fault_sev = sev[y == 1]
        cutoff = np.quantile(fault_sev, drift_severity_quantile)
        is_test = (y == 1) & (sev > cutoff)
        normal_idx = np.flatnonzero(y == 0)
        rng.shuffle(normal_idx)
        test_normals = normal_idx[: len(normal_idx) // 2]
        is_test[test_normals] = True
        pre_idx = np.flatnonzero(~is_test)
        rng.shuffle(pre_idx)
        n_val = int(0.25 * len(pre_idx))
        idx_val, idx_train = pre_idx[:n_val], pre_idx[n_val:]
        idx_test = np.flatnonzero(is_test)
    else:
        raise ValueError(f"Unknown split: {split}")

    scaler = StandardScaler().fit(X.iloc[idx_train])
    scale = lambda idx: pd.DataFrame(
        scaler.transform(X.iloc[idx]), columns=X.columns
    )

    return DatasetSplit(
        name="boiler",
        X_train=scale(idx_train), y_train=y[idx_train],
        X_val=scale(idx_val), y_val=y[idx_val],
        X_test=scale(idx_test), y_test=y[idx_test],
        feature_names=list(X.columns),
        meta={
            "split": split,
            "seed": seed,
            "anomaly_prevalence_target": anomaly_prevalence,
            "n_total": len(data),
            "scaler": scaler,
            "severity_test": data["severity"].to_numpy()[idx_test],
            "mechanism_test": data["mechanism"].to_numpy()[idx_test],
        },
    )


def _stratified_60_20_20(strata: pd.Series, rng: np.random.Generator):
    idx_train, idx_val, idx_test = [], [], []
    for _, group in strata.groupby(strata):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_tr, n_va = int(0.6 * n), int(0.2 * n)
        idx_train.extend(idx[:n_tr])
        idx_val.extend(idx[n_tr : n_tr + n_va])
        idx_test.extend(idx[n_tr + n_va :])
    return np.array(idx_train), np.array(idx_val), np.array(idx_test)
