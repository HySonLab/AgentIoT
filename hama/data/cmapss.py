"""NASA C-MAPSS FD001 loader with real RUL labels.

Replaces the synthetic (uniform-random) RUL targets of the previous
implementation with genuine run-to-failure degradation labels, enabling
MAE/RMSE comparable to the published literature (R1-Q2/Q3, R3-2).

Standard preprocessing (Saxena et al. 2008; common in RUL literature):
  * RUL = failure_cycle - current_cycle, capped at RUL_CAP=125
    (piecewise-linear degradation assumption).
  * Constant / near-constant sensors in FD001 are dropped.
  * Unit-based split: engines never straddle train/val (no leakage).
  * Test RUL: official RUL_FD001.txt at each test engine's last cycle.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

COLS = (
    ["unit", "cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)
# Constant/uninformative in FD001
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19", "op3"]
RUL_CAP = 125


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


def load_cmapss(
    data_dir: str | Path = "dataset/cmapss",
    fd: str = "FD001",
    seed: int = 42,
    val_fraction: float = 0.2,
):
    data_dir = Path(data_dir)
    _require(data_dir / f"train_{fd}.txt", f"NASA C-MAPSS {fd}",
             "Run 'python scripts/get_data.py' to download it automatically.")
    train = pd.read_csv(data_dir / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    test = pd.read_csv(data_dir / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLS)
    rul_last = pd.read_csv(data_dir / f"RUL_{fd}.txt", header=None)[0].to_numpy()

    # Train RUL, capped
    max_cycle = train.groupby("unit")["cycle"].transform("max")
    train["RUL"] = np.minimum(max_cycle - train["cycle"], RUL_CAP)

    # Test RUL for every cycle (offset from official last-cycle RUL), capped
    last_cycle = test.groupby("unit")["cycle"].transform("max")
    unit_rul = pd.Series(rul_last, index=np.arange(1, len(rul_last) + 1))
    test["RUL"] = np.minimum(
        test["unit"].map(unit_rul) + (last_cycle - test["cycle"]), RUL_CAP
    )

    feat_cols = [c for c in COLS if c not in ("unit", "cycle") and c not in DROP_SENSORS]

    # Unit-based train/val split
    rng = np.random.default_rng(seed)
    units = train["unit"].unique()
    rng.shuffle(units)
    n_val = int(val_fraction * len(units))
    val_units = set(units[:n_val])
    is_val = train["unit"].isin(val_units)

    scaler = StandardScaler().fit(train.loc[~is_val, feat_cols])

    def pack(df: pd.DataFrame) -> dict:
        return {
            "X": pd.DataFrame(scaler.transform(df[feat_cols]), columns=feat_cols),
            "RUL": df["RUL"].to_numpy(dtype=float),
            "unit": df["unit"].to_numpy(),
            "cycle": df["cycle"].to_numpy(),
        }

    return {
        "name": f"cmapss_{fd}",
        "train": pack(train[~is_val]),
        "val": pack(train[is_val]),
        "test": pack(test),
        "feature_names": feat_cols,
        "meta": {
            "rul_cap": RUL_CAP,
            "n_train_units": len(units) - n_val,
            "n_val_units": n_val,
            "n_test_units": test["unit"].nunique(),
            "scaler": scaler,
            "official_last_cycle_rul": rul_last,
        },
    }
