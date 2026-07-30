"""Wind Turbine SCADA dataset loader with time-based splits.

Raw data: 49,027 SCADA records (10-min cadence, ~66 channels) and 553
timestamped fault events of 5 types (FF, EF, AF, GF, MF).

The previous implementation inner-merged SCADA and fault tables on *exact*
timestamps and then undersampled to 500 rows, discarding ~99% of the data
and yielding a 100-sample test set (the ceiling effect Reviewer 3 flagged).

Task construction here (documented for the manuscript):
  * y(t) = 1 if any fault event occurs within (t, t + horizon]  (default 60 min)
    -> a genuinely *predictive* label: detect pre-fault signatures.
  * Full time series is kept; NO undersampling.
  * Chronological 60/20/20 split (train on the past, test on the future),
    directly answering R1-Q4's online-deployment concern.

Scaler fit on train only; temporal derivative features are computed on the
time-sorted series before splitting (backward differences only - no lookahead).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .boiler import DatasetSplit

# Availability/production bookkeeping columns (not sensor physics), as
# excluded in the original study, plus identifiers.
EXCLUDE_COLS = [
    "DateTime", "Time", "Error",
    "WEC: ava. windspeed", "WEC: ava. available P from wind",
    "WEC: ava. available P technical reasons",
    "WEC: ava. Available P force majeure reasons",
    "WEC: ava. Available P force external reasons",
    "WEC: max. windspeed", "WEC: min. windspeed",
    "WEC: Operating Hours", "WEC: Production kWh",
    "WEC: Production minutes",
]


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


def load_wind(
    data_dir: str | Path = "dataset/iiot-data-of-wind-turbine",
    horizon_minutes: int = 60,
    key_derivative_cols: int = 8,
) -> DatasetSplit:
    data_dir = Path(data_dir)
    _require(data_dir / "scada_data.csv", "Wind Turbine SCADA",
             "Download from Kaggle (link in scripts/get_data.py); place "
             "scada_data.csv, fault_data.csv, status_data.csv here.")
    scada = pd.read_csv(data_dir / "scada_data.csv")
    fault = pd.read_csv(data_dir / "fault_data.csv")

    scada["DateTime"] = pd.to_datetime(scada["DateTime"], format="mixed")
    fault["DateTime"] = pd.to_datetime(fault["DateTime"], format="mixed")
    scada = scada.sort_values("DateTime").reset_index(drop=True)

    # Restrict to the period covered by fault labels: beyond the last
    # recorded fault event the negative labels are unverifiable (the SCADA
    # series continues ~3 months past the fault log; a chronological split
    # would otherwise yield a zero-anomaly test period).
    label_end = fault["DateTime"].max() + pd.Timedelta(minutes=horizon_minutes)
    scada = scada[scada["DateTime"] <= label_end].reset_index(drop=True)

    # Predictive labels: fault within (t, t + horizon]
    fault_times = np.sort(fault["DateTime"].to_numpy())
    t = scada["DateTime"].to_numpy()
    horizon = np.timedelta64(horizon_minutes, "m")
    # for each record, index of first fault event strictly after t
    nxt = np.searchsorted(fault_times, t, side="right")
    has_next = nxt < len(fault_times)
    y = np.zeros(len(scada), dtype=int)
    y[has_next] = (fault_times[nxt[has_next]] - t[has_next]) <= horizon

    feat_cols = [c for c in scada.columns if c not in EXCLUDE_COLS]
    X = scada[feat_cols].apply(pd.to_numeric, errors="coerce")
    X = X.ffill().bfill()

    # Backward-difference derivatives on the most variable channels
    variances = X.var().sort_values(ascending=False)
    for col in variances.index[:key_derivative_cols]:
        X[f"d_{col}"] = X[col].diff().fillna(0.0)

    n = len(X)
    n_tr, n_va = int(0.6 * n), int(0.2 * n)
    idx_train = np.arange(0, n_tr)
    idx_val = np.arange(n_tr, n_tr + n_va)
    idx_test = np.arange(n_tr + n_va, n)

    scaler = StandardScaler().fit(X.iloc[idx_train])
    scale = lambda idx: pd.DataFrame(scaler.transform(X.iloc[idx]), columns=X.columns)

    return DatasetSplit(
        name="wind",
        X_train=scale(idx_train), y_train=y[idx_train],
        X_val=scale(idx_val), y_val=y[idx_val],
        X_test=scale(idx_test), y_test=y[idx_test],
        feature_names=list(X.columns),
        meta={
            "split": "chronological_60_20_20",
            "horizon_minutes": horizon_minutes,
            "n_total": n,
            "n_fault_events": len(fault),
            "scaler": scaler,
            "train_period": (str(scada["DateTime"].iloc[0]), str(scada["DateTime"].iloc[n_tr - 1])),
            "test_period": (str(scada["DateTime"].iloc[n_tr + n_va]), str(scada["DateTime"].iloc[-1])),
        },
    )
