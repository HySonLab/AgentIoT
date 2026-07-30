# Datasets

Datasets are **not redistributed** in this repository. Place them here as
described below, or run:

```bash
python scripts/get_data.py --check    # status of all three
python scripts/get_data.py            # auto-download NASA C-MAPSS
```

Expected layout once populated:

```
dataset/
├── cmapss/
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── iiot-data-of-wind-turbine/
│   ├── scada_data.csv
│   ├── fault_data.csv
│   └── status_data.csv
└── Boiler_emulator_dataset.csv
```

| Dataset | Source | Notes |
|---|---|---|
| **NASA C-MAPSS** | NASA Prognostics Data Repository | Public, no account needed. `scripts/get_data.py` downloads and extracts it automatically. Backs the paper's RUL benchmark (MAE 11.20 / RMSE 15.95). |
| **Wind Turbine SCADA** | Kaggle (URL printed by `get_data.py`) | Requires a free Kaggle account; third-party terms prevent redistribution here. |
| **Boiler Emulator** | Request from the corresponding author | Not redistributed here. |

**No dataset is required** to verify that the code works — run
`python scripts/verify_install.py`, which exercises the full HAMA pipeline
on synthetic data.
