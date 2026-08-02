# Datasets

One of the three datasets is **bundled** here; the other two are fetched.

```bash
python scripts/get_data.py --check     # status of all three
python scripts/get_data.py             # auto-download NASA C-MAPSS
python scripts/get_data.py --verify    # SHA-256 everything you have
```

| Dataset | How you get it | Redistributed here? |
|---|---|---|
| **Boiler Emulator** | Already in this repo | **Yes** — open access under CC BY (IEEE DataPort, [doi:10.21227/awav-bn36](https://dx.doi.org/10.21227/awav-bn36)). Attribution is a licence condition: see [BOILER_DATASET_LICENSE.md](BOILER_DATASET_LICENSE.md). |
| **NASA C-MAPSS** | `python scripts/get_data.py` (automatic) | No — downloaded from NASA. Public, no account. Backs the RUL benchmark (MAE 11.20 / RMSE 15.95). |
| **Wind Turbine SCADA** | Kaggle, URL printed by `get_data.py` | No — needs a free Kaggle account, and we could not establish its redistribution terms. |

Expected layout once populated:

```
dataset/
├── Boiler_emulator_dataset.csv      <- bundled
├── CHECKSUMS.sha256                 <- bundled
├── BOILER_DATASET_LICENSE.md        <- bundled
├── cmapss/                          <- auto-downloaded
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
└── iiot-data-of-wind-turbine/       <- manual
    ├── scada_data.csv
    ├── fault_data.csv
    └── status_data.csv
```

## Verifying your inputs

`dataset/CHECKSUMS.sha256` records the SHA-256 of every file behind the
published numbers. Run `python scripts/get_data.py --verify` to confirm your
copies are byte-identical. If a hash differs, the numbers will not reproduce
exactly, and that is worth knowing *before* you spend hours on a run rather
than after.

**No dataset is required** to verify that the code works — run
`python scripts/verify_install.py`, which exercises the full HAMA pipeline
on synthetic data.
