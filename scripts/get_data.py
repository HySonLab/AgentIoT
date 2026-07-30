# -*- coding: utf-8 -*-
"""Fetch the datasets needed to reproduce the manuscript's numbers.

Only NASA C-MAPSS can be downloaded automatically: it is public, freely
redistributable, and needs no account. It is also the dataset behind the
paper's only externally-comparable result (the RUL benchmark), so this
script alone is enough to reproduce Tier 1 of the README.

The other two datasets are NOT redistributed in this repository, and are
not auto-downloadable:

  * Wind Turbine SCADA - third-party dataset requiring an account to
    download; see the URL printed below.
  * Boiler Emulator - obtain from the authors (see README); we do not
    redistribute it here.

Usage:
    python scripts/get_data.py            # download C-MAPSS
    python scripts/get_data.py --check    # report status only, download nothing
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset"

CMAPSS_URL = ("https://phm-datasets.s3.amazonaws.com/NASA/"
              "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip")

WIND_URL = "https://www.kaggle.com/datasets/inIT-OWL/wind-turbine-scada-dataset"

TARGETS = {
    "C-MAPSS FD001": DATA / "cmapss" / "train_FD001.txt",
    "Wind SCADA": DATA / "iiot-data-of-wind-turbine" / "scada_data.csv",
    "Boiler Emulator": DATA / "Boiler_emulator_dataset.csv",
}


def status():
    print("Dataset status:")
    missing = []
    for label, path in TARGETS.items():
        if path.exists():
            print(f"  [ OK ] {label:16s} {path.relative_to(ROOT)}")
        else:
            print(f"  [MISS] {label:16s} expected at {path.relative_to(ROOT)}")
            missing.append(label)
    return missing


def download_cmapss():
    target_dir = DATA / "cmapss"
    if (target_dir / "train_FD001.txt").exists():
        print("C-MAPSS already present; nothing to do.")
        return True
    DATA.mkdir(parents=True, exist_ok=True)
    zip_path = DATA / "CMAPSSData.zip"
    print(f"Downloading C-MAPSS (~12 MB) from\n  {CMAPSS_URL}")
    try:
        urllib.request.urlretrieve(CMAPSS_URL, zip_path)
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Download manually from the NASA Prognostics Data Repository "
              "and unzip so that dataset/cmapss/train_FD001.txt exists.")
        return False

    print("Extracting ...")
    raw_dir = DATA / "cmapss_raw"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw_dir)
    # The NASA archive nests a second zip inside a descriptive folder.
    inner = list(raw_dir.rglob("CMAPSSData.zip"))
    if inner:
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(inner[0]) as z:
            z.extractall(target_dir)
    else:
        found = list(raw_dir.rglob("train_FD001.txt"))
        if not found:
            print("  Could not locate train_FD001.txt inside the archive.")
            return False
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in found[0].parent.iterdir():
            f.replace(target_dir / f.name)

    ok = (target_dir / "train_FD001.txt").exists()
    print("  C-MAPSS ready." if ok else "  Extraction did not produce train_FD001.txt.")
    return ok


def instructions(missing):
    if "Wind SCADA" in missing:
        print("\nWind Turbine SCADA (manual download required):")
        print(f"  {WIND_URL}")
        print("  Place scada_data.csv, fault_data.csv and status_data.csv in:")
        print(f"  {(DATA / 'iiot-data-of-wind-turbine').relative_to(ROOT)}/")
    if "Boiler Emulator" in missing:
        print("\nBoiler Emulator (not redistributed here):")
        print("  Request from the corresponding author - see README, "
              "'Getting the data'.")
        print(f"  Place Boiler_emulator_dataset.csv in {DATA.relative_to(ROOT)}/")


if __name__ == "__main__":
    check_only = "--check" in sys.argv
    missing = status()
    if not check_only and "C-MAPSS FD001" in missing:
        print()
        download_cmapss()
        print()
        missing = status()
    instructions(missing)
    print("\nNo datasets are required to run scripts/verify_install.py,")
    print("which exercises the full pipeline on synthetic data.")
