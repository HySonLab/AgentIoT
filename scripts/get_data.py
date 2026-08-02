# -*- coding: utf-8 -*-
"""Fetch and verify the datasets needed to reproduce the manuscript's numbers.

Three datasets, three different situations:

  * Boiler Emulator - BUNDLED in this repository (dataset/). It is open access
    under CC BY (IEEE DataPort, doi:10.21227/awav-bn36), so we may redistribute
    it with attribution; see dataset/BOILER_DATASET_LICENSE.md. Nothing to do.

  * NASA C-MAPSS - downloaded automatically. Public, freely redistributable,
    no account needed. This is the dataset behind the paper's only externally
    comparable result (the RUL benchmark), so this script alone is enough to
    reproduce Tier 1 of the README.

  * Wind Turbine SCADA - NOT redistributed. Third-party dataset whose licence
    terms we could not establish, and which requires a Kaggle account to
    download. The URL is printed below.

Every file this script knows about has a recorded SHA-256 in
dataset/CHECKSUMS.sha256, so you can prove your inputs are byte-identical to
the ones behind the published numbers rather than merely assuming it.

Usage:
    python scripts/get_data.py            # download what can be downloaded
    python scripts/get_data.py --check    # report status only, download nothing
    python scripts/get_data.py --verify   # check SHA-256 of everything present
"""
import hashlib
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

# NOTE: an earlier revision of this script pointed at
# kaggle.com/datasets/inIT-OWL/wind-turbine-scada-dataset, which is dead (404).
# The live source matching the file layout used here is:
WIND_URL = "https://www.kaggle.com/datasets/wasuratme96/iiot-data-of-wind-turbine"

BOILER_DOI = "https://dx.doi.org/10.21227/awav-bn36"

TARGETS = {
    "Boiler Emulator": DATA / "Boiler_emulator_dataset.csv",
    "C-MAPSS FD001": DATA / "cmapss" / "train_FD001.txt",
    "Wind SCADA": DATA / "iiot-data-of-wind-turbine" / "scada_data.csv",
}


def load_checksums():
    """Parse dataset/CHECKSUMS.sha256 -> {relative_path: sha256}."""
    manifest = DATA / "CHECKSUMS.sha256"
    if not manifest.exists():
        return {}
    out = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, rel = line.partition("  ")
        if digest and rel:
            out[rel.strip()] = digest.strip()
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify():
    """Check every manifest entry that is present on disk."""
    expected = load_checksums()
    if not expected:
        print("No dataset/CHECKSUMS.sha256 found; cannot verify.")
        return False
    print("Verifying SHA-256 against dataset/CHECKSUMS.sha256:")
    ok = missing = bad = 0
    for rel, want in expected.items():
        path = DATA / rel
        if not path.exists():
            print(f"  [ -- ] {rel}  (not present)")
            missing += 1
            continue
        got = sha256(path)
        if got == want:
            print(f"  [ OK ] {rel}")
            ok += 1
        else:
            print(f"  [FAIL] {rel}")
            print(f"         expected {want}")
            print(f"         got      {got}")
            bad += 1
    print(f"\n  {ok} verified, {bad} mismatched, {missing} not present.")
    if bad:
        print("  A mismatch means your copy differs from the one behind the")
        print("  published numbers - results will not reproduce exactly.")
    return bad == 0


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
        print("  We do not redistribute this one: it needs a Kaggle account and")
        print("  we could not establish its redistribution terms.")
        print("  Place scada_data.csv, fault_data.csv and status_data.csv in:")
        print(f"  {(DATA / 'iiot-data-of-wind-turbine').relative_to(ROOT)}/")
        print("  Then run:  python scripts/get_data.py --verify")
    if "Boiler Emulator" in missing:
        print("\nBoiler Emulator is normally bundled with this repository but is")
        print("missing from your checkout. Re-clone, or download it from")
        print(f"  {BOILER_DOI}")
        print(f"  and place Boiler_emulator_dataset.csv in {DATA.relative_to(ROOT)}/")
        print("  Cite: R. Shohet, M. Kandil, J. J. McArthur, 'Simulated boiler")
        print("  data for fault detection and classification', IEEE Dataport, 2019.")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(0 if verify() else 1)

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
    print("Run 'python scripts/get_data.py --verify' to confirm the files you")
    print("have are byte-identical to the ones behind the published numbers.")
