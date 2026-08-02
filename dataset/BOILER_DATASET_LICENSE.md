# Boiler Emulator dataset — source, license and attribution

`Boiler_emulator_dataset.csv` in this directory is **not our data**. It is a
third-party, published, open-access dataset, redistributed here unmodified
under the terms of its licence so that this repository is reproducible without
requiring an account on another service.

## Citation (required)

> R. Shohet, M. Kandil, and J. J. McArthur, "Simulated boiler data for fault
> detection and classification," IEEE Dataport, March 30, 2019.
> doi: [10.21227/awav-bn36](https://dx.doi.org/10.21227/awav-bn36)

Landing page:
<https://ieee-dataport.org/open-access/simulated-boiler-data-fault-detection-and-classification>

## Licence

IEEE DataPort **Open Access** datasets are made available under the
**Creative Commons Attribution (CC BY)** licence, which permits redistribution
and reuse — including for reproducibility — provided the original creators are
credited. See the
[IEEE DataPort Terms of Use](https://ieee-dataport.org/ieee-dataport-terms-use).

Attribution is a **condition** of that licence, not a courtesy: if you use this
file, cite Shohet et al. above.

## Integrity

The copy here is byte-identical to the file distributed by IEEE DataPort:

```
SHA-256  83ede6755fb2a4428a75df2536a2757971501a7b6bcf3d0af0803c15e8afddca
bytes    1,343,984
rows     27,280 (+ header)
```

Verify with `python scripts/get_data.py --verify`.

## What it contains

A Matlab/Simulink emulator of a Viessmann Vitorond 200 gas-fired boiler
(VD2 series 380), swept across nominal operation and four fault mechanisms at
graded severities.

| Column | Meaning |
|---|---|
| `Fuel_Mdot` | fuel mass flow rate |
| `Tair` | air temperature (K) |
| `Treturn` | return water temperature (K) |
| `Tsupply` | supply water temperature (K) |
| `Water_Mdot` | water mass flow rate |
| `Condition` | fault severity, `%=0.05` … `%=0.40` |
| `Class` | `Nominal`, `Lean`, `ExcessAir`, `Fouling`, `Scaling` |

### How this study labels it

Anomaly = any `Class != Nominal`, i.e. all four fault mechanisms, each carrying
its graded severity. This is stated in the manuscript's dataset table.
Marking only `Fouling` as anomalous — as an early internal draft of this work
did — silently relabels three genuine fault modes as normal and inflates the
apparent negative class. Use `hama/data/boiler.py`, which implements the
labelling described in the paper.
