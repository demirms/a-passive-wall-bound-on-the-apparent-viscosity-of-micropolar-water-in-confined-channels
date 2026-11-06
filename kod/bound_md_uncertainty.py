# Standard error of the bound 1 + eta_r/eta0 from the molecular-dynamics standard errors of
# Hansen et al. (2010), J. Chem. Phys. 133, 144906, Table II (p. 144906-5), at 284-299 K,
# by first-order propagation with zero covariance between eta0 and eta_r (assumption),
# and the distance of the narrow-tube and square-channel data above the bound in units of it.
# Data: data/water_viscosity_data_public.csv (rows of Table 3).
import csv
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
# T (K): (eta0, SE, eta_r, SE) in mPa s, Hansen et al. (2010) Table II
HANSEN = {284: (0.92, 0.09, 0.17, 0.03), 289: (0.83, 0.09, 0.17, 0.02),
          293: (0.76, 0.04, 0.17, 0.02), 299: (0.70, 0.10, 0.17, 0.03)}

rows = list(csv.DictReader(open(HERE / "data" / "water_viscosity_data_public.csv", encoding="utf-8")))
tubes = [r for r in rows if r["id"] in ("R001", "R002", "R003")]
squares = [r for r in rows if r["in_table3"] == "yes" and r["geometry"].startswith("square")
           and float(r["size_value_nm"]) <= 500 and r["electrolyte"].lower().startswith("none")]
print("narrow tubes:", [(r["id"], r["size_value_nm"], r["eta_ratio"]) for r in tubes])
print("squares 73-491 nm:", [(r["id"], r["size_value_nm"], r["eta_ratio"]) for r in squares])
tmin = min(float(r["eta_ratio"]) for r in tubes)
smin = min(float(r["eta_ratio"]) for r in squares)
print(f"lowest narrow-tube value {tmin:.3f}; lowest square value {smin:.3f}")
for T, (e0, s0, er, sr) in HANSEN.items():
    x = er / e0
    se = x * math.sqrt((s0 / e0) ** 2 + (sr / er) ** 2)
    print(f"T={T} K: bound {1 + x:.3f} +- {se:.3f} (1 SE; rel SE eta0 {s0 / e0:.3f}, eta_r {sr / er:.3f}); "
          f"95% interval {1 + x - 1.96 * se:.3f}-{1 + x + 1.96 * se:.3f}; "
          f"tube {tmin:.3f} lies {(tmin - 1 - x) / se:.2f} SE above; square {smin:.3f} lies {(smin - 1 - x) / se:.2f} SE above")
