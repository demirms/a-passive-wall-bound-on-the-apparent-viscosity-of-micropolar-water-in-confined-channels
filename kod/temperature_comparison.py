"""Temperature dependence: Churaev 1970 Fig. 5 (veri_tablosu.csv, rows with temperature) vs micropolar
prediction with MD coefficients at each temperature (Hansen et al. 2010, Table II, SPC/Fw; model-consistent
eta_r/eta0 and zeta/eta0 ratios), tube with M1 wall. Table II values copied from the PDF (p. 144906-5)."""
import csv, numpy as np
from hansen2010_md_vs_deryagin import solve

# T [K], eta0 [1e-3 Pa s], eta_r [1e-3 Pa s], zeta0, zeta_r [1e-21 kg m/s]
TABLE2 = [(284, 0.92, 0.17, 1.4, 1.6), (289, 0.83, 0.17, 1.2, 1.5), (293, 0.76, 0.17, 1.2, 1.4),
          (299, 0.70, 0.17, 0.9, 1.2), (305, 0.61, 0.16, 0.93, 1.2), (309, 0.56, 0.164, 0.83, 1.0),
          (313, 0.54, 0.161, 0.8, 0.9), (319, 0.48, 0.160, 0.7, 0.92)]

def md_pred(r_nm, row):
    T, e0, er, z0, zr = row
    return 1 / solve(r_nm * 1e-9, e0 * 1e-3, er * 1e-3, (z0 + zr) * 1e-21, "tube")

print("MD prediction (tube, M1) vs T; bound = 1 + eta_r/eta0")
print(f"{'T_C':>5} {'bound':>6} {'r=48nm':>7} {'r=170nm':>8}")
for row in TABLE2:
    print(f"{row[0]-273.15:5.1f} {1+row[2]/row[1]:6.3f} {md_pred(48, row):7.3f} {md_pred(170, row):8.3f}")

rows = [r for r in csv.DictReader(open("../01_arastirma/veri_tablosu.csv", encoding="utf-8-sig"))
        if r["source_key"] == "Churaev_1970" and r["temperature_C"] not in ("", "20", "~20")]
print("\nChuraev 1970 Fig. 5 data (r, T, measured):")
for r in rows:
    print(f"  {r['id']} r={r['size_nm']:>6} nm  T={r['temperature_C']:>6} C  eta/eta0={r['eta_ratio']}")
