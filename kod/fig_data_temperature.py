"""Fig. 4 of the paper: temperature dependence. Churaev et al. 1970 Fig. 5 data (r = 48 +- 3 nm and r = 170 nm, read from the
figure, 01_arastirma/veri_tablosu.csv) against the parameter-free bound 1 + eta_r/eta0 and the micropolar
prediction (tube, M1 wall) with the MD coefficients of Hansen et al. 2010 Table II (284-319 K, MD-MD ratios).
Output: sekil/fig5_data_temperature.pdf/.png and fig5_data_temperature_output.txt
The upper axis gives the temperature in K (T[K] = T[C] + 273.15).
"""
import csv
import os
import numpy as np
from paper_common import TABLE2, water_tube, setup_matplotlib
from params_writer import save_params

HERE = os.path.dirname(os.path.abspath(__file__))
plt = setup_matplotlib()
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "01_arastirma", "veri_tablosu.csv"), encoding="utf-8-sig")))
c = [r for r in rows if r["source_key"] == "Churaev_1970" and r["temperature_C"]
     and not r["temperature_C"].startswith("~")]
out = ["T[C]  bound  MD r=48nm  MD r=170nm"]
T = np.array([t - 273.15 for t, *_ in TABLE2])
bound = np.array([1 + er / e0 for _, e0, er, _, _ in TABLE2])
p48 = np.array([water_tube(48.0, e0, er, z0 + zr) for _, e0, er, z0, zr in TABLE2])
p170 = np.array([water_tube(170.0, e0, er, z0 + zr) for _, e0, er, z0, zr in TABLE2])
for i in range(len(T)):
    out.append(f"{T[i]:5.1f} {bound[i]:.3f} {p48[i]:.4f} {p170[i]:.4f}")

fig, ax = plt.subplots(figsize=(3.35, 3.9))
for rad, mk, lab in (("48", "o", r"data, $r=48$ nm"), ("170", "s", r"data, $r=170$ nm")):
    pts = sorted((float(r["temperature_C"]), float(r["eta_ratio"])) for r in c if r["size_nm"].startswith(rad))
    x, y = np.array(pts).T
    ax.plot(x, y, mk, ms=4, mfc="w" if rad == "48" else "k", mec="k", ls="none", label=lab)
    out.append(f"data r={rad} nm: " + ", ".join(f"{a:.1f}C:{b:.3f}" for a, b in pts))
ax.plot(T, bound, "k-.", label="bound")
ax.plot(T, p48, "k-", label=r"micropolar, $r=48$ nm")
ax.plot(T, p170, "k--", label=r"micropolar, $r=170$ nm")
ax.set_xlabel(r"$T$ ($^\circ$C)")
secx = ax.secondary_xaxis("top", functions=(lambda t: t + 273.15, lambda t: t - 273.15))
secx.set_xlabel(r"$T$ (K)")
secx.set_xticks([280, 300, 320, 340])
ax.set_ylabel(r"$\eta_{\mathrm{app}}/\eta_0$")
ax.set_xlim(0, 72)
ax.set_ylim(0.95, 1.55)
ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2, bbox_to_anchor=(0.45, -0.24), columnspacing=0.8, handlelength=1.8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "..", "sekil", "fig5_data_temperature.pdf"))
fig.savefig(os.path.join(HERE, "..", "sekil", "fig5_data_temperature.png"), dpi=600)
with open(os.path.join(HERE, "fig5_data_temperature_output.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
save_params("Fig. 5", parameters={"coefficients": "Hansen 2010 Table II, 284-319 K (MD-MD ratios)", "wall": "BC1",
                                "radii [nm]": [48, 170]},
       domain="tube, closed form", mesh="closed form (no grid)", tolerances="closed form",
       initial_guess="not applicable")
