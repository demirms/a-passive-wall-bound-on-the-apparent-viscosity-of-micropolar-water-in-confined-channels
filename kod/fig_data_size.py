"""Fig. 3 of the paper: measured apparent viscosity of water in hydrophilic fine channels versus hydraulic diameter,
with the micropolar prediction (M1 wall, MD coefficients) and the parameter-free bound 1 + eta_r/eta0.

Data: 01_arastirma/veri_tablosu.csv (digitize_capillary_data.py). Selection: water, hydrophilic quartz/silica,
no added salt, duplicates removed, Cheng & Giordano (photoresist walls) and Churaev 1984 (hydrophobised)
excluded, Churaev 1970 only at ~20 C (Fig. 4 of the source), Haneveld 5 nm excluded (below the validity
limit of the local model, and channel collapse suspected by its authors). D_h = 2r (tube), 2H (slit/plate),
D_h (square). Error bars: eta_ratio_unc of the table (authors' bar or reading uncertainty, whichever larger).
Model: MD-MD normalisation, Hansen 2010 Table II at 299 K (largest bound and prediction of the
284-299 K rows, i.e. the case most favourable to the micropolar explanation).
Output: sekil/fig4_data_size.pdf/.png and fig4_data_size_output.txt
Plot details:
  - the square-duct curve is the finite-difference solution (N = 160, stretched grid) for D_h <= 200 nm (lambda D_h from 4 to
    about 100) and the large-size law (19) beyond; the law is no longer drawn where it is not valid;
  - the grey bound band is darker (40 % grey, colour 0.6, no transparency), the figure has double-column width (about 17 cm);
  - legend terms follow Table 3: "plate channel" for Li 2012 and Kazoe 2022, "slit" for Tas 2004 and Haneveld 2008;
  - coincident markers of one source (D_h within 3 %) are shifted horizontally by +-1.2 % per rank (display only, data unchanged);
  - every plotted prediction point is asserted to be below 1 + eta_r/eta0 (299 K).
"""
import csv
import os
import numpy as np
from paper_common import WATER_299, WATER_284, water_tube, water_slit, water_slip, eringen, lam, setup_matplotlib
from micropolar_duct_2d import solve_duct, stretch_for
from params_writer import save_params

HERE = os.path.dirname(os.path.abspath(__file__))
plt = setup_matplotlib()
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "01_arastirma", "veri_tablosu.csv"), encoding="utf-8-sig")))


def keep(r):
    if r["liquid"].strip().lower() != "water" or "DUPLICATE" in r["notes"]:
        return False
    if r["source_key"] in ("Churaev_1984", "Cheng_2002"):
        return False
    el = r["electrolyte"].lower()
    if "nacl" in el or "kcl" in el:
        return False
    if r["source_key"] == "Churaev_1970" and not r["temperature_C"].startswith("~20"):
        return False
    if r["source_key"] == "Haneveld_2008" and float(r["size_nm"]) < 6:
        return False
    return True


groups = {}
for r in rows:
    try:
        s, y = float(r["size_nm"]), float(r["eta_ratio"])
    except ValueError:
        continue
    if not keep(r):
        continue
    u = float(r["eta_ratio_unc"]) if r["eta_ratio_unc"] not in ("", None) else 0.0
    g = r["geometry"].lower()
    sq = "square" in g
    dh = s if sq else 2 * s
    key = (r["source_key"], "square" if sq else ("tube" if "capillar" in g else "slit"))
    groups.setdefault(key, []).append((dh, y, u, r["id"]))

LABEL = {("Churaev_1970", "tube"): ("Churaev 1970, tube", "o", "w"),
         ("Tas_2004", "slit"): ("Tas 2004, slit", "s", "w"),
         ("Haneveld_2008", "slit"): ("Haneveld 2008, slit", "^", "w"),
         ("Li_2012", "slit"): ("Li 2012, plate channel", "v", "w"),
         ("Kazoe_2022", "slit"): ("Kazoe 2022, plate channel", "<", "w"),
         ("Li_2012", "square"): ("Li 2012, square", "D", "k"),
         ("Kazoe_2022", "square"): ("Kazoe 2022, square", ">", "k"),
         ("Kazoe_2023", "square"): ("Kazoe 2023, square", "P", "k")}
out = ["source geometry D_h[nm] measured unc id"]
fig, ax = plt.subplots(figsize=(6.7, 5.0))
for key in LABEL:
    if key not in groups:
        continue
    pts = sorted(groups[key])
    x, y, u, ids = zip(*pts)
    lab, mk, fc = LABEL[key]
    xs = np.array(x, dtype=float)
    shifted = xs.copy()
    for i in range(len(xs)):                       # display-only horizontal shift of coincident markers of one source
        cl = [j for j in range(len(xs)) if abs(xs[j] / xs[i] - 1) < 0.03]
        if len(cl) > 1:
            shifted[i] = xs[i] * (1 + 0.012 * (cl.index(i) - (len(cl) - 1) / 2))
    ax.errorbar(shifted, y, yerr=u, fmt=mk, ms=3.5, mfc=fc, mec="k", ecolor="0.5", elinewidth=0.6, capsize=1.5,
                ls="none", label=lab)
    for p in pts:
        out.append(f"{key[0]} {key[1]} {p[0]:.1f} {p[1]:.3f} {p[2]:.3f} {p[3]}")
dh = np.logspace(np.log10(8), np.log10(3000), 200)
b299, c299 = water_slip(*WATER_299)
lo, hi = 1 + WATER_284[1] / WATER_284[0], 1 + WATER_299[1] / WATER_299[0]
mu_, k_, g_ = eringen(*WATER_299)
L299 = lam(mu_, k_, g_)
# square duct: finite differences up to D_h = 200 nm, large-size law (19) beyond
DH_FD = np.logspace(np.log10(8), np.log10(200), 22)
sq_fd = np.array([solve_duct(float(d), float(d), mu_, k_, g_, 0.0, 160, 160, stretch_for(d * L299), stretch_for(d * L299))[0] / WATER_299[0]
                  for d in DH_FD])
dh_law = dh[dh > 200]
sq_x = np.concatenate([DH_FD, dh_law])
sq_y = np.concatenate([sq_fd, 1 + c299 / dh_law])
tube_y, slit_y = water_tube(dh / 2, *WATER_299), water_slit(dh / 2, *WATER_299)
for nm, arr in (("tube", tube_y), ("slit", slit_y), ("square", sq_y)):
    assert np.all(arr < hi), f"{nm} curve reaches the bound {hi:.4f}: max {arr.max():.4f}"
ax.plot(dh, tube_y, "k-", label="micropolar, tube")
ax.plot(dh, slit_y, "k--", label="micropolar, slit / plate channel")
ax.plot(sq_x, sq_y, "k:", lw=1.3, label="micropolar, square")
ax.axhspan(lo, hi, color="0.6", zorder=0, lw=0, label="bound (284–299 K)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(8, 3000)
ax.set_ylim(0.8, 7)
ax.set_yticks([0.8, 1, 1.5, 2, 3, 4, 5, 6])
ax.set_yticklabels(["0.8", "1", "1.5", "2", "3", "4", "5", "6"])
ax.minorticks_off()
ax.set_xlabel(r"hydraulic diameter $D_h$ (nm)")
ax.set_ylabel(r"$\eta_{\mathrm{app}}/\eta_0$")
h_, l_ = ax.get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, fontsize=8.5, ncol=3, loc="lower center", columnspacing=1.2, handlelength=1.8)
fig.tight_layout(rect=(0, 0.22, 1, 1))
fig.savefig(os.path.join(HERE, "..", "sekil", "fig4_data_size.pdf"))
fig.savefig(os.path.join(HERE, "..", "sekil", "fig4_data_size.png"), dpi=600)
out.append(f"bound band {lo:.4f}-{hi:.4f}; large-size coefficient c(299 K) = {c299:.4f} nm, b = {b299:.4f} nm")
for d in (20, 82, 100, 234, 500):
    out.append(f"D_h = {d} nm: tube {water_tube(d / 2, *WATER_299):.4f}, slit {water_slit(d / 2, *WATER_299):.4f}, "
               f"square law {1 + c299 / d:.4f}")
out.append("square-duct curve: FD (N = 160) at D_h = " + ", ".join(f"{d:.1f}" for d in DH_FD[::3]) + " nm ...; law (19) for D_h > 200 nm")
out.append(f"junction D_h = 200 nm: FD {sq_fd[-1]:.5f} vs law (19) {1 + c299 / 200:.5f} (difference {abs(sq_fd[-1] - 1 - c299 / 200):.1e})")
out.append(f"FD square values: D_h = 8 nm {sq_fd[0]:.4f} (lambda D_h = {8 * L299:.1f}); 20 nm "
           f"{np.interp(20, DH_FD, sq_fd):.4f}; 73 nm {np.interp(73, DH_FD, sq_fd):.4f} (interpolated)")
out.append(f"assertion: every plotted point of the tube, slit and square curves is below the 299 K bound {hi:.4f} "
           f"(largest: tube {tube_y.max():.4f}, slit {slit_y.max():.4f}, square {sq_y.max():.4f}): passed")
with open(os.path.join(HERE, "fig4_data_size_output.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out[-9:]))
save_params("Fig. 4", parameters={"eta0 [mPa s]": WATER_299[0], "eta_r [mPa s]": WATER_299[1],
                                "zeta0+zeta_r [mPa s nm^2]": WATER_299[2], "T [K]": 299, "wall": "BC1",
                                "bound band": "284-299 K, Hansen 2010 Table II"},
       domain="tube, slit: closed forms; square: 2D cross-section, finite differences for D_h <= 200 nm, large-size law 1 + c/D_h beyond",
       mesh="square: stretched grid, 160 x 160 intervals; tube, slit: closed forms",
       tolerances="square law within 3e-4 of the 2D solver at D_h = 200 nm (junction, printed in the output)",
       initial_guess="not applicable")
