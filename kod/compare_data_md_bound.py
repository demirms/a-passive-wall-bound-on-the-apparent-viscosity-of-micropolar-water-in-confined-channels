"""Compare measured apparent viscosity of water in hydrophilic fine channels (01_arastirma/veri_tablosu.csv)
with (i) the parameter-free passivity bound 1 + eta_r/eta0 and (ii) the micropolar prediction with MD water
coefficients (Hansen et al. 2010, 298 K), wall u = 0, omega = 0 (M1).
Tube: finite differences; slit: Hansen Eq. (23) (= exact closed form); square: bound only (no 2D solver yet)."""
import csv, numpy as np
from hansen2010_md_vs_deryagin import solve

eta0, eta_r, zeta = 1.0e-3, 1.7e-4, 2.5e-21
K = np.sqrt(4 * eta_r * eta0 / ((eta0 + eta_r) * zeta))
BOUND_LO, BOUND_HI = 1 + eta_r / eta0, 1.24   # MD/experimental eta0 vs MD/MD ratio (Table II, 299 K)

def slit(H):
    kh = K * H / 2
    return 1 / (1 - 3 * eta_r * (kh - np.tanh(kh)) / (np.tanh(kh) * (eta0 + eta_r) * kh**2))

rows = list(csv.DictReader(open("../01_arastirma/veri_tablosu.csv", encoding="utf-8-sig")))
out = []
for r in rows:
    if r["liquid"].strip().lower() != "water" or "DUPLICATE" in r["notes"]:
        continue
    if "hydrophob" in (r["wall_treatment"] + r["notes"]).lower() or r["source_key"] == "Churaev_1984":
        continue
    try:
        s, y = float(r["size_nm"]), float(r["eta_ratio"])
    except ValueError:
        continue
    g = r["geometry"].lower()
    if "tube" in g or "capillar" in g:
        pred = 1 / solve(s * 1e-9, eta0, eta_r, zeta, "tube")
    elif "slit" in g or "plate" in g:
        pred = slit(s * 1e-9)
    else:
        pred = float("nan")
    out.append((r["id"], r["source_key"], g[:22], r["size_symbol"][:4], s, y, r["eta_ratio_unc"], r["electrolyte"][:12], pred))

print(f"bound: {BOUND_LO:.2f}-{BOUND_HI:.2f};  1/K = {1/K*1e9:.2f} nm")
print(f"{'id':5} {'source':14} {'geometry':22} {'sym':4} {'size_nm':>8} {'meas':>6} {'unc':>6} {'electrolyte':12} {'MD':>6} {'>bound':>6}")
for o in out:
    flag = "YES" if o[5] > BOUND_HI else ("edge" if o[5] > BOUND_LO else "no")
    print(f"{o[0]:5} {o[1]:14} {o[2]:22} {o[3]:4} {o[4]:8.1f} {o[5]:6.3f} {o[6]:>6} {o[7]:12} {o[8]:6.3f} {flag:>6}")
