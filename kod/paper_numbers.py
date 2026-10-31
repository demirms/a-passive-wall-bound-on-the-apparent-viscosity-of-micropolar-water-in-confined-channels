"""Derived numbers quoted in the manuscript that no other script prints (simple arithmetic on published values
and on the outputs of the other scripts). Output: paper_numbers_output.txt
"""
import csv
import os
import re
import numpy as np
from paper_common import TABLE2, WATER_284, WATER_299, eringen, lam, water_slip

HERE = os.path.dirname(os.path.abspath(__file__))
out = []
log = out.append

# (1) Hansen 2010 Eq. (23): saturation of the flow reduction for h -> 0 (298 K set of their Fig. 3:
#     eta0 = 1.0, eta_r = 0.17 mPa s): 1 - Q/Q^P -> eta_r/(eta0 + eta_r)
e0, er = 1.0, 0.17
log(f"(1) Eq. (23) limit h->0 (298 K set): {100 * er / (e0 + er):.2f} %")

# (2) rotational-viscosity ratio needed to put the largest measured value below the bound
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "01_arastirma", "veri_tablosu.csv"), encoding="utf-8-sig")))
water = [r for r in rows if r["liquid"].strip().lower() == "water" and "DUPLICATE" not in r["notes"]]
r081 = [r for r in water if r["id"] == "R081"][0]
ymax = float(r081["eta_ratio"])
need = ymax - 1
eps = [er_ / e0_ for _, e0_, er_, _, _ in TABLE2[:4]]
log(f"(2) largest measured (Kazoe 2023, D_h = 119 nm) = {ymax:.3f}; eta_r/eta0 needed >= {need:.3f}; "
    f"MD 284-299 K: {min(eps):.3f}-{max(eps):.3f}; needed/MD = {need / max(eps):.1f}-{need / min(eps):.1f}")

# (3) Migun / Kolpashchikov fit (M1): kappa/mu = 5.25, eta_r/eta_n = 0.724; length 1/kappa_tilde,
#     kappa_tilde = 70.3e6 1/m (Kolpashchikov et al. 1983, Sec. 5)
kt = 70.3e6
L_fit = 1e9 / kt
ratio_fit = 0.724
Ls = [1 / lam(*eringen(a, b, c + d)) for _, a, b, c, d in TABLE2[:4]]
log(f"(3) fitted length 1/kappa_tilde = {L_fit:.2f} nm; MD 1/lambda (284-299 K) = {min(Ls):.3f}-{max(Ls):.3f} nm; "
    f"length ratio {L_fit / max(Ls):.1f}-{L_fit / min(Ls):.1f}; eta_r ratio {ratio_fit / max(eps):.2f}-{ratio_fit / min(eps):.2f}")

# (4) immobile-layer equivalent of the micropolar wall layer: a layer of thickness h0 at the wall acts, to first
#     order, as a slip length b = -h0 (tube and slit), so h0_eq = -b
bs = [water_slip(a, b, c + d)[0] for _, a, b, c, d in TABLE2[:4]]
log(f"(4) equivalent immobile layer h0_eq = -b = {-max(bs):.3f}-{-min(bs):.3f} nm (284-299 K); "
    f"Churaev et al. 1970 model h0 = 8 nm -> ratio {8 / -min(bs):.0f}-{8 / -max(bs):.0f}")

# (5) lambda D_h of the water channels (299 K and 284 K), square D_h = 73-491 nm, tube r = 41-60 nm
for name, w in (("299 K", WATER_299), ("284 K", WATER_284)):
    l = lam(*eringen(*w))
    log(f"(5) {name}: 1/lambda = {1 / l:.3f} nm; square lambda D_h = {73 * l:.0f}-{491 * l:.0f}; "
        f"tube lambda R (r = 41-60 nm) = {41 * l:.0f}-{60 * l:.0f}")

# (6) sensitivity to alpha + beta in the 2D sweep (c) of micropolar_duct_2d_output.txt
txt = open(os.path.join(HERE, "micropolar_duct_2d_output.txt"), encoding="utf-8").read()
sec = txt.split("(c) Bound sweep")[1].split("bound violations")[0]
vals = {}
for m in re.finditer(r"^\s*(-?[\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)", sec, re.M):
    mu_, k_, c_, la_, ma_ = map(float, m.groups())
    vals.setdefault((mu_, k_, la_), []).append(ma_)
for la_ in (30.0, 100.0):
    rel = max((max(v) - min(v)) / min(v) for (m_, k_, l_), v in vals.items() if l_ == la_ and m_ > 0)
    log(f"(6) alpha+beta in {{0, -0.9 gamma, 10 gamma}}: max relative change of mu_app at lambda a = {la_:g}: {rel:.1e}")

# (7) counts of hydrophilic-water data rows above the bound (water_slip_length_output.txt, section (3), 299 K)
wtxt = open(os.path.join(HERE, "water_slip_length_output.txt"), encoding="utf-8").read()
sec3 = wtxt.split("(3) Every data row")[1].split("(4) Headline")[0]
tot = above = 0
for line in sec3.splitlines():
    m = re.match(r"\s*(R\d{3})\s.*\|\s*([YNn])/([YNn])\s*$", line)
    if m:
        tot += 1
        above += m.group(3) == "Y"
log(f"(7) water rows compared (no duplicates, all electrolytes): {tot}; above the 299 K bound 1.243: {above}")
with open(os.path.join(HERE, "paper_numbers_output.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
