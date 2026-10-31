"""Water: negative Navier slip length of the micropolar wall layer and the large-channel coefficient,
under the TWO normalisations used here, plus every data comparison
(tube, slit, plate, square) re-evaluated under both normalisations.

Purpose:
 (1) b = -kappa / [2 (mu + kappa) lambda]  (exact for the slit, check_slit_asymptotics.py),
     lambda^2 = kappa (2 mu + kappa) / [gamma (mu + kappa)] = 4 eta_r eta0 / [(eta0 + eta_r) zeta]
     (micropolar_duct_2d.lam0), map mu = eta0 - eta_r, kappa = 2 eta_r, gamma = zeta0 + zeta_r.
 (2) large-channel law for the square / circle (Pohozaev: a * oint|d_n w0|^2 / int|grad w0|^2 = 8 / D_h * a):
     mu_app/eta0 ~ 1 - 8 b / D_h = 1 + c / D_h,  c = 4 kappa / [(mu + kappa) lambda].
     Reproduce the quoted "1 + 2.41 nm / D_h".
 (3) bound 1 + eta_r/eta0 and the M1 (zero wall spin) MD prediction for every data row used in
     compare_data_md_bound.py and micropolar_duct_2d.py (d), under
       A  "experimental eta0":  eta0 = 1.0 mPa s (experimental bulk, ~298 K), eta_r = 0.17 mPa s,
          zeta = zeta0 + zeta_r = 1.1 + 1.4 = 2.5 mPa s nm^2 (Hansen et al. 2010, 298 K values as used
          in compare_data_md_bound.py, md_prediction_vs_datasets.py, hansen2010_md_vs_deryagin.py,
          fig_comparison.py, micropolar_duct_2d.py (d));
       B  "MD-MD ratio":  Hansen et al. 2010 Table II (SPC/Fw) eta0, eta_r, zeta0 + zeta_r at 284 ... 299 K
          (as in temperature_comparison.py, check_tube_m3_bound.py N4, check_hansen2010_table_eq23.py).
Prediction methods: tube = closed form with s_hat = 0 (M1); slit/plate = Hansen
Eq. (23) (= exact closed form, half-gap h = H/2); square = 2D solver of micropolar_duct_2d.py (N = 160,
Dh <= 491 nm as in Table 3) and the large-channel law for every square row.
Units: mPa s, nm.  Output: water_slip_length_output.txt
"""
import csv
import os
import numpy as np
import sympy as sp
from scipy.special import ive

from micropolar_duct_2d import solve_duct, lam0, stretch_for

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []


def log(s=""):
    print(s)
    OUT.append(s)


# ---------------------------------------------------------------- coefficient sets
# Hansen et al. 2010 Table II: T [K], eta0, eta_r [mPa s], zeta0, zeta_r [1e-21 kg m/s = mPa s nm^2]
TABLE2 = [(284, 0.92, 0.17, 1.4, 1.6), (289, 0.83, 0.17, 1.2, 1.5),
          (293, 0.76, 0.17, 1.2, 1.4), (299, 0.70, 0.17, 0.9, 1.2)]
SETS = {"A (exp. eta0, 298 K)": (1.0, 0.17, 2.5)}
for T, e0, er, z0, zr in TABLE2:
    SETS[f"B (MD-MD, {T} K)"] = (e0, er, z0 + zr)


def coeffs(e0, er, z):
    mu, k, g = e0 - er, 2 * er, z
    lam = lam0(mu, k, g)
    b = -k / (2 * (mu + k) * lam)
    c = 4 * k / ((mu + k) * lam)
    return mu, k, g, lam, b, c


# ---------------------------------------------------------------- (0) symbolic identities
log("(0) Symbolic identities (sympy simplify of the difference)")
mu_, k_, g_, e0_, er_, z_ = sp.symbols("mu kappa gamma eta0 eta_r zeta", positive=True)
lam_s = sp.sqrt(k_ * (2 * mu_ + k_) / (g_ * (mu_ + k_)))
b_s = -k_ / (2 * (mu_ + k_) * lam_s)
c_s = 4 * k_ / ((mu_ + k_) * lam_s)
sub = {mu_: e0_ - er_, k_: 2 * er_, g_: z_}
d1 = sp.simplify(-8 * b_s - c_s)
d2 = sp.simplify((lam_s**2).subs(sub) - 4 * er_ * e0_ / ((e0_ + er_) * z_))
lamw = sp.sqrt(4 * er_ * e0_ / ((e0_ + er_) * z_))
d3 = sp.simplify(b_s.subs(sub) - (-er_ / ((e0_ + er_) * lamw)))
log(f"    -8 b - c                                   = {d1}")
log(f"    lambda^2 (water map) - 4 eta_r eta0/((eta0+eta_r) zeta) = {d2}")
log(f"    b (water map) - [-eta_r/((eta0+eta_r) lambda)]          = {d3}")

# ---------------------------------------------------------------- (1) slip length and coefficient
log("\n(1) Negative slip length b = -kappa/[2(mu+kappa)lambda] and large-channel coefficient c = -8b")
log("    set                      mu     kappa  gamma   1/lam[nm]  b[nm]     c=-8b[nm]  slit -6b[nm]  bound 1+eta_r/eta0")
for name, (e0, er, z) in SETS.items():
    mu, k, g, lam, b, c = coeffs(e0, er, z)
    log(f"    {name:24s} {mu:5.3f}  {k:5.3f}  {g:5.2f}   {1/lam:7.4f}   {b:+8.4f}  {c:8.4f}   {-6*b:8.4f}      {1+er/e0:.4f}")
muA, kA, gA, lamA, bA, cA = coeffs(*SETS["A (exp. eta0, 298 K)"])
log(f"\n    Reproduction of '1 + 2.41 nm/D_h': normalisation A gives c = {cA:.4f} nm -> '{cA:.2f}' "
    f"(= 4*{kA:.2f}/{muA+kA:.2f} * {1/lamA:.4f} nm).")
cB = [coeffs(*SETS[f'B (MD-MD, {T} K)'])[5] for T, *_ in TABLE2]
bB = [coeffs(*SETS[f'B (MD-MD, {T} K)'])[4] for T, *_ in TABLE2]
log(f"    Normalisation B (284-299 K): c = {min(cB):.3f}-{max(cB):.3f} nm, b = {max(bB):+.3f} ... {min(bB):+.3f} nm")
log("    Slit (half-gap h, height H = 2h): mu_app/eta0 ~ 1 - 3b/h = 1 - 6b/H; square/circle: 1 - 8b/D_h.")

# ---------------------------------------------------------------- predictions


def tube_m1(r, e0, er, z):
    """Tube closed form with s_hat = 0: mu_app = (mu+k)/[1 + k e - 4 k e J], J = I2(a)/(a I1(a)), a = lam r."""
    mu, k, g, lam, _, _ = coeffs(e0, er, z)
    a = lam * r
    J = ive(2, a) / (a * ive(1, a))
    e = 1 / (2 * mu + k)
    return (mu + k) / (1 + k * e - 4 * k * e * J) / e0


def slit_eq23(H, e0, er, z):
    """Hansen 2010 Eq. (23), half-gap h = H/2; returns mu_app/eta0 = Q^P/Q."""
    mu, k, g, lam, _, _ = coeffs(e0, er, z)
    kh = lam * H / 2
    red = -3 * er * (np.tanh(kh) - kh) / (np.tanh(kh) * (e0 + er) * kh**2)
    return 1 / (1 - red)


def square_2d(Dh, e0, er, z, N=160):
    mu, k, g, lam, _, _ = coeffs(e0, er, z)
    s = stretch_for(Dh * lam)
    return solve_duct(float(Dh), float(Dh), mu, k, g, 0.0, N, N, s, s)[0] / e0


def square_asym(Dh, e0, er, z):
    return 1 + coeffs(e0, er, z)[5] / Dh


# cross-checks against existing outputs
log("\n(2) Cross-checks against existing outputs (normalisation A)")
A = SETS["A (exp. eta0, 298 K)"]
for r, ref in [(48.0, "1.024 (compare_data_md_bound, FD)"), (50.0, "1.023 (hansen2010_md_vs_deryagin, FD)"),
               (40.0, "1.029 (hansen2010_md_vs_deryagin, FD)")]:
    log(f"    tube r = {r:4.0f} nm: closed form {tube_m1(r, *A):.5f}  vs  {ref};  large-channel law 1 + c/(2r) = {1 + cA/(2*r):.5f}")
for H, ref in [(234.0, "1.008"), (10.5, "1.119"), (52.6, "1.033")]:
    log(f"    slit H = {H:5.1f} nm: Eq. (23) {slit_eq23(H, *A):.5f}  vs  {ref} (compare_data_md_bound)")
log(f"    square Dh = 73 nm: 2D {square_2d(73, *A):.5f} vs 1.03084 (micropolar_duct_2d (d), N=160); law {square_asym(73, *A):.5f}")

# ---------------------------------------------------------------- (3) data rows
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "01_arastirma", "veri_tablosu.csv"), encoding="utf-8-sig")))
data = []
for r in rows:   # same filter as compare_data_md_bound.py
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
        kind = "tube(r)"
    elif "slit" in g:
        kind = "slit(H)"
    elif "plate" in g:
        kind = "plate(D)"
    else:
        kind = "square(Dh)"
    data.append((r["id"], r["source_key"], kind, s, y, r["temperature_C"][:10], r["electrolyte"][:10]))

KEYS = ["A (exp. eta0, 298 K)", "B (MD-MD, 299 K)", "B (MD-MD, 284 K)"]
cache2d = {}


def pred(kind, s, key):
    e0, er, z = SETS[key]
    if kind == "tube(r)":
        return tube_m1(s, e0, er, z), ""
    if kind in ("slit(H)", "plate(D)"):
        return slit_eq23(s, e0, er, z), ""
    if s <= 491:
        if (s, key) not in cache2d:
            cache2d[(s, key)] = square_2d(s, e0, er, z)
        return cache2d[(s, key)], "2D"
    return square_asym(s, e0, er, z), "law"


log("\n(3) Every data row of compare_data_md_bound.py (hydrophilic water), M1 prediction and bound, both normalisations")
log("    bound: A = %.3f ; B(299 K) = %.3f ; B(284 K) = %.3f" % tuple(1 + SETS[k][1] / SETS[k][0] for k in KEYS))
log("    id    source         geometry   size[nm]  T[C]       electrolyte meas  | MD_A    meas/MD_A | MD_B299 meas/MD_B299 | MD_B284 | >bound A/B299")
for (i, src, kind, s, y, T, el) in data:
    pA, tag = pred(kind, s, KEYS[0])
    pB, _ = pred(kind, s, KEYS[1])
    pB2, _ = pred(kind, s, KEYS[2])
    bA = 1 + SETS[KEYS[0]][1] / SETS[KEYS[0]][0]
    bB = 1 + SETS[KEYS[1]][1] / SETS[KEYS[1]][0]
    flag = ("Y" if y > bA else "n") + "/" + ("Y" if y > bB else "n")
    log(f"    {i:5} {src:14} {kind:10} {s:8.1f}  {T:10} {el:10} {y:5.3f} | {pA:.4f}{tag:>3} {y/pA:5.2f}     | {pB:.4f}  {y/pB:5.2f}        | {pB2:.4f}  | {flag}")

# ---------------------------------------------------------------- (4) headline comparisons
log("\n(4) Headline comparisons, both normalisations")


def rng(vals):
    return f"{min(vals):.3f}-{max(vals):.3f}"


groups = [
    ("Churaev tube r = 41-60 nm (~20 C, Fig. 4)", "tube(r)", [41.2, 53.0, 59.8], "1.31-1.48 (incl. Fig. 5 r=48)"),
    ("Churaev tube r = 48 nm (Fig. 5)", "tube(r)", [48.0], "1.31-1.48 at 19-46 C"),
    ("Churaev tube r = 170-255 nm", "tube(r)", [170.0, 181.0, 199.5, 225.6, 241.6, 255.1], "1.10-1.23"),
    ("Kazoe 2022 plate D = 234 nm", "plate(D)", [234.0], "1.31 +- 0.07"),
    ("Haneveld slit H = 10.5 nm", "slit(H)", [10.5], "1.32 +- 0.11"),
    ("Tas slit H = 53 nm (DI water)", "slit(H)", [52.6, 54.3], "1.19-1.27"),
    ("Li 2012 plate D = 25-415 nm", "plate(D)", [25.0, 50.0, 100.0, 189.0, 415.0], "0.94-1.12"),
    ("Square Dh = 73-491 nm (Kazoe, Li)", "square(Dh)", [73.0, 119.0, 190.0, 200.0, 394.0, 491.0], "1.9-5.5"),
]
log("    comparison                               measured        | A: MD       bound | B 299 K: MD  bound | B 284 K: MD  bound")
for name, kind, sizes, meas in groups:
    cols = []
    for key in KEYS:
        v = [pred(kind, s, key)[0] for s in sizes]
        cols.append((rng(v), 1 + SETS[key][1] / SETS[key][0]))
    log(f"    {name:40s} {meas:15s} | {cols[0][0]:11s} {cols[0][1]:.3f} | {cols[1][0]:11s} {cols[1][1]:.3f} | {cols[2][0]:11s} {cols[2][1]:.3f}")

log("\n    Square channel, measured / MD (2D, N = 160):")
sq = [("Kazoe2023", 73, 4.7), ("Kazoe2023", 119, 5.5), ("Kazoe2022", 190, 3.6),
      ("Li2012", 200, 2.5), ("Kazoe2022", 394, 2.8), ("Li2012", 491, 1.9)]
for src, Dh, m in sq:
    vA, vB = pred("square(Dh)", float(Dh), KEYS[0])[0], pred("square(Dh)", float(Dh), KEYS[1])[0]
    log(f"    {src:9s} Dh={Dh:4d}  meas {m:4.2f} | A: MD {vA:.5f} ratio {m/vA:.2f} | B299: MD {vB:.5f} ratio {m/vB:.2f}"
        f" | law A {square_asym(Dh, *SETS[KEYS[0]]):.5f}, B299 {square_asym(Dh, *SETS[KEYS[1]]):.5f}")
for key in KEYS[:2]:
    e0, er, _ = SETS[key]
    log(f"    eta_r/eta0 needed to put 5.5 below the bound: >= 4.5 (independent of normalisation); {key}: eta_r/eta0 = {er/e0:.3f}")

with open(os.path.join(HERE, "water_slip_length_output.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(OUT) + "\n")
