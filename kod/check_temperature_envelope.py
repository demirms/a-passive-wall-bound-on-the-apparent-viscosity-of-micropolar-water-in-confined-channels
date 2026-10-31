"""Temperature envelope of the BC1 predictions and related numbers.

Coefficients: Hansen et al. 2010 Table II (SPC/Fw), MD-MD normalisation, mapping mu = eta0 - eta_r, kappa = 2 eta_r,
gamma = zeta0 + zeta_r, 284, 289, 293, 299 K (paper_common.TABLE2[:4]).
Predictions: tube and slit closed forms (paper_common), square: 2D solver (micropolar_duct_2d.solve_duct, N = 160,
stretched grid) for D_h <= 491 nm and the large-size law (19) 1 + c/D_h beyond, as in Table 3.
(a) Table 3 groups and Fig. 4 lines at the four temperatures: prediction per temperature, the largest over T,
    the temperature that gives it, the largest excess over the 299 K value; check that the 299 K column reproduces
    the values printed in Table 3 (DIFF line).
(b) Fig. 5: r = 48 and 170 nm, all eight temperatures of Table II (284-319 K): values, monotonicity.
(c) bound 1 + 0.17/0.890 (experimental bulk viscosity at 298 K) and 1 + 0.17/1.0.
(d) tube passive wall-spin range n*, r = 40-500 nm, 284-299 K, for beta = 0 and beta = zeta0 - zeta_r.
    n* = s* (mu + kappa) / (1 + s* kappa), s* from paper_common.s_star.
check_tube_m3_bound.py is NOT imported (it runs an 83 s computation at module level); the same formulas are in paper_common.
Output: check_temperature_envelope_output.txt
"""
import csv
import gc
import os
import numpy as np
from paper_common import TABLE2, eringen, lam, tube_muapp, s_star, water_tube, water_slit, water_slip
from micropolar_duct_2d import solve_duct, stretch_for

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []


def P(s=""):
    print(s, flush=True)
    OUT.append(s)


SETS = {T: (e0, er, z0 + zr) for T, e0, er, z0, zr in TABLE2[:4]}      # 284, 289, 293, 299 K
TS = list(SETS)
rows = {r["id"]: r for r in csv.DictReader(open(os.path.join(HERE, "..", "01_arastirma", "veri_tablosu.csv"), encoding="utf-8-sig"))}
size = lambda i: float(rows[i]["size_nm"])
ids = lambda a, b: [f"R{n:03d}" for n in range(a, b + 1)]


def square_fd(dh, T):
    e0, er, z = SETS[T]
    mu, k, g = eringen(e0, er, z)
    L = lam(mu, k, g)
    s = stretch_for(float(dh * L))
    v = solve_duct(float(dh), float(dh), mu, k, g, 0.0, 160, 160, s, s)[0] / e0
    gc.collect()
    return v


FD_CACHE = {}


def pred(kind, s, T):
    e0, er, z = SETS[T]
    if kind == "tube":
        return water_tube(s, e0, er, z)
    if kind == "slit":
        return water_slit(s, e0, er, z)
    if s <= 491:                                   # square: 2D solver
        if (s, T) not in FD_CACHE:
            FD_CACHE[(s, T)] = square_fd(s, T)
        return FD_CACHE[(s, T)]
    return 1 + water_slip(e0, er, z)[1] / s        # large-size law (19)


# Table 3 groups (label, kind, ids, printed prediction at 299 K)
GROUPS = [
    ("Churaev tube r 41-60", "tube", ids(1, 3), "1.025-1.036"),
    ("Churaev tube r 181-255", "tube", ids(4, 8), "1.006-1.008"),
    ("Churaev tube r 547-998", "tube", ids(9, 11), "1.002-1.003"),
    ("Haneveld slit H 10.5", "slit", ["R046"], "1.162"),
    ("Haneveld slit H 22.4", "slit", ["R047"], "1.092"),
    ("Haneveld slit H 46", "slit", ["R048"], "1.048"),
    ("Tas slit H 53", "slit", ids(49, 52) + ["R070"], "1.041-1.042"),
    ("Tas slit H 102-161", "slit", ids(53, 60) + ["R071"], "1.014-1.022"),
    ("Tas 0.1 M NaCl slit H 53-161", "slit", ids(61, 69), "1.014-1.042"),
    ("Li plate H 25-415", "slit", ids(40, 44), "1.006-1.084"),
    ("Kazoe plate H 234", "slit", ["R077"], "1.010"),
    ("Kazoe plate H 417-745", "slit", ["R078", "R079"], "1.003-1.005"),
    ("Kazoe square Dh 73", "sq", ["R080"], "1.040"),
    ("Kazoe square Dh 119", "sq", ["R081"], "1.025"),
    ("Kazoe square Dh 190", "sq", ["R072"], "1.016"),
    ("Li square Dh 200", "sq", ["R034"], "1.015"),
    ("Li square Dh 393", "sq", ["R035"], "1.008"),
    ("Kazoe square Dh 394", "sq", ["R073"], "1.008"),
    ("Li square Dh 491", "sq", ["R036"], "1.006"),
    ("Kazoe square Dh 760-1691", "sq", ["R074", "R075"], "1.002-1.004"),
    ("Li square Dh 838-1530", "sq", ids(37, 39), "1.002-1.004"),
]


def fmt(lo, hi):
    a, b = f"{lo:.3f}", f"{hi:.3f}"
    return a if a == b else f"{a}-{b}"


# =========================================================================== (a)
P("(a) BC1 predictions of the Table 3 groups at 284, 289, 293, 299 K (MD-MD normalisation, Hansen 2010 Table II)")
P("    per group: range over the members at each T; 'max T' = temperature of the largest member prediction; 'excess' = (max over T) - (299 K) for that member")
P(f"    {'group':32s} " + " ".join(f"{T:>13d} K" for T in TS) + "   max T   max excess over 299 K   | 299 K as printed: check")
env_lo, env_hi = [], []
exc_all = []
ndiff = 0
for lab, kind, members, printed_txt in GROUPS:
    sizes = [size(i) for i in members]
    mat = np.array([[pred(kind, s, T) for T in TS] for s in sizes])      # members x T
    cols = [fmt(mat[:, j].min(), mat[:, j].max()) for j in range(len(TS))]
    imax = int(np.argmax(mat.max(axis=0)))
    exc = (mat.max(axis=1) - mat[:, -1]).max()
    env = fmt(mat.max(axis=1).min(), mat.max(axis=1).max())             # per-member largest over T
    ok299 = cols[-1] == printed_txt
    ndiff += (not ok299)
    P(f"    {lab:32s} " + " ".join(f"{c:>15s}" for c in cols) + f"   {TS[imax]} K   {exc:+.4f}                  | {cols[-1]} vs printed {printed_txt}: {'identical' if ok299 else 'DIFF'}   envelope(max over T) {env}")
    env_lo.append(mat.max(axis=1).min()); env_hi.append(mat.max(axis=1).max()); exc_all.append((exc, lab))
P(f"    largest excess of the T-maximum over the 299 K value among all groups: {max(exc_all)[0]:.4f} ({max(exc_all)[1]})")
P(f"    299 K column vs Table 3: {len(GROUPS)-ndiff}/{len(GROUPS)} groups identical to the printed 3-decimal values, {ndiff} DIFF")

P("\n    Fig. 4 lines (closed forms; square: large-size law (19)) at four temperatures; D_h = 2r (tube), 2H (slit)")
P(f"    {'curve':18s} " + " ".join(f"{T:>9d} K" for T in TS) + "   max T   (max over T) - (299 K)")
for dh in (20, 82, 100, 234, 500):
    for kind, nm in (("tube", "tube"), ("slit", "slit"), ("law", "square law")):
        if kind == "law":
            v = [1 + water_slip(*SETS[T])[1] / dh for T in TS]
        else:
            v = [pred(kind, dh / 2, T) for T in TS]
        P(f"    {nm + ' D_h=' + str(dh):18s} " + " ".join(f"{x:11.5f}" for x in v) + f"   {TS[int(np.argmax(v))]} K   {max(v) - v[-1]:+.5f}")
P("\n    Bound 1 + eta_r/eta0 at 284, 289, 293, 299 K: " + ", ".join(f"{1 + SETS[T][1] / SETS[T][0]:.4f}" for T in TS) + "  (largest at 299 K)")

# =========================================================================== (b)
P("\n(b) Fig. 5: tube r = 48 and 170 nm, all eight temperatures of Table II (BC1)")
for r in (48.0, 170.0):
    v = [water_tube(r, e0, er, z0 + zr) for T, e0, er, z0, zr in TABLE2]
    d = np.diff(v)
    P(f"    r = {r:.0f} nm: " + ", ".join(f"{T}K {x:.5f}" for (T, *_), x in zip(TABLE2, v)))
    P(f"      monotone increasing: {bool(np.all(d > 0))}; steps that decrease: " +
      (", ".join(f"{TABLE2[i][0]}->{TABLE2[i+1][0]} K ({d[i]:+.5f})" for i in range(len(d)) if d[i] <= 0) or "none") +
      f"; overall change {v[0]:.4f} -> {v[-1]:.4f}")
    v4 = v[:4]
    P(f"      284-299 K: monotone {bool(np.all(np.diff(v4) > 0))}; maximum at {TS[int(np.argmax(v4))]} K")
bd = [1 + er / e0 for _, e0, er, _, _ in TABLE2]
P("    bound 1 + eta_r/eta0 over 284-319 K: " + ", ".join(f"{b:.4f}" for b in bd) + f"; monotone increasing: {bool(np.all(np.diff(bd) > 0))}")

# =========================================================================== (c)
P("\n(c) bound with the experimental bulk viscosity: 1 + 0.17/0.890 = %.4f; with the 1.0 mPa s reference of Hansen et al. Fig. 3: 1 + 0.17/1.0 = %.4f"
  % (1 + 0.17 / 0.890, 1 + 0.17 / 1.0))

# =========================================================================== (d)
P("\n(d) Tube, passive wall-spin range n* = s*(mu+kappa)/(1+s* kappa) (upper end of the passive range of the spin-slip parameter; n* < 1/2 means that n = 1/2 lies just outside it, i.e. is weakly active)")


def nstar(mu, k, a, bg):
    s = s_star(mu, k, a, bg)
    return s * (mu + k) / (1 + s * k)


radii = (40, 50, 60, 100, 200, 300, 500)
res = {"0": [], "b": []}
P("    T[K]  beta[mPa s nm^2]  beta/gamma | " + " ".join(f"r={r:<4d}" for r in radii) + " (n*, beta = 0 / beta = zeta0 - zeta_r)")
for T, e0, er, z0, zr in TABLE2[:4]:
    mu, k, g = eringen(e0, er, z0 + zr)
    L = lam(mu, k, g)
    bg = (z0 - zr) / (z0 + zr)
    a0 = [nstar(mu, k, L * r, 0.0) for r in radii]
    ab = [nstar(mu, k, L * r, bg) for r in radii]
    res["0"] += a0
    res["b"] += ab
    P(f"    {T}   {z0 - zr:+.2f}            {bg:+.4f}   | " + " ".join(f"{x:.4f}/{y:.4f}" for x, y in zip(a0, ab)))
P(f"    r = 40-500 nm, 284-299 K: n* (beta = 0) in [{min(res['0']):.4f}, {max(res['0']):.4f}]; n* (beta = zeta0 - zeta_r) in [{min(res['b']):.4f}, {max(res['b']):.4f}]")
P("    mapping check (check_hansen2010_mapping_output.txt): beta = 0 -> [0.4750, 0.4984]; beta = zeta0 - zeta_r -> [0.4734, 0.4982]")
def nstar_T(T_row, r, bg_zero):
    T, e0, er, z0, zr = T_row
    mu, k, g = eringen(e0, er, z0 + zr)
    return nstar(mu, k, lam(mu, k, g) * r, 0.0 if bg_zero else (z0 - zr) / (z0 + zr))


r40 = [nstar_T(row, r, True) for r in (40, 50, 60) for row in TABLE2[:4]]
r40b = [nstar_T(row, r, False) for r in (40, 50, 60) for row in TABLE2[:4]]
P(f"    r = 40-60 nm: n* (beta = 0) = {min(r40):.4f}-{max(r40):.4f}; n* (beta = zeta0 - zeta_r) = {min(r40b):.4f}-{max(r40b):.4f}; all below 1/2: {max(r40 + r40b + res['0'] + res['b']) < 0.5}")
P("    (paper_common.s_star is used; check_tube_m3_bound.py was not imported.)")
with open(os.path.join(HERE, "check_temperature_envelope_output.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")
