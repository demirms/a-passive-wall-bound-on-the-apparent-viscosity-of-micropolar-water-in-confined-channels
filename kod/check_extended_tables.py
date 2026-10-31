"""Extended tables and wall-condition checks used in the paper.

(1) Table 1 extended to all temperatures of Hansen et al. 2010 Table II (284-319 K).
(2) Unrounded bound at 284 K and its two-decimal rounding.
(3) Migun / Kolpashchikov fit (kappa/mu = 5.25, N = 7.03e7 1/m) with the tube closed form.
(4) Equivalent slip length of the slit wall layer for BC1 (zero spin), BC3 (spin slip) and BC4 (zero couple).
(5) Additional data groups (every hydrophilic-water point of the data-size figure), predictions at 299 K.
(6) Size of the BC2 tube scan grid and number of grid points that fall on the pole n_p.
(7) Tube, BC2: lam R above which the negative-n violation interval disappears (water, 284-299 K).
"""
import numpy as np
from scipy.optimize import brentq
from paper_common import TABLE2, eringen, lam, tube_muapp, J_, water_tube, water_slit, water_slip

out = []
P = out.append

# (1) -------------------------------------------------------------------------------------------
P("(1) Table 1, all temperatures (MD-MD normalisation)")
P("    T[K]  eta0   eta_r  zeta    ratio    1/lam[nm]  b[nm]     c[nm]    bound")
for T, e0, er, z0, zr in TABLE2:
    z = z0 + zr
    mu, k, g = eringen(e0, er, z)
    b, c = water_slip(e0, er, z)
    P(f"    {T}   {e0:.2f}   {er:.3f}  {z:.2f}    {er/e0:.6f}   {1/lam(mu, k, g):.6f}  {b:+.6f}  {c:.6f}  {1+er/e0:.6f}")

# (2) -------------------------------------------------------------------------------------------
b284 = 1 + 0.17 / 0.92
b299 = 1 + 0.17 / 0.70
P(f"\n(2) bound 284 K = {b284:.6f} -> two decimals {b284:.2f}; 299 K = {b299:.6f} -> {b299:.2f}")

# (3) -------------------------------------------------------------------------------------------
mu, k = 1.0, 5.25
for L in (1e9 / 7.03e7, 14.2):
    vals = [tube_muapp(mu, k, r / L) / (mu + k / 2) for r in (300.0, 40.0)]
    P(f"(3) Migun fit, kappa/mu = 5.25, 1/lam = {L:.4f} nm: r = 300 nm {vals[0]:.5f}; r = 40 nm {vals[1]:.5f}")


# (4) -------------------------------------------------------------------------------------------
def slit_q(cond, h, m, kk, g, ls=0.0):
    """Flow rate of the slit -h<y<h with G = 1 (same closed form as check_slit_wall_conditions.py)."""
    lm = np.sqrt(kk * (2 * m + kk) / (g * (m + kk)))
    s, c = np.sinh(lm * h), np.cosh(lm * h)
    if cond == 'BC1':
        C = -h / ((2 * m + kk) * s)
    elif cond == 'BC3':
        C = -(h + ls) / ((2 * m + kk) * (s + ls * lm * c))
    else:  # BC4
        C = -1.0 / ((2 * m + kk) * lm * c)
    I1 = h * c / lm - s / lm ** 2
    return 2 / (m + kk) * (h ** 3 / 3 + kk * h ** 3 / (3 * (2 * m + kk)) + kk * C * I1)


P("\n(4) Slit, equivalent slip length b*lam from mu_app/(mu+k/2) = 1 - 3 b/h (mu = gamma = 1)")
for kk in (0.41, 1.0, 10.0):
    lm = np.sqrt(kk * (2 + kk) / (1 + kk))
    bth = -kk / (2 * (1 + kk))
    for hl in (100.0, 200.0, 400.0):
        h = hl / lm
        row = []
        for cond, ls in (('BC1', 0.0), ('BC3', 0.1 / lm), ('BC3', 1.0 / lm), ('BC3', 10.0 / lm), ('BC4', 0.0)):
            q = slit_q(cond, h, 1.0, kk, 1.0, ls)
            ratio = (2 * h ** 3 / (3 * q)) / (1 + kk / 2)
            row.append((1 - ratio) * h / 3 * lm)
        P(f"    k/mu={kk:5.2f} h*lam={hl:6.0f}: BC1 {row[0]:+.5f} (Eq. 18: {bth:+.5f}); "
          f"BC3 lam*ls=0.1 {row[1]:+.5f}, 1 {row[2]:+.5f}, 10 {row[3]:+.5f}; BC4 {row[4]:+.6f}")

# (5) -------------------------------------------------------------------------------------------
W299 = (0.70, 0.17, 2.1)
c299 = water_slip(*W299)[1]
P("\n(5) Additional data groups of the data-size figure (values from veri_tablosu.csv, listed in fig4_data_size_output.txt)")
groups = {
    "Churaev tube r 547-998 nm": [("R009", "tube", 547.1, 1.030, 0.015), ("R010", "tube", 739.9, 1.013, 0.015),
                                  ("R011", "tube", 997.5, 1.006, 0.015)],
    "Haneveld slit H 22.4 nm": [("R047", "slit", 22.4, 1.114, 0.050)],
    "Haneveld slit H 46 nm": [("R048", "slit", 46.0, 1.020, 0.040)],
    "Tas slit H 102-161 nm (pure water)": [("R053", "slit", 104.5, 1.137, 0.11), ("R054", "slit", 102.4, 1.067, 0.11),
                                           ("R055", "slit", 110.9, 1.099, 0.11), ("R056", "slit", 111.2, 1.058, 0.11),
                                           ("R057", "slit", 152.6, 1.076, 0.11), ("R058", "slit", 154.9, 1.060, 0.11),
                                           ("R059", "slit", 161.2, 1.086, 0.11), ("R060", "slit", 152.7, 1.034, 0.11),
                                           ("R071", "slit", 152.0, 1.070, 0.06)],
    "Kazoe plate H 417-745 nm": [("R078", "slit", 417.0, 0.897, 0.101), ("R079", "slit", 745.0, 1.099, 0.090)],
    "Kazoe square D_h 760-1691 nm": [("R074", "sq", 760.0, 1.211, 0.101), ("R075", "sq", 1691.0, 1.133, 0.022)],
    "Li square D_h 838-1530 nm": [("R037", "sq", 838.0, 1.120, 0.072), ("R038", "sq", 1112.0, 0.855, 0.038),
                                  ("R039", "sq", 1530.0, 0.976, 0.034)],
}
for name, rows in groups.items():
    preds, devs = [], []
    for rid, geo, size, meas, unc in rows:
        if geo == "tube":
            p = water_tube(size, *W299)
        elif geo == "slit":
            p = water_slit(size, *W299)
        else:
            p = 1 + c299 / size          # large-size law (19)
        preds.append(p)
        devs.append((meas - p) / unc)
        P(f"    {rid} {geo:4s} {size:7.1f} meas {meas:.3f} +- {unc:.3f} pred {p:.5f} (meas-pred)/unc {devs[-1]:+.2f}"
          f" above bound 1.2429: {meas > 1.2429}")
    ms = [r[3] for r in rows]
    P(f"  -> {name}: measured {min(ms):.3f}-{max(ms):.3f}, predicted {min(preds):.5f}-{max(preds):.5f}, "
      f"(meas-pred)/unc {min(devs):+.2f} .. {max(devs):+.2f}")

# (6) -------------------------------------------------------------------------------------------
kl = np.geomspace(0.1, 10, 21); al = np.geomspace(0.1, 100, 31); nl = np.linspace(-6, 16, 881)
on_pole = sum(int(np.sum(np.abs(nl - (1 + kk) / kk) < 1e-9)) for kk in kl)
total = len(kl) * len(al) * len(nl) * 3
P(f"\n(6) BC2 tube scan: full grid {total}; grid n on the pole for {on_pole} kappa/mu values "
  f"-> omitted {on_pole * len(al) * 3}; evaluated {total - on_pole * len(al) * 3} (28 225 + 1 692 089 = {28225 + 1692089})")

# (7) -------------------------------------------------------------------------------------------
P("\n(7) BC2 tube: negative-n violation interval exists iff kappa(1/4 - J) < (2 mu + kappa) J")
for T, e0, er, z0, zr in TABLE2[:4]:
    mu, k, g = eringen(e0, er, z0 + zr)
    f = lambda a: k * (0.25 - J_(a)) - (2 * mu + k) * J_(a)
    a_star = brentq(f, 0.5, 200.0)
    P(f"    {T} K: kappa/mu = {k/mu:.4f}; interval of negative n only for lam R < {a_star:.3f}")
P("    (experimental tubes r = 41-60 nm: lam R = 18-31, paper_numbers_output.txt (5))")

with open("check_extended_tables_output.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(out) + "\n")
print("\n".join(out))
