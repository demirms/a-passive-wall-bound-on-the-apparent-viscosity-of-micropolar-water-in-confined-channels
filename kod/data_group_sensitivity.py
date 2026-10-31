"""Sensitivity of the data comparison to the rotational viscosity and to the spin viscosity.

For every member row of Table 3 (data_table3_members.csv, written by data_table3_extended.py) and for the 284 K and
299 K coefficient sets (Hansen et al. 2010 Table II, MD-MD normalisation, mu = eta0 - eta_r, kappa = 2 eta_r,
gamma = zeta0 + zeta_r), with eta0 and zeta held at their MD values unless stated:

 (a) f_bound = (measured - 1)/(eta_r/eta0): factor on eta_r that puts the bound 1 + f eta_r/eta0 at the measured value
     (for the squares of the paper: '19-25 times'); also with the measured value minus its own error bar.
 (b) f_pred: factor on eta_r (eta0, zeta fixed; mu = eta0 - f eta_r, kappa = 2 f eta_r) at which the BC1 prediction
     reaches the measured value. As f -> infinity the prediction does NOT grow without limit: lambda^2 ->
     4 eta0/zeta and the prediction tends to a finite ceiling (tube: 1/(1-4J(a_inf)); slit: closed form;
     square: 1 + 8/(lambda_inf D_h)); if the ceiling is below the measured value: 'no finite factor'.
 (c) g_pred: factor on zeta (eta_r fixed) at which the BC1 prediction reaches the measured value. The spin length is
     1/lambda = sqrt[gamma (mu + kappa)/(kappa (2 mu + kappa))], so a larger zeta means a longer spin length and a larger
     effect; the prediction increases with zeta and saturates at the bound 1 + eta_r/eta0 as zeta -> infinity (lambda a -> 0).
     If the measured value is at or above the bound: 'none' (no zeta reaches it). For squares the large-size law
     1 + c/D_h, c proportional to zeta^(1/2), is used and marked '*' where lambda D_h < 30 (law no longer valid).
 (d) f_pred jointly with zeta factors g = 0.25, 1, 4, 16 (prediction reaches the measured value).

Predictions are evaluated with the stable closed forms in this file (tube M1: (1+e)/(1+e(1-4J)), e = f eta_r/eta0;
slit: Hansen Eq. (23); square: large-size law 1 + c/D_h, c = 4 kappa/((mu+kappa) lambda)); at f = 1, g = 1 they are
checked against paper_common. The square law is the large-size limit: lambda D_h >= 32 for all square channels of
the data set at the MD values, and the finite-difference solution is used for the checks at the end (f = 1, 3, 20;
D_h = 73, 119 nm; 299 K), run one after the other.
Output: data_group_sensitivity_output.txt
"""
import csv
import gc
import os
import numpy as np
from scipy.optimize import brentq
from scipy.special import ive
from paper_common import TABLE2, eringen, lam, water_tube, water_slit, water_slip

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []


def P(s=""):
    print(s, flush=True)
    OUT.append(s)


SETS = {T: (e0, er, z0 + zr) for T, e0, er, z0, zr in TABLE2 if T in (284, 299)}


def lam2(e0, er, z):
    return 4 * er * e0 / ((e0 + er) * z)


def J_(a):
    return ive(2, a) / (a * ive(1, a))


def tube_pred(r, e0, er, z):
    eps = er / e0
    a = np.sqrt(lam2(e0, er, z)) * r
    return (1 + eps) / (1 + eps * (1 - 4 * J_(a)))


def slit_pred(H, e0, er, z):
    kh = np.sqrt(lam2(e0, er, z)) * H / 2
    red = 3 * er * (kh - np.tanh(kh)) / (np.tanh(kh) * (e0 + er) * kh ** 2)
    return 1 / (1 - red)


def sq_pred(D, e0, er, z):
    c = 8 * er / ((e0 + er) * np.sqrt(lam2(e0, er, z)))
    return 1 + c / D


PRED = {"tube": tube_pred, "slit": slit_pred, "sq": sq_pred}


def pred(kind, s, coef, f=1.0, g=1.0):
    e0, er, z = coef
    return PRED[kind](s, e0, f * er, g * z)


def ceiling(kind, s, coef, g=1.0):
    """limit f -> infinity (eta0, zeta fixed): evaluate with a huge f (lambda^2 -> 4 eta0/zeta) and check convergence."""
    a = pred(kind, s, coef, f=1e7, g=g)
    b = pred(kind, s, coef, f=1e9, g=g)
    return b if abs(a - b) <= 1e-3 * abs(b) else float("nan")


def solve_f(kind, s, coef, target, g=1.0):
    """smallest f > 0 with prediction = target (prediction increases with f); returns ('le1', None), ('f', value), ('none', ceiling)."""
    p1 = pred(kind, s, coef, 1.0, g)
    if target <= p1:
        return ("le1", brentq(lambda f: pred(kind, s, coef, f, g) - target, 1e-8, 1.0) if target > pred(kind, s, coef, 1e-8, g) else 0.0)
    cmax = ceiling(kind, s, coef, g)
    if target >= cmax:
        return ("none", cmax)
    return ("f", brentq(lambda f: pred(kind, s, coef, f, g) - target, 1.0, 1e7))


def solve_g(kind, s, coef, target):
    """factor g on zeta (eta_r fixed); the prediction increases with g and saturates at the bound.
    returns ('none', bound) | ('le1', g) | ('g', g) | ('law', g, valid)."""
    e0, er, z = coef
    bound = 1 + er / e0
    if target >= bound:
        return ("none", bound)
    if kind == "sq":
        c0 = 8 * er / ((e0 + er) * np.sqrt(lam2(e0, er, z)))
        g = ((target - 1) * s / c0) ** 2                       # 1 + c0 sqrt(g)/D = target
        valid = np.sqrt(lam2(e0, er, z) / g) * s >= 30.0
        return ("law", g, bool(valid))
    p1 = pred(kind, s, coef, 1.0, 1.0)
    fn = lambda g: pred(kind, s, coef, 1.0, g) - target
    if target <= p1:
        return ("le1", brentq(fn, 1e-9, 1.0) if fn(1e-9) < 0 else 0.0)
    return ("g", brentq(fn, 1.0, 1e12))


# ---------------------------------------------------------------- sanity checks (closed forms vs paper_common)
P("(0) Checks of the closed forms used here against paper_common (f = 1, g = 1, 299 K):")
c299 = SETS[299]
for kind, s, ref in (("tube", 48.0, water_tube(48.0, *c299)), ("tube", 41.25, water_tube(41.25, *c299)), ("slit", 10.5, water_slit(10.5, *c299)),
                     ("slit", 234.0, water_slit(234.0, *c299)), ("sq", 73.0, 1 + water_slip(*c299)[1] / 73.0)):
    v = pred(kind, s, c299)
    P(f"    {kind:5s} size {s:7.2f} nm: here {v:.6f}  paper_common {ref:.6f}  diff {v - ref:+.1e}")
    assert abs(v - ref) < 1e-9
a1, a2 = pred("tube", 48.0, c299, f=1e4), water_tube(48.0, c299[0], 1e4 * c299[1], c299[2])
P(f"    large-f tube (f = 1e4): here {a1:.6f}  paper_common {a2:.6f}")
# monotonicity of the prediction in f and g (tube r = 48, slit H = 53, square 73 nm)
for kind, s in (("tube", 48.0), ("slit", 53.0), ("sq", 73.0)):
    fs = np.array([0.1, 0.5, 1, 2, 5, 20, 100, 1e3, 1e5, 1e7])
    pf = [pred(kind, s, c299, f=f) for f in fs]
    gs = np.array([1e-3, 0.01, 0.1, 1, 10, 100, 1e3, 1e6])
    pg = [pred(kind, s, c299, g=g) for g in gs]
    P(f"    {kind:5s} {s:g} nm: increasing in f: {bool(np.all(np.diff(pf) > 0))}; increasing in g (zeta): {bool(np.all(np.diff(pg) > 0))}; "
      f"f=1e7 -> {pf[-1]:.5f}; bound {1 + c299[1] / c299[0]:.4f}")

# ---------------------------------------------------------------- members
mem = list(csv.DictReader(open(os.path.join(HERE, "data_table3_members.csv"), encoding="utf-8")))
groups = []
for m in mem:
    if not groups or groups[-1][0] != m["group"]:
        groups.append((m["group"], []))
    groups[-1][1].append(m)


def rng(vals, nd=2):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    lo, hi = min(vals), max(vals)
    a, b = f"{lo:.{nd}f}", f"{hi:.{nd}f}"
    return a if a == b else f"{a}-{b}"


P("\n(a)-(d) Per group. eps = eta_r/eta0 = %.4f (299 K), %.4f (284 K); bound = 1 + eps." % (SETS[299][1] / SETS[299][0], SETS[284][1] / SETS[284][0]))
P("    f_bound = (meas - 1)/eps; 'f_pred' = factor on eta_r at which the BC1 prediction reaches the measured value (zeta fixed); ")
P("    'none' = measured value above the finite ceiling of the prediction (ceiling given); '<=1' = measured value at or below the BC1 prediction.")
hdr = f"    {'group':34s} {'meas':>10s} | {'f_bound 299':>11s} {'f_bound 284':>11s} {'(meas-unc) 299':>14s} | {'f_pred 299':>11s} {'ceiling 299':>12s} | {'g (zeta) 299':>16s}"
P(hdr)
rows_out = []
per_member = []
for gname, ms in groups:
    fb299, fb284, fb299lo, fp299, fp284, ceil299, gz = [], [], [], [], [], [], []
    nnone = 0
    nle1 = 0
    gnone = 0
    gstar = 0
    for m in ms:
        kind, s, y, u = m["kind"], float(m["size_nm"]), float(m["eta_ratio"]), float(m["unc"])
        e299 = SETS[299][1] / SETS[299][0]
        e284 = SETS[284][1] / SETS[284][0]
        fb299.append((y - 1) / e299)
        fb284.append((y - 1) / e284)
        fb299lo.append((y - u - 1) / e299)
        st, v = solve_f(kind, s, SETS[299], y)
        st2, v2 = solve_f(kind, s, SETS[284], y)
        ceil299.append(ceiling(kind, s, SETS[299]))
        if st == "f":
            fp299.append(v)
        elif st == "none":
            nnone += 1
        else:
            nle1 += 1
        sg = solve_g(kind, s, SETS[299], y)
        if sg[0] == "none":
            gnone += 1
        else:
            gz.append(sg[1])
            if sg[0] == "law" and not sg[2]:
                gstar += 1
        per_member.append((gname, m["id"], kind, s, y, u, (y - 1) / e299, (y - 1) / e284, st, v, st2, v2, sg[0], sg[1]))
    fp = (rng(fp299) if fp299 else "-") + (f" (+{nnone} none)" if nnone else "") + (f" ({nle1} <=1)" if nle1 else "")
    gtxt = (rng(gz, 2) if gz else "none") + (f" (+{gnone} none)" if gnone and gz else "") + (f" *{gstar}" if gstar else "")
    P(f"    {gname:34s} {rng([float(m['eta_ratio']) for m in ms]):>10s} | {rng(fb299):>11s} {rng(fb284):>11s} {rng(fb299lo):>14s} | {fp:>11s} {rng(ceil299, 3):>12s} | {gtxt:>16s}")

P("\n    Notes: g (zeta) for squares uses the large-size law and is only indicative where marked *n (the law requires lambda D_h >= 30); 'none' = measured value at or above the bound, which no zeta can exceed; g > 1 = longer spin length.")

P("\n(a') Per member (299 K / 284 K): measured, f_bound, status of f_pred")
for pm in per_member:
    gname, i, kind, s, y, u, fb299, fb284, st, v, st2, v2, sg, vg = pm
    stxt = {"f": f"f_pred 299 K = {v:.2f}", "le1": "f_pred <= 1 (data at or below BC1 prediction)", "none": f"no finite f; ceiling {v:.3f}"}[st]
    stxt2 = {"f": f"284 K: {v2:.2f}", "le1": "284 K: <=1", "none": f"284 K: none (ceiling {v2:.3f})"}[st2]
    P(f"    {i} {gname[:30]:30s} {kind:4s} {s:7.1f} nm meas {y:6.3f} +- {u:5.3f} | f_bound {fb299:6.2f} / {fb284:6.2f} | {stxt}; {stxt2}")

# ---------------------------------------------------------------- (d) joint
P("\n(d) Joint change: factor on eta_r (f) needed to reach the measured value for a prescribed factor g on zeta (299 K).")
P("    g < 1: longer spin length (larger lambda^-1); g > 1: shorter. Per group the range over members; 'none' = measured above the ceiling for that g.")
GS = (0.25, 1.0, 4.0, 16.0)
P(f"    {'group':34s} " + " ".join(f"{'g=' + str(g):>16s}" for g in GS))
for gname, ms in groups:
    cols = []
    for g in GS:
        vals, nn, nl = [], 0, 0
        for m in ms:
            st, v = solve_f(m["kind"], float(m["size_nm"]), SETS[299], float(m["eta_ratio"]), g)
            if st == "f":
                vals.append(v)
            elif st == "none":
                nn += 1
            else:
                nl += 1
        t = rng(vals) if vals else "-"
        if nn:
            t += f"+{nn}none"
        if nl:
            t += f"+{nl}<=1"
        cols.append(t)
    P(f"    {gname:34s} " + " ".join(f"{c:>16s}" for c in cols))

# ---------------------------------------------------------------- headline numbers
P("\n(e) Headline numbers")
tube = [pm for pm in per_member if pm[2] == "tube" and pm[3] <= 60]
P(f"    tubes r = 41-60 nm (Churaev, ~20 C): f_bound = {rng([p[6] for p in tube])} (299 K), {rng([p[7] for p in tube])} (284 K); "
  f"needed eta_r/eta0 = {min(p[4] for p in tube) - 1:.3f}-{max(p[4] for p in tube) - 1:.3f} (MD 0.185-0.243)")
tf = [(p[8], p[9], p[10], p[11]) for p in tube]
P("    tubes r = 41-60 nm: BC1 prediction reaches the measured value at f = " + ", ".join(
    (f"{p[1]:.1f}" if p[0] == "f" else f"none (ceiling {p[1]:.3f})") for p in tf) + " (299 K); ceiling of the tube BC1 prediction at r = 41-60 nm: "
  + rng([ceiling('tube', pm[3], SETS[299]) for pm in tube], 3))
sq = [pm for pm in per_member if pm[2] == "sq" and pm[3] <= 491 and pm[4] > 1.7]
P(f"    squares D_h <= 491 nm (n = {len(sq)}): f_bound = {rng([p[6] for p in sq], 1)} (299 K), {rng([p[7] for p in sq], 1)} (284 K) (19-25 for the single largest value 5.55);"
  f" no finite f_pred for any of them: {all(p[8] == 'none' for p in sq)}; ceilings {rng([ceiling('sq', p[3], SETS[299]) for p in sq], 3)} vs measured {rng([p[4] for p in sq])}")
sl = [pm for pm in per_member if pm[2] == "slit" and pm[1] in ("R049", "R050", "R051", "R052", "R070")]
P(f"    Tas slits H = 53 nm: f_bound {rng([p[6] for p in sl])} (299 K), {rng([p[7] for p in sl])} (284 K); f_pred {rng([p[9] for p in sl if p[8] == 'f'])}")
h = [pm for pm in per_member if pm[1] == "R046"][0]
P(f"    Haneveld 10.5 nm slit: f_bound {h[6]:.2f} (299 K) / {h[7]:.2f} (284 K); f_pred {h[9]:.2f} ({h[8]})")
pl = [pm for pm in per_member if pm[1] == "R077"][0]
P(f"    Kazoe plate 234 nm: f_bound {pl[6]:.2f} (299 K) / {pl[7]:.2f} (284 K); f_pred {pl[9]:.2f} ({pl[8]})")
ch = [pm for pm in per_member if pm[1] in ("R004", "R005", "R006", "R007", "R008")]
P(f"    Churaev tubes r = 181-255 nm: measured {rng([p[4] for p in ch])}; f_bound {rng([p[6] for p in ch])}; f_pred {rng([p[9] for p in ch if p[8] == 'f'])}")

# ---------------------------------------------------------------- FD checks of the square law (sequential)
P("\n(f) Finite-difference checks of the square large-size law (299 K coefficients; stretched grid N = 160; solved one after the other)")
from micropolar_duct_2d import solve_duct, stretch_for
for Dh in (73.0, 119.0):
    for f in (1.0, 3.0, 20.0):
        e0, er, z = SETS[299]
        e0_, er_ = e0, f * er
        mu, k, g = e0_ - er_, 2 * er_, z
        L = np.sqrt(k * (2 * mu + k) / (g * (mu + k)))
        st = stretch_for(float(Dh * L))
        val = solve_duct(float(Dh), float(Dh), mu, k, g, 0.0, 160, 160, st, st)[0] / e0
        law = sq_pred(Dh, e0, er_, z)
        P(f"    D_h = {Dh:5.0f} nm, f = {f:4.0f} (mu = {mu:+.3f}, kappa = {k:.3f} mPa s, lambda D_h = {L * Dh:.1f}): FD {val:.5f}  law {law:.5f}  diff {val - law:+.5f}")
        gc.collect()
e0, er, z = SETS[299]
P(f"    ceiling of the square law (f -> infinity): 1 + 8/(lambda_inf D_h), lambda_inf = 2 sqrt(eta0/zeta) = {2 * np.sqrt(e0 / z):.4f} /nm: "
  f"D_h = 73 nm {1 + 8 / (2 * np.sqrt(e0 / z) * 73):.4f}; 119 nm {1 + 8 / (2 * np.sqrt(e0 / z) * 119):.4f}")

with open(os.path.join(HERE, "data_group_sensitivity_output.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")
