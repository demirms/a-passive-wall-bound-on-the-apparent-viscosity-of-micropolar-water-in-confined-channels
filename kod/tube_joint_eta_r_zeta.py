"""Joint change of the rotational viscosity eta_r and the spin viscosity zeta in the tube (BC1 wall).

Definitions (all other quantities as in Table 1 of the manuscript):
    f : factor on eta_r   ->  mu = eta0 - f*eta_r,  kappa = 2 f eta_r   (eta0 fixed)
    g : factor on zeta    ->  gamma = g*(zeta0 + zeta_r)               (alpha + beta does not enter BC1 tube flow)
    internal length 1/lambda, lambda^2 = kappa (2 mu + kappa) / [gamma (mu + kappa)]   (manuscript definition;
    NO extra factor 2)
    tube radius R, a = lambda R, BC1 (zero wall spin):  mu_app/eta0 from paper_common.tube_muapp.
Because BC1 sets the wall spin to zero, the BC1 tube value does not depend on beta at all (beta enters only the
wall power of BC2/BC4 walls; see khanukaeva_n07_beta_power.py).  Temperatures: 299 K and 284 K coefficients of Table 1,
eta0 fixed at the bulk value of that temperature.

Parts:
 (a) per tube member (41-60 nm, measured 1.339-1.407) and for a grid of values: smallest zeta factor g_min (f -> infinity)
     and the pairs (f, g) that reach the value; also g_min for capped f.
 (b) one pair (f, g) that reaches both ends of the 41-60 nm set (end values of that range), and a chi^2 fit to the three members.
 (c) the Migun/Kolpashchikov fit (kappa/mu = 5.25, 1/lambda = 14.22 nm) as an (f, g) pair; 1.485 at 40 nm.
 (d) ceilings of the tube prediction for g = 1, 10, 100, 1000 (f -> infinity).
 (e) squares: necessary condition f >= f_bound for every square member (diff against data_group_sensitivity_output.txt),
     and the zeta factor g needed for f = 1.1 f_bound (law (19) and finite differences).
Output: tube_joint_eta_r_zeta_output.txt
"""
import csv
import gc
import os
import re
import numpy as np
from scipy.optimize import brentq, least_squares, minimize

from paper_common import TABLE2, eringen, lam, tube_muapp, J_
from micropolar_duct_2d import solve_duct, stretch_for

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []


def P(s=""):
    print(s, flush=True)
    OUT.append(str(s))


SETS = {T: (e0, er, z0 + zr) for T, e0, er, z0, zr in TABLE2}
COEF = {299: SETS[299], 284: SETS[284]}


def pred_tube(R, coef, f=1.0, g=1.0):
    """BC1 tube mu_app/eta0 with eta_r -> f eta_r, zeta -> g zeta, eta0 fixed.
    Cancellation-free form (2 mu + kappa = 2 eta0 and mu + kappa = eta0 + f eta_r exactly); identical to
    paper_common.tube_muapp(eringen(...)) for moderate f (checked below)."""
    e0, er, z = coef
    k = 2.0 * f * er
    s2 = 2.0 * e0
    mk = e0 + f * er
    a = np.sqrt(k * s2 / (g * z * mk)) * R
    return (mk / (1.0 + k / s2 - 4.0 * k / s2 * J_(a))) / e0


def ceiling_tube(R, coef, g=1.0):
    """f -> infinity limit: lambda^2 -> 4 eta0/gamma; value 1/(1 - 4 J(lambda_inf R))."""
    e0, er, z = coef
    a = 2.0 * np.sqrt(e0 / (g * z)) * R
    return 1.0 / (1.0 - 4.0 * J_(a))


def inv_lambda(coef, f, g):
    e0, er, z = coef
    mu, k, gam = eringen(e0, f * er, g * z)
    return 1.0 / lam(mu, k, gam)


def inv_lambda_inf(coef, g):
    e0, er, z = coef
    return 1.0 / (2.0 * np.sqrt(e0 / (g * z)))


def gmin(R, coef, m):
    if ceiling_tube(R, coef, 1.0) >= m:
        return 1.0
    return float(np.exp(brentq(lambda lg: ceiling_tube(R, coef, np.exp(lg)) - m, 0.0, 20.0, xtol=1e-12)))


def gmin_fcap(R, coef, m, fcap):
    """smallest g such that pred(R, f=fcap, g) >= m (pred increases with f and with g)."""
    if pred_tube(R, coef, fcap, 1.0) >= m:
        return 1.0
    if ceiling_tube(R, coef, 1e12) < m:
        return np.nan
    fn = lambda lg: pred_tube(R, coef, fcap, np.exp(lg)) - m
    hi = 25.0
    if fn(hi) < 0:
        return np.nan
    return float(np.exp(brentq(fn, 0.0, hi, xtol=1e-12)))


def f_needed(R, coef, m, g):
    """smallest f with pred(R, f, g) = m (increasing in f); nan if above the ceiling for this g."""
    if pred_tube(R, coef, 1.0, g) >= m:
        return 1.0
    if ceiling_tube(R, coef, g) <= m:
        return np.nan
    return float(np.exp(brentq(lambda lf: pred_tube(R, coef, np.exp(lf), g) - m, 0.0, 25.0, xtol=1e-12)))


# ------------------------------------------------------------------------------------------ sanity
P("tube_joint_eta_r_zeta.py: BC1 tube, joint change of eta_r (factor f) and zeta (factor g); eta0 fixed.")
P("Sets (Table 1): " + "; ".join(f"{T} K: eta0 {c[0]}, eta_r {c[1]}, zeta {c[2]:.2f}" for T, c in COEF.items()))
for T, c in COEF.items():
    P(f"  {T} K: eta_r/eta0 = {c[1]/c[0]:.4f}, bound 1+eta_r/eta0 = {1+c[1]/c[0]:.4f}, 1/lambda(MD) = {inv_lambda(c, 1, 1):.3f} nm")
# check: BC1 tube value at f=g=1 against the printed paper value (299 K, 40 nm: 1.0368 in the independent recomputation)
P(f"  check f=g=1: 299 K R=40 {pred_tube(40, COEF[299]):.4f} (independent recomputation: 1.0368); 284 K R=40 {pred_tube(40, COEF[284]):.4f} (independent recomputation: 1.0337)")
assert abs(pred_tube(40, COEF[299]) - 1.0368) < 5e-5
for _f, _g in ((1, 1), (2.5, 30), (4.0, 100)):
    _mu, _k, _ga = eringen(COEF[299][0], _f * COEF[299][1], _g * COEF[299][2])
    assert abs(tube_muapp(_mu, _k, lam(_mu, _k, _ga) * 45.0) / COEF[299][0] - pred_tube(45.0, COEF[299], _f, _g)) < 1e-12 and abs(pred_tube(40, COEF[284]) - 1.0337) < 5e-5
# monotonicity in f and g
for R in (41.25, 60):
    fs = np.logspace(0, 8, 200)
    v = np.array([pred_tube(R, COEF[299], f, 1.0) for f in fs])
    gs = np.logspace(0, 4, 200)
    w = np.array([pred_tube(R, COEF[299], 1.0, g) for g in gs])
    P(f"  R={R}: increasing in f (g=1): {bool(np.all(np.diff(v) > -1e-12))}, limit f=1e8 {v[-1]:.5f} vs ceiling {ceiling_tube(R, COEF[299]):.5f}; increasing in g: {bool(np.all(np.diff(w) > -1e-12))}")

# ------------------------------------------------------------------------------------------ (a)
P("\n(a) Smallest zeta factor g_min (f -> infinity) for the tube value m at radius R; compare the independent recomputation")
INDEP_A = {(299, 41, 1.31): 9.61, (299, 41, 1.34): 11.24, (299, 41, 1.41): 15.34, (299, 41, 1.48): 19.80,
        (299, 60, 1.31): 20.58, (299, 60, 1.34): 24.07, (299, 60, 1.41): 32.85, (299, 60, 1.48): 42.40,
        (284, 41, 1.31): 8.84, (284, 41, 1.34): 10.34, (284, 41, 1.41): 14.11, (284, 41, 1.48): 18.21,
        (284, 60, 1.31): 18.93, (284, 60, 1.34): 22.14, (284, 60, 1.41): 30.22, (284, 60, 1.48): 39.01}
P(f"    {'T':>4s} {'R':>3s} {'m':>5s} {'g_min here':>11s} {'indep.':>7s} {'rel.diff':>9s} {'1/lam_inf [nm]':>15s}")
maxrel = 0.0
for (T, R, m), ref in INDEP_A.items():
    gm = gmin(R, COEF[T], m)
    rel = gm / ref - 1
    maxrel = max(maxrel, abs(rel))
    P(f"    {T:4d} {R:3d} {m:5.2f} {gm:11.2f} {ref:7.2f} {100*rel:+8.2f}% {inv_lambda_inf(COEF[T], gm):15.2f}")
P(f"    max relative difference to the independent recomputation: {100*maxrel:.2f} % (acceptance < 3 %)")
assert maxrel < 0.03

P("\n    Tube members of Table 3 (Churaev [1], r = 41-60 nm; measured value, own read-off sigma 0.015):")
mem = list(csv.DictReader(open(os.path.join(HERE, "data_table3_members.csv"), encoding="utf-8")))
tubes = [m for m in mem if m["kind"] == "tube" and float(m["size_nm"]) <= 60]
assert len(tubes) == 3
FG = (1, 3, 10, 30, 100, 300, 1000)
for T in (299, 284):
    P(f"    --- {T} K (eta_r/eta0 = {COEF[T][1]/COEF[T][0]:.4f}; ceiling at g=1: see (d))")
    for m in tubes:
        R, y = float(m["size_nm"]), float(m["eta_ratio"])
        gm = gmin(R, COEF[T], y)
        capped = ", ".join(f"f<={fc}: " + (f"{gmin_fcap(R, COEF[T], y, fc):.1f}" if np.isfinite(gmin_fcap(R, COEF[T], y, fc)) else "none") for fc in (3, 5, 10, 30, 100))
        P(f"    {m['id']} R = {R:6.2f} nm  meas {y:.3f}: g_min(f->inf) = {gm:6.2f} (1/lam_inf {inv_lambda_inf(COEF[T], gm):.2f} nm);  g_min for capped f: {capped}")
        row = []
        for g in (gm * 1.05, gm * 1.5, gm * 2, 100, 300):
            fn = f_needed(R, COEF[T], y, g)
            row.append(f"g={g:7.1f}: f=" + (f"{fn:8.2f} (1/lam {inv_lambda(COEF[T], fn, g):6.2f} nm)" if np.isfinite(fn) else "none"))
        P("        " + " | ".join(row))

# ------------------------------------------------------------------------------------------ (b)
P("\n(b) One pair (f, g) reaching both ends of the 41-60 nm set (2 equations, 2 unknowns); compare the independent recomputation (fsolve)")
INDEP_B = {(299, 1.41, 1.34): (2.14, 165.6, 19.05), (299, 1.41, 1.37): (1.88, 359.8, 29.34), (299, 1.38, 1.34): (1.76, 301.9, 27.48),
        (284, 1.41, 1.34): (2.82, 152.4, 19.05), (284, 1.41, 1.37): (2.47, 331.0, 29.34), (284, 1.38, 1.34): (2.32, 277.8, 27.48)}
P(f"    {'T':>4s} {'m(41)':>6s} {'m(60)':>6s} {'f':>7s} {'g':>8s} {'1/lam[nm]':>9s} | indep. f, g, 1/lam | max rel.diff")
for (T, m1, m2), (rf, rg, rl) in INDEP_B.items():
    def res(p):
        f, g = np.exp(p)
        return [pred_tube(41, COEF[T], f, g) - m1, pred_tube(60, COEF[T], f, g) - m2]
    best = None
    for guess in ([1.0, 5.0], [0.7, 4.0], [2.0, 6.0]):
        sol = least_squares(res, guess, xtol=1e-14, ftol=1e-14, gtol=1e-14)
        if np.max(np.abs(sol.fun)) < 1e-9:
            best = sol
            break
    f, g = np.exp(best.x)
    il = inv_lambda(COEF[T], f, g)
    rd = max(abs(f / rf - 1), abs(g / rg - 1), abs(il / rl - 1))
    P(f"    {T:4d} {m1:6.2f} {m2:6.2f} {f:7.3f} {g:8.1f} {il:9.2f} | {rf}, {rg}, {rl} | {100*rd:.2f} %   residual {np.max(np.abs(best.fun)):.1e}")
    assert rd < 0.03
P("    -> the pair reaching both ends has 1/lambda = 19 nm (end values 1.41/1.34), 27-29 nm for flatter end values;")
P("       the two ends fix 2 numbers, the prediction falls with R, so a single pair cannot follow a rise with R (members are not monotone).")

P("\n    Least-squares fit of one pair (f, g) to the three Table-3 tube members (weights 1/0.015; box 1 <= f, g <= 1000; the unconstrained minimum runs to g -> infinity with a flat 1.369):")
for T in (299, 284):
    Rv = np.array([float(m["size_nm"]) for m in tubes])
    yv = np.array([float(m["eta_ratio"]) for m in tubes])
    sg = np.array([float(m["unc"]) for m in tubes])
    fun = lambda p: (np.array([pred_tube(R, COEF[T], np.exp(p[0]), np.exp(p[1])) for R in Rv]) - yv) / sg
    best = None
    for guess in ([0.5, 3.0], [0.7, 4.0], [0.2, 5.0], [1.0, 6.0]):
        sol = least_squares(fun, np.log([1.0,5.0]) if False else guess, bounds=([0.0, 0.0], [np.log(1e3), np.log(1e3)]), xtol=1e-14, ftol=1e-14)
        if best is None or sol.cost < best.cost:
            best = sol
    f, g = np.exp(best.x)
    P(f"    {T} K: f = {f:.2f}, g = {g:.0f}, 1/lambda = {inv_lambda(COEF[T], f, g):.1f} nm, chi^2 = {2*best.cost:.2f} (3 points, 2 parameters); predictions "
      + ", ".join(f"{pred_tube(R, COEF[T], f, g):.3f}" for R in Rv) + " vs " + ", ".join(f"{y:.3f}" for y in yv))

# the minimum of 1/lambda over all pairs that reach the data one-sidedly (pred >= m - sigma for every member): STOP criterion
P("\n    Smallest internal length 1/lambda among pairs that reach EVERY member of the 41-60 nm set within one error bar from below")
P("    (prediction >= measured - 0.015 for all three members, the binding member is the one at the largest R and value):")
for T in (299, 284):
    Rv = np.array([float(m["size_nm"]) for m in tubes])
    yv = np.array([float(m["eta_ratio"]) for m in tubes])
    sg = np.array([float(m["unc"]) for m in tubes])
    for fcap in (3, 5, 10, 30, 100, np.inf):
        # for each f find the smallest g reaching all (g_min is largest over members), then 1/lambda
        gneed = []
        for R, y, s in zip(Rv, yv, sg):
            ymin = y - s
            gneed.append(gmin(R, COEF[T], ymin) if not np.isfinite(fcap) else gmin_fcap(R, COEF[T], ymin, fcap))
        if not np.all(np.isfinite(gneed)):
            P(f"    {T} K, f <= {fcap}: no g reaches all three")
            continue
        gg = max(gneed)
        fu = fcap if np.isfinite(fcap) else np.inf
        il = inv_lambda_inf(COEF[T], gg) if not np.isfinite(fcap) else inv_lambda(COEF[T], fcap, gg)
        P(f"    {T} K, f <= {fcap}: g = {gg:7.1f}, 1/lambda = {il:6.2f} nm")

# ------------------------------------------------------------------------------------------ (c)
P("\n(c) Migun [18] / Kolpashchikov [19] fit as an (f, g) pair: kappa/mu = 5.25 (eta_r/eta0 = 0.7241), 1/lambda = 14.22 nm")
x = 0.5 * 5.25 / (1 + 0.5 * 5.25)
P(f"    eta_r/eta0 = (kappa/2)/(mu+kappa/2) = {x:.4f}")
for T in (299, 284):
    e0, er, z = COEF[T]
    f = x / (er / e0)
    gam = 4 * e0 * x * 14.22 ** 2 / (1 + x)       # lambda^2 = 4 eta_r eta0/[(eta0+eta_r) gamma] for f eta_r
    g = gam / z
    P(f"    {T} K: f = {f:.2f}, zeta needed = {gam:.1f} mPa s nm^2 = {g:.0f} x MD; check 1/lambda = {inv_lambda(COEF[T], f, g):.2f} nm; BC1 tube at 40 nm: {pred_tube(40, COEF[T], f, g):.4f}; at 300 nm: {pred_tube(300, COEF[T], f, g):.4f}")
    assert abs(inv_lambda(COEF[T], f, g) - 14.22) < 1e-9
P("    independent recomputation: 299 K x2.98 and x113; 284 K x3.92 and x104; manuscript: 3.0-3.9 and 'fitted length 6.2-7.3 x MD'.")
P(f"    fitted length / MD length: {14.22/inv_lambda(COEF[299],1,1):.2f} (299 K), {14.22/inv_lambda(COEF[284],1,1):.2f} (284 K)")
mu_m, ka_m = 1.0, 5.25
ga_m = ka_m * (2 * mu_m + ka_m) / ((mu_m + ka_m) * (1 / 14.22) ** 2)
P(f"    own normalisation (eta0 = mu + kappa/2 = 1): BC1 tube at 40 nm {tube_muapp(mu_m, ka_m, lam(mu_m, ka_m, ga_m)*40)/(mu_m+ka_m/2):.4f}, at 300 nm {tube_muapp(mu_m, ka_m, lam(mu_m, ka_m, ga_m)*300)/(mu_m+ka_m/2):.4f}")
assert abs(tube_muapp(mu_m, ka_m, lam(mu_m, ka_m, ga_m) * 40) / (mu_m + ka_m / 2) - 1.4847) < 5e-4

# ------------------------------------------------------------------------------------------ (d)
P("\n(d) Ceiling of the BC1 tube prediction (f -> infinity) for zeta factor g (eta0 fixed)")
P(f"    {'T':>4s} {'R [nm]':>7s} " + " ".join(f"{'g=' + str(g):>9s}" for g in (1, 10, 100, 1000)) + "   (also bound 1+eta_r/eta0 at f=1, g -> infinity)")
for T in (299, 284):
    for R in (40, 41.25, 48, 53.03, 59.76, 60):
        P(f"    {T:4d} {R:7.2f} " + " ".join(f"{ceiling_tube(R, COEF[T], g):9.4f}" for g in (1, 10, 100, 1000)))
cg1 = [ceiling_tube(R, COEF[T], 1.0) for T in (299, 284) for R in (41.25, 59.76)]
P(f"    g = 1 ceiling over 41-60 nm, 284 and 299 K: {min(cg1):.3f}-{max(cg1):.3f}; (independent recomputation over 40-60 nm and four T: 1.060-1.098)")
for T in (299, 284):
    for R in (41.25, 59.76):
        v = [pred_tube(R, COEF[T], f, 1.0) for f in (1e6, 1e8)]
        assert abs(v[1] - ceiling_tube(R, COEF[T], 1.0)) < 1e-6

# ------------------------------------------------------------------------------------------ (e)
P("\n(e) Squares: necessary condition f >= f_bound (bound 1 + f eta_r/eta0 must reach the measured value), independent of zeta")
ref = {}
for line in open(os.path.join(HERE, "data_group_sensitivity_output.txt"), encoding="utf-8"):
    mm = re.match(r"\s+(R\d{3}) .* \| f_bound\s+(-?[\d.]+) /\s+(-?[\d.]+) \|", line)
    if mm:
        ref[mm.group(1)] = (float(mm.group(2)), float(mm.group(3)))
sqm = [m for m in mem if m["kind"] == "sq"]
P(f"    {'id':5s} {'group':32s} {'Dh [nm]':>8s} {'meas':>6s} {'f_bound 299':>11s} {'ref':>6s} {'f_bound 284':>11s} {'ref':>6s} diff")
nd = 0
for m in sqm:
    y = float(m["eta_ratio"])
    fb = [(y - 1) / (COEF[T][1] / COEF[T][0]) for T in (299, 284)]
    r = ref[m["id"]]
    ok = abs(round(fb[0], 2) - r[0]) < 1.01e-2 and abs(round(fb[1], 2) - r[1]) < 1.01e-2
    nd += (not ok)
    P(f"    {m['id']:5s} {m['group'][:32]:32s} {float(m['size_nm']):8.0f} {y:6.3f} {fb[0]:11.2f} {r[0]:6.2f} {fb[1]:11.2f} {r[1]:6.2f} {'identical' if ok else 'DIFFERENT'}")
P(f"    diff vs data_group_sensitivity_output.txt: {nd} differing members (acceptance: 0)")
assert nd == 0
big = [m for m in sqm if float(m["size_nm"]) <= 491 and float(m["eta_ratio"]) > 1.7]
fbs = [(float(m["eta_ratio"]) - 1) / (COEF[299][1] / COEF[299][0]) for m in big]
fbs4 = [(float(m["eta_ratio"]) - 1) / (COEF[284][1] / COEF[284][0]) for m in big]
P(f"    squares D_h <= 491 nm, measured > 1.7 (n = {len(big)}): f_bound = {min(fbs):.1f}-{max(fbs):.1f} (299 K), {min(fbs4):.1f}-{max(fbs4):.1f} (284 K)  [plan: 3.1-18.7 at 299 K]")

P("\n    Square with f = 1.1 f_bound: zeta factor g at which the BC1 prediction reaches the measured value (299 K; eta0 fixed).")
P("    Law (19): 1 + c/D_h with c = 4 kappa/[(mu+kappa) lambda] (large-size limit, valid for lambda D_h >> 1); FD: finite differences N = 80 (root finding), 160 and 320 (check).")
e0, er, z = COEF[299]


def law19(Dh, f, g):
    mu, k, gam = eringen(e0, f * er, g * z)
    return 1.0 + 4 * k / ((mu + k) * lam(mu, k, gam)) / Dh


def fd_square(Dh, f, g, N):
    mu, k, gam = eringen(e0, f * er, g * z)
    L = lam(mu, k, gam)
    st = stretch_for(float(Dh * L))
    v = solve_duct(float(Dh), float(Dh), mu, k, gam, 0.0, N, N, st, st)[0] / e0
    gc.collect()
    return v


for Dh, y in ((73.0, 4.74), (119.0, 5.55)):
    fb = (y - 1) / (er / e0)
    f = 1.1 * fb
    bound_f = 1 + f * er / e0
    # law (19) root in g (analytic: c ~ g^(1/2) only for lambda ~ g^(-1/2), i.e. exact for the formula)
    lg = brentq(lambda t: law19(Dh, f, np.exp(t)) - y, -10, 30, xtol=1e-12)
    g19 = float(np.exp(lg))
    mu, k, gam = eringen(e0, f * er, g19 * z)
    L19 = lam(mu, k, gam)
    P(f"    D_h = {Dh:.0f} nm, measured {y}: f_bound {fb:.2f}, f = 1.1 f_bound = {f:.2f} (eta_r = {f*er:.2f} mPa s, mu = {e0-f*er:+.2f}, bound {bound_f:.3f}); law (19): g = {g19:.0f}, 1/lambda = {1/L19:.1f} nm, lambda D_h = {L19*Dh:.2f}")
    if L19 * Dh >= 4.0:
        P("      lambda D_h >= 4: FD used.")
    else:
        P("      lambda D_h < 4: law (19) is not valid here (large-size limit); FD evaluated for information and marked 'FD not converged' if N = 80/160/320 differ by > 1 %.")
    # FD root find on N = 80
    try:
        fn = lambda t: fd_square(Dh, f, np.exp(t), 80) - y
        lo, hi = np.log(max(g19 / 50, 1.0)), np.log(g19 * 50 + 10)
        flo, fhi = fn(lo), fn(hi)
        if flo * fhi > 0:
            P(f"      FD N=80: no sign change in g = [{np.exp(lo):.1f}, {np.exp(hi):.1f}] (values {flo+y:.3f}, {fhi+y:.3f})")
            continue
        lgf = brentq(fn, lo, hi, xtol=1e-4)
        gf = float(np.exp(lgf))
        v = [fd_square(Dh, f, gf, N) for N in (80, 160, 320)]
        rich = v[2] + (v[2] - v[1]) / 3
        mu, k, gam = eringen(e0, f * er, gf * z)
        P(f"      FD g(N=80) = {gf:.0f} (1/lambda = {1/lam(mu,k,gam):.1f} nm, lambda D_h = {lam(mu,k,gam)*Dh:.2f}); prediction at this g: N=80/160/320 = {v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}; Richardson {rich:.4f}; "
          + ("converged within 1 %" if abs(v[2] / v[1] - 1) < 0.01 else "FD not converged (N=160 vs 320 differ by > 1 %)"))
    except Exception as exc:  # report, do not hide
        P(f"      FD failed: {exc!r}")

with open(os.path.join(HERE, "tube_joint_eta_r_zeta_output.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")
