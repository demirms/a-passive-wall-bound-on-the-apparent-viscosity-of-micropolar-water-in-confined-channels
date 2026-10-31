"""Wall power of the BC2 wall with n = 0.7 as a function of beta/gamma.

Model: tube of radius R, BC2 wall nu_R = n * (couple stress)/..., written through the wall-spin ratio
    s_hat(n) = n / (mu + (1 - n) kappa)                                  (same as check_khanukaeva2023_walls.py)
Wall power per unit length:  2 pi R P,  P = nu_R [ gamma (e (1 - X)/2 + nu_R X) - beta nu_R ]  (R = G = 1),
    e = 1/(2 mu + kappa), X = a I0(a)/I1(a) - 1, a = lambda R, nu_R = s_hat/2,
    Q = pi / (8 mu_app) (Newtonian tube pi R^4/8), mu_app from paper_common.tube_muapp.
Ratio = 2 pi R P / (G Q): wall power divided by the pressure work (> 0 means the wall delivers power, i.e. is active).
The formulas are copied from check_khanukaeva2023_walls.py (section E; that script executes at import and is not
imported) and checked here against its printed output (1.727 / 1.101 / 0.474).
Fitted constants of Migun [18] / Kolpashchikov [19]: kappa/mu = 5.25 (eta0 = mu + kappa/2 = 1.0 after scaling:
mu = 1, kappa = 5.25), 1/lambda = 14.22 nm, R = 40 nm.  beta/gamma in [-1, 1] (admissible range gamma >= |beta|).
Output: khanukaeva_n07_beta_power_output.txt
"""
import os
import numpy as np
from scipy.optimize import brentq
from paper_common import TABLE2, tube_muapp, X_, s_star

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []


def P(s=""):
    print(s, flush=True)
    OUT.append(str(s))


def shat_of_n(m, k, n):
    return n / (m + (1 - n) * k)


def wall_power_ratio(m, k, a, shv, bg):
    g = k * (2 * m + k) / ((m + k) * a ** 2)
    e = 1 / (2 * m + k)
    X = X_(a)
    nr = shv / 2
    Pw = nr * (g * (e * (1 - X) / 2 + nr * X) - bg * g * nr)
    Q = np.pi / (8 * tube_muapp(m, k, a, shv))
    return 2 * np.pi * Pw / Q


def nstar(m, k, a, bg):
    ss = s_star(m, k, a, bg)
    return ss * (m + k) / (1 + ss * k)


m, k = 1.0, 5.25
ell = 14.22
P("n = 0.7 wall of the fitted constants (kappa/mu = 5.25, 1/lambda = 14.22 nm): wall power / pressure work")
for R in (40.0,):
    a = R / ell
    P(f"  R = {R:.0f} nm, lambda R = {a:.4f}")
    P(f"  {'n':>4s} {'beta/gamma = -1':>16s} {'0':>8s} {'+1':>8s}   mu_app/(mu+kappa/2)")
    for n in (0.5, 0.6, 0.7, 0.8):
        sh = shat_of_n(m, k, n)
        r = [wall_power_ratio(m, k, a, sh, bg) for bg in (-1.0, 0.0, 1.0)]
        P(f"  {n:4.1f} {r[0]:16.3f} {r[1]:8.3f} {r[2]:8.3f}   {tube_muapp(m, k, a, sh) / (m + k / 2):.3f}")
    sh = shat_of_n(m, k, 0.7)
    r = [wall_power_ratio(m, k, a, sh, bg) for bg in (-1.0, 0.0, 1.0)]
    P(f"  n = 0.7: beta = -gamma, 0, +gamma -> {r[0]:.3f} / {r[1]:.3f} / {r[2]:.3f}   (independent recomputation: 1.73 / 1.10 / 0.47)")
    assert abs(r[0] - 1.727) < 5e-3 and abs(r[1] - 1.101) < 5e-3 and abs(r[2] - 0.474) < 5e-3
    # the ratio is linear in beta/gamma: threshold ratio = 1
    fn = lambda bg: wall_power_ratio(m, k, a, sh, bg) - 1.0
    thr = brentq(fn, -1.0, 1.0, xtol=1e-12)
    thr_lin = (1 - r[1]) / (r[2] - r[1])          # linear interpolation through bg = 0 and +1
    P(f"  threshold beta/gamma at which wall power = pressure work: {thr:.4f}  (linear check {thr_lin:.4f}; independent recomputation: about 0.16)")
    P(f"  -> wall power exceeds the pressure work for beta/gamma < {thr:.3f} (including beta = 0), and stays below it for beta/gamma > {thr:.3f}.")
    # table over beta/gamma
    P("  beta/gamma  ratio:  " + "  ".join(f"{bg:+.2f}: {wall_power_ratio(m, k, a, sh, bg):.3f}" for bg in (-1, -0.5, -0.25, 0, 0.1, 0.16, 0.2, 0.5, 1)))
    # fine sweep step 0.01 for the whole range
    bgs = np.round(np.arange(-1.0, 1.0 + 1e-9, 0.01), 2)
    rs = np.array([wall_power_ratio(m, k, a, sh, bg) for bg in bgs])
    P(f"  sweep beta/gamma in [-1, 1], step 0.01 ({len(bgs)} values): ratio from {rs.max():.3f} (beta/gamma = -1) to {rs.min():.3f} (+1); "
      f"largest beta/gamma with ratio >= 1: {bgs[rs >= 1].max():+.2f}")

P("\nn > 1/2 is active for every acceptable beta (P > 0), fitted constants, R = 40 nm - 1 um, beta/gamma in [-1, 1], n in (1/2, 0.8]:")
minP = np.inf
cnt = 0
for R in np.logspace(np.log10(40), 3, 60):
    a = R / ell
    for bg in np.linspace(-1, 1, 41):
        for n in np.linspace(0.5001, 0.8, 31):
            sh = shat_of_n(m, k, n)
            r = wall_power_ratio(m, k, a, sh, bg)
            minP = min(minP, r)
            cnt += 1
            assert r > 0, (R, bg, n, r)
P(f"  {cnt} cases, minimum ratio {minP:.2e} > 0: P > 0 everywhere (assert passed)")
P("  upper end n* of the passive range at R = 40 nm - 1 um (fitted constants):")
for bg in (-1.0, 0.0, 1.0):
    v = [nstar(m, k, R / ell, bg) for R in np.logspace(np.log10(40), 3, 60)]
    P(f"    beta/gamma = {bg:+.0f}: n* = {min(v):.4f} .. {max(v):.4f}  (never above 1/2: {max(v) <= 0.5 + 1e-12})")
    assert max(v) <= 0.5 + 1e-12

P("\nMD coefficients (Table 1; 299 K: eta0 0.70, eta_r 0.17, zeta 2.1, beta = zeta0 - zeta_r = -0.3; 284 K: 0.92, 0.17, 3.0, beta = -0.2), R = 40 nm, n = 0.7:")
for T, e0, er, z0, zr in TABLE2[:4]:
    mu, kk, gam = e0 - er, 2 * er, z0 + zr
    lm = np.sqrt(kk * (2 * mu + kk) / (gam * (mu + kk)))
    a = lm * 40
    beta = z0 - zr
    sh = shat_of_n(mu / e0, kk / e0, 0.7)
    P(f"  {T} K: beta/gamma = {beta/gam:+.3f}; ratio at beta = 0: {wall_power_ratio(mu/e0, kk/e0, a, sh, 0.0):.3f}; at beta = {beta:+.2f}: {wall_power_ratio(mu/e0, kk/e0, a, sh, beta/gam):.3f}; at -gamma, +gamma: "
      f"{wall_power_ratio(mu/e0, kk/e0, a, sh, -1.0):.3f}, {wall_power_ratio(mu/e0, kk/e0, a, sh, 1.0):.3f}  (independent recomputation at beta = 0: 0.029-0.034)")

with open(os.path.join(HERE, "khanukaeva_n07_beta_power_output.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")
