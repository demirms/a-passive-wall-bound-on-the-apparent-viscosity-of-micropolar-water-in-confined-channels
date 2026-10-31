"""Finite-difference check of the square-channel ceiling.
Square D_h = 73 nm, 299 K coefficients (eta0 0.70, eta_r 0.17, zeta 2.1), zeta factor g = 1, eta_r factor f = 10, 30, 100
(and 1000 for information), BC1 wall, N = 80/160/320, Richardson extrapolation (order estimated from the three grids).
Reference limit: law (19) 1 + 4 kappa/[(mu+kappa) lambda D_h]; for f -> infinity it is 1 + 8/(lambda_inf D_h).
Grid stretch: micropolar_duct_2d.stretch_for(lambda D_h) (own solver, unchanged).  Solutions run one after the other and
memory is released after each.  Output: square_ceiling_fd_check_output.txt
"""
import gc
import os
import time
import numpy as np
from paper_common import eringen, lam
from micropolar_duct_2d import solve_duct, stretch_for

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []


def P(s=""):
    print(s, flush=True)
    OUT.append(str(s))


e0, er, z = 0.70, 0.17, 2.1
Dh = 73.0
P("square_ceiling_fd_check.py: square D_h = 73 nm, 299 K, zeta factor 1, BC1; eta_r factor f; mu = eta0 - f eta_r (negative for f > 4.1)")
P("Independent recomputation (own solver, theta 0.12): f=100: 1.18608, 1.11813, 1.10313 ; f=1000: 2.15878, 1.34290, 1.15342 (N = 80, 160, 320)")
P(f"{'f':>6s} {'lam [1/nm]':>10s} {'lam*Dh':>7s} {'stretch':>7s} | {'N=80':>8s} {'N=160':>8s} {'N=320':>8s} | {'order p':>7s} {'Richardson':>10s} {'N320/Rich':>9s} | {'law (19)':>8s} {'diff Rich-(19)':>14s}  note")
rows = []
for f in (10, 30, 100, 1000):
    mu, k, gam = eringen(e0, f * er, z)
    L = lam(mu, k, gam)
    st = stretch_for(float(L * Dh))
    law = 1.0 + 4 * k / ((mu + k) * L) / Dh
    v = []
    for N in (80, 160, 320):
        t0 = time.time()
        v.append(solve_duct(Dh, Dh, mu, k, gam, 0.0, N, N, st, st)[0] / e0)
        gc.collect()
        P(f"   (f={f}, N={N}: {v[-1]:.5f}, {time.time()-t0:.0f} s)")
    d1, d2 = v[1] - v[0], v[2] - v[1]
    p = np.log2(d1 / d2) if d1 * d2 > 0 else np.nan
    rich = v[2] + d2 / (2 ** p - 1) if np.isfinite(p) and p > 0.2 else np.nan
    xr = v[2] / rich if np.isfinite(rich) else np.nan
    conv = abs(v[2] / v[1] - 1) < 0.01
    note = "converged (N=160 vs 320 within 1 %)" if conv else f"NOT converged; value on N=320 correct only up to factor {xr:.3f}" if np.isfinite(xr) else "NOT converged, no usable Richardson"
    P(f"{f:6d} {L:10.4f} {L*Dh:7.1f} {st:7.2f} | {v[0]:8.5f} {v[1]:8.5f} {v[2]:8.5f} | {p:7.2f} {rich:10.5f} {xr:9.4f} | {law:8.5f} {rich-law:+14.5f}  {note}")
    rows.append((f, v, p, rich, xr, law, conv))
lim = 1.0 + 8.0 / (2 * np.sqrt(e0 / z) * Dh)
P(f"\nLimit f -> infinity of law (19): 1 + 8/(lambda_inf D_h) = {lim:.5f} (lambda_inf = 2 sqrt(eta0/zeta) = {2*np.sqrt(e0/z):.4f} 1/nm)")
for f, v, p, rich, xr, law, conv in rows:
    P(f"  f = {f}: Richardson / law(19) - 1 = {100*(rich/law-1):+.2f} %" if np.isfinite(rich) else f"  f = {f}: no Richardson value")
with open(os.path.join(HERE, "square_ceiling_fd_check_output.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")
