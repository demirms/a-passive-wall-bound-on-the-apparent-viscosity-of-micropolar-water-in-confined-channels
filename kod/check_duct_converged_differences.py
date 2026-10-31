"""Converged (Richardson-extrapolated) square-duct numbers quoted in the paper.

Square duct, BC1 wall (w = 0, nu = 0), mu = gamma = 1 (so lambda a = a * lam0), kappa/mu varied.
Solver: micropolar_duct_2d.solve_duct (second-order central differences, tanh-stretched grid
x = (L/2)[1 + tanh(s xi)/tanh(s)], xi in [-1, 1]; s = stretch_for(lambda a); mu_app = Q_N(eta=1)/Q on the same grid).
Grids N = 80, 160, 320 intervals per side (Table 2 also N = 40). Each solve is done in turn and the memory is
freed afterwards (one N = 320 case has about 3e5 unknowns).

Richardson extrapolation with two orders:
  formal  p = 2 :  R2 = f320 + (f320 - f160)/3
  observed p    :  p = log2[(f160 - f80)/(f320 - f160)],  Rp = f320 + (f320 - f160)/(2^p - 1)
Sections:
  (a) deviation of the FD solution from the large-size law Eq. (19), mu_app/(mu+kappa/2) = 1 + 4 kappa/[(mu+kappa) lambda a]
      (absolute |FD - law| and relative |FD - law|/FD of the normalised ratio), kappa/mu = 0.41, 1, 10; lambda a = 30, 100
  (b) sensitivity to alpha+beta in {0, -0.9 gamma, 10 gamma}: |mu_app(c) - mu_app(0)| (absolute, units of mu) and
      |mu_app(c) - mu_app(0)| / mu_app(0) (relative), converged values and the single-grid N = 80 values
  (c) Table 2: last column |mu_160 - mu_R| with R from (N = 80, 160) and p = 2 and with the observed order
  (d) tanh stretching parameter s for every lambda a used
Output: check_duct_converged_differences_output.txt.  Cache of solves (resume): json given by --cache.
"""
import gc
import json
import os
import sys
import time
import numpy as np
from micropolar_duct_2d import solve_duct, stretch_for, lam0

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = sys.argv[sys.argv.index("--cache") + 1] if "--cache" in sys.argv else os.path.join(HERE, "cikti", "duct_converged_cache.json")
os.makedirs(os.path.dirname(CACHE), exist_ok=True)
cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
OUT = []
T0 = time.time()


def P(s=""):
    print(s)
    sys.stdout.flush()
    OUT.append(s)


def mu_app(mu, k, g, c, la, N):
    key = f"{mu}|{k}|{g}|{c}|{la}|{N}"
    if key not in cache:
        a = la / lam0(mu, k, g)
        s = stretch_for(la)
        cache[key] = float(solve_duct(a, a, mu, k, g, c, N, N, s, s)[0])
        gc.collect()
        json.dump(cache, open(CACHE, "w"))
        print(f"      [solved {key} t={time.time()-T0:.0f} s]", flush=True)
    return cache[key]


def rich(f80, f160, f320):
    r2 = f320 + (f320 - f160) / 3.0
    p = np.log2((f160 - f80) / (f320 - f160))
    rp = f320 + (f320 - f160) / (2.0 ** p - 1.0)
    return r2, rp, p


mu, g = 1.0, 1.0
P("Square duct, BC1 wall, mu = gamma = 1; grids N = 80/160/320; Richardson with formal p = 2 (R2) and observed p (Rp)")

# ------------------------------------------------------------------ (d) stretching parameter
P("\n(d) tanh stretching parameter s = stretch_for(lambda a): s = 0 for lambda a <= 3, else min(3, 0.8 + 0.5 ln(lambda a));")
P("    grid x = (L/2)[1 + tanh(s xi)/tanh(s)], xi uniform in [-1, 1] (micropolar_duct_2d.grid1d)")
for la in (1, 10, 30, 100):
    P(f"    lambda a = {la:4d}: s = {stretch_for(float(la)):.4f}")

# ------------------------------------------------------------------ (c) Table 2
P("\n(c) Table 2: columns N = 40, 80, 160 and the last column; R from (80, 160) with p = 2")
P("    lam*a k/mu |  N=40        N=80        N=160       N=320      | p(40,80,160) p(80,160,320) | R(80,160;p=2)  |mu160-R| | R2(80..320)  Rp(80..320)  |mu160-R2| |mu160-Rp|")
t2 = {}
for la, k in [(1, 1.0), (10, 1.0), (100, 1.0), (10, 10.0)]:
    f = {N: mu_app(mu, k, g, 0.0, float(la), N) for N in (40, 80, 160, 320)}
    p1 = np.log2((f[80] - f[40]) / (f[160] - f[80]))
    R_pub = f[160] + (f[160] - f[80]) / 3.0
    r2, rp, p2 = rich(f[80], f[160], f[320])
    t2[(la, k)] = (f, p1, p2, R_pub, r2, rp)
    P(f"    {la:5d} {k:5.1f} | {f[40]:.8f} {f[80]:.8f} {f[160]:.8f} {f[320]:.8f} | {p1:6.2f} {p2:6.2f} | {R_pub:.8f} | {abs(f[160]-R_pub):.2e} | "
      f"{r2:.8f} {rp:.8f} {abs(f[160]-r2):.2e} {abs(f[160]-rp):.2e}")
P("    Table 2 last column (printed): 1.1e-06, 6.4e-05, 7.6e-05, 8.5e-04.")
P("    |mu160 - R| for every row, three variants:")
for (la, k), (f, p1, p2, R_pub, r2, rp) in t2.items():
    P(f"    lambda a = {la}, k/mu = {k}: (80,160) p = 2 -> {abs(f[160]-R_pub):.2e}; (80,160,320) p = 2 -> {abs(f[160]-r2):.2e}; "
      f"(80,160,320) observed p = {p2:.3f} -> {abs(f[160]-rp):.2e}; "
      f"(40,80,160) observed p = {p1:.3f} -> {abs(f[160]-(f[160]+(f[160]-f[80])/(2.0**p1-1.0))):.2e}")

# ------------------------------------------------------------------ (a) Eq. (19)
P("\n(a) FD (Richardson) against Eq. (19): mu_app/(mu+kappa/2) vs 1 + 4 kappa/[(mu+kappa) lambda a]")
P("    lam*a k/mu |  FD(R2)      FD(Rp)     law        | abs diff (R2)  abs diff (Rp) | rel diff (R2)  rel diff (Rp) | N=160 single grid abs")
worst = {30: [0, 0], 100: [0, 0]}
for la in (30, 100):
    for k in (0.41, 1.0, 10.0):
        f = {N: mu_app(mu, k, g, 0.0, float(la), N) for N in (80, 160, 320)}
        r2, rp, p = rich(f[80], f[160], f[320])
        law = 1 + 4 * k / ((mu + k) * la)
        n2, np_, n160 = r2 / (mu + k / 2), rp / (mu + k / 2), f[160] / (mu + k / 2)
        ab2, abp = abs(n2 - law), abs(np_ - law)
        P(f"    {la:5d} {k:5.2f} | {n2:.6f} {np_:.6f} {law:.6f} | {ab2:.2e}      {abp:.2e}     | {ab2/n2:.2e}      {abp/np_:.2e}     | {abs(n160-law):.2e}")
        worst[la][0] = max(worst[la][0], ab2)
        worst[la][1] = max(worst[la][1], ab2 / n2)
for la in (30, 100):
    P(f"    largest over kappa/mu at lambda a = {la}: absolute {worst[la][0]:.2e}, relative {worst[la][1]:.2e}   (single grid N = 160: 3.5e-3 and 2.5e-4)")

# ------------------------------------------------------------------ (b) alpha+beta
P("\n(b) alpha+beta in {0, -0.9 gamma, 10 gamma}: change of mu_app relative to alpha+beta = 0")
P("    absolute = |mu_app(c) - mu_app(0)| (units of mu = 1); relative = absolute / mu_app(0); converged = Richardson (R2) of N = 80/160/320")
P("    lam*a k/mu |  c/gamma  mu_app(0) conv | abs conv     rel conv   | rel single N=80")
best = {30: [0, 0, 0], 100: [0, 0, 0]}
spread = {30: [0, 0], 100: [0, 0]}      # (max - min)/min over the three alpha+beta values (paper_numbers.py (6))
for la in (30, 100):
    for k in (0.1, 0.41, 1.0, 10.0):
        base = {N: mu_app(mu, k, g, 0.0, float(la), N) for N in (80, 160, 320)}
        b2 = rich(*[base[N] for N in (80, 160, 320)])[0]
        vals3 = [b2]
        v80 = [base[80]]
        for cf in (-0.9, 10.0):
            fc = {N: mu_app(mu, k, g, cf * g, float(la), N) for N in (80, 160, 320)}
            c2 = rich(*[fc[N] for N in (80, 160, 320)])[0]
            ab, rel = abs(c2 - b2), abs(c2 - b2) / b2
            rel80 = abs(fc[80] - base[80]) / base[80]
            P(f"    {la:5d} {k:5.2f} | {cf:6.1f}   {b2:.6f}        | {ab:.2e}   {rel:.2e}   | {rel80:.2e}")
            best[la][0] = max(best[la][0], ab)
            best[la][1] = max(best[la][1], rel)
            best[la][2] = max(best[la][2], rel80)
            vals3.append(c2)
            v80.append(fc[80])
        spread[la][0] = max(spread[la][0], (max(vals3) - min(vals3)) / min(vals3))
        spread[la][1] = max(spread[la][1], (max(v80) - min(v80)) / min(v80))
for la in (30, 100):
    P(f"    largest at lambda a = {la} (kappa/mu = 0.1, 0.41, 1, 10): absolute {best[la][0]:.2e}, relative (converged) {best[la][1]:.2e}, relative single N=80 {best[la][2]:.2e}")
for la in (30, 100):
    P(f"    spread metric (max - min)/min over the three alpha+beta values, kappa/mu = 0.1, 0.41, 1, 10: lambda a = {la}: converged {spread[la][0]:.2e}; single grid N = 80 {spread[la][1]:.2e}")
P("    (the quoted 8.1e-04 and 3.8e-05 are this spread metric on N = 80 for kappa/mu = 0.1, 1, 10 (without 0.41); it is not relative to mu_app(alpha+beta = 0).)")
P(f"\n[total time {time.time()-T0:.0f} s]")
with open(os.path.join(HERE, "check_duct_converged_differences_output.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")
