# BC2 constant n needed to reach the measured tube values 1.31 and 1.48 (Churaev et al. 1970, 19-54 C)
# at the data radii of the narrow tubes (41.25, 53.03, 59.76 nm; data/water_viscosity_data_public.csv R001-R003)
# with the molecular-dynamics coefficients at 284-299 K (Table 1), beta = 0, from the tube closed form Eq. (13),
# and the wall power per pressure work 2 pi R P / (G Q) from Eq. (14). Sanity check: r = 40 nm reproduces
# check_tube_m3_bound_output.txt (284 K: n = 23.9387, ratio 2.041).
import math
from scipy.special import iv

HANSEN = {284: (0.92, 0.17, 3.0), 289: (0.83, 0.17, 2.7), 293: (0.76, 0.17, 2.6), 299: (0.70, 0.17, 2.1)}


def solve(T, R, target, beta=0.0):
    e0, er, zeta = HANSEN[T]
    mu, k, g = e0 - er, 2 * er, zeta
    lam = math.sqrt(k * (2 * mu + k) / (g * (mu + k)))
    q = lam * R
    J = iv(2, q) / (q * iv(1, q))
    X = q * iv(0, q) / iv(1, q) - 1
    e = 1 / (2 * mu + k)
    D = (mu + k) / (target * e0)
    s = e + (D - 1 - k * e) / (4 * k * J)
    n = s * (mu + k) / (1 + s * k)
    G = 1.0
    nuR = s * G * R / 2
    AP = g * G * e * (X - 1) / 2
    BP = (g * X - beta) / R
    P = nuR * (-AP + BP * nuR)
    Q = G * (math.pi * R ** 4 / 8) / (target * e0)
    return n, 2 * math.pi * R * P / (G * Q), 1 + mu / k


print("sanity r = 40 nm, 284 K, target 1.31:", ["%.4f" % v for v in solve(284, 40.0, 1.31)])
ns, ratios = [], []
for T in HANSEN:
    for R in (41.25, 53.03, 59.76):
        for tgt in (1.31, 1.48):
            n, rat, npole = solve(T, R, tgt)
            ns.append(n); ratios.append(rat)
            print(f"T={T} K R={R:6.2f} nm target {tgt}: n = {n:8.4f} (pole {npole:.3f}), wall power / GQ = {rat:.3f}")
print(f"data radii 41.25-59.76 nm, 284-299 K: n = {min(ns):.2f}-{max(ns):.2f}; wall power / GQ = {min(ratios):.2f}-{max(ratios):.2f}")
