"""Micropolar (extended NS) flow reduction for water with MD coefficients (Hansen et al. 2010)
vs. the Deryagin-Churaev quartz-capillary data as fitted by Migun (1981).
Mapping: kappa = 2 eta_r, mu = eta0 - eta_r, gamma = zeta0 + zeta_r; wall: u = 0, omega = 0 (M1).
Solved by finite differences (slit and tube) and checked against Hansen's slit numbers."""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def solve(h, eta0, eta_r, zeta, geom, n=1500):
    mu, ka, ga = eta0 - eta_r, 2 * eta_r, zeta
    x = np.linspace(0, h, n + 1); dx = x[1] - x[0]
    # unknowns: u_0..u_n, w_0..w_n ; G = 1 (pressure gradient)
    A = lil_matrix((2 * (n + 1), 2 * (n + 1))); b = np.zeros(2 * (n + 1))
    U = lambda i: i; W = lambda i: n + 1 + i
    for i in range(n + 1):
        if i == n:
            A[U(i), U(i)] = 1; A[W(i), W(i)] = 1; continue
        if i == 0:  # symmetry: u' = 0, w = 0
            A[U(0), U(0)] = -1; A[U(0), U(1)] = 1
            A[W(0), W(0)] = 1; continue
        r = x[i]
        c = 1 / r if geom == "tube" else 0.0
        # momentum: (mu+ka)(u'' + c u') + ka (w' + c w) + G = 0
        A[U(i), U(i - 1)] = (mu + ka) * (1 / dx**2 - c / (2 * dx))
        A[U(i), U(i)] = -2 * (mu + ka) / dx**2
        A[U(i), U(i + 1)] = (mu + ka) * (1 / dx**2 + c / (2 * dx))
        A[U(i), W(i - 1)] = -ka / (2 * dx); A[U(i), W(i + 1)] = ka / (2 * dx); A[U(i), W(i)] = ka * c
        b[U(i)] = -1.0
        # spin: ga (w'' + c w' - c^2 w) - ka u' - 2 ka w = 0   (w = microrotation, sign as Eringen)
        A[W(i), W(i - 1)] = ga * (1 / dx**2 - c / (2 * dx))
        A[W(i), W(i)] = -2 * ga / dx**2 - ga * c**2 - 2 * ka
        A[W(i), W(i + 1)] = ga * (1 / dx**2 + c / (2 * dx))
        A[W(i), U(i - 1)] = ka / (2 * dx); A[W(i), U(i + 1)] = -ka / (2 * dx)
    s = spsolve(A.tocsr(), b); u = s[:n + 1]
    wgt = x if geom == "tube" else np.ones_like(x)
    Q = np.trapezoid(u * wgt, x)
    Qn = h**4 / (16 * eta0) if geom == "tube" else h**3 / (3 * eta0)
    return Q / Qn

# Hansen Fig. 3 coefficients (T = 298 K)
eta0, eta_r, zeta = 1.0e-3, 1.7e-4, 1.1e-21 + 1.4e-21
K = np.sqrt(4 * eta_r * eta0 / (zeta * (eta0 + eta_r)))
print(f"MD: kappa/mu = {2*eta_r/(eta0-eta_r):.3f},  1/K = {1/K*1e9:.2f} nm,  l = sqrt(zeta/eta0) = {np.sqrt(zeta/eta0)*1e9:.2f} nm")
print("Slit check vs Hansen (5% @200 nm, 15% @100 nm, 50% @20 nm; height = 2h):")
for H in [200e-9, 100e-9, 20e-9]:
    q = solve(H / 2, eta0, eta_r, zeta, "slit")
    print(f"  height {H*1e9:5.0f} nm: 1-Q/Qn = {100*(1-q):5.1f} %")
print("Tube (quartz capillary radii), MD coefficients vs Migun fit / data:")
for r in [5e-7, 3e-7, 1e-7, 5e-8, 4e-8]:
    q = solve(r, eta0, eta_r, zeta, "tube")
    print(f"  r0 {r*1e9:5.0f} nm: eta_e/eta_n MD = {1/q:.3f}")
