"""2D steady pressure-driven micropolar flow in a rectangular duct, M1 walls (w = 0, nu = 0).

Unknowns: axial velocity w(x,y), in-plane microrotation (a, b) = (nu_x, nu_y); nu_z decouples (= 0).
Field equations (Eringen, incompressible, unidirectional; c = alpha + beta):
  (mu+k) Lap w + k (b_x - a_y) + G = 0
  gamma Lap a + c (a_xx + b_xy) + k w_y - 2k a = 0
  gamma Lap b + c (a_xy + b_yy) - k w_x - 2k b = 0
Discretisation: second-order central differences on a tanh-stretched tensor grid (clustering at
the walls), Dirichlet data eliminated, sparse direct solve. Flow rate by the trapezoid rule.
mu_app = Q_N(eta=1)/Q with the Newtonian flow rate computed on the SAME grid (errors cancel);
the exact-series Newtonian value is also reported.

Sections (output in micropolar_duct_2d_output.txt):
 (a) wide-aspect-ratio limit vs the 1D slit closed form
 (b) grid convergence (square)
 (c) bound sweep: mu+k/2 <= refined lower <= mu_app <= mu+k, kappa/mu up to 10, alpha+beta != 0, mu < 0
 (e) large-size asymptotic cross-check (effective slip of the wall layer)
 (d) MD coefficients for water in square / measured rectangular channels vs data
"""
import sys
import time
import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from scipy.integrate import quad

OUT = []
def log(s=""):
    print(s); sys.stdout.flush(); OUT.append(s)


def grid1d(L, N, s):
    xi = np.linspace(-1.0, 1.0, N + 1)
    if s <= 0:
        x = L / 2 * (1 + xi); xp = np.full_like(xi, L / 2); xpp = np.zeros_like(xi)
    else:
        t = np.tanh(s)
        x = L / 2 * (1 + np.tanh(s * xi) / t)
        sech2 = 1 - np.tanh(s * xi)**2
        xp = L / 2 * s * sech2 / t
        xpp = L / 2 * (-2 * s**2 * np.tanh(s * xi) * sech2) / t
    return x, xp, xpp, 2.0 / N


def ops1d(L, N, s):
    x, xp, xpp, dxi = grid1d(L, N, s)
    n = N - 1
    e = np.ones(n)
    Dxi = sps.diags([-e[1:], e[1:]], [-1, 1]) / (2 * dxi)
    Dxixi = sps.diags([e[1:], -2 * e, e[1:]], [-1, 0, 1]) / dxi**2
    xpi, xppi = xp[1:-1], xpp[1:-1]
    D1 = sps.diags(1 / xpi) @ Dxi
    D2 = sps.diags(1 / xpi**2) @ Dxixi - sps.diags(xppi / xpi**3) @ Dxi
    wq = xpi * dxi                      # trapezoid weights (boundary values are zero)
    return D1.tocsr(), D2.tocsr(), wq, x


def solve_duct(Lx, Ly, mu, k, g, c=0.0, Nx=80, Ny=80, sx=0.0, sy=0.0, G=1.0, fields=False):
    D1x, D2x, wx, xg = ops1d(Lx, Nx, sx)
    D1y, D2y, wy, yg = ops1d(Ly, Ny, sy)
    Ix, Iy = sps.identity(Nx - 1), sps.identity(Ny - 1)
    Dx, Dxx = sps.kron(D1x, Iy), sps.kron(D2x, Iy)
    Dy, Dyy = sps.kron(Ix, D1y), sps.kron(Ix, D2y)
    Dxy = Dx @ Dy
    Lap = Dxx + Dyy
    M = (Nx - 1) * (Ny - 1)
    I = sps.identity(M)
    A = sps.bmat([[(mu + k) * Lap, -k * Dy, k * Dx],
                  [k * Dy, g * Lap + c * Dxx - 2 * k * I, c * Dxy],
                  [-k * Dx, c * Dxy, g * Lap + c * Dyy - 2 * k * I]], format='csc')
    rhs = np.concatenate([-G * np.ones(M), np.zeros(2 * M)])
    sol = spsolve(A, rhs)
    W = np.kron(wx, wy)
    Q = W @ sol[:M]
    wN = spsolve(Lap.tocsc(), -G * np.ones(M))
    QN = W @ wN
    if fields:
        return QN / Q, Q, QN, sol, (xg, yg)
    return QN / Q, Q, QN


def qn_rect_exact(Lx, Ly, G=1.0, eta=1.0, nmax=401):
    a, b = min(Lx, Ly), max(Lx, Ly)
    s = sum(np.tanh(n * np.pi * b / (2 * a)) / n**5 for n in range(1, nmax, 2))
    return G * a**3 * b / (12 * eta) * (1 - 192 * a / (np.pi**5 * b) * s)


def lam0(mu, k, g):
    return np.sqrt(k * (2 * mu + k) / (g * (mu + k)))


def slit_muapp(h, mu, k, g):
    lm = lam0(mu, k, g)
    s, ch = np.sinh(lm * h), np.cosh(lm * h)
    C = -h / ((2 * mu + k) * s)
    I1 = h * ch / lm - s / lm**2
    Q = 2 / (mu + k) * (h**3 / 3 + k * h**3 / (3 * (2 * mu + k)) + k * C * I1)
    return 2 * h**3 / (3 * Q)


def stretch_for(lam_a):
    # cluster more when the wall layer (1/lam) is thin compared to the duct
    return 0.0 if lam_a <= 3 else min(3.0, 0.8 + 0.5 * np.log(lam_a))


if __name__ == "__main__":
    t0 = time.time()
    mu, g = 1.0, 1.0

    # ---------------- (b) grid convergence, square ----------------
    log("(b) Grid convergence, square side a, mu=1, gamma=1, alpha+beta=0, M1")
    log("    lam*a  k/mu   N   s    mu_app(same-grid N)   mu_app(exact-series N)   QN_grid/QN_exact-1")
    for k in [1.0, 10.0]:
        for lam_a in [1.0, 10.0, 100.0]:
            a = lam_a / lam0(mu, k, g)
            s = stretch_for(lam_a)
            vals = []
            for N in [40, 80, 160]:
                ma, Q, QN = solve_duct(a, a, mu, k, g, 0.0, N, N, s, s)
                QNe = qn_rect_exact(a, a)
                vals.append(ma)
                log(f"    {lam_a:6.1f} {k:5.1f} {N:4d} {s:4.2f}   {ma:.8f}          {QNe/Q:.8f}             {QN/QNe-1:+.2e}")
            p = np.log(abs(vals[0] - vals[1]) / abs(vals[1] - vals[2])) / np.log(2) if vals[1] != vals[2] else np.nan
            rich = vals[2] + (vals[2] - vals[1]) / 3
            log(f"      observed order {p:.2f}; Richardson (N=160,80) {rich:.8f}; |N160 - Rich| = {abs(vals[2]-rich):.1e}")
    log(f"    [t = {time.time()-t0:.0f} s]")

    # ---------------- (a) wide aspect ratio vs slit ----------------
    log("\n(a) Wide rectangle (thickness 2h = 1, width AR) vs slit closed form, mu=1, gamma=1, M1")
    for k, lh in [(1.0, 1.0), (1.0, 5.0), (10.0, 2.0)]:
        h = 0.5
        g_ = (k * (2 * mu + k) / (mu + k)) * h**2 / lh**2     # choose gamma so that lam*h = lh
        ref = slit_muapp(h, mu, k, g_)
        res = []
        for AR in [10, 20, 40]:
            ma, Q, QN = solve_duct(AR * 1.0, 1.0, mu, k, g_, 0.0, Nx=240, Ny=60, sx=3.0, sy=stretch_for(lh))
            res.append(ma)
            log(f"    k/mu={k:4.1f} lam*h={lh:4.1f} AR={AR:3d}: mu_app={ma:.6f}")
        ext = 2 * res[2] - res[1]
        log(f"      extrapolated (linear in 1/AR from 20,40): {ext:.6f}; slit closed form {ref:.6f}; rel diff {(ext-ref)/ref:+.2e}")
    log(f"    [t = {time.time()-t0:.0f} s]")

    # ---------------- (c) bound sweep ----------------
    log("\n(c) Bound sweep, square, N=80 (stretched for lam*a>3). gm = gamma + min(0, alpha+beta);")
    log("    refined lower = mu+k - k^2/(2k + gm*lam1), lam1 = 2 pi^2/a^2")
    log("    mu    k     c=a+b   lam*a     mu_app       mu+k/2    refined     mu+k    ok   monotone")
    nviol, ncase = 0, 0
    cases = []
    for (mu_, k) in [(1.0, 0.1), (1.0, 1.0), (1.0, 10.0), (-0.3, 1.0)]:
        for c in [0.0, -0.9 * g, 10.0 * g]:
            prev = None
            mono = True
            for lam_a in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
                a = lam_a / lam0(mu_, k, g)
                s = stretch_for(lam_a)
                ma = solve_duct(a, a, mu_, k, g, c, 80, 80, s, s)[0]
                gm = g + min(0.0, c)
                low = mu_ + k - k**2 / (2 * k + gm * 2 * np.pi**2 / a**2)
                tol = 1e-6 * (mu_ + k)
                ok = (mu_ + k / 2 - tol <= low) and (low - tol <= ma) and (ma <= mu_ + k + tol)
                if prev is not None and ma > prev + tol:
                    mono = False
                prev = ma
                ncase += 1; nviol += (not ok)
                log(f"    {mu_:4.1f} {k:5.1f} {c:7.2f} {lam_a:7.1f}   {ma:.6f}   {mu_+k/2:.4f}   {low:.6f}   {mu_+k:.4f}  {ok}")
            log(f"      -> mu_app non-increasing with size: {mono}")
    log(f"    bound violations: {nviol} / {ncase}")
    log(f"    [t = {time.time()-t0:.0f} s]")

    # ---------------- (e) large-size asymptotics ----------------
    log("\n(e) Large-size asymptotics: wall layer acts as slip b = -k/(2(mu+k) lam) (fitted in slit, check_slit_wall_conditions)")
    log("    first order: mu_app/(mu+k/2) ~ 1 - b * R/a, R = a*oint|d_n w0|^2 / int|grad w0|^2 (Newtonian square)")
    # R for the square by series
    nmax = 301
    def dwdx0(yv):        # d w/dx at x=0, square [0,1]^2, G/eta = 1
        s_ = 0.5
        for n_ in range(1, nmax, 2):
            s_ -= 4 / (n_**2 * np.pi**2) * np.cosh(n_ * np.pi * (yv - 0.5)) / np.cosh(n_ * np.pi / 2)
        return s_
    wall = 4 * quad(lambda yy: dwdx0(yy)**2, 0, 1, limit=200)[0]
    Q0 = qn_rect_exact(1.0, 1.0)
    R = wall / Q0     # int|grad w0|^2 = G Q0/eta
    log(f"    R(square) = {R:.6f}  (slit analogue: 3 per half-gap h)")
    for k in [0.41, 1.0, 10.0]:
        for lam_a in [30.0, 100.0]:
            a = lam_a / lam0(mu, k, g)
            ma = solve_duct(a, a, mu, k, g, 0.0, 160, 160, stretch_for(lam_a), stretch_for(lam_a))[0]
            asym = 1 + k / (2 * (mu + k)) * R / lam_a
            log(f"    k/mu={k:5.2f} lam*a={lam_a:5.0f}: 2D {ma/(mu+k/2):.6f}  asymptotic {asym:.6f}  diff {ma/(mu+k/2)-asym:+.2e}")
    log(f"    [t = {time.time()-t0:.0f} s]")

    # ---------------- (d) MD coefficients, water ----------------
    log("\n(d) Water, MD coefficients (Hansen et al. 2010, 298 K): eta0=1.0e-3, eta_r=1.7e-4 Pa s, zeta=2.5e-21 kg m/s")
    log("    mapping mu = eta0 - eta_r, kappa = 2 eta_r, gamma = zeta, alpha = beta = 0 (M1: only alpha+beta enters)")
    log("    units: mPa s and nm  -> gamma = 2.5 mPa s nm^2")
    eta0, etar, zeta = 1.0, 0.17, 2.5
    muw, kw, gw = eta0 - etar, 2 * etar, zeta
    log(f"    1/lam0 = {1/lam0(muw, kw, gw):.3f} nm ; upper bound (mu+k)/eta0 = {(muw+kw)/eta0:.3f}")
    data = [("Kazoe2023", 73, 4.7, (73, 72)), ("Kazoe2023", 119, 5.5, (130, 110)),
            ("Kazoe2022", 190, 3.6, (163, 230)), ("Li2012", 200, 2.5, None),
            ("Kazoe2022", 394, 2.8, (410, 380)), ("Li2012", 491, 1.9, None)]
    log("    source     Dh[nm]  measured | square: MD mu_app/eta0  (N=120,160) | rectangle W x D: MD | measured/MD | refined lower (sq)")
    for src, Dh, meas, rect in data:
        a = float(Dh)
        lam_a = a * lam0(muw, kw, gw)
        s = stretch_for(lam_a)
        m1 = solve_duct(a, a, muw, kw, gw, 0.0, 120, 120, s, s)[0]
        m2 = solve_duct(a, a, muw, kw, gw, 0.0, 160, 160, s, s)[0]
        low = muw + kw - kw**2 / (2 * kw + gw * 2 * np.pi**2 / a**2)
        rtxt = "      -      "
        if rect is not None:
            W_, D_ = rect
            mr = solve_duct(float(W_), float(D_), muw, kw, gw, 0.0, 160, 160, s, s)[0]
            rtxt = f"{W_}x{D_}: {mr/eta0:.4f}"
        log(f"    {src:9s} {Dh:6d}   {meas:5.2f}   | {m1/eta0:.5f}, {m2/eta0:.5f}                  | {rtxt} | {meas/(m2/eta0):.2f} | {low/eta0:.5f}")
    log(f"    [t = {time.time()-t0:.0f} s]")

    with open(__file__.replace('.py', '_output.txt'), 'w', encoding='utf-8') as f:
        f.write("\n".join(OUT) + "\n")
