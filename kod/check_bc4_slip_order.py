"""Order of the equivalent slip length in the wide slit for BC1, BC3 and BC4.

Slit -h < y < h, v = w(y) e_z, nu = psi(y) e_x; Eqs. (3)-(4) give (mu+k) w'' - k psi' + G = 0,
gamma psi'' + k w' - 2k psi = 0; couple traction on y = +h (n = e_y): m_(n),x = gamma psi'(h).
First integral (odd psi): (mu+k) w' - k psi = -G y; psi = -G y/(2mu+k) + C sinh(lam y).
  BC1  psi(h) = 0;  BC3  psi(h) = -l_s psi'(h)  (nu = -(l_s/gamma) m_(n));  BC4  psi'(h) = 0.
Q = -int y w' dy (w(+-h) = 0); mu_app/(mu+k/2) = (2h^3/3) G / (Q (mu+k/2)).
Large-gap expansion: tanh(lam h) -> 1 (exponentially small error), eps = 1/(lam h), L = lam l_s fixed.
Prints: S1 the residuals of the ODEs and wall conditions; S2 BC1 equals Eq. (17) exactly (up to exp. small terms)
and its slip length equals Eq. (18); S3 BC4: first-order term 0, second-order coefficient; S4 BC3: slip length
b(L) and its limits L -> 0 (BC1), L -> oo (BC4); S5 BC3 energy identity int Phi = G Q + 2P, P = -(gamma/l_s) psi(h)^2.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import sympy as sp
import mpmath as mp

t0 = time.time()
out = []


def say(s=""):
    print(s)
    out.append(s)


y, h, lam, G, ls = sp.symbols('y h lambda G l_s', positive=True)
mu, ka, ga = sp.symbols('mu kappa gamma', positive=True)
C = sp.symbols('C')
psi = -G * y / (2 * mu + ka) + C * sp.sinh(lam * y)
lam_def = sp.sqrt(ka * (2 * mu + ka) / (ga * (mu + ka)))
wp = (-G * y + ka * psi) / (mu + ka)          # w'
conds = {'BC1': psi.subs(y, h),
         'BC3': psi.subs(y, h) + ls * sp.diff(psi, y).subs(y, h),
         'BC4': sp.diff(psi, y).subs(y, h)}
sols = {k: sp.solve(cnd, C)[0] for k, cnd in conds.items()}

say("S1 residuals (lambda^2 = k(2mu+k)/[gamma(mu+k)])")
for k, Cv in sols.items():
    ps = psi.subs(C, Cv)
    wpk = wp.subs(C, Cv)
    r_ang = sp.simplify((ga * sp.diff(ps, y, 2) + ka * wpk - 2 * ka * ps).subs(lam, lam_def))
    r_lin = sp.simplify((mu + ka) * sp.diff(wpk, y) - ka * sp.diff(ps, y) + G)
    r_bc = sp.simplify(conds[k].subs(C, Cv))
    say(f"   {k}: angular momentum {r_ang}, z-momentum {r_lin}, wall condition {r_bc}")

eps_, L, T, Ch = sp.symbols('epsilon L T Ch', positive=True)
muinf = mu + ka / 2
ratio = {}
for k, Cv in sols.items():
    Q = sp.simplify(-sp.integrate(sp.expand((y * wp).subs(C, Cv)), (y, -h, h)))
    rk = sp.simplify((2 * h**3 / 3) * G / (Q * muinf))
    # large gap: sinh(lam h) = T Ch, cosh(lam h) = Ch, T = tanh(lam h) -> 1; l_s = L/lam, h = 1/(lam eps)
    rk = sp.simplify(rk.subs({sp.sinh(h * lam): T * Ch, sp.cosh(h * lam): Ch, sp.tanh(h * lam): T}))
    rk = sp.simplify(rk.subs(ls, L / lam).subs(h, 1 / (lam * eps_)).subs(sp.tanh(1 / eps_), T))
    ratio[k] = rk
say("")
say("S2-S4 large-gap expansion of mu_app/(mu+k/2), tanh(lam h) = T -> 1, eps = 1/(lam h), L = lam l_s")
for k, rk in ratio.items():
    say(f"   {k}: exact ratio in T = tanh(lam h): {rk};  contains sinh/cosh/Ch: {rk.has(sp.sinh, sp.cosh, Ch)}")
inv = {k: sp.simplify(1 / ratio[k].subs(T, 1)) for k in ratio}
ser = {k: sp.series(inv[k], eps_, 0, 3).removeO() for k in inv}
eq17 = 1 - 3 * ka / (2 * (mu + ka)) * eps_ + 3 * ka / (2 * (mu + ka)) * eps_**2
say(f"S2 BC1: [mu_app/(mu+k/2)]^-1 (T = 1) - Eq. (17) bracket = {sp.simplify(inv['BC1'] - eq17)}")
b = {k: sp.simplify(sp.diff(ser[k], eps_).subs(eps_, 0) / 3 / lam) for k in ser}   # 1 + 3b/h = 1 + 3 b lam eps
say(f"   slip length from the first-order term, 1/(1+3b/h): b_BC1 = {b['BC1']}")
say(f"   residual b_BC1 - Eq. (18) [-k/(2(mu+k) lam)] = {sp.simplify(b['BC1'] + ka / (2 * (mu + ka) * lam))}")
c2 = {k: sp.simplify(ser[k].coeff(eps_, 2)) for k in ser}
say(f"S3 BC4: series of [mu_app/(mu+k/2)]^-1 = {sp.simplify(ser['BC4'])}")
say(f"   first-order coefficient (b_BC4 lam) = {sp.simplify(ser['BC4'].coeff(eps_, 1))}  -> no slip at first order")
say(f"   second-order coefficient = {c2['BC4']}  -> correction of order (lambda h)^-2")
say(f"   exact BC4 inverse ratio (T = 1) = {sp.simplify(inv['BC4'])}")
say(f"S4 BC3: b_BC3 = {sp.factor(b['BC3'])}")
say(f"   residual b_BC3 - b_BC1/(1 + lam l_s) = {sp.simplify(b['BC3'] - b['BC1'] / (1 + L))}")
say(f"   limit L = lam l_s -> 0: {sp.simplify(sp.limit(b['BC3'], L, 0) - b['BC1'])} (difference to BC1);"
    f"  L -> oo: {sp.limit(b['BC3'], L, sp.oo)} (BC4)")
say(f"   |b_BC3|/|b_BC1| = {sp.simplify(b['BC3'] / b['BC1'])} < 1 for l_s > 0")
say(f"   BC3 second-order coefficient = {sp.factor(c2['BC3'])}")

# S5 energy identity BC3 (numerical, 30 digits; the closed form is used with tanh kept exact)
mp.mp.dps = 30
say("")
say("S5 BC3 energy identity int Phi dy = G Q + 2P, P = gamma psi'(h) psi(h) = -(gamma/l_s) psi(h)^2 (mpmath, 30 digits)")
for (muv, kav, gav, hv, lsv) in [(1, 1, 1, 1, 0.5), (1, 10, 1, 3, 2), (-0.3, 1, 0.7, 0.8, 0.1)]:
    muv, kav, gav, hv, lsv = [mp.mpf(q) for q in (muv, kav, gav, hv, lsv)]
    Gv = mp.mpf(1)
    lv = mp.sqrt(kav * (2 * muv + kav) / (gav * (muv + kav)))
    Cv = Gv * (hv + lsv) / ((2 * muv + kav) * (mp.sinh(lv * hv) + lsv * lv * mp.cosh(lv * hv)))
    ps = lambda q: -Gv * q / (2 * muv + kav) + Cv * mp.sinh(lv * q)
    dps = lambda q: -Gv / (2 * muv + kav) + Cv * lv * mp.cosh(lv * q)
    wpf = lambda q: (-Gv * q + kav * ps(q)) / (muv + kav)
    Q = -mp.quad(lambda q: q * wpf(q), [-hv, hv])
    Phi = lambda q: (muv + kav / 2) * wpf(q)**2 + 2 * kav * (wpf(q) / 2 - ps(q))**2 + gav * dps(q)**2
    I = mp.quad(Phi, [-hv, hv])
    P = gav * dps(hv) * ps(hv)
    P2 = -(gav / lsv) * ps(hv)**2
    say(f"   (mu,k,g,h,l_s)=({float(muv)},{float(kav)},{float(gav)},{float(hv)},{float(lsv)}): "
        f"int Phi - GQ - 2P = {mp.nstr(I - Gv * Q - 2 * P, 3)}, P - [-(g/l_s)psi^2] = {mp.nstr(P - P2, 3)}, "
        f"P = {mp.nstr(P, 8)} (<= 0)")
say(f"runtime {time.time() - t0:.1f} s")

with open(__file__.replace('.py', '_output.txt'), 'w', encoding='utf-8') as fh:
    fh.write("\n".join(out) + "\n")
