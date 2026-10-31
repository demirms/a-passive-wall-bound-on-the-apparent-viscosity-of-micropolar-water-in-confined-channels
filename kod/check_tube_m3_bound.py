"""Tube (Hagen-Poiseuille) flow of a micropolar fluid with a no-slip wall and a spin wall condition:
upper bound mu_app < mu + kappa for every PASSIVE wall, failure only for ACTIVE walls.

Notation (Eringen 1966): t_kl = -p d_kl + mu (v_k,l + v_l,k) + kappa (v_l,k - e_klm nu_m),
m_kl = alpha nu_r,r d_kl + beta nu_k,l + gamma nu_l,k.  v = w(r) e_z, nu = nu(r) e_theta, -dp/dz = G > 0.
Wall condition family M3: nu = n curl v at r = R, i.e. nu(R) = -n w'(R)
(Kline/Migun alpha = 2n; Chaturani-Mahajan 1982 / Erdogan: dv/dr = -(2+s) w  <=>  n = 1/(2+s)).
Wall power into the fluid per unit area (n = e_r faces the wall, velocity no-slip):
P = m_(n).nu = nu(R) m_r_theta(R) = nu(R) [gamma nu'(R) - beta nu(R)/R]; passive iff P <= 0.

Parts
 S  : symbolic derivation and identities (sympy; "zero" means simplify returned 0)
 N1 : independent finite-difference solver vs closed form, energy identity by quadrature
 N2 : scans kappa/mu = 0.1..10, lam R = 0.1..100, n in [-6, 16], beta/gamma in {-1, 0, 1}
 N3 : limits (n -> 0 gives the M1 tube result, lam R -> inf gives mu + kappa/2)
 N4 : water, Hansen et al. 2010 Table II (SPC/Fw), r = 40..60 nm, vs Churaev 1970 (1.31..1.48)
"""
import numpy as np
import sympy as sp
from scipy.special import iv, ive
from scipy.integrate import quad
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

out = []
P_ = out.append

# ----------------------------------------------------------------------------------------- S
r, R, G, a, x = sp.symbols('r R G a x', positive=True)
mu, ka, ga = sp.symbols('mu kappa gamma', real=True)
be, n, C, nuR = sp.symbols('beta n C nu_R', real=True)
lam = sp.symbols('lambda', positive=True)
lam2 = ka * (2 * mu + ka) / (ga * (mu + ka))
e = 1 / (2 * mu + ka)

# general regular solution (C free), first integral (mu+k) w' + k nu = -G r / 2
nu = G * r * e / 2 + C * sp.besseli(1, lam * r)
wp = -(G * r / 2 + ka * nu) / (mu + ka)
w = sp.integrate(wp, (r, R, r))
res_mom = (mu + ka) * (sp.diff(w, r, 2) + sp.diff(w, r) / r) + ka * sp.diff(r * nu, r) / r + G
res_ang = ga * (sp.diff(nu, r, 2) + sp.diff(nu, r) / r - nu / r**2) - ka * sp.diff(w, r) - 2 * ka * nu
res_ang = res_ang.subs(lam, sp.sqrt(lam2))
P_(f"S1 momentum residual (general C): {sp.simplify(res_mom)}")
P_(f"S1 angular-momentum residual (general C, lam^2 = k(2mu+k)/(g(mu+k))): {sp.simplify(res_ang)}")
P_(f"S1 w(R) = {sp.simplify(w.subs(r, R))}")

# flow rate: Q = 2 pi int w r dr = (pi/(mu+k)) [G R^4/8 + k int r^2 nu dr]
Q = 2 * sp.pi * sp.integrate(w * r, (r, 0, R))
Qf = sp.pi / (mu + ka) * (G * R**4 / 8 + ka * sp.integrate(r**2 * nu, (r, 0, R)))
P_(f"S2 Q - (pi/(mu+k))[G R^4/8 + k int r^2 nu dr] = {sp.simplify(Q - Qf)}")
P_(f"S2 int_0^R r^2 I1(lam r) dr - R^2 I2(lam R)/lam = "
   f"{sp.simplify(sp.integrate(r**2 * sp.besseli(1, lam * r), (r, 0, R)) - R**2 * sp.besseli(2, lam * R) / lam)}")

# closed form in terms of the wall spin: nu_R = (G R/2) shat, J(a) = I2(a)/(a I1(a)), a = lam R
shat, J = sp.symbols('shat J', real=True)
C_of = (nuR - G * R * e / 2) / sp.besseli(1, lam * R)
mu_app_def = sp.pi * G * R**4 / (8 * Qf.subs(C, C_of))
mu_app_closed = (mu + ka) / (1 + ka * e + 4 * ka * (shat - e) * J)
diff_ma = mu_app_def.subs(nuR, G * R * shat / 2) - mu_app_closed.subs(
    J, sp.besseli(2, lam * R) / (lam * R * sp.besseli(1, lam * R)))
P_(f"S3 mu_app = (mu+k)/[1 + k e + 4k (shat - e) J],  e = 1/(2mu+k): difference = {sp.simplify(diff_ma)}")
# two-way: closed form -> Q and back
P_(f"S3' back-substitution Q(closed) - Q(def) = "
   f"{sp.simplify((sp.pi*G*R**4/(8*mu_app_closed)).subs(J, sp.besseli(2, lam*R)/(lam*R*sp.besseli(1, lam*R))) - Qf.subs(C, C_of).subs(nuR, G*R*shat/2))}")
P_(f"S3'' d mu_app / d shat = {sp.factor(sp.diff(mu_app_closed, shat))}  (negative for J > 0: mu_app decreases with wall spin)")

# bound criterion: mu_app < mu+k  <=>  e/4 + (shat - e) J > 0  <=>  int r^2 nu dr > 0
crit = sp.simplify((mu + ka) / mu_app_closed - 1 - 4 * ka * (e / 4 + (shat - e) * J))
P_(f"S4 (mu+k)/mu_app - 1 - 4k[e/4 + (shat-e) J] = {crit}")

# Bessel bounds by series: J < 1/4 and X = a I0/I1 - 1 > 1 (termwise comparison)
k = sp.symbols('k', integer=True, nonnegative=True)
c_I2 = 1 / (sp.factorial(k) * sp.factorial(k + 2))            # coeff of (a/2)^(2k+2) in I2
c_aI1_4 = 1 / (2 * sp.factorial(k) * sp.factorial(k + 1))     # coeff of (a/2)^(2k+2) in (a/4) I1
P_(f"S5 coeff[(a/4) I1 - I2] / coeff[I2] = {sp.simplify(c_aI1_4 / c_I2 - 1)}  (= k/2 >= 0, > 0 for k >= 1 => I2 < a I1/4, J < 1/4)")
c_I0 = 1 / sp.factorial(k)**2
c_2I1_a = 1 / (sp.factorial(k) * sp.factorial(k + 1))         # coeff of (a/2)^(2k) in 2 I1 / a
P_(f"S5 coeff[I0 - 2 I1/a] / coeff[2 I1/a] = {sp.simplify(c_I0 / c_2I1_a - 1)}  (= k => a I0 > 2 I1, X > 1)")
P_(f"S5 recurrence I2 - (I0 - 2 I1/a) = {sp.simplify(sp.besseli(2, a) - (sp.besseli(0, a) - 2*sp.besseli(1, a)/a))}")
P_(f"S5 I1'(a) - (I0 - I1/a) = {sp.simplify(sp.diff(sp.besseli(1, a), a) - (sp.besseli(0, a) - sp.besseli(1, a)/a))}")

# M3: nu(R) = -n w'(R)
nuR_m3 = sp.solve(sp.Eq(nuR, -n * (-(G * R / 2 + ka * nuR) / (mu + ka))), nuR)[0]
P_(f"S6 M3 wall spin nu_R = {sp.factor(nuR_m3)};  shat(n) = {sp.factor(2*nuR_m3/(G*R))}")
P_(f"S6 d shat/dn = {sp.factor(sp.diff(2*nuR_m3/(G*R), n))}  (> 0 on each branch; pole n_p = (mu+k)/k)")

# wall power: nu'(R) in terms of nu_R, P = nu_R [gamma nu'(R) - beta nu_R / R]
nu_g = nu.subs(C, C_of)
dnuR = sp.diff(nu_g, r).subs(r, R)
X = sp.symbols('X', positive=True)   # X = a I0(a)/I1(a) - 1 = a I1'(a)/I1(a)
dnuR_X = G * e / 2 * (1 - X) + nuR * X / R
dnu_diff = sp.simplify((dnuR - dnuR_X.subs(X, lam*R*sp.besseli(0, lam*R)/sp.besseli(1, lam*R) - 1)).rewrite(sp.besseli))
P_(f"S7 nu'(R) - [G e (1-X)/2 + nu_R X/R] = {dnu_diff}")
Pw = sp.expand(nuR * (ga * dnuR_X - be * nuR / R))
A_ = ga * G * e * (X - 1) / 2
B_ = (ga * X - be) / R
P_(f"S7 P - nu_R(-A + B nu_R), A = g G e (X-1)/2, B = (g X - b)/R : {sp.simplify(Pw - nuR*(-A_ + B_*nuR))}")
s_star = ga * e * (X - 1) / (ga * X - be)
P_(f"S7 passive <=> 0 <= nu_R <= A/B  <=>  0 <= shat <= s* = {s_star}")
n_star = sp.simplify(s_star * (mu + ka) / (1 + s_star * ka))
T_ = 1 / X
Del = ka / mu
n_star_guide = (1 + Del) * (1 - T_) / (2 + 2 * Del - Del * T_)
P_(f"S8 n*(beta=0) - tube formula [(1+D)(1-T)/(2+2D-DT), T = I1/(lam I0 - I1) = 1/X] = "
   f"{sp.simplify(n_star.subs(be, 0) - n_star_guide)}")
P_(f"S8 n*(beta = gamma) = {sp.simplify(n_star.subs(be, ga))}")
# M5 (m_(n) = 0) is exactly the P = 0 end: gamma nu'(R) - beta nu_R/R = 0
nuR_m5 = sp.solve(sp.Eq(ga * dnuR_X - be * nuR / R, 0), nuR)[0]
P_(f"S9 M5 wall spin / (G R/2) - s* = {sp.simplify(2*nuR_m5/(G*R) - s_star)}  (M5 is the upper end of the passive set)")

# Chaturani & Mahajan 1982 Eqs (8),(10), with mu_CM = mu_s + k, mapping kappa = 2k, mu = mu_s:
#   mu_a = (mu_s+k)/[1 - k g sbar/(mu_s+2k)],  g = -4T, T = 2/a^2 - I0/(a I1),
#   sbar = [1 + 2 mu_CM/(s mu_CM + s k)]^(-1), n = 1/(2+s)
s = sp.symbols('s', positive=True)
mus, kk = mu, ka / 2
muCM = mus + kk
sbar = 1 / (1 + 2 * muCM / (s * muCM + s * kk))
Tch = 2 / a**2 - sp.besseli(0, a) / (a * sp.besseli(1, a))
mu_a_CM = (mus + kk) / (1 - kk * (-4 * Tch) * sbar / (mus + 2 * kk))
shat_s = sp.factor((2 * nuR_m3 / (G * R)).subs(n, 1 / (2 + s)))
mine = mu_app_closed.subs({shat: shat_s, J: sp.besseli(2, a) / (a * sp.besseli(1, a))})
P_(f"S10 shat(n = 1/(2+s)) - e(1 - sbar) = {sp.simplify(shat_s - e*(1 - sbar))}")
P_(f"S10 mu_app(this work) - mu_a(Chaturani-Mahajan Eq. 10, mapped) = {sp.simplify(mine - mu_a_CM)}")
P_(f"S10 reverse: mu_a_CM/mine - 1 = {sp.simplify(mu_a_CM/mine - 1)}")
P_(f"S10 J + T_CM = {sp.simplify(sp.besseli(2, a)/(a*sp.besseli(1, a)) + Tch)}")
# general cross-section reduction (no wall condition on nu used): Q = [G Q_N(1) - k int nu.curl(phi e_z)]/(mu+k)
# tube check: phi = (R^2 - r^2)/4, curl(phi e_z) = -phi' e_theta = (r/2) e_theta
phi = (R**2 - r**2) / 4
QN1 = 2 * sp.pi * sp.integrate(phi * r, (r, 0, R))
Qgen = (G * QN1 - ka * 2 * sp.pi * sp.integrate(nu * (-sp.diff(phi, r)) * r, (r, 0, R))) / (mu + ka)
# sign: curl(phi e_z)_theta = -phi'; the identity reads Q = [G Q_N - k int nu.curl(phi e_z)]/(mu+k)?
P_(f"S11 general reduction, tube check: Q - [G Q_N(1) + k int nu (r/2) dA]/(mu+k) = "
   f"{sp.simplify(Q - (G*QN1 + ka*2*sp.pi*sp.integrate(nu*(r/2)*r, (r, 0, R)))/(mu + ka))}")

# ----------------------------------------------------------------------------------------- numerics
def J_(aa):
    return ive(2, aa) / (aa * ive(1, aa))      # exponentially scaled: safe for large aa


def X_(aa):
    return aa * ive(0, aa) / ive(1, aa) - 1


def Dden(m, k, aa, shv):
    ee = 1 / (2 * m + k)
    return 1 + k * ee + 4 * k * (shv - ee) * J_(aa)


def viol_intervals(m, k, aa):
    """n-intervals where 0 < D < 1 (mu_app > mu+k); shat(n) = n/(m+(1-n)k); D < 0 means reversed flow."""
    ee = 1 / (2 * m + k); Jv = J_(aa)
    sc = -ee * (0.25 - Jv) / Jv                  # D = 1
    s0 = ee - (1 + k * ee) / (4 * k * Jv)         # D = 0
    ninv = lambda sv: sv * (m + k) / (1 + sv * k)
    sm = -1 / k                                    # shat(n -> +-inf)
    res = []
    lo, hi = max(s0, sm), min(sc, 0.0)             # negative-n branch: shat in (sm, 0)
    if lo < hi:
        res.append(('n<0', -np.inf if lo == sm else ninv(lo), ninv(hi)))
    lo, hi = s0, min(sc, sm)                       # above-pole branch: shat in (-inf, sm)
    if lo < hi:
        res.append(('n>pole', ninv(lo), np.inf if hi == sm else ninv(hi)))
    return res, sc, s0


def closed(m, k, aa, shv):
    ee = 1 / (2 * m + k)
    return (m + k) / (1 + k * ee + 4 * k * (shv - ee) * J_(aa))


def shat_n(m, k, nn):
    return nn / (m + (1 - nn) * k)


def sstar(m, k, aa, bg):            # bg = beta/gamma
    ee = 1 / (2 * m + k)
    XX = X_(aa)
    return ee * (XX - 1) / (XX - bg)


def nstar(m, k, aa, bg):
    ss = sstar(m, k, aa, bg)
    return ss * (m + k) / (1 + ss * k)


def power(m, k, aa, shv, bg, g=1.0, Rr=1.0, Gv=1.0):
    ee = 1 / (2 * m + k)
    XX = X_(aa)
    nr = Gv * Rr * shv / 2
    return nr * (g * (Gv * ee * (1 - XX) / 2 + nr * XX / Rr) - bg * g * nr / Rr)


def fd(m, k, aa, nn=None, mode='M3', bg=0.0, N=2000, ls=0.0):
    """Independent FD solver of the full 2nd-order system on a uniform grid, R = G = 1."""
    g = k * (2 * m + k) / ((m + k) * aa**2)
    rr = np.linspace(0, 1, N + 1); h = rr[1] - rr[0]
    A = lil_matrix((2 * (N + 1), 2 * (N + 1))); b = np.zeros(2 * (N + 1))
    U = lambda i: i; W = lambda i: N + 1 + i
    for i in range(N + 1):
        if i == 0:
            A[U(0), U(0)] = -3; A[U(0), U(1)] = 4; A[U(0), U(2)] = -1   # w'(0) = 0
            A[W(0), W(0)] = 1; continue                                  # nu(0) = 0
        if i == N:
            A[U(N), U(N)] = 1                                            # w(1) = 0
            if mode == 'M1':
                A[W(N), W(N)] = 1
            elif mode == 'M3':   # nu + n w' = 0, one-sided 2nd order
                A[W(N), W(N)] = 1
                A[W(N), U(N)] += nn * 3 / (2 * h); A[W(N), U(N - 1)] += -nn * 4 / (2 * h); A[W(N), U(N - 2)] += nn / (2 * h)
            elif mode in ('M5', 'M4'):  # gamma nu' - beta nu/R = -(gamma/ls) nu (M4) ; M5: ls -> inf
                A[W(N), W(N)] = g * 3 / (2 * h) - bg * g + (g / ls if mode == 'M4' else 0.0)
                A[W(N), W(N - 1)] = -g * 4 / (2 * h); A[W(N), W(N - 2)] = g / (2 * h)
            continue
        c = 1 / rr[i]
        A[U(i), U(i - 1)] = (m + k) * (1 / h**2 - c / (2 * h)); A[U(i), U(i)] = -2 * (m + k) / h**2
        A[U(i), U(i + 1)] = (m + k) * (1 / h**2 + c / (2 * h))
        A[U(i), W(i - 1)] = -k / (2 * h); A[U(i), W(i + 1)] = k / (2 * h); A[U(i), W(i)] = k * c
        b[U(i)] = -1.0
        A[W(i), W(i - 1)] = g * (1 / h**2 - c / (2 * h)); A[W(i), W(i)] = -2 * g / h**2 - g * c**2 - 2 * k
        A[W(i), W(i + 1)] = g * (1 / h**2 + c / (2 * h))
        A[W(i), U(i - 1)] = k / (2 * h); A[W(i), U(i + 1)] = -k / (2 * h)
    sol = spsolve(A.tocsr(), b); wv = sol[:N + 1]; nv = sol[N + 1:]
    Qv = 2 * np.pi * np.trapezoid(wv * rr, rr)
    return np.pi / (8 * Qv), nv[N]


P_("\nN1 closed form vs independent FD (N = 1000, 2000, 4000; Richardson), mu = 1")
worst = 0.0
for k, aa, nn in [(1.0, 1.0, 0.0), (1.0, 1.0, 0.2), (1.0, 3.0, -0.5), (10.0, 10.0, 0.3), (0.1, 30.0, 0.45),
                  (1.0, 0.3, -1.0), (1.0, 2.0, 3.0), (-0.0 + 0.5, 100.0, 0.4)]:
    vals = [fd(1.0, k, aa, nn, N=Nn)[0] for Nn in (1000, 2000, 4000)]
    rich = vals[2] + (vals[2] - vals[1]) / 3
    cf = closed(1.0, k, aa, shat_n(1.0, k, nn))
    worst = max(worst, abs(rich / cf - 1))
    P_(f"  k={k:5.1f} lamR={aa:6.1f} n={nn:5.2f}: FD {vals[0]:.8f} {vals[1]:.8f} {vals[2]:.8f} Rich {rich:.8f} | closed {cf:.8f} | rel {rich/cf-1:+.1e}")
P_(f"N1 worst relative difference (Richardson vs closed): {worst:.1e}")
# M5 / M4 FD vs closed form with nu_R from the passive-set formula
for k, aa, bg in [(1.0, 1.0, 0.0), (1.0, 5.0, 0.5), (10.0, 20.0, -0.7)]:
    m5 = fd(1.0, k, aa, mode='M5', bg=bg, N=4000)[0]
    cf = closed(1.0, k, aa, sstar(1.0, k, aa, bg))
    P_(f"  M5 k={k} lamR={aa} beta/gamma={bg}: FD {m5:.7f} closed(shat = s*) {cf:.7f} rel {m5/cf-1:+.1e}")

# energy identity: int Phi dA = G Q + 2 pi R P, Phi = (mu+k) w'^2 + 2k nu^2 + 2k nu w' + g (nu'^2 + nu^2/r^2) - 2 b nu nu'/r
P_("\nN1' energy identity int Phi dA = G Q + 2 pi R P (quadrature), relative error")
def energy(m, k, aa, nn, bg):
    g = k * (2 * m + k) / ((m + k) * aa**2); ee = 1 / (2 * m + k)
    sh = shat_n(m, k, nn); nr = sh / 2
    Cv = (nr - ee / 2) / iv(1, aa)
    def nuf(q):
        return q * ee / 2 + Cv * iv(1, aa * q)
    def dnu(q):
        return ee / 2 + Cv * aa * (iv(0, aa * q) - iv(1, aa * q) / (aa * q))
    def Phi(q):
        wp = -(q / 2 + k * nuf(q)) / (m + k)
        nv, dv = nuf(q), dnu(q)
        return ((m + k) * wp**2 + 2 * k * nv**2 + 2 * k * nv * wp
                + g * (dv**2 + (nv / q)**2) - 2 * bg * g * nv * dv / q)
    D = 2 * np.pi * quad(lambda q: Phi(q) * q, 1e-12, 1, epsabs=1e-14, epsrel=1e-12, limit=400)[0]
    Qv = np.pi / 8 / closed(m, k, aa, sh)
    Pv = power(m, k, aa, sh, bg, g=g)
    return (D - (Qv + 2 * np.pi * Pv)) / D, Pv, D, Qv


for args in [(1.0, 1.0, 2.0, 0.2, 0.0), (1.0, 1.0, 2.0, -0.4, 0.6), (1.0, 4.0, 8.0, 0.3, -0.8), (-0.3, 1.0, 3.0, 0.1, 0.3)]:
    err, Pv, Dv, Qv = energy(*args)
    P_(f"  mu={args[0]} k={args[1]} lamR={args[2]} n={args[3]} beta/gamma={args[4]}: int Phi = {Dv:.10f}, G Q = {Qv:.10f}, 2 pi R P = {2*np.pi*Pv:+.10f}, rel err {err:.1e}")

# ----------------------------------------------------------------------------------------- N2 scans
P_("\nN2 scan: mu = 1, kappa/mu in geomspace(0.1,10,21), lamR in geomspace(0.1,100,31), n in linspace(-6,16,881)\\{pole}, beta/gamma in {-1,0,1}")
kl = np.geomspace(0.1, 10, 21); al = np.geomspace(0.1, 100, 31); nl = np.linspace(-6, 16, 881)
stats = dict(passive=0, pass_viol=0, active=0, act_viol=0, act_ok=0, viol_neg=0, viol_pole=0)
max_pass = (0, None); min_pass_low = (1e9, None)
for bg in (-1.0, 0.0, 1.0):
    for k in kl:
        pole = (1 + k) / k
        for aa in al:
            for nn in nl:
                if abs(nn - pole) < 1e-9 or abs(1 + (1 - nn) * k) < 1e-9:
                    continue
                sh = shat_n(1.0, k, nn)
                ma = closed(1.0, k, aa, sh)
                Pv = power(1.0, k, aa, sh, bg)
                passive = Pv <= 1e-15
                Dv = Dden(1.0, k, aa, sh)
                viol = 0 < Dv < 1
                stats['reverse'] = stats.get('reverse', 0) + (Dv <= 0)
                if passive:
                    stats['passive'] += 1; stats['pass_viol'] += viol
                    rr_ = ma / (1 + k)
                    if rr_ > max_pass[0]:
                        max_pass = (rr_, (k, aa, nn, bg))
                    lo = ma / (1 + k / 2)
                    if lo < min_pass_low[0]:
                        min_pass_low = (lo, (k, aa, nn, bg))
                else:
                    stats['active'] += 1; stats['act_viol'] += viol; stats['act_ok'] += (not viol)
                    if viol:
                        stats['viol_neg' if nn < 0 else 'viol_pole'] += 1
P_(f"  passive cases {stats['passive']}, of which mu_app > mu+k: {stats['pass_viol']}")
P_(f"  active cases {stats['active']}: mu_app > mu+k in {stats['act_viol']} (n<0: {stats['viol_neg']}, n>pole: {stats['viol_pole']}); active but bound holds: {stats['act_ok']}")
P_(f"  active cases with reversed flow (D <= 0, mu_app < 0): {stats.get('reverse', 0)}")
P_(f"  max passive mu_app/(mu+k) = {max_pass[0]:.6f} at (k/mu, lamR, n, beta/gamma) = {max_pass[1]}")
P_(f"  min passive mu_app/(mu+k/2) = {min_pass_low[0]:.12f} at {min_pass_low[1]}  (Y4 lower bound: must be >= 1)")
# passive set independence of beta for the bound: shat in [0, s*] for every beta in [-gamma, gamma]

P_("\nN2' thresholds: D = 1 + k e + 4k(shat - e)J; mu_app > mu+k iff 0 < D < 1 (shat in (s0, sc)); D < 0: reversed flow (mu_app < 0)")
P_("  k/mu   lamR    n*(b=0)  n*(b=-g)  n*(b=g)  M1/(mu+k)  M5(b=0)/(mu+k)  n-intervals with mu_app > mu+k")
for k in (0.1, 1.0, 10.0):
    for aa in (0.1, 1.0, 3.0, 10.0, 30.0, 100.0):
        iv_, sc, s0 = viol_intervals(1.0, k, aa)
        txt = "; ".join(f"{b}: ({lo:.4f}, {hi:.4f})" for b, lo, hi in iv_)
        P_(f"  {k:5.1f} {aa:6.1f}   {nstar(1,k,aa,0):.5f}  {nstar(1,k,aa,-1):.5f}  {nstar(1,k,aa,1):.5f}   "
           f"{closed(1,k,aa,0)/(1+k):.6f}   {closed(1,k,aa,sstar(1,k,aa,0))/(1+k):.6f}     {txt}")
# brute-force confirmation of the first violation on a fine n grid
P_("  brute-force first violation (n grid step 1e-4, from 0 downward / from pole upward), beta = 0:")
for k, aa in [(1.0, 0.1), (1.0, 1.0), (1.0, 3.0), (10.0, 1.0), (0.1, 1.0)]:
    ng = np.arange(0, -50, -1e-4)
    v = [nn for nn in ng if closed(1, k, aa, shat_n(1, k, nn)) > 1 + k]
    first_neg = v[0] if v else None
    pole = (1 + k) / k
    ng2 = pole + np.arange(1e-4, 50, 1e-4)
    v2 = [nn for nn in ng2 if closed(1, k, aa, shat_n(1, k, nn)) > 1 + k]
    up = (v2[0], v2[-1]) if v2 else None
    nl_ = v[-1] if v else None
    ana = [(b, round(lo, 4), round(hi, 4)) for b, lo, hi in viol_intervals(1, k, aa)[0]]
    P_(f"    k={k} lamR={aa}: n<0 violations {None if first_neg is None else round(first_neg,4)} .. {None if nl_ is None else round(nl_,4)} (scan to -50); above pole {None if up is None else (round(up[0],4), round(up[1],4))} (scan to pole+50); analytic {ana}")

P_("\nN2'' mu < 0 test (mu = -0.3, kappa = 1, 2mu+k = 0.4 > 0)")
cnt = 0; worst = 0
for aa in al:
    for bg in (-1, 0, 1):
        for nn in np.linspace(0, nstar(-0.3, 1.0, aa, bg), 41):
            ma = closed(-0.3, 1.0, aa, shat_n(-0.3, 1.0, nn))
            cnt += 1; worst = max(worst, ma / 0.7)
P_(f"  {cnt} passive cases, max mu_app/(mu+k) = {worst:.6f}")

# ----------------------------------------------------------------------------------------- N3 limits
P_("\nN3 limits")
for k in (0.1, 1.0, 10.0):
    for aa in (0.5, 5.0):
        m1fd = fd(1.0, k, aa, mode='M1', N=4000)[0]
        P_(f"  n -> 0: k={k} lamR={aa}: closed n=1e-8 {closed(1,k,aa,shat_n(1,k,1e-8)):.8f}, closed M1 {closed(1,k,aa,0):.8f}, FD M1 {m1fd:.8f}")
for k in (0.1, 1.0, 10.0):
    vals = [closed(1, k, aa, shat_n(1, k, 0.3)) / (1 + k / 2) for aa in (1e2, 1e3, 1e4, 1e5)]
    P_(f"  lamR -> inf (n = 0.3): k={k}: mu_app/(mu+k/2) = " + ", ".join(f"{v:.6f}" for v in vals))
    vals = [closed(1, k, aa, 0) / (1 + k) for aa in (1e-1, 1e-2, 1e-3)]
    P_(f"  lamR -> 0 (M1): k={k}: mu_app/(mu+k) = " + ", ".join(f"{v:.8f}" for v in vals))
P_("  narrow-tube passive range: n*(b=0) at lamR = 0.1, 0.01: " + ", ".join(f"{nstar(1,1.0,aa,0):.3e}" for aa in (0.1, 0.01)))

# ----------------------------------------------------------------------------------------- N4 water
P_("\nN4 water (Hansen et al. 2010 Table II, SPC/Fw): mu = eta0 - eta_r, kappa = 2 eta_r, gamma = zeta0 + zeta_r")
TABLE2 = [(284, 0.92, 0.17, 1.4, 1.6), (289, 0.83, 0.17, 1.2, 1.5), (293, 0.76, 0.17, 1.2, 1.4), (299, 0.70, 0.17, 0.9, 1.2)]
P_("  T[K] eta_r/eta0 1/lam[nm]  r[nm]  lamR   M1=max passive  M5(b=0)  n*(b=0)  abs bound   n needed for 1.31 / 1.48  (wall power / pressure work)")
for T, e0, er, z0, zr in TABLE2:
    e0 *= 1e-3; er *= 1e-3; gg = (z0 + zr) * 1e-21
    m, k = e0 - er, 2 * er
    lm = np.sqrt(k * (2 * m + k) / (gg * (m + k)))
    for rn in (40, 50, 60):
        aa = lm * rn * 1e-9
        m1 = closed(m, k, aa, 0) / e0
        m5 = closed(m, k, aa, sstar(m, k, aa, 0)) / e0
        need = []
        for target in (1.31, 1.48):
            # mu_app/e0 = target -> shat
            ee = 1 / (2 * m + k); Jv = J_(aa)
            sh = (((m + k) / (target * e0)) - 1 - k * ee) / (4 * k * Jv) + ee
            nn = sh * (m + k) / (1 + sh * k)
            Rr = rn * 1e-9
            Pv = power(m, k, aa, sh, 0.0, g=gg, Rr=Rr, Gv=1.0)
            Qv = np.pi * Rr**4 / (8 * target * e0)
            need.append(f"n={nn:+.4f} ({2*np.pi*Rr*Pv/Qv:+.3f})" if (1 + sh * k) != 0 else "pole")
        P_(f"  {T}   {er/e0:.3f}     {1/lm*1e9:6.3f}   {rn:4d}  {aa:6.2f}   {m1:.5f}        {m5:.5f}  {nstar(m,k,aa,0):.4f}   {1+er/e0:.4f}    " + " / ".join(need))
P_("  Churaev 1970 (tube, r = 40..60 nm): 1.31..1.48 (data table)")
P_("  water: n-intervals with mu_app > mu+k (eta-units), r = 40/50/60 nm at 284 K and 299 K:")
for T, e0, er, z0, zr in (TABLE2[0], TABLE2[-1]):
    e0 *= 1e-3; er *= 1e-3; gg = (z0 + zr) * 1e-21
    m, k = e0 - er, 2 * er
    lm = np.sqrt(k * (2 * m + k) / (gg * (m + k)))
    for rn in (40, 50, 60):
        ivs = viol_intervals(m, k, lm * rn * 1e-9)[0]
        P_(f"    T={T} r={rn}: pole n_p = {(m+k)/k:.4f}; " + "; ".join(f"{b}: ({lo:.4f}, {hi:.4f})" for b, lo, hi in ivs))

# ----------------------------------------------------------------------------------------- B  Bessonov & Frolova 2022
P_("\nB1 Bessonov & Frolova (2022) Eqs (3.29)-(3.46): notation mapping and GBC as an M3 member (symbolic)")
muB, mur, mub = sp.symbols('mu_B mu_r mu_b', positive=True)
# their stress: mu_B (v_k,n + v_n,k) + mu_r (v_k,n - v_n,k - 2 e w)  vs Eringen mu(.)+kappa(v_l,k - e nu):
# Eringen symmetric part (mu + kappa/2)(v_k,l + v_l,k), antisymmetric (kappa/2)(v_l,k - v_k,l) - kappa e nu
mapB = {muB: mu + ka / 2, mur: ka / 2}
P_(f"  symmetric coeff: mu_B - (mu + kappa/2) = {sp.simplify(muB.subs(mapB) - (mu + ka/2))}; antisym: 2 mu_r - kappa = {sp.simplify((2*mur).subs(mapB) - ka)}")
P_(f"  (3.42) coefficients: (mu_B + mu_r) - (mu + kappa) = {sp.simplify((muB + mur).subs(mapB) - (mu + ka))}; 2 mu_r - kappa = {sp.simplify((2*mur).subs(mapB) - ka)}")
P_(f"  (3.43): -2 mu_r (2 w + v') vs Eringen -2 kappa nu + kappa (curl v)_theta = -2 kappa nu - kappa v': difference = "
   f"{sp.simplify((-2*mur*(2*sp.Symbol('w') + sp.Symbol('vp'))).subs(mapB) - (-2*ka*sp.Symbol('w') - ka*sp.Symbol('vp')))}")
Lb2 = ga * (1 / muB + 1 / mur)
P_(f"  L^2 lambda^2 = {sp.simplify((Lb2.subs(mapB)) * lam2)}  (L = 2/lambda, hbar_e = h_e/L = 2R/L = lambda R)")
nB = (muB + mur - mub) / (2 * mur)
P_(f"  GBC (3.44): omega_phi = -[(mu_B + mu_r - mu_b)/(2 mu_r)] v_z' = n (curl v)_theta  =>  n = {sp.simplify(nB.subs(mapB))}")
mub_of_n = sp.solve(sp.Eq(nB.subs(mapB), n), mub)[0]
P_(f"  => mu_b = {sp.expand(mub_of_n)}  (= mu + (1-n) kappa: the M3 denominator; shat = n/mu_b)")
P_(f"  check shat(n) - n/mu_b = {sp.simplify(2*nuR_m3/(G*R) - n/mub_of_n)}")
Kb = 1 - muB / mub
mue_B = 1 / (1 - 4 * Kb * J)              # their (3.46), relative to mu_B
mine_rel = (mu_app_closed / (mu + ka / 2)).subs(shat, n / mub_of_n)
P_(f"  (3.46) vs this work (mu_app/(mu + kappa/2)) with mu_b = mu + (1-n)kappa: difference = "
   f"{sp.simplify(mue_B.subs(mapB).subs(mub, mub_of_n) - mine_rel)}")
P_(f"  reverse (n expressed by mu_b): difference = "
   f"{sp.simplify(mue_B.subs(mapB) - (mu_app_closed/(mu + ka/2)).subs(shat, nB.subs(mapB)/mub))}")
P_(f"  limits of (3.46): J -> 1/4: {sp.simplify(mue_B.subs(J, sp.Rational(1, 4)))} (= mu_b/mu_B), J -> 0: {mue_B.subs(J, 0)}")
eps = sp.symbols('epsilon', positive=True)   # eta_r/eta0 = (kappa/2)/(mu + kappa/2)
nst = sp.symbols('n_star', positive=True)
mub_rel = sp.simplify((mub_of_n / (mu + ka / 2)).subs(ka, 2 * eps * (mu + ka / 2)))
mub_rel = sp.simplify(sp.expand(mub_of_n / (mu + ka/2)).subs(mu, sp.Symbol('eta0') * (1 - eps)).subs(ka, 2 * eps * sp.Symbol('eta0')))
P_(f"  mu_b/mu_B in terms of n and eps = eta_r/eta0: {sp.factor(mub_rel)};  passive n in [0, n*]  <=>  mu_b/mu_B in [1 + eps(1 - 2 n*), 1 + eps]")

P_("\nB2 Bessonov & Frolova Table 3.1 row 10 (Derjaguin water/quartz) vs water MD eps = eta_r/eta0 (beta = 0)")
def bess_mue(mubar, abar):
    Kv = 1 - 1 / mubar
    return 1 / (1 - 4 * Kv * J_(abar))
for case, mubar, Lnm, hs in (("(a)", 1.6, 32.0, 0.0), ("(b)", 0.09, 3.0, 20.0)):
    P_(f"  case {case}: mu_b/mu = {mubar}, L = {Lnm} nm, h_s = {hs} nm")
    for T, e0, er, z0, zr in TABLE2:
        ep = er / e0
        n_req = (1 + ep - mubar) / (2 * ep)
        e0s = e0 * 1e-3; ers = er * 1e-3
        m, k = e0s - ers, 2 * ers
        gB = (Lnm * 1e-9)**2 * e0s * ers / (e0s + ers)      # gamma implied by their L and the MD eps
        rows = []
        for rn in (40, 50, 60):
            he = 2 * rn - 2 * hs
            aa = he / Lnm
            ns = nstar(m, k, aa, 0.0)
            mue = bess_mue(mubar, aa)
            sh = n_req / (m + (1 - n_req) * k)
            Rr = he / 2 * 1e-9
            Pv = power(m, k, aa, sh, 0.0, g=gB, Rr=Rr, Gv=1.0)
            Qv = np.pi * Rr**4 / (8 * mue * e0s)
            nominal = mue * (2 * rn / he)**4
            pas = (0 <= n_req <= ns)
            rows.append(f"r={rn}: hbar={aa:.3f} n*={ns:.4f} {'passive' if pas else 'ACTIVE'} mu_e={mue:.4f}"
                        + (f" (nominal-diameter {nominal:.3f})" if hs > 0 else "") + f" Pwall/(GQ)={2*np.pi*Rr*Pv/Qv:+.3f}")
        P_(f"    T={T} eps={ep:.3f}: n = {n_req:+.4f} (pole {(1+ep)/(2*ep):.4f}); passive mu_b/mu range at r=50: "
           f"[{1 + ep*(1 - 2*nstar(m, k, (100 - 2*hs)/Lnm, 0.0)):.4f}, {1+ep:.4f}]; bound 1+eps = {1+ep:.4f}")
        for rw in rows:
            P_("      " + rw)
# smallest eps for which mu_b/mu = 1.6 is passive at hbar = 2R/L, R = 40..60 nm (L = 32 nm)
P_("  smallest eta_r/eta0 making mu_b/mu = 1.6 passive (beta = 0), L = 32 nm:")
for rn in (40, 50, 60):
    aa = 2 * rn / 32.0
    epsg = np.linspace(0.6, 30, 294001)
    ok = None
    for ep in epsg:
        m, k = 1 - ep, 2 * ep
        nn = (1 + ep - 1.6) / (2 * ep)
        if 0 <= nn <= nstar(m, k, aa, 0.0):
            ok = ep; break
    P_(f"    r={rn} nm (hbar={aa:.3f}): eps_min = {ok}")
P_("  same mu_b/mu = 1.6 but with the MD length L = 2/lambda (water 299 K, 1.959 nm -> L = 3.918 nm): mu_e at r=40/50/60 = "
   + ", ".join(f"{bess_mue(1.6, 2*rn/3.918):.4f}" for rn in (40, 50, 60)))

with open(__file__.replace('.py', '_output.txt'), 'w', encoding='utf-8') as f:
    for line in out:
        print(line)
        f.write(line + "\n")
