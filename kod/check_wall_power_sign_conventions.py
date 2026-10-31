"""Sign chain of the energy identity (8) and of the wall power P = m_(n) . nu.

Conventions (Eringen 1966, J. Math. Mech. 16, 1-18):
  linear momentum   t_kl,k + rho (f_l - dv_l/dt) = 0                 (2.2), p. 2
  angular momentum  m_rk,r + eps_klr t_lr + rho (l_k - dsigma_k/dt) = 0  (4.9), p. 7
  surface couple    m_rk n_r = m_k on S, n exterior normal           (4.13), p. 7; (2.9)-(2.10), p. 4
  dissipation       t_kl (v_l,k - eps_klr nu_r) + m_kl nu_l,k         (4.10), p. 7
  constitutive      (4.14) and (4.17) = Eqs. (1) and (2) of the paper; v_k,l = d v_k / d x_l (2.6), p. 3.

Part A  general cross-section, fully developed flow v = w(x,y) e_z, p = p1(x,y) - G z, nu = nu(x,y) (3 components):
  A1 pointwise identity  div_xy(F) + G w = Phi_E + v_l t_kl,k + nu_l (m_kl,k + eps_lmn t_mn),
     F_k = t_kl v_l + m_kl nu_l (k = x, y)                                        -> residual 0
  A2 Phi_E (Eringen) equals Eq. (5) of the paper (+ 2 kappa nu_z^2 + gamma |grad nu_z|^2)   -> residual 0
  A3 z-momentum t_kz,k = left side of Eq. (3); A4 in-plane angular momentum = left side of Eq. (4) -> 0
  A5 exact test on the square [-1,1]^2 with polynomial fields (w = 0 on the wall, nu != 0 on the wall):
     int Phi - G Q - oint m_(n).nu ds + int R dA = 0 with the OUTWARD normal (R = balance residuals),
     and the same with the inward normal is not zero.
  Hence for a solution (R = 0) and a no-slip wall: int Phi dA = G Q + oint P ds, P = m_(n).nu, n outward  (Eq. 8).
Part B  BC3: nu = -(l_s/gamma) m_(n)  =>  P = -(l_s/gamma)|m_(n)|^2 = -(gamma/l_s)|nu|^2 <= 0; limits l_s -> 0, l_s -> oo.
  B2 bilinear (polarized) identity: first variation of int Phi gives the wall term oint m_(n)[X].phi ds,
     Phi_bil symmetric; the functional 2G int u - int Phi - (gamma/l_s) oint |phi|^2 has BC3 as natural condition,
     and its value at the solution equals G Q.
Part C  BC2 in the tube: m_(n).nu for nu = nu(r) e_theta equals nu (gamma nu' - beta nu / R) (symbolic);
  P from the closed-form solution against Eq. (14); sign of P along n (indefinite).
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


x, y, z = sp.symbols('x y z', real=True)
mu, ka, al, be, ga, G = sp.symbols('mu kappa alpha beta gamma G', real=True)
X = (x, y, z)
eps = sp.LeviCivita


def stresses(v, nu, p):
    """Eqs. (1), (2): t_kl, m_kl with v_k,l = d v_k / d x_l."""
    dv = [[sp.diff(v[k], X[l]) for l in range(3)] for k in range(3)]
    dn = [[sp.diff(nu[k], X[l]) for l in range(3)] for k in range(3)]
    divn = sum(dn[r][r] for r in range(3))
    t = [[-p * sp.KroneckerDelta(k, l) + mu * (dv[k][l] + dv[l][k])
          + ka * (dv[l][k] - sum(eps(k, l, m) * nu[m] for m in range(3)))
          for l in range(3)] for k in range(3)]
    m = [[al * divn * sp.KroneckerDelta(k, l) + be * dn[k][l] + ga * dn[l][k]
          for l in range(3)] for k in range(3)]
    return t, m, dv, dn


def phi_eringen(t, m, dv, dn, nu):
    """Eringen (4.10): t_kl (v_l,k - eps_klr nu_r) + m_kl nu_l,k."""
    return sum(t[k][l] * (dv[l][k] - sum(eps(k, l, r) * nu[r] for r in range(3)))
               + m[k][l] * dn[l][k] for k in range(3) for l in range(3))


# ---------------------------------------------------------------- Part A
say("=" * 78)
say("Part A. Energy identity in a general section (Eringen conventions, outward normal)")
say("=" * 78)
w = sp.Function('w')(x, y)
n1, n2, n3 = [sp.Function(f'nu{i}')(x, y) for i in (1, 2, 3)]
p1 = sp.Function('p1')(x, y)
v = [0, 0, w]
nu = [n1, n2, n3]
p = p1 - G * z
t, m, dv, dn = stresses(v, nu, p)
PhiE = phi_eringen(t, m, dv, dn, nu)
lin = [sum(sp.diff(t[k][l], X[k]) for k in range(3)) for l in range(3)]                  # t_kl,k
ang = [sum(sp.diff(m[r][k], X[r]) for r in range(3))
       + sum(eps(k, l, r) * t[l][r] for l in range(3) for r in range(3)) for k in range(3)]  # (4.9)
F = [sum(t[k][l] * v[l] + m[k][l] * nu[l] for l in range(3)) for k in range(3)]
divxy = sp.diff(F[0], x) + sp.diff(F[1], y)
dFz = sp.diff(F[2], z)
say(f"A0 d/dz of the axial flux t_zl v_l + m_zl nu_l = {sp.simplify(dFz)}   (expected G*w)")
R = sum(v[l] * lin[l] for l in range(3)) + sum(nu[k] * ang[k] for k in range(3))
resA1 = sp.simplify(sp.expand(divxy + G * w - (PhiE + R)))
say(f"A1 residual div_xy(t.v + m.nu) + G w - [Phi_E + v.(div t) + nu.(div m + eps:t)] = {resA1}")
# A2 Eq. (5) of the paper (in-plane nu) plus the decoupled axial spin
curl_half = (sp.diff(w, y) / 2, -sp.diff(w, x) / 2)
grad2 = sp.diff(w, x)**2 + sp.diff(w, y)**2
Phi5 = ((mu + ka / 2) * grad2 + 2 * ka * ((curl_half[0] - n1)**2 + (curl_half[1] - n2)**2)
        + al * (sp.diff(n1, x) + sp.diff(n2, y))**2
        + be * sum(dn[k][l] * dn[l][k] for k in range(2) for l in range(2))
        + ga * sum(dn[k][l]**2 for k in range(2) for l in range(2)))
extra_z = 2 * ka * n3**2 + ga * (sp.diff(n3, x)**2 + sp.diff(n3, y)**2)
resA2 = sp.simplify(sp.expand(PhiE - Phi5 - extra_z))
say(f"A2 residual Phi_E - [Eq. (5) + 2 kappa nu_z^2 + gamma |grad nu_z|^2] = {resA2}")
eq3 = (mu + ka) * (sp.diff(w, x, 2) + sp.diff(w, y, 2)) + ka * (sp.diff(n2, x) - sp.diff(n1, y)) + G
say(f"A3 residual t_kz,k - LHS Eq. (3) = {sp.simplify(sp.expand(lin[2] - eq3))}")
curlw = (sp.diff(w, y), -sp.diff(w, x))
divn = sp.diff(n1, x) + sp.diff(n2, y)
eq4 = [ga * (sp.diff(nu[i], x, 2) + sp.diff(nu[i], y, 2)) + (al + be) * sp.diff(divn, X[i])
       + ka * curlw[i] - 2 * ka * nu[i] for i in range(2)]
say(f"A4 residual angular momentum (4.9) x, y - LHS Eq. (4) = "
    f"{sp.simplify(sp.expand(ang[0] - eq4[0]))}, {sp.simplify(sp.expand(ang[1] - eq4[1]))}")
say(f"A4' axial spin: (4.9)_z = gamma Lap nu_z - 2 kappa nu_z, residual "
    f"{sp.simplify(sp.expand(ang[2] - (ga * (sp.diff(n3, x, 2) + sp.diff(n3, y, 2)) - 2 * ka * n3)))}")
say("   -> A1 integrated over the section with the divergence theorem and the OUTWARD normal n:")
say("      oint (t_kl n_k v_l + m_kl n_k nu_l) ds = int Phi dA - G Q + int R dA;")
say("      no-slip (v = 0) removes the first term, R = 0 for a solution: int Phi = G Q + oint m_(n).nu ds  (Eq. 8).")

# A5 exact test on the square [-1,1]^2
vals = {mu: sp.Rational(7, 5), ka: sp.Rational(3, 4), al: sp.Rational(1, 3), be: sp.Rational(-1, 5),
        ga: sp.Rational(9, 7), G: sp.Rational(5, 2)}
wt = (1 - x**2) * (1 - y**2) * (1 + x / 3 - y**2 / 2 + x * y / 4)
n1t = sp.Rational(1, 2) + x * y - y**3 / 3 + x**2 / 5
n2t = -x / 2 + x**2 * y / 3 + sp.Rational(1, 7) * y**2
n3t = sp.Rational(1, 4) - x**2 / 6 + x * y**2 / 5
p1t = x * y / 3
sub = {w: wt, n1: n1t, n2: n2t, n3: n3t, p1: p1t}


def S(e):
    return sp.expand(e.subs(sub).doit().subs(vals))


Phi_t = S(PhiE)
R_t = S(R)
mt = [[S(m[k][l]) for l in range(3)] for k in range(3)]
nut = [S(nu[k]) for k in range(3)]
edges = [((1, 0), x, 1, y), ((-1, 0), x, -1, y), ((0, 1), y, 1, x), ((0, -1), y, -1, x)]


def wall_power(sign):
    tot = 0
    for (nx, ny), var, val, par in edges:
        n_ = (sign * nx, sign * ny, 0)
        Pk = sum(mt[k][l] * n_[k] * nut[l] for k in range(3) for l in range(3))
        tot += sp.integrate(Pk.subs(var, val), (par, -1, 1))
    return sp.nsimplify(tot)


intPhi = sp.integrate(Phi_t, (x, -1, 1), (y, -1, 1))
GQ = vals[G] * sp.integrate(wt, (x, -1, 1), (y, -1, 1))
intR = sp.integrate(R_t, (x, -1, 1), (y, -1, 1))
Pout = wall_power(+1)
Pin = wall_power(-1)
say("A5 exact test, square [-1,1]^2, polynomial fields, (mu,kappa,alpha,beta,gamma,G) = (7/5,3/4,1/3,-1/5,9/7,5/2):")
say(f"   int Phi = {intPhi} = {sp.N(intPhi, 12)}")
say(f"   G Q     = {sp.N(GQ, 12)}")
say(f"   oint m_(n).nu ds, outward n = {sp.N(Pout, 12)}   (inward n: {sp.N(Pin, 12)})")
say(f"   int R dA (balance residuals of the test field) = {sp.N(intR, 12)}")
res_out = sp.simplify(intPhi - GQ - Pout + intR)
res_in = sp.simplify(intPhi - GQ - Pin + intR)
say(f"   residual int Phi - G Q - oint P + int R, outward normal: {res_out}")
say(f"   same with the inward normal (wrong orientation): {sp.N(res_in, 12)}  (not zero)")
wall_w = [sp.simplify(wt.subs(var, val)) for (_, var, val, _) in edges]
say(f"   w on the four edges: {wall_w}")

# ---------------------------------------------------------------- Part B
say("")
say("=" * 78)
say("Part B. BC3 wall power, limits, and the functional with the wall term")
say("=" * 78)
ls = sp.symbols('l_s', positive=True)
gp = sp.symbols('gamma_p', positive=True)
mx, my, mz = sp.symbols('m_x m_y m_z', real=True)
mvec = sp.Matrix([mx, my, mz])
nuv = -(ls / gp) * mvec
P_b = (mvec.T * nuv)[0]
say(f"B1 P = m_(n).nu with nu = -(l_s/gamma) m_(n): P = {sp.simplify(P_b)}")
say(f"   residual P + (l_s/gamma)|m|^2 = {sp.simplify(P_b + ls / gp * (mvec.T * mvec)[0])}")
say(f"   residual P + (gamma/l_s)|nu|^2 = {sp.simplify(P_b + gp / ls * (nuv.T * nuv)[0])}")
say("   P <= 0 for l_s > 0 and gamma > 0 (a negative multiple of a sum of squares).")
say(f"   limit l_s -> 0 at fixed m_(n): nu -> {[sp.limit(c, ls, 0) for c in nuv]},  P -> {sp.limit(P_b, ls, 0)}  (BC1: nu = 0, P = 0)")
nx_, ny_, nz_ = sp.symbols('n_x n_y n_z', real=True)
nfix = sp.Matrix([nx_, ny_, nz_])
mfromnu = -(gp / ls) * nfix
P_nu = (mfromnu.T * nfix)[0]
say(f"   limit l_s -> oo at fixed nu: m_(n) -> {[sp.limit(c, ls, sp.oo) for c in mfromnu]},  P -> {sp.limit(P_nu, ls, sp.oo)}  (BC4: m_(n) = 0, P = 0)")

# B2 bilinear identity
u = sp.Function('u')(x, y)
f1, f2, f3 = [sp.Function(f'phi{i}')(x, y) for i in (1, 2, 3)]
tX, mX, dvX, dnX = stresses([0, 0, w], [n1, n2, n3], 0)
tY, mY, dvY, dnY = stresses([0, 0, u], [f1, f2, f3], 0)
vY = [0, 0, u]
nuY = [f1, f2, f3]
nuX = [n1, n2, n3]


def phi_bil(t_, m_, dvB, dnB, nuB):
    return sum(t_[k][l] * (dvB[l][k] - sum(eps(k, l, r) * nuB[r] for r in range(3)))
               + m_[k][l] * dnB[l][k] for k in range(3) for l in range(3))


BXY = phi_bil(tX, mX, dvY, dnY, nuY)
BYX = phi_bil(tY, mY, dvX, dnX, nuX)
say(f"B2a symmetry Phi_bil(X,Y) - Phi_bil(Y,X) = {sp.simplify(sp.expand(BXY - BYX))}")
linX = [sum(sp.diff(tX[k][l], X[k]) for k in range(3)) for l in range(3)]
angX = [sum(sp.diff(mX[r][k], X[r]) for r in range(3))
        + sum(eps(k, l, r) * tX[l][r] for l in range(3) for r in range(3)) for k in range(3)]
FXY = [sum(tX[k][l] * vY[l] + mX[k][l] * nuY[l] for l in range(3)) for k in range(3)]
resB2 = sp.diff(FXY[0], x) + sp.diff(FXY[1], y) - (BXY + sum(vY[l] * linX[l] for l in range(3))
                                                   + sum(nuY[k] * angX[k] for k in range(3)))
say(f"B2b residual div_xy(t[X].v[Y] + m[X].nu[Y]) - [Phi_bil(X,Y) + v[Y].div t[X] + nu[Y].(div m[X] + eps:t[X])] = "
    f"{sp.simplify(sp.expand(resB2))}")
say("   -> for a solution X and a variation Y with u = 0 on the wall:")
say("      d/de [2G int(w+e u) - int Phi(X+eY) - (gamma/l_s) oint |nu+e phi|^2]_(e=0)")
say("        = -2 oint (m_(n)[X] + (gamma/l_s) nu) . phi ds,")
say("      so the natural wall condition is m_(n) = -(gamma/l_s) nu, i.e. nu = -(l_s/gamma) m_(n) = BC3 (sign checked in B1).")
Qs, Ws = sp.symbols('Q W_s', real=True)   # W_s = oint |nu|^2 ds
Jsol = 2 * G * Qs - (G * Qs - gp / ls * Ws) - gp / ls * Ws
say(f"   value at the solution: 2GQ - int Phi - (gamma/l_s) oint|nu|^2 with int Phi = GQ - (gamma/l_s) oint|nu|^2: "
    f"J - GQ = {sp.simplify(Jsol - G * Qs)}")

# ---------------------------------------------------------------- Part C
say("")
say("=" * 78)
say("Part C. BC2 in the tube: wall power and its sign")
say("=" * 78)
r, th = sp.symbols('r theta', positive=True)
Rr = sp.symbols('R', positive=True)
nfun = sp.Function('nu')
# Cartesian field nu = nu(r) e_theta at a wall point, contract m_kl n_k nu_l with n = e_r
xs, ys = sp.symbols('xs ys', real=True)
rr = sp.sqrt(xs**2 + ys**2)
nuc = [-nfun(rr) * ys / rr, nfun(rr) * xs / rr, 0]
Xc = (xs, ys, z)
dn_c = [[sp.diff(nuc[k], Xc[l]) for l in range(3)] for k in range(3)]
divc = sum(dn_c[q][q] for q in range(3))
m_c = [[al * divc * sp.KroneckerDelta(k, l) + be * dn_c[k][l] + ga * dn_c[l][k] for l in range(3)] for k in range(3)]
ncart = [xs / rr, ys / rr, 0]
Pc = sum(m_c[k][l] * ncart[k] * nuc[l] for k in range(3) for l in range(3))
Pc_polar = sp.simplify(Pc.subs({xs: Rr * sp.cos(th), ys: Rr * sp.sin(th)}))
Pexp = nfun(Rr) * (ga * sp.Subs(sp.Derivative(nfun(r), r), r, Rr).doit() - be * nfun(Rr) / Rr)
say(f"C1 residual m_kl n_k nu_l - nu (gamma nu' - beta nu/R) at r = R (outward n = e_r): "
    f"{sp.simplify(sp.expand(Pc_polar - Pexp))}")
say(f"   div nu for nu(r) e_theta = {sp.simplify(divc)} (alpha does not enter)")

# C2 closed-form solution, P against Eq. (14), and the sign along n
mp.mp.dps = 30


def tube_P(muv, kav, gav, bev, Gv, Rv, n):
    lam = mp.sqrt(kav * (2 * muv + kav) / (gav * (muv + kav)))
    a = lam * Rv
    e = 1 / (2 * muv + kav)
    s_hat = n / (muv + (1 - n) * kav)
    nuR = s_hat * Gv * Rv / 2
    C = (nuR - Gv * Rv * e / 2) / mp.besseli(1, a)
    nu_r = lambda rv: Gv * rv * e / 2 + C * mp.besseli(1, lam * rv)
    dnu = mp.diff(nu_r, Rv)
    P_direct = nu_r(Rv) * (gav * dnu - bev * nu_r(Rv) / Rv)
    Xv = a * mp.besseli(0, a) / mp.besseli(1, a) - 1
    AP = gav * Gv * e * (Xv - 1) / 2
    BP = (gav * Xv - bev) / Rv
    P14 = nuR * (-AP + BP * nuR)
    sstar = gav * (Xv - 1) / ((2 * muv + kav) * (gav * Xv - bev))
    nstar = sstar * (muv + kav) / (1 + sstar * kav)
    # check that nu solves the axisymmetric angular-momentum equation with the first integral
    wp = lambda rv: (-Gv * rv / 2 - kav * nu_r(rv)) / (muv + kav)
    res = gav * (mp.diff(nu_r, Rv / 2, 2) + mp.diff(nu_r, Rv / 2) / (Rv / 2) - nu_r(Rv / 2) / (Rv / 2)**2) \
        - kav * wp(Rv / 2) - 2 * kav * nu_r(Rv / 2)
    return P_direct, P14, nstar, s_hat, res


say("C2 tube, closed form nu = G r e/2 + C I1(lam r), BC2 nu_R = n (curl v)_theta; P from d nu/dr at the wall vs Eq. (14)")
say("   (mu, kappa, gamma, G, R) = (1, 1, 1, 1, 1) -> lam R = sqrt(1.5); pole n_p = 1 + mu/kappa = 2")
say(f"   {'beta':>5} {'n':>7} {'s_hat':>12} {'P (direct)':>16} {'P (Eq. 14)':>16} {'rel.diff':>9} {'ODE res':>9}  sign")
worst = mp.mpf(0)
for bev in (-1, 0, 1):
    _, _, nstar, _, _ = tube_P(mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(bev), mp.mpf(1), mp.mpf(1), mp.mpf('0.1'))
    for n in (mp.mpf('-0.5'), mp.mpf(0), nstar / 2, nstar, (nstar + 2) / 2, mp.mpf('1.9'), mp.mpf(3)):
        Pd, P14, nst, sh, res = tube_P(mp.mpf(1), mp.mpf(1), mp.mpf(1), mp.mpf(bev), mp.mpf(1), mp.mpf(1), n)
        rel = abs(Pd - P14) / max(abs(P14), mp.mpf('1e-40'))
        worst = max(worst, rel) if abs(P14) > 1e-15 else worst
        sgn = ("P=0 (n = n*, BC4 end)" if n == nstar else "P=0 (BC1)") if abs(P14) < 1e-15 \
            else ("passive P<0" if P14 < 0 else "ACTIVE P>0")
        rels = f"{float(rel):>9.1e}" if abs(P14) > 1e-15 else f"{'-':>9}"
        say(f"   {bev:>5} {float(n):>7.4f} {float(sh):>12.5e} {float(Pd):>16.8e} {float(P14):>16.8e} "
            f"{rels} {float(abs(res)):>9.1e}  {sgn}")
    say(f"   beta = {bev}: n* = {float(nstar):.6f}")
say(f"   largest relative difference P(direct) vs Eq. (14): {float(worst):.1e}")
say("   -> BC2 is passive only for 0 <= n <= n*; P > 0 for n < 0, for n* < n < n_p and for n > n_p (s_hat < 0).")
say(f"   (beta = gamma gives n* = 1/2 exactly, cf. Sec. III.)")
say("")
say(f"runtime {time.time() - t0:.1f} s")

with open(__file__.replace('.py', '_output.txt'), 'w', encoding='utf-8') as fh:
    fh.write("\n".join(out) + "\n")
