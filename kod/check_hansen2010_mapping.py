"""Map the constitutive relations of Hansen et al. 2010 (J. Chem. Phys. 133, 144906) onto Eringen's
micropolar fluid, term by term, and evaluate the consequences of the resulting beta for the tube.

Source relations (Hansen 2010, read from the page images):
  Eq. (1)  P^d  = -eta_r (curl u - 2 Omega)                 (vector dual of antisymmetric pressure tensor)
  Eq. (2)  Q^os = -2 zeta0 (grad Omega)^os,  Q^d = -zeta_r curl Omega
  Eq. (12) P^d  = (P^a_yz, P^a_zx, P^a_xy)  ->  P^a_ij = eps_ijk P^d_k (same convention for Q)
  Eq. (22a,b) slit balances with zeta = zeta0 + zeta_r.
  The symmetric traceless shear part is P^os = -2 eta0 (grad u)^os; the trace part of Q
  (bulk spin viscosity zeta_v) is not reported in Hansen 2010 and is kept as a free symbol.
Conventions: momentum balance rho du/dt = -div P with (div P)_j = d_i P_ij (Irving-Kirkwood);
spin balance rho I dOmega/dt = -div Q - 2 P^d. Eringen: t_kl,k + rho f_l = 0, m_kl,k + eps_lmn t_mn = 0,
with t_kl and m_kl as in Eqs. (1)-(2) of the manuscript. Hence t = -P and m = -Q component-wise.

Units for numbers: mPa s, nm (zeta in 1e-21 kg m/s = mPa s nm^2).
"""
import sympy as sp
import numpy as np
from paper_common import TABLE2, eringen, lam, tube_muapp, s_star

print("=" * 78)
print("Part A. Symbolic term-by-term mapping (sympy)")
print("=" * 78)
x = sp.symbols('x0:3', real=True)
mu, ka, al, be, ga = sp.symbols('mu kappa alpha beta gamma', real=True)
e0, er, z0, zr, zv = sp.symbols('eta0 eta_r zeta0 zeta_r zeta_v', real=True)
p = sp.Function('p')(*x)
u = [sp.Function(f'u{i}')(*x) for i in range(3)]
Om = [sp.Function(f'O{i}')(*x) for i in range(3)]
eps = sp.LeviCivita
d = lambda f, i: sp.diff(f, x[i])
I3 = sp.eye(3)


def curl(v):
    return [sum(eps(j, i, k) * d(v[k], i) for i in range(3) for k in range(3)) for j in range(3)]


Gu = sp.Matrix(3, 3, lambda i, j: d(u[j], i))        # (grad u)_ij = d_i u_j
GO = sp.Matrix(3, 3, lambda i, j: d(Om[j], i))       # (grad Omega)_ij = d_i Omega_j
sym0 = lambda A: (A + A.T) / 2 - A.trace() / 3 * I3
div_u = Gu.trace()

# Hansen tensors
Pd = [-er * (curl(u)[k] - 2 * Om[k]) for k in range(3)]
Pa = sp.Matrix(3, 3, lambda i, j: sum(eps(i, j, k) * Pd[k] for k in range(3)))
# incompressible fluid: (grad u)^os and the symmetric part differ only by (div u)/3 I = 0
P = p * I3 - e0 * (Gu + Gu.T) + Pa
Qd = [-zr * curl(Om)[k] for k in range(3)]
Qa = sp.Matrix(3, 3, lambda i, j: sum(eps(i, j, k) * Qd[k] for k in range(3)))
Q = -zv * GO.trace() * I3 - 2 * z0 * sym0(GO) + Qa

# check the dual convention of Eq. (12): P^d_x = P^a_yz
print("Eq.(12) convention P^d = (P^a_yz, P^a_zx, P^a_xy):",
      [sp.simplify(Pa[1, 2] - Pd[0]), sp.simplify(Pa[2, 0] - Pd[1]), sp.simplify(Pa[0, 1] - Pd[2])])

# Eringen tensors (manuscript Eqs. 1-2): t_kl, m_kl ; v_{l,k} = d_k v_l = Gu[k,l]
t = sp.Matrix(3, 3, lambda k, l: -p * I3[k, l] + mu * (Gu[l, k] + Gu[k, l])
              + ka * (Gu[k, l] - sum(eps(k, l, m) * Om[m] for m in range(3))))
m = sp.Matrix(3, 3, lambda k, l: al * GO.trace() * I3[k, l] + be * GO[l, k] + ga * GO[k, l])

# solve t + P = 0 and m + Q = 0 identically for the Eringen coefficients
Rt = (t + P).applyfunc(sp.expand)
Rm = (m + Q).applyfunc(sp.expand)
derivs = set()
for R in (Rt, Rm):
    for e in R:
        derivs |= e.atoms(sp.Derivative)
eqs = []
for R in (Rt, Rm):
    for e in R:
        e2 = e.subs({dd: sp.Symbol(f'D{n}') for n, dd in enumerate(sorted(derivs, key=str))})
        e2 = e2.subs({Om[i]: sp.Symbol(f'W{i}') for i in range(3)})
        syms = [s for s in e2.free_symbols if str(s)[0] in 'DW']
        if syms:
            eqs += sp.Poly(e2, *syms).coeffs()
sol = sp.solve(list(set(eqs)), [mu, ka, al, be, ga], dict=True)
print("unique solution of t = -P, m = -Q for (mu, kappa, alpha, beta, gamma):")
for s in sol:
    for k_, v_ in s.items():
        print(f"   {k_} = {v_}")
S = sol[0]
print("residual t + P with this map:", set(sp.simplify(e.subs(S)) for e in Rt))
print("residual m + Q with this map:", set(sp.simplify(e.subs(S)) for e in Rm))

# balance laws: Eringen vs Hansen (no inertia, no body force/couple)
mom_E = [sum(d(t[k, l], k) for k in range(3)) for l in range(3)]
mom_H = [-sum(d(P[i, j], i) for i in range(3)) for j in range(3)]
spin_E = [sum(d(m[k, l], k) for k in range(3)) + sum(eps(l, a_, b_) * t[a_, b_] for a_ in range(3) for b_ in range(3))
          for l in range(3)]
spin_H = [-sum(d(Q[i, j], i) for i in range(3)) - 2 * Pd[j] for j in range(3)]
print("momentum balance residual (Eringen - Hansen):",
      set(sp.simplify((mom_E[j] - mom_H[j]).subs(S)) for j in range(3)))
print("spin balance residual (Eringen - Hansen):",
      set(sp.simplify((spin_E[j] - spin_H[j]).subs(S)) for j in range(3)))

# reduction to Hansen Eq. (22): u = u_x(z) e_x, Omega = Omega_y(z) e_y, steady, no force, p = 0
uxf, Wyf = sp.Function('ux'), sp.Function('Wy')
rep_fun = {u[0]: uxf(x[2]), u[1]: sp.S(0), u[2]: sp.S(0), Om[0]: sp.S(0), Om[1]: Wyf(x[2]), Om[2]: sp.S(0), p: sp.S(0)}
mE = sp.simplify(mom_E[0].subs(S).subs(rep_fun).doit())
sE = sp.simplify(spin_E[1].subs(S).subs(rep_fun).doit())
l22a = ((e0 + er) * sp.diff(uxf(x[2]), x[2], 2) - 2 * er * sp.diff(Wyf(x[2]), x[2]))
l22b = (2 * er * (sp.diff(uxf(x[2]), x[2]) - 2 * Wyf(x[2])) + (z0 + zr) * sp.diff(Wyf(x[2]), x[2], 2))
print("slit reduction vs Hansen Eq. (22a), (22b) [zeta = zeta0 + zeta_r]:",
      sp.simplify(mE - l22a), sp.simplify(sE - l22b))

# second-law check: gamma +- beta and 3 alpha + beta + gamma in Hansen's coefficients
print("gamma + beta =", sp.simplify((ga + be).subs(S)), "; gamma - beta =", sp.simplify((ga - be).subs(S)),
      "; 3 alpha + beta + gamma =", sp.simplify((3 * al + be + ga).subs(S)),
      "; alpha + beta =", sp.simplify((al + be).subs(S)))

print()
print("=" * 78)
print("Part B. Numbers from Table II: beta = zeta0 - zeta_r, gamma = zeta0 + zeta_r")
print("=" * 78)
# standard errors of Table II (zeta0, zeta_r), page 144906-5
SE = {284: (0.1, 0.2), 289: (0.1, 0.2), 293: (0.2, 0.1), 299: (0.1, 0.2),
      305: (0.09, 0.2), 309: (0.07, 0.2), 313: (0.1, 0.1), 319: (0.1, 0.08)}
print(" T[K]  zeta0  zeta_r  gamma  beta    s.e.(beta)  beta/gamma   (gamma-|beta|)>0")
for T, a0, ar, q0, qr in TABLE2:
    s0, sr = SE[T]
    b = q0 - qr
    g = q0 + qr
    print(f" {T}  {q0:5.2f}  {qr:5.2f}  {g:5.2f}  {b:+.2f}   {np.hypot(s0, sr):.2f}        {b/g:+.4f}     {g-abs(b)>0}")
print("Fig. 3 coefficients (298 K): zeta0 = 1.1, zeta_r = 1.4 -> gamma = 2.5, beta = -0.3, beta/gamma = -0.12")

print()
print("=" * 78)
print("Part C. Tube: passive range n* and BC4 (M5) value with beta = zeta0 - zeta_r vs beta = 0")
print("        (BC1 value does not depend on beta; slit M1 does not depend on alpha, beta)")
print("=" * 78)


def nstar(m_, k_, a_, bg):
    s = s_star(m_, k_, a_, bg)
    return s * (m_ + k_) / (1 + s * k_)


radii = (40, 50, 60, 100, 200, 300, 500)
print(" T[K] r[nm]  lamR    BC1/eta0   BC4(b=0)/eta0  BC4(b)/eta0   n*(b=0)   n*(b)    dn*")
allr = {'n0': [], 'nb': [], 'm50': [], 'm5b': []}
for T, a0, ar, q0, qr in TABLE2[:4]:
    mu_, k_, g_ = eringen(a0, ar, q0 + qr)
    bg = (q0 - qr) / (q0 + qr)
    L = lam(mu_, k_, g_)
    for r in radii:
        a_ = L * r
        m1 = tube_muapp(mu_, k_, a_, 0.0) / a0
        m50 = tube_muapp(mu_, k_, a_, s_star(mu_, k_, a_, 0.0)) / a0
        m5b = tube_muapp(mu_, k_, a_, s_star(mu_, k_, a_, bg)) / a0
        n0, nb = nstar(mu_, k_, a_, 0.0), nstar(mu_, k_, a_, bg)
        if r <= 60:
            allr['n0'].append(n0); allr['nb'].append(nb); allr['m50'].append(m50); allr['m5b'].append(m5b)
        print(f" {T}  {r:4d}  {a_:6.2f}   {m1:.5f}    {m50:.5f}        {m5b:.5f}      {n0:.4f}   {nb:.4f}  {nb-n0:+.4f}")
print(f"r = 40-60 nm, 284-299 K: n*(b=0) = {min(allr['n0']):.4f}-{max(allr['n0']):.4f};"
      f" n*(b) = {min(allr['nb']):.4f}-{max(allr['nb']):.4f};"
      f" BC4(b=0) = {min(allr['m50']):.4f}-{max(allr['m50']):.4f}; BC4(b) = {min(allr['m5b']):.4f}-{max(allr['m5b']):.4f}")
# range over r = 40-500 nm for the n = 1/2 statement
lo = {'0': 1, 'b': 1}; hi = {'0': 0, 'b': 0}
for T, a0, ar, q0, qr in TABLE2[:4]:
    mu_, k_, g_ = eringen(a0, ar, q0 + qr)
    bg = (q0 - qr) / (q0 + qr)
    for r in np.linspace(40, 500, 461):
        a_ = lam(mu_, k_, g_) * r
        for key, bb in (('0', 0.0), ('b', bg)):
            v = nstar(mu_, k_, a_, bb); lo[key] = min(lo[key], v); hi[key] = max(hi[key], v)
print(f"r = 40-500 nm, 284-299 K: n*(b=0) in [{lo['0']:.4f}, {hi['0']:.4f}], n*(b=zeta0-zeta_r) in [{lo['b']:.4f}, {hi['b']:.4f}]")
# sensitivity: beta at +-1 s.e.
print("beta/gamma at central value -+ 1 s.e. (284 and 299 K, r = 40 nm): n*")
for T, a0, ar, q0, qr in (TABLE2[0], TABLE2[3]):
    mu_, k_, g_ = eringen(a0, ar, q0 + qr)
    a_ = lam(mu_, k_, g_) * 40
    s0, sr = SE[T]
    for bb in (q0 - qr - np.hypot(s0, sr), q0 - qr, q0 - qr + np.hypot(s0, sr)):
        print(f"   T={T} beta={bb:+.3f} beta/gamma={bb/(q0+qr):+.4f}  n*={nstar(mu_, k_, a_, bb/(q0+qr)):.4f}")

print()
print("=" * 78)
print("Part D. Coefficient sets in one unit system (mPa s, mPa s nm^2 = 1e-21 kg m/s, nm)")
print("=" * 78)
sets = [("Hansen 2010 Table II, 284 K (MD-MD)", 0.92, 0.17, 3.0),
        ("Hansen 2010 Table II, 299 K (MD-MD)", 0.70, 0.17, 2.1),
        ("Hansen 2010 Fig. 3 / Eq. (23), 298 K", 1.00, 0.17, 2.5),
        ("Hansen 2015 SI Table I, 298.7 K", 0.70, 0.17, 2.1),
        ("Hansen 2011a Table I, 328.4 K (L=3.57 nm)", 0.32, 0.1678, 1.01),
        ("Bonthuis 2009: eta=1.0, eta_r/eta~0.1, zeta=a^2 eta_r, a=0.3 nm", 1.00, 0.10, 0.3 ** 2 * 0.10),
        ("Hansen 2010 note 9: Bertolini-Tani eta_r with zeta ~ 9e-24", 1.00, 0.19, 0.009)]
print(" set                                                           eta0   eta_r  zeta     eta_r/eta0  1/lam[nm]  bound 1+eta_r/eta0")
for name, a0, ar, z in sets:
    mu_, k_, g_ = eringen(a0, ar, z)
    print(f" {name:62s} {a0:5.2f}  {ar:6.4f} {z:7.4f}  {ar/a0:8.3f}   {1/lam(mu_, k_, g_):7.3f}    {1+ar/a0:.3f}")
print(" Bonthuis' own spin screening length (eta_r/eta << 1): a/2 = 0.15 nm")
print(f" ratio of spin viscosities Hansen 299 K / Bonthuis: {2.1/(0.3**2*0.10):.0f}")
