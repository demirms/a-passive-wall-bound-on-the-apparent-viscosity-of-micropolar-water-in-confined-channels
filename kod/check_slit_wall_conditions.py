"""Plane Poiseuille flow of a micropolar fluid (gap -h<y<h), closed form, several wall conditions.

Equations: (mu+kappa) u'' + kappa psi' + G = 0 ;  gamma psi'' - kappa u' - 2 kappa psi = 0
(psi = nu_z, curl_z v = -u').  First integral (symmetry): (mu+kappa) u' + kappa psi = -G y.
psi = G y/(2mu+kappa) + C sinh(lam y),  lam^2 = kappa(2mu+kappa)/(gamma(mu+kappa)).
Wall conditions at y = h (u = 0 always):
  M1  psi = 0 ;  M3  psi = -n u' (psi = n * vorticity) ;  M4  psi = -ls psi' (ls >= 0) ;  M5  psi' = 0.
Q = (2/(mu+k)) [G h^3/3 + k G h^3/(3(2mu+k)) + k C I1],  I1 = h cosh(lam h)/lam - sinh(lam h)/lam^2.
mu_app = 2 G h^3/(3 Q). Wall power into the fluid per wall P = gamma psi(h) psi'(h); passive iff P <= 0.
Symbolic part: ODE residuals for general C, Q formula, M1 limits. Numeric part: energy identity
D = G Q + 2P by quadrature, bound scans, search for passive M3 cases with mu_app > mu + kappa.
"""
import sympy as sp
import numpy as np
from scipy.integrate import quad

out = []
y, h, G = sp.symbols('y h G', positive=True)
mu, ka, ga = sp.symbols('mu kappa gamma', positive=True)
C = sp.symbols('C')
lam = sp.sqrt(ka * (2 * mu + ka) / (ga * (mu + ka)))
psi = G * y / (2 * mu + ka) + C * sp.sinh(lam * y)
up = (-G * y - ka * psi) / (mu + ka)
u = sp.integrate(up, (y, h, y))
r1 = sp.simplify((mu + ka) * sp.diff(u, y, 2) + ka * sp.diff(psi, y) + G)
r2 = sp.simplify(ga * sp.diff(psi, y, 2) - ka * sp.diff(u, y) - 2 * ka * psi)
out.append(f"S1 ODE residuals (general C): {r1}, {r2}; u(h) = {sp.simplify(u.subs(y, h))}; u'(0) = {sp.simplify(up.subs(y, 0))}")
Qs = 2 * sp.integrate(u, (y, 0, h))
L = sp.symbols('L', positive=True)
I1 = h * sp.cosh(L * h) / L - sp.sinh(L * h) / L**2
Qf = 2 / (mu + ka) * (G * h**3 / 3 + ka * G * h**3 / (3 * (2 * mu + ka)) + ka * C * I1)
out.append(f"S2 Q formula check: {sp.simplify((Qs - Qf.subs(L, lam)).rewrite(sp.exp))}")
CM1 = -G * h / ((2 * mu + ka) * sp.sinh(lam * h))
muapp = 2 * G * h**3 / (3 * Qf.subs(L, lam).subs(C, CM1))
num = {mu: 1, ka: sp.Rational(3, 2), ga: sp.Rational(7, 10), G: 1}
out.append(f"S3 M1 limit h->inf: {sp.limit(muapp.subs(num), h, sp.oo)} (expect mu+k/2 = 7/4)")
out.append(f"S3 M1 limit h->0  : {sp.limit(muapp.subs(num), h, 0)} (expect mu+k = 5/2)")


def sol(cond, H, m, k, g, n=0.0, ll=0.0, Gv=1.0):
    lm = np.sqrt(k * (2 * m + k) / (g * (m + k)))
    s, c = np.sinh(lm * H), np.cosh(lm * H)
    if cond == 'M1':
        Cv = -Gv * H / ((2 * m + k) * s)
    elif cond == 'M3':
        psih = n * Gv * H / (m + k) / (1 - n * k / (m + k))
        Cv = (psih - Gv * H / (2 * m + k)) / s
    elif cond == 'M4':
        Cv = -Gv * (H + ll) / ((2 * m + k) * (s + ll * lm * c))
    elif cond == 'M5':
        Cv = -Gv / ((2 * m + k) * lm * c)
    I1v = H * c / lm - s / lm**2
    Q = 2 / (m + k) * (Gv * H**3 / 3 + k * Gv * H**3 / (3 * (2 * m + k)) + k * Cv * I1v)
    ps = lambda yy: Gv * yy / (2 * m + k) + Cv * np.sinh(lm * yy)
    pp = lambda yy: Gv / (2 * m + k) + Cv * lm * np.cosh(lm * yy)
    P = g * ps(H) * pp(H)
    return 2 * Gv * H**3 / (3 * Q), P, Q, ps, pp, lm


def energy(cond, H, m, k, g, n=0.0, ll=0.0):
    ma, P, Q, ps, pp, lm = sol(cond, H, m, k, g, n, ll)
    upf = lambda yy: -(yy + k * ps(yy)) / (m + k)
    Phi = lambda yy: (m + k) * upf(yy)**2 + 2 * k * ps(yy)**2 + 2 * k * ps(yy) * upf(yy) + g * pp(yy)**2
    D = 2 * quad(Phi, 0, H, epsabs=1e-13, epsrel=1e-12)[0]
    return (D - (Q + 2 * P)) / D


for cond, kw in [('M1', {}), ('M3', {'n': -0.3}), ('M3', {'n': 0.3}), ('M4', {'ll': 0.5}), ('M5', {})]:
    out.append(f"N0 energy identity rel. error {cond} {kw}: {energy(cond, 1.3, 1.0, 1.5, 0.7, **kw):.2e}")

out.append("\nTable A: M1 (mu=1, gamma=1): mu+k/2 <= refined lower <= mu_app <= mu+k ;  refined = mu+k-k^2/(2k+gamma*pi^2/(4h^2))")
okA = True
for k in [0.1, 1.0, 10.0]:
    lm = np.sqrt(k * (2 + k) / (1 + k))
    for Hl in [0.01, 0.1, 1, 3, 10, 100]:
        H = Hl / lm
        m1, P = sol('M1', H, 1.0, k, 1.0)[:2]
        low = 1 + k - k**2 / (2 * k + np.pi**2 / (4 * H**2))
        ok = (1 + k / 2 <= low + 1e-12) and (low <= m1 + 1e-12) and (m1 <= 1 + k + 1e-12)
        okA &= ok
        out.append(f"  k/mu={k:5.1f} h*lam={Hl:7.2f}: mu_app={m1:.6f} | mu+k/2={1+k/2:.4f} refined={low:.6f} mu+k={1+k:.4f} | ok={ok} P={P:.1e}")
out.append(f"Table A all ok: {okA}")

out.append("\nTable B: M3 (psi = -n u'), mu=1, gamma=1. passive iff P<=0.")
viol = []
for k in [0.5, 1.0, 10.0]:
    lm = np.sqrt(k * (2 + k) / (1 + k))
    for Hl in [0.3, 1, 3, 10, 30]:
        H = Hl / lm
        for nn in [-1.0, -0.5, -0.2, -0.05, 0.05, 0.2, 0.45]:
            m3, P = sol('M3', H, 1.0, k, 1.0, nn)[:2]
            passive = P <= 1e-14
            fl = ('  >mu+k' if m3 > 1 + k + 1e-12 else '') + ('  <mu+k/2' if m3 < 1 + k / 2 - 1e-12 else '')
            if m3 > 1 + k + 1e-12 and passive:
                viol.append((k, Hl, nn, m3 / (1 + k), P))
            out.append(f"  k={k:5.1f} h*lam={Hl:5.1f} n={nn:6.2f}: mu_app/(mu+k)={m3/(1+k):.5f} mu_app/(mu+k/2)={m3/(1+k/2):.5f} P={P:+.3e} {'passive' if passive else 'ACTIVE'}{fl}")
out.append(f"PASSIVE M3 cases with mu_app > mu+k: {len(viol)}")
for v in viol:
    out.append(f"   k={v[0]} h*lam={v[1]} n={v[2]} mu_app/(mu+k)={v[3]:.6f} P={v[4]:.3e}")
cnt, best, nact_hi = 0, None, 0
for k in np.geomspace(0.05, 20, 25):
    lm = np.sqrt(k * (2 + k) / (1 + k))
    for Hl in np.geomspace(0.05, 100, 40):
        for nn in np.linspace(-3, 0.5, 71):
            if abs(1 - nn * k / (1 + k)) < 1e-6:
                continue
            m3, P = sol('M3', Hl / lm, 1.0, k, 1.0, nn)[:2]
            if m3 > 1 + k:
                if P <= 0:
                    cnt += 1
                    r = m3 / (1 + k)
                    if best is None or r > best[0]:
                        best = (r, k, Hl, nn, P)
                else:
                    nact_hi += 1
out.append(f"Fine scan (25 k x 40 h x 71 n in [-3,0.5]): mu_app>mu+k with passive wall: {cnt}; with active wall: {nact_hi}; max passive ratio {best}")

out.append("\nTable C: M4 (psi = -ls psi'), M5, mu=1, gamma=1")
bad = 0
for k in [0.1, 1.0, 10.0]:
    lm = np.sqrt(k * (2 + k) / (1 + k))
    for Hl in [0.01, 0.1, 1, 10, 100]:
        H = Hl / lm
        row = []
        for ll in [0.1, 1.0, 10.0]:
            m4, P = sol('M4', H, 1.0, k, 1.0, 0, ll * H)[:2]
            row.append(m4)
            bad += (not (1 + k / 2 - 1e-12 <= m4 <= 1 + k + 1e-12)) or P > 1e-14
        m5, P5 = sol('M5', H, 1.0, k, 1.0)[:2]
        bad += (not (1 + k / 2 - 1e-12 <= m5 <= 1 + k + 1e-12)) or abs(P5) > 1e-12
        m1 = sol('M1', H, 1.0, k, 1.0)[0]
        out.append(f"  k={k:5.1f} h*lam={Hl:7.2f}: M1/(mu+k)={m1/(1+k):.5f}  M4(ls/h=0.1,1,10)/(mu+k)=" + ", ".join(f"{r/(1+k):.5f}" for r in row) + f"  M5/(mu+k)={m5/(1+k):.5f}")
out.append(f"M4/M5 bound or passivity violations: {bad}")

out.append("\nSlip-equivalent length of the M1 wall layer: b defined by mu_app/(mu+k/2) = 1 - 3 b/h (h -> inf)")
for k in [0.41, 1.0, 10.0]:
    lm = np.sqrt(k * (2 + k) / (1 + k))
    for Hl in [100.0, 200.0, 400.0]:
        H = Hl / lm
        m1 = sol('M1', H, 1.0, k, 1.0)[0]
        b = -(H / 3) * (m1 / (1 + k / 2) - 1)
        out.append(f"  k/mu={k}: h*lam={Hl}: b*lam = {b*lm:.6f}")

with open(__file__.replace('.py', '_output.txt'), 'w', encoding='utf-8') as f:
    for line in out:
        print(line)
        f.write(line + "\n")
