"""Tube: Fourier orthogonality of the dissipation and of the wall power.

Symbolic (sympy), general radial profiles (undefined functions of r), no numerical solution needed.
Fields in polar components, d/dz = 0:
  forced part X0 :  w0(r) e_z,  nu = g0(r) e_theta                      (axisymmetric azimuthal part, carries G Q)
  rest  Y0       :  nu = f0(r) e_r + h0(r) e_z                          (axisymmetric radial and axial spin)
        Ym, m>=1 :  w, nu_r, nu_theta, nu_z = (c(r) cos m theta + s(r) sin m theta), 8 free profiles
Checks (theta-integrals are exact: cos/sin -> Laurent polynomial in E = exp(i theta), keep the E^0 coefficient):
  O1  int_0^2pi [Phi(X0+Ym) - Phi(X0) - Phi(Ym)] dtheta = 0, m = 1, 2, 3
  O2  Phi(X0+Y0) - Phi(X0) - Phi(Y0) = 0 pointwise
  O3  wall power cross terms at r = R: theta-integral 0 (m >= 1), pointwise 0 (Y0)
  O4  int_0^2pi w(Ym) dtheta = 0 (the rest carries no flow rate)
  O5  the operator of Eqs. (3)-(4) commutes with the theta-average (projection on m = 0):
      theta-average of L(Ym) = 0 and L(X0 + Y0) is theta-independent; L(X0) has no r- or z-spin component,
      L(Y0) has no z-momentum or theta-spin component (the m = 0 radial/axial spin is unforced and decoupled)
  O6  pointwise form of Phi in the section: Phi = (mu+k/2)|grad w|^2 + 2k|curl(w e_z)/2 - nu_in|^2 + 2k nu_z^2
      + [(2a+b+g)/2 tr A^2 + (g+b)|A^os|^2 + (g-b)|A^a|^2] + g |grad nu_z|^2, A = in-plane spin gradient -> residual 0
Together with the pointwise energy identity of check_wall_power_sign_conventions.py (A1) these give the
step "integral passivity of the whole field => passivity of the axisymmetric part" used after Eq. (16).
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import time
import sympy as sp

t0 = time.time()
out = []


def say(s=""):
    print(s)
    out.append(s)


r, th, R = sp.symbols('r theta R', positive=True)
mu, ka, al, be, ga, G = sp.symbols('mu kappa alpha beta gamma G', real=True)
E = sp.symbols('E')
eps = sp.LeviCivita
c, s = sp.cos(th), sp.sin(th)


def dx(f):
    return c * sp.diff(f, r) - s / r * sp.diff(f, th)


def dy(f):
    return s * sp.diff(f, r) + c / r * sp.diff(f, th)


def D(f, l):
    return [dx, dy, lambda q: 0][l](f)


def field(w, nr, nt, nz):
    """Cartesian (v, nu) from polar components."""
    return [0, 0, w], [nr * c - nt * s, nr * s + nt * c, nz]


def stresses(v, nu):
    dv = [[D(v[k], l) for l in range(3)] for k in range(3)]
    dn = [[D(nu[k], l) for l in range(3)] for k in range(3)]
    divn = dn[0][0] + dn[1][1]
    t = [[mu * (dv[k][l] + dv[l][k]) + ka * (dv[l][k] - sum(eps(k, l, m) * nu[m] for m in range(3)))
          for l in range(3)] for k in range(3)]
    m = [[al * divn * sp.KroneckerDelta(k, l) + be * dn[k][l] + ga * dn[l][k] for l in range(3)] for k in range(3)]
    return t, m, dv, dn


def phi_bil(FX, FY):
    (vX, nX), (vY, nY) = FX, FY
    tX, mX, _, _ = stresses(vX, nX)
    _, _, dvY, dnY = stresses(vY, nY)
    return sum(tX[k][l] * (dvY[l][k] - sum(eps(k, l, q) * nY[q] for q in range(3))) + mX[k][l] * dnY[l][k]
               for k in range(3) for l in range(3))


def wall_bil(FX, FY):
    (vX, nX), (vY, nY) = FX, FY
    _, mX, _, _ = stresses(vX, nX)
    n = [c, s, 0]
    return sum(mX[k][l] * n[k] * nY[l] for k in range(3) for l in range(3))


def to_laurent(e):
    def rep(q):
        k = sp.simplify(q.args[0] / th)
        if isinstance(q, sp.cos):
            return (E**k + E**(-k)) / 2
        return (E**k - E**(-k)) / (2 * sp.I)
    return sp.expand(e.replace(lambda q: isinstance(q, (sp.cos, sp.sin)) and q.args[0].has(th), rep))


def theta_integral(e):
    """Exact int_0^2pi e dtheta for e polynomial in cos(k theta), sin(k theta)."""
    lp = to_laurent(e)
    tot = 0
    for term in sp.Add.make_args(lp):
        coeff, ex = term.as_coeff_exponent(E)
        if ex == 0:
            tot += coeff
    return sp.simplify(2 * sp.pi * tot)


def operator(F):
    """Left sides of z-momentum (3) and angular momentum (4) (Cartesian x, y, z), G = 0."""
    v, nu = F
    t, m, _, _ = stresses(v, nu)
    zmom = sum(D(t[k][2], k) for k in range(3))
    ang = [sum(D(m[q][k], q) for q in range(3)) + sum(eps(k, l, q) * t[l][q] for l in range(3) for q in range(3))
           for k in range(3)]
    # polar components of the spin balance
    return [zmom, ang[0] * c + ang[1] * s, -ang[0] * s + ang[1] * c, ang[2]]


w0, g0, f0, h0 = [sp.Function(n)(r) for n in ('w0', 'g0', 'f0', 'h0')]
X0 = field(w0, 0, g0, 0)
Y0 = field(0, f0, 0, h0)


def mode(m):
    prof = {q: sp.Function(f'{q}{m}')(r) for q in ('Wc', 'Ws', 'Ac', 'As', 'Bc', 'Bs', 'Cc', 'Cs')}
    cm, sm = sp.cos(m * th), sp.sin(m * th)
    return field(prof['Wc'] * cm + prof['Ws'] * sm, prof['Ac'] * cm + prof['As'] * sm,
                 prof['Bc'] * cm + prof['Bs'] * sm, prof['Cc'] * cm + prof['Cs'] * sm)


say("=" * 78)
say("Tube: theta-orthogonality of dissipation and wall power (general radial profiles)")
say("=" * 78)
allzero = True
for m in (1, 2, 3):
    Ym = mode(m)
    crossD = phi_bil(X0, Ym) + phi_bil(Ym, X0)
    sym = sp.simplify(sp.expand(phi_bil(X0, Ym) - phi_bil(Ym, X0)))
    O1 = theta_integral(sp.expand(crossD))
    crossP = (wall_bil(X0, Ym) + wall_bil(Ym, X0)).subs(r, R)
    O3 = theta_integral(sp.expand(crossP.doit()))
    O4 = theta_integral(Ym[0][2])
    say(f"m = {m}: O1 int dtheta [dissipation cross term] = {O1};  O3 int dtheta [wall-power cross term at r=R] = {O3};"
        f"  O4 int dtheta w = {O4};  (Phi_bil symmetric: {sym})")
    allzero &= (O1 == 0 and O3 == 0 and O4 == 0 and sym == 0)
crossD0 = sp.simplify(sp.expand(phi_bil(X0, Y0) + phi_bil(Y0, X0)))
crossP0 = sp.simplify(sp.expand((wall_bil(X0, Y0) + wall_bil(Y0, X0)).subs(r, R).doit()))
say(f"m = 0 radial/axial spin Y0: O2 pointwise dissipation cross term = {crossD0};  O3 pointwise wall cross term = {crossP0}")
allzero &= (crossD0 == 0 and crossP0 == 0)

say("")
say("O5 operator and theta-average (projection on m = 0)")
for m in (1, 2, 3):
    Lm = operator(mode(m))
    avg = [theta_integral(sp.expand(q)) for q in Lm]
    say(f"   m = {m}: theta-integral of [z-momentum, spin_r, spin_theta, spin_z] of the mode = {avg}")
    allzero &= all(a == 0 for a in avg)
L0 = [sp.simplify(sp.expand(q)) for q in operator(X0)]
L00 = [sp.simplify(sp.expand(q)) for q in operator(Y0)]
dth = [sp.simplify(sp.diff(q, th)) for q in L0 + L00]
say(f"   L(X0): spin_r = {L0[1]}, spin_z = {L0[3]}   (forced part has only z-momentum and spin_theta)")
say(f"   L(Y0): z-momentum = {L00[0]}, spin_theta = {L00[2]}   (radial/axial spin decoupled and unforced)")
say(f"   d/dtheta of all components of L(X0), L(Y0) = {dth}")
allzero &= (L0[1] == 0 and L0[3] == 0 and L00[0] == 0 and L00[2] == 0 and all(q == 0 for q in dth))
zm = sp.simplify(L0[0] - ((mu + ka) * (sp.diff(w0, r, 2) + sp.diff(w0, r) / r) + ka / r * sp.diff(r * g0, r)))
st = sp.simplify(L0[2] - (ga * (sp.diff(g0, r, 2) + sp.diff(g0, r) / r - g0 / r**2) - ka * sp.diff(w0, r) - 2 * ka * g0))
say(f"   L(X0) against the tube equations, step 1 (residuals): {zm}, {st}")
allzero &= (zm == 0 and st == 0)

say("")
say("O6 pointwise form of the dissipation in the section (Cartesian, general fields)")
x, y = sp.symbols('x y', real=True)
wf = sp.Function('w')(x, y)
a1, a2, a3 = [sp.Function(f'n{i}')(x, y) for i in (1, 2, 3)]
Xc = (x, y, sp.Symbol('zz'))


def Dc(f, l):
    return sp.diff(f, Xc[l]) if l < 2 else 0


vC, nC = [0, 0, wf], [a1, a2, a3]
dv = [[Dc(vC[k], l) for l in range(3)] for k in range(3)]
dn = [[Dc(nC[k], l) for l in range(3)] for k in range(3)]
divn = dn[0][0] + dn[1][1]
tC = [[mu * (dv[k][l] + dv[l][k]) + ka * (dv[l][k] - sum(eps(k, l, m) * nC[m] for m in range(3))) for l in range(3)]
      for k in range(3)]
mC = [[al * divn * sp.KroneckerDelta(k, l) + be * dn[k][l] + ga * dn[l][k] for l in range(3)] for k in range(3)]
PhiC = sum(tC[k][l] * (dv[l][k] - sum(eps(k, l, q) * nC[q] for q in range(3))) + mC[k][l] * dn[l][k]
           for k in range(3) for l in range(3))
A = sp.Matrix(2, 2, lambda i, j: dn[i][j])
trA = A.trace()
Aos = (A + A.T) / 2 - trA / 2 * sp.eye(2)
Aa = (A - A.T) / 2
fro = lambda M: sum(q**2 for q in M)
form = ((mu + ka / 2) * (sp.diff(wf, x)**2 + sp.diff(wf, y)**2)
        + 2 * ka * ((sp.diff(wf, y) / 2 - a1)**2 + (-sp.diff(wf, x) / 2 - a2)**2) + 2 * ka * a3**2
        + (2 * al + be + ga) / 2 * trA**2 + (ga + be) * fro(Aos) + (ga - be) * fro(Aa)
        + ga * (sp.diff(a3, x)**2 + sp.diff(a3, y)**2))
O6 = sp.simplify(sp.expand(PhiC - form))
say(f"   residual Phi - [sum of the squares above] = {O6}")
say("   -> Phi >= 0 pointwise for 2mu+k >= 0, k >= 0, g >= |b|, 2a+b+g >= 0 (2a+b+g = (2/3)(3a+b+g) + (1/3)(b+g)).")
allzero &= (O6 == 0)

say("")
say(f"ALL ZERO: {allzero}")
say("Consequence (with the pointwise energy identity A1 of check_wall_power_sign_conventions.py):")
say("  the theta-average of a solution solves the forced problem (O5), so solution = X0 + rest with the rest unforced;")
say("  int Phi = int Phi(X0) + int Phi(rest) and oint P = P(X0) + P(rest) (O1-O3); G Q comes from X0 only (O4);")
say("  rest: int Phi(rest) = oint P(rest) >= 0 (O6); hence oint P <= 0  =>  P(X0) = oint P - int Phi(rest) <= 0,")
say("  i.e. the wall spin of the axisymmetric part lies in the passive range (15), and (16) applies.")
say(f"runtime {time.time() - t0:.1f} s")

with open(__file__.replace('.py', '_output.txt'), 'w', encoding='utf-8') as fh:
    fh.write("\n".join(out) + "\n")
