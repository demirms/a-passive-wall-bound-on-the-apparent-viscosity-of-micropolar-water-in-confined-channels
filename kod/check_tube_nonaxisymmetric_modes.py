"""Which coefficient conditions does the non-axisymmetric-mode argument of the tube bound need?

Claim checked (manuscript, tube paragraph after Eq. (16)): the pressure gradient forces only the
axisymmetric azimuthal part; every other Fourier mode / component is unforced, its wall power equals its
dissipation (energy identity without G Q) and is therefore >= 0; a passive wall (total wall power <= 0)
then leaves the axisymmetric part with wall power <= 0, i.e. inside the passive range (15).

Steps:
 1. sympy: Phi_g = (3a+b+g)/3 (tr G)^2 + (g+b) |G^os|^2 + (g-b) |G^a|^2 for a general 3x3 gradient G,
    so Phi_g >= 0 for every G  <=>  3a+b+g >= 0, g+b >= 0, g-b >= 0  (i.e. g >= |b|).
 2. Necessity inside the fully developed setting (d/dz = 0, nu_z = 0): the three parts are realised by
    the in-plane fields (x, y), (x, -y), (-y, x).
 3. Exact unforced solutions of Eqs. (3)-(4) in the tube (G = 0, w = 0 on the wall) for mode m >= 1:
    (A) irrotational nu = grad(phi), phi = I_m(c r) cos(m t), w = 0, c^2 = 2k/(a+b+g);
    (B) solenoidal  nu = curl(psi e_z), psi = A I_m(lam r) cos(m t) + B r^m cos(m t),
        w = C I_m(lam r) cos(m t) + D r^m cos(m t), C = k A/(mu+k), B = D/2, w(R) = 0.
    Residuals of Eqs. (3)-(4) at random points; energy identity  int Phi = G Q + wall power;
    with valid coefficients int Phi >= 0; counter-examples with g+b < 0 (A) and g-b < 0 (B) give
    int Phi < 0, i.e. a mode that a passive wall (wall power < 0) can sustain.
 4. Cross terms: dissipation and wall power of (forced axisymmetric BC1 solution + mode) equal the sums
    of the parts, so only the integral (total) passivity of the wall is used.
"""
import numpy as np
import sympy as sp
from scipy.special import iv

print("=" * 78)
print("1. Decomposition of the spin-gradient dissipation (sympy, general 3x3 G)")
print("=" * 78)
a, b, g = sp.symbols('alpha beta gamma', real=True)
Gm = sp.Matrix(3, 3, sp.symbols('G0:9', real=True))     # G_kl = nu_{l,k}
trG = Gm.trace()
Gs = (Gm + Gm.T) / 2 - trG / 3 * sp.eye(3)
Ga = (Gm - Gm.T) / 2
dd = lambda A, B: sum(A[i, j] * B[i, j] for i in range(3) for j in range(3))
# Phi_g = a (nu_{r,r})^2 + b nu_{k,l} nu_{l,k} + g nu_{k,l} nu_{k,l}
phig = a * trG ** 2 + b * dd(Gm, Gm.T) + g * dd(Gm, Gm)
dec = (3 * a + b + g) / 3 * trG ** 2 + (g + b) * dd(Gs, Gs) + (g - b) * dd(Ga, Ga)
print("residual Phi_g - [(3a+b+g)/3 trG^2 + (g+b)|G^os|^2 + (g-b)|G^a|^2] =", sp.expand(phig - dec))

print()
print("=" * 78)
print("2. The three parts inside the fully developed setting (in-plane nu, d/dz = 0)")
print("=" * 78)
X, Y = sp.symbols('x y', real=True)
for name, nu in (("trace     nu=(x, y)", (X, Y)), ("sym-trless nu=(x,-y)", (X, -Y)), ("antisym   nu=(-y, x)", (-Y, X))):
    G2 = sp.zeros(3, 3)
    for k_, var in enumerate((X, Y)):
        for l_ in range(2):
            G2[k_, l_] = sp.diff(nu[l_], var)
    val = phig.subs({Gm[i]: G2[i // 3, i % 3] for i in range(9)})
    print(f"   {name}: Phi_g = {sp.factor(val)}")
A2 = sp.Matrix(2, 2, sp.symbols('A0:4', real=True))
trA = A2.trace()
A2s = (A2 + A2.T) / 2 - trA / 2 * sp.eye(2)
A2a = (A2 - A2.T) / 2
d2 = lambda P_, Q_: sum(P_[i, j] * Q_[i, j] for i in range(2) for j in range(2))
phig2 = a * trA ** 2 + b * d2(A2, A2.T) + g * d2(A2, A2)
dec2 = (2 * a + b + g) / 2 * trA ** 2 + (g + b) * d2(A2s, A2s) + (g - b) * d2(A2a, A2a)
print("   in-plane gradient (d/dz = 0, nu_z = 0): residual Phi_g - [(2a+b+g)/2 trA^2 + (g+b)|A^os|^2 + (g-b)|A^a|^2] =",
      sp.expand(phig2 - dec2))
print("   -> in the fully developed section the pointwise conditions are 2a+b+g >= 0, g+b >= 0, g-b >= 0;")
print("      2a+b+g = (2/3)(3a+b+g) + (1/3)(b+g), so the thermodynamic set (3a+b+g >= 0, g >= |b|) is sufficient;")
print("      the trace condition actually needed is the weaker 2a+b+g >= 0 (nu_z, which carries d/dz-free axial spin, decouples).")

print()
print("=" * 78)
print("3./4. Exact unforced modes in the tube, energy identity, counter-examples, cross terms")
print("=" * 78)
r_, t_ = sp.symbols('r t', positive=True)
xs, ys = sp.symbols('x y', real=True)
R = 1.0


def fields(kind, m, mu, k, al, be, ga, Rw=1.0):
    """Return sympy (w, nu_x, nu_y) in Cartesian x, y for an unforced exact mode."""
    rr = sp.sqrt(xs ** 2 + ys ** 2)
    cm = sp.re(sp.expand((xs + sp.I * ys) ** m)) / rr ** m          # cos(m t)
    if kind == 'A':
        c = sp.sqrt(2 * k / (al + be + ga))
        phi = sp.besseli(m, c * rr) * cm
        return sp.S(0), sp.diff(phi, xs), sp.diff(phi, ys)
    lam = sp.sqrt(k * (2 * mu + k) / (ga * (mu + k)))
    A = 1
    C = k * A / (mu + k)
    D = -C * sp.besseli(m, lam * Rw) / Rw ** m      # w(R) = 0
    B = D / 2
    harm = sp.re(sp.expand((xs + sp.I * ys) ** m))                    # r^m cos(m t)
    psi = A * sp.besseli(m, lam * rr) * cm + B * harm
    w = C * sp.besseli(m, lam * rr) * cm + D * harm
    return w, sp.diff(psi, ys), -sp.diff(psi, xs)


def axisym(mu, k, ga, Gv=1.0, Rw=1.0):
    """Forced axisymmetric BC1 solution (nu(R) = 0), Cartesian."""
    lam = np.sqrt(k * (2 * mu + k) / (ga * (mu + k)))
    rr = sp.sqrt(xs ** 2 + ys ** 2)
    C = -Gv * Rw / (2 * (2 * mu + k) * iv(1, lam * Rw))
    nu = Gv * rr / (2 * (2 * mu + k)) + C * sp.besseli(1, lam * rr)
    w = (Gv * (Rw ** 2 - rr ** 2) / 4 + k * Gv * (Rw ** 2 - rr ** 2) / (4 * (2 * mu + k))
         + k * C * (iv(0, lam * Rw) - sp.besseli(0, lam * rr)) / lam) / (mu + k)
    return w, -nu * ys / rr, nu * xs / rr


def residuals(w, nx, ny, mu, k, al, be, ga, Gv):
    lap = lambda f: sp.diff(f, xs, 2) + sp.diff(f, ys, 2)
    div = sp.diff(nx, xs) + sp.diff(ny, ys)
    curlz = sp.diff(ny, xs) - sp.diff(nx, ys)
    e3 = (mu + k) * lap(w) + k * curlz + Gv
    e4x = ga * lap(nx) + (al + be) * sp.diff(div, xs) + k * sp.diff(w, ys) - 2 * k * nx
    e4y = ga * lap(ny) + (al + be) * sp.diff(div, ys) - k * sp.diff(w, xs) - 2 * k * ny
    f = sp.lambdify((xs, ys), [e3, e4x, e4y], modules=['scipy', 'numpy'])
    rng = np.random.default_rng(1)
    pts = rng.uniform(-0.6, 0.6, size=(20, 2))
    return max(max(abs(np.asarray(f(px, py), dtype=float))) for px, py in pts)


def quad_forms(w, nx, ny, mu, k, al, be, ga, nr=80, nt=96, Rw=1.0):
    """Return lambdified integrand pieces for dissipation and wall power."""
    wx, wy = sp.diff(w, xs), sp.diff(w, ys)
    Gk = [[sp.diff(nx, xs), sp.diff(ny, xs)], [sp.diff(nx, ys), sp.diff(ny, ys)]]   # G_kl = d_k nu_l
    f = sp.lambdify((xs, ys), [w, wx, wy, nx, ny, Gk[0][0], Gk[0][1], Gk[1][0], Gk[1][1]],
                    modules=['scipy', 'numpy'])
    return f


def integrals(flist, coeffs, Gv, nr=80, nt=96, Rw=1.0):
    """Dissipation integral, G Q and wall power for the superposition of the given field lambdas."""
    mu, k, al, be, ga = coeffs
    xg, wg = np.polynomial.legendre.leggauss(nr)
    rr = 0.5 * Rw * (xg + 1); wr = 0.5 * Rw * wg
    tt = np.linspace(0, 2 * np.pi, nt, endpoint=False); wt = 2 * np.pi / nt
    Rg, Tg = np.meshgrid(rr, tt, indexing='ij')
    Xg, Yg = Rg * np.cos(Tg), Rg * np.sin(Tg)
    vals = sum(np.array([np.broadcast_to(np.asarray(v, dtype=float), Xg.shape) for v in f(Xg, Yg)]) for f in flist)
    w, wx, wy, nx, ny, g00, g01, g10, g11 = vals
    div = g00 + g11
    phig = al * div ** 2 + be * (g00 ** 2 + g11 ** 2 + 2 * g01 * g10) + ga * (g00 ** 2 + g01 ** 2 + g10 ** 2 + g11 ** 2)
    hx, hy = 0.5 * wy, -0.5 * wx                    # half of curl(w e_z)
    phi = (mu + k / 2) * (wx ** 2 + wy ** 2) + 2 * k * ((hx - nx) ** 2 + (hy - ny) ** 2) + phig
    D = np.sum(phi * Rg * wr[:, None]) * wt
    GQ = Gv * np.sum(w * Rg * wr[:, None]) * wt
    # wall power: nu_l m_kl n_k on r = R, ds = R dt
    Tw = tt
    Xw, Yw = Rw * np.cos(Tw), Rw * np.sin(Tw)
    vw = sum(np.array([np.broadcast_to(np.asarray(v, dtype=float), Xw.shape) for v in f(Xw, Yw)]) for f in flist)
    w_, wx_, wy_, nx_, ny_, a00, a01, a10, a11 = vw
    n0, n1 = np.cos(Tw), np.sin(Tw)
    dv = a00 + a11
    Gk = [[a00, a01], [a10, a11]]
    nu = [nx_, ny_]
    P = 0.0
    for l in range(2):
        ml = sum((al * dv * (1 if kk == l else 0) + be * Gk[l][kk] + ga * Gk[kk][l]) * (n0, n1)[kk] for kk in range(2))
        P = P + nu[l] * ml
    Pw = np.sum(P) * Rw * wt
    return D, GQ, Pw


cases = [
    ("valid (mu,k,a,b,g)=(1,1,0,0,1), A m=2", 'A', 2, (1.0, 1.0, 0.0, 0.0, 1.0)),
    ("valid (1,1,0,-0.5,1),        B m=1", 'B', 1, (1.0, 1.0, 0.0, -0.5, 1.0)),
    ("valid (1,1,0,0.9,1),         B m=2", 'B', 2, (1.0, 1.0, 0.0, 0.9, 1.0)),
    ("g+b<0 (1,1,0.6,-1.5,1),      A m=2", 'A', 2, (1.0, 1.0, 0.6, -1.5, 1.0)),
    ("g-b<0 (1,1,-1,1.5,1),        B m=1", 'B', 1, (1.0, 1.0, -1.0, 1.5, 1.0)),
    ("g-b<0 (1,1,-1,1.5,1),        B m=2", 'B', 2, (1.0, 1.0, -1.0, 1.5, 1.0)),
]
print(" case                                   residual(3,4)   intPhi        wall power    intPhi-P     sign")
for name, kind, m, (mu, k, al, be, ga) in cases:
    w, nx, ny = fields(kind, m, mu, k, al, be, ga)
    res = residuals(w, nx, ny, mu, k, al, be, ga, 0.0)
    f = quad_forms(w, nx, ny, mu, k, al, be, ga)
    D, GQ, Pw = integrals([f], (mu, k, al, be, ga), 0.0)
    print(f" {name}  {res:.1e}      {D:+.6e}  {Pw:+.6e}  {D-Pw:+.1e}  {'>=0' if D >= 0 else 'NEGATIVE'}"
          f"  [3a+b+g={3*al+be+ga:+.1f}, g+b={ga+be:+.1f}, g-b={ga-be:+.1f}]")

print()
print("cross terms: forced axisymmetric BC1 solution (G = 1) + unforced mode (coefficients valid case 2)")
for (name, kind, m, co) in (cases[1], cases[0], cases[2]):
    mu, k, al, be, ga = co
    wa, nxa, nya = axisym(mu, k, ga)
    res_a = residuals(wa, nxa, nya, mu, k, al, be, ga, 1.0)
    fa = quad_forms(wa, nxa, nya, mu, k, al, be, ga)
    w, nx, ny = fields(kind, m, mu, k, al, be, ga)
    fm = quad_forms(w, nx, ny, mu, k, al, be, ga)
    Da, GQa, Pa = integrals([fa], co, 1.0)
    Dm, _, Pm = integrals([fm], co, 0.0)
    Dt, GQt, Pt = integrals([fa, fm], co, 1.0)
    print(f"   {name.split(',')[-1].strip()} : axisym residual {res_a:.1e}; energy identity axisym "
          f"{Da-GQa-Pa:+.1e}; D_tot-(D_a+D_m) = {Dt-Da-Dm:+.1e}; P_tot-(P_a+P_m) = {Pt-Pa-Pm:+.1e};"
          f" GQ_tot-GQ_a = {GQt-GQa:+.1e}")
print("   axisymmetric non-azimuthal component (m = 0 irrotational, nu radial) + axisym BC1:")
mu, k, al, be, ga = cases[1][3]
wa, nxa, nya = axisym(mu, k, ga)
fa = quad_forms(wa, nxa, nya, mu, k, al, be, ga)
w0, nx0, ny0 = fields('A', 0, mu, k, al, be, ga)
print(f"   m=0 radial mode residual {residuals(w0, nx0, ny0, mu, k, al, be, ga, 0.0):.1e}")
f0 = quad_forms(w0, nx0, ny0, mu, k, al, be, ga)
Da, _, Pa = integrals([fa], (mu, k, al, be, ga), 1.0)
D0, _, P0 = integrals([f0], (mu, k, al, be, ga), 0.0)
Dt, _, Pt = integrals([fa, f0], (mu, k, al, be, ga), 1.0)
print(f"   D_tot-(D_a+D_0) = {Dt-Da-D0:+.1e}; P_tot-(P_a+P_0) = {Pt-Pa-P0:+.1e}; D_0 = {D0:+.4e} = P_0 = {P0:+.4e}")
print()
print("counter-example scan over the tube radius R (and mu for family B); exact unforced solutions:")


def min_over_R(kind, m, co, Rlist):
    mu, k, al, be, ga = co
    out = []
    for Rw in Rlist:
        w, nx, ny = fields(kind, m, mu, k, al, be, ga, Rw)
        f = quad_forms(w, nx, ny, mu, k, al, be, ga, Rw=Rw)
        D, _, Pw = integrals([f], co, 0.0, Rw=Rw)
        out.append((D / (np.pi * Rw ** 2), Rw, D - Pw))
    return min(out), out


Rlist = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
for co in ((1.0, 1.0, 0.6, -1.5, 1.0), (1.0, 1.0, 0.0, -1.2, 1.0)):
    (dmin, Rm, idres), _ = min_over_R('A', 2, co, Rlist)
    w, nx, ny = fields('A', 2, *co, Rm)
    print(f"   A m=2 (mu,k,a,b,g)={co}: g+b={co[4]+co[3]:+.1f}; min over R of intPhi/(pi R^2) = {dmin:+.4e} at R = {Rm}"
          f" ({'NEGATIVE' if dmin < 0 else '>=0'}); residual {residuals(w, nx, ny, *co, 0.0):.1e}; intPhi - wall power = {idres:+.1e}")
best = None
for mu in (1.0, 0.0, -0.3, -0.4, -0.45, -0.48):
    for be in (2.0, 4.0, 8.0):
        co = (mu, 1.0, 0.0, be, 1.0)
        for m in (1, 2):
            (dmin, Rm, idres), _ = min_over_R('B', m, co, Rlist)
            if best is None or dmin < best[0]:
                best = (dmin, Rm, co, m, idres)
dmin, Rm, co, m, idres = best
w, nx, ny = fields('B', m, *co, Rm)
print(f"   B (g < b), scan mu in [-0.48, 1], b in {{2, 4, 8}}, g = 1, m = 1, 2: most negative intPhi/(pi R^2) = {dmin:+.4e}"
      f" at mu={co[0]}, b={co[3]}, m={m}, R={Rm} ({'NEGATIVE' if dmin < 0 else '>=0'}); residual {residuals(w, nx, ny, *co, 0.0):.1e};"
      f" intPhi - wall power = {idres:+.1e}")
print()
print("Conclusion: with 2a+b+g >= 0 (implied by 3a+b+g >= 0 and g+b >= 0) and g >= |b| every unforced part has wall power")
print("= dissipation >= 0, and cross terms vanish, so total wall power <= 0 implies axisymmetric wall power <= 0.")
print("Necessity: g+b < 0 -> exact unforced m = 2 mode with negative dissipation (scan above), the argument fails;")
print("g < b -> Phi_g < 0 for the admissible in-plane field (-y, x) (part 2), but no exact unforced m = 1, 2 solenoidal")
print("mode with negative dissipation was found in the scan; necessity of g >= b for the mode argument is not shown.")
