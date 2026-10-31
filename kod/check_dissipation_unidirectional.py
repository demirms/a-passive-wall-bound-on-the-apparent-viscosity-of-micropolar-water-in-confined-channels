"""Symbolic checks for unidirectional micropolar duct flow v = w(x,y) e_z, nu = (a(x,y), b(x,y), 0).

Checks (each reported as simplify(...) == 0):
 C1  Eringen dissipation Phi = t_kl (v_{l,k} - eps_{jkl} nu_j) + m_kl nu_{l,k} equals
     (mu+kappa)|grad w|^2 + 2 kappa |nu|^2 - 2 kappa (a w_y - b w_x) + alpha (div nu)^2
     + beta nu_{k,l} nu_{l,k} + gamma |grad nu|^2.
 C2  Same Phi equals (mu+kappa/2)|grad w|^2 + 2 kappa |curl v /2 - nu|^2 + gradient part.
 C3  Euler-Lagrange equations of L = Phi/2 - G w equal minus the Eringen balance equations
     (linear momentum z, angular momentum x, y)  -> operator is symmetric (variational).
 C4  nu_z decouples: with nu_z present, its equations do not contain w, a, b.
 C5  Pointwise 2D gradient form: alpha tr^2 + beta A:A^T + gamma A:A
     = (alpha+(beta+gamma)/2) tr^2 + (beta+gamma)|dev sym|^2 + (gamma-beta)|skew|^2.
 C6  For a test field vanishing on the unit-square boundary:
     int |grad nu|^2 = int (div nu)^2 + int (curl nu)^2  and  int nu_{k,l}nu_{l,k} = int (div nu)^2.
 C7  min_s [(mu+kappa) A^2 - 2 kappa A s + (2 kappa + g lam) s^2] = (mu+kappa - kappa^2/(2kappa+g lam)) A^2.
 C8  Domain-scaling identity: with x = t X, w = t^2 W, nu = t N the functional
     J = 2G int w - int Phi equals t^4 [2G int W - B(W,N) - t^-2 Grad(N)].
"""
import sympy as sp
from sympy.calculus.euler import euler_equations

x, y, z = sp.symbols('x y z', real=True)
mu, ka, al, be, ga, G = sp.symbols('mu kappa alpha beta gamma G', real=True)
w = sp.Function('w')(x, y)
a = sp.Function('a')(x, y)
b = sp.Function('b')(x, y)
c = sp.Function('c')(x, y)   # nu_z, used in C4

X = [x, y, z]
def d(f, k):
    return sp.diff(f, X[k])

def build(v, nu):
    eps = sp.LeviCivita
    gv = [[d(v[l], k) for l in range(3)] for k in range(3)]       # gv[k][l] = v_{l,k}
    gn = [[d(nu[l], k) for l in range(3)] for k in range(3)]      # gn[k][l] = nu_{l,k}
    divv = sum(gv[k][k] for k in range(3))
    divn = sum(gn[k][k] for k in range(3))
    # t_kl = mu (v_{k,l}+v_{l,k}) + kappa (v_{l,k} - eps_{klm} nu_m)   (pressure dropped)
    t = [[mu * (gv[l][k] + gv[k][l]) + ka * (gv[k][l] - sum(eps(k, l, m) * nu[m] for m in range(3)))
          for l in range(3)] for k in range(3)]
    # m_kl = alpha nu_{r,r} delta_kl + beta nu_{k,l} + gamma nu_{l,k}
    mm = [[al * divn * (1 if k == l else 0) + be * gn[l][k] + ga * gn[k][l] for l in range(3)] for k in range(3)]
    Phi = sum(t[k][l] * (gv[k][l] - sum(eps(j, k, l) * nu[j] for j in range(3))) for k in range(3) for l in range(3)) \
        + sum(mm[k][l] * gn[k][l] for k in range(3) for l in range(3))
    mom = [sum(d(t[k][l], k) for k in range(3)) for l in range(3)]
    ang = [sum(d(mm[k][l], k) for k in range(3)) + sum(eps(l, m, n) * t[m][n] for m in range(3) for n in range(3))
           for l in range(3)]
    return sp.expand(Phi), mom, ang, divv

res = []
v = [0, 0, w]
nu = [a, b, 0]
Phi, mom, ang, divv = build(v, nu)
wx, wy = sp.diff(w, x), sp.diff(w, y)
divn = sp.diff(a, x) + sp.diff(b, y)
curln = sp.diff(b, x) - sp.diff(a, y)
gradn2 = sp.diff(a, x)**2 + sp.diff(a, y)**2 + sp.diff(b, x)**2 + sp.diff(b, y)**2
trans = sp.diff(a, x)**2 + 2 * sp.diff(a, y) * sp.diff(b, x) + sp.diff(b, y)**2   # nu_{k,l} nu_{l,k}
Gradpart = al * divn**2 + be * trans + ga * gradn2
Phi1 = (mu + ka) * (wx**2 + wy**2) + 2 * ka * (a**2 + b**2) - 2 * ka * (a * wy - b * wx) + Gradpart
res.append(("C1 Phi expanded form", sp.simplify(Phi - Phi1)))
Phi2 = (mu + ka / 2) * (wx**2 + wy**2) + 2 * ka * ((wy / 2 - a)**2 + (-wx / 2 - b)**2) + Gradpart
res.append(("C2 Phi = (mu+k/2)|gw|^2 + 2k|curl/2-nu|^2 + grad", sp.simplify(Phi - Phi2)))
res.append(("    div v = 0", sp.simplify(divv)))

L = Phi / 2 - G * w
el = euler_equations(L, [w, a, b], [x, y])
# euler_equations returns Eq(dL/df - div(...), 0); compare with -(balance)
res.append(("C3 EL(w) + (mom_z + G)", sp.simplify(el[0].lhs + (mom[2] + G))))
res.append(("C3 EL(a) + ang_x", sp.simplify(el[1].lhs + ang[0])))
res.append(("C3 EL(b) + ang_y", sp.simplify(el[2].lhs + ang[1])))
res.append(("    mom_x, mom_y identically zero", sp.simplify(mom[0]) + sp.simplify(mom[1])))

# C4: add nu_z = c(x,y)
Phi_c, mom_c, ang_c, _ = build(v, [a, b, c])
eq_c = sp.expand(ang_c[2])
dep = [s for s in (w, a, b) if eq_c.has(s)]
res.append(("C4 ang_z depends on w,a,b? (0 = no)", len(dep)))
res.append(("C4 mom_z gains c-terms? (0 = no)", sp.simplify(mom_c[2] - mom[2])))
res.append(("C4 ang_x gains c-terms? (0 = no)", sp.simplify(ang_c[0] - ang[0])))

# C5 pointwise gradient form in 2D
A11, A12, A21, A22 = sp.symbols('A11 A12 A21 A22', real=True)
tr = A11 + A22
form = al * tr**2 + be * (A11**2 + 2 * A12 * A21 + A22**2) + ga * (A11**2 + A12**2 + A21**2 + A22**2)
dev2 = ((A11 - A22) / 2)**2 * 2 + 2 * ((A12 + A21) / 2)**2
skew2 = 2 * ((A12 - A21) / 2)**2
res.append(("C5 2D gradient form decomposition", sp.simplify(form - ((al + (be + ga) / 2) * tr**2 + (be + ga) * dev2 + (ga - be) * skew2))))

# C6 integral identities with test fields vanishing on unit square boundary
p, q = sp.symbols('p q')
bub = x * (1 - x) * y * (1 - y)
at = bub * (1 + 2 * x - 3 * y**2 + x * y)
bt = bub * (2 - x**2 + y + 5 * x * y**3)
I = lambda f: sp.integrate(sp.integrate(sp.expand(f), (x, 0, 1)), (y, 0, 1))
dv = sp.diff(at, x) + sp.diff(bt, y)
cu = sp.diff(bt, x) - sp.diff(at, y)
g2 = sp.diff(at, x)**2 + sp.diff(at, y)**2 + sp.diff(bt, x)**2 + sp.diff(bt, y)**2
tr2 = sp.diff(at, x)**2 + 2 * sp.diff(at, y) * sp.diff(bt, x) + sp.diff(bt, y)**2
res.append(("C6 int|grad nu|^2 - int div^2 - int curl^2", sp.simplify(I(g2) - I(dv**2) - I(cu**2))))
res.append(("C6 int nu_{k,l}nu_{l,k} - int div^2", sp.simplify(I(tr2) - I(dv**2))))

# C7 quadratic minimisation
A_, s_, lam, gm = sp.symbols('A s lambda g_m', positive=True)
quad = (mu + ka) * A_**2 - 2 * ka * A_ * s_ + (2 * ka + gm * lam) * s_**2
smin = sp.solve(sp.diff(quad, s_), s_)[0]
res.append(("C7 min_s quad - (mu+k-k^2/(2k+g lam))A^2",
            sp.simplify(quad.subs(s_, smin) - (mu + ka - ka**2 / (2 * ka + gm * lam)) * A_**2)))
res.append(("C7 second derivative in s (must be >0): 2(2k+g lam)", sp.simplify(sp.diff(quad, s_, 2) - 2 * (2 * ka + gm * lam))))

# C8 scaling identity (pointwise densities): x = t X, so d/dx = (1/t) d/dX, dA = t^2 dA_X
t = sp.symbols('t', positive=True)
WX, WY, N1, N2, N1X, N1Y, N2X, N2Y = sp.symbols('W_X W_Y N1 N2 N1X N1Y N2X N2Y', real=True)
W0 = sp.symbols('W0', real=True)
# physical gradients in terms of scaled ones: w = t^2 W => w_x = t W_X ; nu = t N => nu_{,x} = N_X
wx_, wy_ = t * WX, t * WY
a_, b_ = t * N1, t * N2
ax, ay, bx, by = N1X, N1Y, N2X, N2Y
dvn = ax + by
Phi_phys = (mu + ka) * (wx_**2 + wy_**2) + 2 * ka * (a_**2 + b_**2) - 2 * ka * (a_ * wy_ - b_ * wx_) \
    + al * dvn**2 + be * (ax**2 + 2 * ay * bx + by**2) + ga * (ax**2 + ay**2 + bx**2 + by**2)
dens_phys = (2 * G * t**2 * W0 - Phi_phys) * t**2      # times dA = t^2 dA_X
B = (mu + ka) * (WX**2 + WY**2) + 2 * ka * (N1**2 + N2**2) - 2 * ka * (N1 * WY - N2 * WX)
Gr = al * dvn**2 + be * (ax**2 + 2 * ay * bx + by**2) + ga * (ax**2 + ay**2 + bx**2 + by**2)
res.append(("C8 scaling identity", sp.simplify(dens_phys - t**4 * (2 * G * W0 - B - Gr / t**2))))

with open(__file__.replace('.py', '_output.txt'), 'w') as f:
    for name, val in res:
        line = f"{name}: {val}"
        print(line); f.write(line + "\n")
    ok = all((val == 0) for _, val in res)
    print("ALL ZERO:", ok); f.write(f"ALL ZERO: {ok}\n")
