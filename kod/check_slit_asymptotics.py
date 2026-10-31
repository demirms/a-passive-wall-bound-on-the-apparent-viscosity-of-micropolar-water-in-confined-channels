"""Large-gap asymptotics of the M1 slit (half-gap h): with coth(lam h) -> 1 (exponentially small error),
Q * 3 mu_inf/(2 h^3) = 1 - 3k/(2(mu+k) lam h) + 3k/(2(mu+k) lam^2 h^2),  mu_inf = mu + k/2,
i.e. the wall layer is equivalent to a Navier slip length b = -k/(2(mu+k) lam) (negative).
Also checks the narrow-gap expansion mu_app = (mu+k) - c2 (lam h)^2 + ...
"""
import sympy as sp

h, mu, ka, ga = sp.symbols('h mu kappa gamma', positive=True)
lam = sp.symbols('lam', positive=True)
CT = sp.symbols('coth')   # coth(lam h)
# Q from check_slit_wall_conditions (G = 1), with C*I1 = -h/(2mu+k) (h coth/lam - 1/lam^2)
CI1 = -h / (2 * mu + ka) * (h * CT / lam - 1 / lam**2)
Q = 2 / (mu + ka) * (h**3 / 3 + ka * h**3 / (3 * (2 * mu + ka)) + ka * CI1)
muinf = mu + ka / 2
lhs = sp.simplify(Q.subs(CT, 1) * 3 * muinf / (2 * h**3))
rhs = 1 - 3 * ka / (2 * (mu + ka) * lam * h) + 3 * ka / (2 * (mu + ka) * lam**2 * h**2)
out = [f"A1 large-gap identity residual: {sp.simplify(lhs - rhs)}"]
# narrow gap: exact coth series
x = sp.symbols('x', positive=True)
Qx = Q.subs(CT, sp.coth(x)).subs(h, x / lam)
muapp = 2 * (x / lam)**3 / (3 * Qx)
lamval = sp.sqrt(ka * (2 * mu + ka) / (ga * (mu + ka)))
ser = sp.series(muapp, x, 0, 3).removeO()
out.append(f"A2 narrow-gap series mu_app = {sp.simplify(ser)}")
out.append(f"A2 series at x=0 minus (mu+k): {sp.simplify(ser.subs(x, 0) - (mu + ka))}")
out.append(f"A2 series minus (mu+k)(1 - k x^2/(15(2mu+k))): {sp.simplify(ser - (mu + ka) * (1 - ka * x**2 / (15 * (2 * mu + ka))))}")
with open(__file__.replace('.py', '_output.txt'), 'w', encoding='utf-8') as f:
    for s in out:
        print(s); f.write(s + "\n")
