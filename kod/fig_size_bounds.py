"""Fig. 1 of the paper: size dependence of the apparent viscosity (M1 wall) for the slit, the tube and the square duct.

Normalised excess Psi = (mu_app - mu - kappa/2)/(kappa/2): Psi = 1 is the upper bound mu + kappa, Psi = 0 the
large-size limit mu + kappa/2. Abscissa lambda D_h (D_h = 4h slit, 2R tube, side a square).
kappa/mu = 1, mu = 1, gamma = 1, alpha + beta = 0.
Curves: slit and tube closed forms; square from the 2D finite-difference solver (micropolar_duct_2d.solve_duct,
N = 80, stretched grid as in its sweep (c)); refined lower bound of the square,
mu + kappa - kappa^2/(2 kappa + gamma lambda_1), lambda_1 = 2 pi^2 / a^2; large-size negative-slip laws
mu_app/(mu + kappa/2) ~ 1 + c/D_h (square, circle) and 1 + 1.5 c/D_h (slit), c = 4 kappa/[(mu+kappa) lambda].
Output: sekil/fig2_size_bounds.pdf/.png and fig2_size_bounds_output.txt
Eq. (19) (tube, square) and Eq. (20) (slit) have separate legend entries.
"""
import os
import numpy as np
from paper_common import lam, tube_muapp, slit_muapp, setup_matplotlib
from micropolar_duct_2d import solve_duct, stretch_for
from params_writer import save_params

HERE = os.path.dirname(os.path.abspath(__file__))
plt = setup_matplotlib()
mu, k, g = 1.0, 1.0, 1.0
L = lam(mu, k, g)
psi = lambda m: (m - mu - k / 2) / (k / 2)
x = np.logspace(-1, np.log10(300), 300)                      # lambda D_h
slit = psi(slit_muapp(mu, k, g, x / (4 * L)))
tube = psi(tube_muapp(mu, k, x / 2))
c = 4 * k / ((mu + k) * L)
asq = psi((mu + k / 2) * (1 + c * L / x))
asl = psi((mu + k / 2) * (1 + 1.5 * c * L / x))
lam1 = 2 * np.pi ** 2 / (x / L) ** 2
low = psi(mu + k - k ** 2 / (2 * k + g * lam1))

xs = np.array([0.1, 0.3, 1, 3, 10, 30, 100])
sq = []
out = ["lambda*a   square mu_app (N=80)   Psi"]
for xa in xs:
    a = xa / L
    s = stretch_for(xa)
    m = solve_duct(a, a, mu, k, g, 0.0, 80, 80, s, s)[0]
    sq.append(psi(m))
    out.append(f"{xa:8.1f}   {m:.6f}   {psi(m):.6f}")
sq = np.array(sq)
out.append(f"monotone non-increasing (square): {bool(np.all(np.diff(sq) <= 1e-12))}")
out.append(f"monotone non-increasing (slit, tube): {bool(np.all(np.diff(slit) <= 1e-12))}, "
           f"{bool(np.all(np.diff(tube) <= 1e-12))}")
for xx in (30, 100, 300):
    i = np.argmin(abs(x - xx))
    out.append(f"lambda D_h = {x[i]:.1f}: tube Psi {tube[i]:.5f} vs law {asq[i]:.5f}; slit Psi {slit[i]:.5f} vs law {asl[i]:.5f}")

fig, ax = plt.subplots(figsize=(3.35, 3.6))
ax.semilogx(x, slit, "k-", label="slit")
ax.semilogx(x, tube, "k--", label="tube")
ax.semilogx(xs, sq, "ks", ms=4, mfc="w", label="square (2D)")
ax.semilogx(x, low, ":", color="0.4", label="square, lower bound")
m = x > 3
ax.semilogx(x[m], asq[m], "-.", color="0.45", lw=0.9, label="tube, square: Eq. (19)")
ax.semilogx(x[m], asl[m], ls=(0, (6, 1.8, 1, 1.8, 1, 1.8)), color="0.45", lw=0.9, label="slit: Eq. (20)")
ax.axhline(1.0, color="k", lw=0.6)
ax.axhline(0.0, color="k", lw=0.6)
ax.set_ylim(-0.05, 1.12)
ax.set_xlim(0.1, 300)
ax.set_xlabel(r"$\lambda D_h$")
ax.set_ylabel(r"$(\mu_{\mathrm{app}}-\mu-\kappa/2)/(\kappa/2)$")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, fontsize=8.5, columnspacing=0.8, handlelength=1.8)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "..", "sekil", "fig2_size_bounds.pdf"))
fig.savefig(os.path.join(HERE, "..", "sekil", "fig2_size_bounds.png"), dpi=600)
with open(os.path.join(HERE, "fig2_size_bounds_output.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
save_params("Fig. 2", parameters={"mu": mu, "kappa": k, "gamma": g, "alpha+beta": 0,
                                "lambda D_h": "0.1-300 (closed forms), square at 0.1, 0.3, 1, 3, 10, 30, 100"},
       domain="slit and tube: closed forms; square: side a, 2D cross-section",
       mesh="square: stretched grid, 80 x 80 intervals, second order",
       tolerances="sparse direct solve; grid error at most 0.00085 at N = 160 (micropolar_duct_2d.py (b))",
       initial_guess="not applicable (linear problem)")
