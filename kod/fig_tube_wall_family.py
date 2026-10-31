"""Fig. 2 of the paper: tube, apparent viscosity versus the wall spin for every no-slip wall condition.

mu_app/(mu+kappa) against sigma = s_hat (2 mu + kappa), where nu_R = (G R / 2) s_hat is the wall spin.
sigma = 0 is M1 (zero wall spin); sigma = sigma* (upper end of the passive range) is M5 (zero couple traction).
Passive walls: 0 <= sigma <= sigma* (solid thick); active walls: dashed. beta = 0.
Output: sekil/fig3_tube_wall_family.pdf/.png and fig3_tube_wall_family_output.txt
"""
import os
import numpy as np
from paper_common import tube_muapp, s_star, setup_matplotlib
from params_writer import save_params

HERE = os.path.dirname(os.path.abspath(__file__))
plt = setup_matplotlib()

mu = 1.0
CASES = [(1.0, 1.0, "-"), (1.0, 10.0, "--"), (10.0, 10.0, "-.")]   # kappa/mu, lam R, style
MARK = {0: "o", 1: "s", 2: "^"}
out = []
fig, ax = plt.subplots(figsize=(3.35, 2.9))
sig = np.linspace(-2.6, 1.4, 1601)
for i, (k, a, ls) in enumerate(CASES):
    e = 1 / (2 * mu + k)
    y = tube_muapp(mu, k, a, sig * e) / (mu + k)
    y = np.where(y > 0, y, np.nan)
    ax.plot(sig, y, ls, color="0.55", lw=0.8)
    ss = s_star(mu, k, a) / e
    pas = (sig >= 0) & (sig <= ss)
    ax.plot(sig[pas], y[pas], ls, color="k", lw=2.0,
            label=rf"$\kappa/\mu={k:g}$, $\lambda R={a:g}$")
    m1 = tube_muapp(mu, k, a, 0.0) / (mu + k)
    m5 = tube_muapp(mu, k, a, ss * e) / (mu + k)
    ax.plot([0, ss], [m1, m5], MARK[i], ms=4, mfc="w", mec="k", ls="none")
    # sigma where mu_app = mu + kappa (D = 1): s_c = -e (1/4 - J)/J
    from paper_common import J_
    Jv = J_(a)
    sc = -(0.25 - Jv) / Jv
    out.append(f"kappa/mu={k:g} lamR={a:g}: M1/(mu+k)={m1:.6f} M5/(mu+k)={m5:.6f} sigma*={ss:.5f} "
               f"(mu+k/2)/(mu+k)={(mu + k / 2) / (mu + k):.6f} bound crossed at sigma={sc:.5f}")
ax.axhline(1.0, color="k", lw=0.6)
ax.axvline(0.0, color="k", lw=0.4, ls=":")
ax.text(-2.55, 1.04, r"$\mu_{\mathrm{app}}=\mu+\kappa$", fontsize=11.5)
ax.set_xlim(-2.6, 1.4)
ax.set_ylim(0.3, 1.6)
ax.set_xlabel(r"wall spin $\sigma=2(2\mu+\kappa)\nu_R/(GR)$")
ax.set_ylabel(r"$\mu_{\mathrm{app}}/(\mu+\kappa)$")
ax.legend(frameon=False, loc="upper right", fontsize=8.5, handlelength=2.6)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "..", "sekil", "fig3_tube_wall_family.pdf"))
fig.savefig(os.path.join(HERE, "..", "sekil", "fig3_tube_wall_family.png"), dpi=600)
with open(os.path.join(HERE, "fig3_tube_wall_family_output.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
save_params("Fig. 3", parameters={"mu": mu, "kappa/mu": [1, 1, 10], "lambda R": [1, 10, 10], "beta/gamma": 0,
                                "sigma range": [-2.6, 1.4]},
       domain="tube 0 <= r <= R (closed form, modified Bessel functions)", mesh="closed form; 1601 sigma points",
       tolerances="closed form (agrees with finite differences to 1.1e-9, check_tube_m3_bound.py N1)",
       initial_guess="not applicable")
