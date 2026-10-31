"""Check the corrected reading of Migun (1981) / Kolpashchikov et al. (1983):
delta = 2 kappa (1 - alpha) / [2(mu + kappa) - kappa alpha]  ->  alpha = 0: delta = kappa / (mu + kappa).
With delta = 0.84 and N = 7.03e7 1/m the tube solver must reproduce eta_e/eta_n = 1.080 (300 nm), 1.485 (40 nm)."""
from hansen2010_md_vs_deryagin import solve

delta, N = 0.84, 7.03e7
mu = 1.0
ka = delta * mu / (1 - delta)                      # alpha = 0 (M1)
ga = ka * (2 * mu + ka) / ((mu + ka) * N**2)
eta_n, eta_r = mu + ka / 2, ka / 2
print(f"kappa/mu = {ka:.3f}, eta_r/eta_n = {eta_r/eta_n:.3f}, narrow-gap bound 1 + eta_r/eta_n = {1 + eta_r/eta_n:.3f}")
for r, ref in [(3e-7, 1.080), (4e-8, 1.485)]:
    q = solve(r, eta_n, eta_r, ga, "tube", n=3000)
    print(f"r0 = {r*1e9:.0f} nm: eta_e/eta_n = {1/q:.3f}  (Migun curve {ref})")
