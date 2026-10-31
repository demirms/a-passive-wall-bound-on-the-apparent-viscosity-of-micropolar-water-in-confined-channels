"""Wu et al. 2017 (PNAS 114, 3358): what does their own interfacial-viscosity model give for wide pores?

Source: literatur/Wu_2017_Wettability_effect_on_nanoconfined_water_flow.pdf
  Eq. (1) (p. 3359)  l_st = C / (cos(theta) + 1)^2, C = 0.41 (nm; MD fit, Methods p. 3363)
  Eq. (2) (p. 3359)  mu(d) = mu_i * A_i/A_t + mu_inf * (1 - A_i/A_t)   [area-weighted effective viscosity]
  Eq. (3) (p. 3360)  mu_i / mu_inf = -0.018 * theta[deg] + 3.25
  Eq. (7) (p. 3361)  enhancement factor eps = Q_s/Q_n = (1 + 8 l_st/d) * mu_inf/mu(d)
  critical (interface) thickness 0.7 nm (p. 3359, SI Text section C, not available here)
  p. 3362: "for the water confined in pores with a diameter of larger than 200 nm and a contact angle
            of smaller than 150 deg, the enhancement factor by our model is smaller than 2".

Assumption (not printed in the main text; SI not available): the interface region is the annulus of
width t = 0.7 nm at the wall of the circular pore, A_i/A_t = 1 - ((d - 2 t)/d)^2.

Parts
 (a) mu(d)/mu_inf by Wu's own Eq. (2)-(3): d = 200, 300, 500 nm; theta = 0, 30, 60, 90 deg.
 (b) enhancement factor eps of Eq. (7) (flow, includes slip) -> where "smaller than 2" comes from.
 (c) scan of eps over d >= 200 nm, theta < 150 deg: is it below 2?
 (d) our own cross-check, NOT Wu's definition: exact no-slip Poiseuille tube with a wall layer of
     viscosity ratio r = mu_i/mu_inf and thickness t (flux-based apparent viscosity).
     Q = (pi G / 2) * int_0^R rho^3 / mu(rho) d rho  ->  mu_app/mu_inf = 1 / [ (1-x)^4 + (1-(1-x)^4)/r ], x = t/R.
"""
import numpy as np

T_NM = 0.7        # critical thickness [nm], Wu p. 3359
C_NM = 0.41       # slip constant [nm], Wu Eq. (1) + Methods


def area_fraction(d, t=T_NM):
    return 1.0 - ((d - 2.0 * t) / d) ** 2


def ratio_interface(theta_deg):
    return -0.018 * theta_deg + 3.25            # Eq. (3)


def mu_eff_over_inf(d, theta_deg):
    f = area_fraction(d)
    return 1.0 + f * (ratio_interface(theta_deg) - 1.0)        # Eq. (2) divided by mu_inf


def slip_true(theta_deg):
    return C_NM / (np.cos(np.radians(theta_deg)) + 1.0) ** 2   # Eq. (1), nm


def enhancement(d, theta_deg):
    return (1.0 + 8.0 * slip_true(theta_deg) / d) / mu_eff_over_inf(d, theta_deg)   # Eq. (7)


def mu_app_composite(d, theta_deg, t=T_NM):
    x = t / (d / 2.0)
    r = ratio_interface(theta_deg)
    return 1.0 / ((1 - x) ** 4 + (1 - (1 - x) ** 4) / r)


print("Wu et al. 2017, Eq. (2)-(3): mu(d)/mu_inf with t = 0.7 nm, A_i/A_t = 1 - ((d-2t)/d)^2")
print("A_i/A_t: " + ", ".join(f"d={d}: {area_fraction(d):.5f}" for d in (200, 300, 500)))
print()
print("(a) mu(d)/mu_inf (Wu's area-weighted definition)")
print("  d[nm]  " + "".join(f"theta={th:3d}   " for th in (0, 30, 60, 90)))
for d in (200, 300, 500):
    print(f"  {d:5d}  " + "".join(f"{mu_eff_over_inf(d, th):9.4f}   " for th in (0, 30, 60, 90)))
print("  mu_i/mu_inf = " + ", ".join(f"{ratio_interface(th):.3f} (theta={th})" for th in (0, 30, 60, 90)))
v = mu_eff_over_inf(200, 0)
print(f"  200 nm, theta = 0: {v:.4f}  (acceptance criterion <= 1.05: {'met' if v <= 1.05 else 'NOT met'};"
      f" independent estimate ~1.03)")
print(f"  theta at which mu_i = mu_inf: {(3.25 - 1) / 0.018:.1f} deg")
print()
print("(b) enhancement factor eps of Eq. (7), flow-rate ratio to no-slip bulk Hagen-Poiseuille")
print("  true slip length l_st [nm] = 0.41/(cos(theta)+1)^2: " +
      ", ".join(f"{th}: {slip_true(th):.3f}" for th in (0, 30, 60, 90, 120, 140, 150)))
print("  d[nm]  " + "".join(f"theta={th:3d}   " for th in (0, 30, 60, 90, 120, 140, 150)))
for d in (100, 200, 300, 500):
    print(f"  {d:5d}  " + "".join(f"{enhancement(d, th):9.4f}   " for th in (0, 30, 60, 90, 120, 140, 150)))
print()
print("(c) scan d = 200-20000 nm (log grid, 400 points) x theta = 0-149.99 deg (3000 points)")
dd = np.logspace(np.log10(200), np.log10(20000), 400)
tt = np.linspace(0, 149.99, 3000)
D, TH = np.meshgrid(dd, tt)
E = enhancement(D, TH)
i = np.unravel_index(np.argmax(E), E.shape)
print(f"  max eps = {E.max():.4f} at d = {D[i]:.1f} nm, theta = {TH[i]:.2f} deg; eps < 2 everywhere: {bool(E.max() < 2)}")
print(f"  at theta = 150 deg exactly: eps(200 nm) = {enhancement(200, 150):.4f}")
print(f"  max of mu(d)/mu_inf over theta in [0, 90] deg, d >= 200 nm: "
      f"{mu_eff_over_inf(200, 0):.4f} (theta = 0, d = 200 nm)")
print(f"  eps >= 1 only if the slip term outweighs the viscosity term; for theta <= 90 deg at 200 nm eps = "
      f"{enhancement(200, 0):.4f} ... {enhancement(200, 90):.4f}, i.e. a flow REDUCTION by up to "
      f"{100 * (1 - enhancement(200, 0)):.1f} %")
print()
print("(d) our cross-check (flux-based, exact no-slip two-layer Poiseuille tube; NOT Wu's definition)")
print("  d[nm]  " + "".join(f"theta={th:3d}   " for th in (0, 30, 60, 90)))
for d in (200, 300, 500):
    print(f"  {d:5d}  " + "".join(f"{mu_app_composite(d, th):9.4f}   " for th in (0, 30, 60, 90)))
print(f"  ratio Wu-area-weighted / flux-based at 200 nm, theta = 0: "
      f"{(mu_eff_over_inf(200, 0) - 1) / (mu_app_composite(200, 0) - 1):.3f} (excess over bulk)")
print("  For a thin layer the excess over bulk is 2x(r-1) (Wu, area-weighted) against 4x(1-1/r) (flux-based), x = t/R,")
print("  so the ratio of the two excesses is r/2: Wu's value is the larger one for r = mu_i/mu_inf > 2 (theta < 69 deg),")
print("  the flux-based one for 1 < r < 2 (at most twice Wu's tiny excess). Both stay below 1.032 at d >= 200 nm.")
print(f"  max over theta in [0, 90] deg and d in {{200, 300, 500}} nm of either definition: "
      f"{max(max(mu_eff_over_inf(d, th), mu_app_composite(d, th)) for d in (200, 300, 500) for th in range(0, 91, 5)):.4f}")
