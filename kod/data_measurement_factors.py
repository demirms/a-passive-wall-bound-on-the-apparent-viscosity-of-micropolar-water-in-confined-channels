"""Measurement-specific factors that could affect the measured apparent viscosities.

Inputs are quoted from the full texts of the sources (page numbers are journal pages, or PDF pages where the journal
page is not printed); nothing is fitted. Output: data_measurement_factors_output.txt

(1) Kazoe et al. 2023 [5]: sensitivity of the inferred viscosity to D_h (mu ~ D_h^4 in the resistance formula),
    including the propagated SEM/AFM uncertainty of W and H with the exact rectangular-duct flow law;
    fraction of blocked channels or error of D_h that would be needed to explain the measured 4.74 and 5.55.
(2) Kazoe 2022 [4]: D_h sensitivity of the 190 nm square channel; 99 % unbroken channels.
(3) Churaev et al. 1970 [1]: eta ~ r^2 (their Eq. 4), radius uncertainty 3-4 %, radius error needed for 1.34-1.41.
(4) Capillary-filling sets (Tas [2], Haneveld [3]): contact-angle and height terms; Li [6]: slope method.
(5) Normalisation: sensitivity of the ratios to the bulk viscosity used (0.890 vs 1.002 mPa s).
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []


def P(s=""):
    print(s, flush=True)
    OUT.append(s)


def f_rect(W, H, nmax=401):
    """Q = dP W H^3 / (12 mu L) * F, F = 1 - 192 H/(pi^5 W) sum_{n odd} tanh(n pi W/(2H))/n^5 (exact Newtonian rectangle)."""
    n = np.arange(1, nmax, 2)
    return 1 - 192 * H / (np.pi ** 5 * W) * np.sum(np.tanh(n * np.pi * W / (2 * H)) / n ** 5)


def res_exact(W, H):
    """mu L / (R-factor): resistance R = 12 mu L/(W H^3 F)."""
    return W * H ** 3 * f_rect(W, H) / 12.0


def mu_factor_formula(W, H):
    """Kazoe formula: C = 128 mu L/(4.5 D_h^4) -> mu_inferred = C 4.5 D_h^4/(128 L) (per unit C/L)."""
    Dh = 2 * W * H / (W + H)
    return 4.5 * Dh ** 4 / 128.0


P("(1) Kazoe et al. 2023 [5] (Anal. Sci. 39, 779; p. 782: mass flowmetry, Q = 4.5 D_h^4 dP/(128 mu L); 200,000 channels of 70 nm and 150,000 of 100 nm;")
P("    D_h = 73 +- 2 nm (W = 73 +- 3, H = 72 +- 1), D_h = 119 +- 1 nm (W = 130 +- 3, H = 110 +- 1); widths by SEM, depths by AFM)")
for (W, dW, H, dH, ratio, lab) in ((73.0, 3.0, 72.0, 1.0, 4.736, "70 nm channel"), (130.0, 3.0, 110.0, 1.0, 5.549, "100 nm channel")):
    Dh = 2 * W * H / (W + H)
    # stated D_h uncertainty -> mu ~ D_h^4
    dDh = {73.0: 2.0, 130.0: 1.0}[W]
    Dh_stated = {73.0: 73.0, 130.0: 119.0}[W]
    rel_stated = 4 * dDh / Dh_stated
    # propagate W, H with the exact rectangular law: mu_inf ~ res_exact(W,H) * R (R measured): d ln mu = d ln res_exact
    base = res_exact(W, H)
    dlnW = (np.log(res_exact(W * 1.001, H)) - np.log(base)) / 0.001
    dlnH = (np.log(res_exact(W, H * 1.001)) - np.log(base)) / 0.001
    rel_exact = np.hypot(dlnW * dW / W, dlnH * dH / H)
    # same with the Kazoe formula (mu ~ D_h^4)
    bf = mu_factor_formula(W, H)
    flW = (np.log(mu_factor_formula(W * 1.001, H)) - np.log(bf)) / 0.001
    flH = (np.log(mu_factor_formula(W, H * 1.001)) - np.log(bf)) / 0.001
    rel_form = np.hypot(flW * dW / W, flH * dH / H)
    Dh_true = Dh_stated / ratio ** 0.25
    phi = 1 - 1 / ratio
    P(f"    {lab}: D_h(stated) = {Dh_stated:g} +- {dDh:g} nm -> mu ~ D_h^4: +-{100 * rel_stated:.1f} % (4 dD_h/D_h); "
      f"propagating W +- {dW:g}, H +- {dH:g} nm: formula {100 * rel_form:.1f} %, exact rectangle {100 * rel_exact:.1f} % "
      f"(d ln mu/d ln W = {dlnW:.2f}, d ln mu/d ln H = {dlnH:.2f})")
    P(f"      measured ratio {ratio:.2f} would be 1.00 only if D_h were {Dh_true:.1f} nm (i.e. {100 * (1 - Dh_true / Dh_stated):.0f} % smaller than stated, "
      f"{(Dh_stated - Dh_true) / dDh:.0f} times the stated D_h uncertainty), or if a fraction phi = 1 - 1/ratio = {100 * phi:.1f} % of the channels carried no flow "
      f"(inferred viscosity ~ 1/(1 - phi)); a 1 % unflowing fraction changes it by {100 * (1 / 0.99 - 1):.2f} %.")
P("    Formula check (existing output of check_kazoe_resistance_formula.py, not re-run): exact/formula flow rate 0.9997 (73 x 72) and 1.0007 (130 x 110); the 4.5 D_h^4/128 law is not the source of a factor of 5.")

P("\n(2) Kazoe et al. 2022 [4] (Anal. Sci. 38, 281; Table 1, p. 283: 190 +- 8 nm square, N = 11,100; p. 283: 'approximately 99% of the nanochannels were successfully fabricated without breaking')")
ratio = 3.60
P(f"    D_h = 190 +- 8 nm -> mu ~ D_h^4: +-{100 * 4 * 8 / 190:.1f} %; 1 % broken channels would raise the viscosity by {100 * (1 / 0.99 - 1):.2f} %; "
  f"ratio {ratio} would need phi = {100 * (1 - 1 / ratio):.0f} % blocked or D_h = {190 / ratio ** 0.25:.0f} nm.")
P("    Same paper, p. 286: the 10 mM KCl solution (Debye length ~3 nm) gives 3.07e-3 Pa s (text: 3.08e-3), 'almost the same' as pure water (3.21e-3, Debye length ~100 nm); the authors conclude that the electroviscous effect is not dominant in the square channels (but is for the plate channel, p. 286).")

P("\n(3) Churaev et al. 1970 [1] (pp. 215-216)")
P("    Eq. (4): v l_1 = r^2 (P_c - P)/(8 eta): eta from the slope of the v(P) lines (the capillary pressure, i.e. the contact angle and the meniscus term, sets the intercept, not the slope).")
P("    r from the bubble-detachment pressure P_0 (r_0 = 2 sigma/P_0); systematic error from the adsorption film 'not more than 3-4 %' even for the finest capillaries (p. 215);")
P("    different liquids in the same capillary agree to ~1 % (p. 215). eta ~ r^2:")
for dr in (0.03, 0.04):
    P(f"      radius error {100 * dr:.0f} % -> viscosity error {100 * ((1 + dr) ** 2 - 1):.1f} %")
for meas in (1.339, 1.361, 1.407):
    P(f"      measured ratio {meas:.3f} would be 1.00 only if the radius were {100 * (1 - 1 / np.sqrt(meas)):.1f} % smaller than stated (r_true/r = {1 / np.sqrt(meas):.3f})")
for meas in (1.16, 1.23):
    P(f"      ratio {meas:.2f} (r = 170 nm peak): radius {100 * (1 - 1 / np.sqrt(meas)):.1f} % smaller")
P("    Gas-column correction, Eq. (5): the second term is 'small compared to 1' ('the difference between l_1 and l is not more than a few per cent', p. 216).")

P("\n(4) Capillary-filling and filling-under-pressure sets")
P("    Tas et al. 2004 [2] (Appl. Phys. Lett. 85, 3274; p. 3275): eta_app from the slope a of x = a t^(1/2) with cos(theta) = 0.94 +- 0.02 (low-speed advancing angle, macroscopic measurement) and bulk viscosity at the substrate temperature (+-0.5 C);")
P(f"      error bars (+-0.11 at 53 nm) 'represent the random errors, mainly ... in the channel height, the contact angle, and in the bulk viscosity'; height 53 +- 6 nm (random, {100 * 6 / 53:.0f} %), 5 % systematic height error is NOT in the bars;")
P(f"      cos(theta) alone: {100 * 0.02 / 0.94:.1f} % in eta; eta ~ h cos(theta)/a^2, so a 5 % height error is a 5 % error in eta.")
P(f"    Haneveld et al. 2008 [3] (J. Appl. Phys. 104, 014309; pp. 014309-3/-4): modified Washburn x = (1/C) sqrt(gamma cos(theta) h_0/(3 eta)) t^(1/2) - b t, the loss term b is fitted, cos(theta) = 0.99 +- 0.01 (macroscopic), "
  f"C = apparent-viscosity factor; 10.5 +- 0.8 nm channel ({100 * 0.8 / 10.5:.1f} % in h_0, i.e. in C) vs the quoted C = 1.32 +- 0.11 ({100 * 0.11 / 1.32:.1f} %). Each point of their Fig. 5 is one chip (2-3 chips per height); the text does not say whether +-0.11 is a standard deviation or an extreme range.")
P("      The authors list four possible causes of C > 1: dynamic contact angle, electroviscous effect, geometry (wall deformation), and wall-liquid interaction; for 5 nm they suspect channel collapse (p. 014309-4/-5).")
P("    Li et al. 2012 [6] (J. Phys. Chem. Lett.; PDF p. 3): viscosity from the SLOPE and surface tension force gamma cos(theta) from the INTERCEPT of Dx^2/Dt versus the external pressure P_ex (0-2 MPa); a constant dynamic contact angle is assumed and the linearity is taken as evidence; the contact angle therefore does not enter the viscosity if constant.")
P("    Hence only the data of Refs. [1, 4, 5] (pressure-driven flow, viscosity from the resistance or the slope of flow rate versus pressure) are directly comparable with the fully developed Poiseuille flow of the model;")
P("    those of Refs. [2, 3] (spontaneous filling) depend on the contact angle and on the front loss and are kept as supporting data; Ref. [6] (slope method) is intermediate.")

P("\n(5) Normalisation viscosity (Kazoe et al. do not state the temperature; Li et al.: 26.1 C, 0.868 mPa s)")
for lab, mu in (("Kazoe 2022 190 nm (3.21 mPa s x 1e-3)", 3.21), ("Kazoe 2022 394 nm (Table 1: 2.50)", 2.50)):
    for t, e0 in ((20, 1.002), (25, 0.890), (26.1, 0.868)):
        P(f"    {lab:42s} bulk viscosity {e0:.3f} mPa s ({t} C, handbook): ratio {mu / e0:.2f}")
P(f"    The ratios 4.74 and 5.55 of Kazoe 2023 are read from Fig. 4 (normalisation: bulk viscosity from the literature [19], temperature not stated). A change of the bulk viscosity between 20 and 25 C changes the ratios by {100 * (1.002 / 0.890 - 1):.1f} %: "
  f"4.74 -> {4.736 * 0.890 / 1.002:.2f} (if normalised at 0.890 but 1.002 is right) or {4.736 * 1.002 / 0.890:.2f} (reverse); 5.55 -> {5.549 * 0.890 / 1.002:.2f} or {5.549 * 1.002 / 0.890:.2f}; all stay far above the bound 1.243.")

with open(os.path.join(HERE, "data_measurement_factors_output.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(OUT) + "\n")
