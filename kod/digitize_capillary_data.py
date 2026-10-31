"""Apparent viscosity data of water in capillaries and nanochannels: figure reading (pixel -> value) and table conversions.

Purpose: build the eta_app/eta_bulk data table for water (and control liquids) from the source publications and write it
to ../01_arastirma/veri_tablosu.csv. Pixel coordinates were read by hand on figure crops (the dpi is stated in each
block); the axis calibration of every figure uses the positions of the tick marks. Values given in tables or in the
text are entered directly.
Run: py -I digitize_capillary_data.py
"""
import csv
import math
import pathlib
import statistics as st

OUT = pathlib.Path(__file__).resolve().parents[1] / "01_arastirma" / "veri_tablosu.csv"
rows = []

COLS = ["id", "source_key", "location", "data_origin", "liquid", "wall_material", "wall_treatment",
        "geometry", "size_symbol", "size_nm", "size_unc_nm", "eta_ratio", "eta_ratio_unc",
        "unc_type", "reported_quantity", "conversion", "temperature_C", "electrolyte", "method", "notes"]


def add(**kw):
    r = {c: "" for c in COLS}
    r.update(kw)
    r["id"] = "R%03d" % (len(rows) + 1)
    for k in ("size_nm", "size_unc_nm", "eta_ratio", "eta_ratio_unc"):
        if isinstance(r[k], float):
            r[k] = "%.4g" % r[k]
    rows.append(r)


def lin(p, p0, v0, p1, v1):
    """Linear axis calibration through two ticks (pixel p0 -> value v0, p1 -> v1)."""
    return v0 + (p - p0) * (v1 - v0) / (p1 - p0)


def logc(p, p0, v0, p1, v1):
    """Log10 axis calibration through two ticks."""
    return 10 ** (math.log10(v0) + (p - p0) * (math.log10(v1) - math.log10(v0)) / (p1 - p0))


# ------------------------------------------------------------------ Churaev 1970
S = "Churaev_1970"
W = dict(wall_material="fused quartz (>99.99% SiO2)", geometry="cylindrical capillary", size_symbol="r (radius)",
         method="pressure-driven meniscus flow (gas pressure); eta from slope of v(P) at fixed l",
         electrolyte="none (distilled water, not specified)")
# Fig. 4 (PDF page 6; 450 dpi crop): x ticks 0.5 um -> 878 px, 1.0 um -> 1472 px; y ticks 1.2 -> 752 px, 1.4 -> 508 px
fx = lambda x: lin(x, 878, 0.5, 1472, 1.0)
fy = lambda y: lin(y, 752, 1.2, 508, 1.4)
water4 = [(347, 500), (355, 555), (333, 582), (499, 762), (521, 838), (552, 802), (571, 873), (587, 843),
          (934, 960), (1163, 980), (1469, 989)]
for i, (x, y) in enumerate(water4, 1):
    add(source_key=S, location="Fig. 4 (p.218), open circles = water", data_origin="figure read-off (450 dpi crop)",
        liquid="water", wall_treatment="none (clean quartz; hydrophilic)", size_nm=1000 * fx(x), size_unc_nm=15.0,
        eta_ratio=fy(y), eta_ratio_unc=0.015, unc_type="read-off (axis calibration + centring)",
        reported_quantity="eta/eta0 (read directly)", conversion="none", temperature_C="~20 (caption: t~20 C)",
        notes="Fig4 water point %d/11; radius from figure axis (r0 from bubble-detachment pressure; film correction <=3-4%%)" % i, **W)
# CCl4 and benzene: symbols merged with the eta/eta0 = 1 line; positions approximate (900 dpi zoom)
for liq, rr, note in [("CCl4", 47.5, "filled circle, left blob (zoom read)"),
                      ("benzene", 66.0, "phi symbol, partly merged with the CCl4 blob"),
                      ("CCl4 and/or benzene (merged)", 190.0, "merged blob at r~0.19 um; CCl4 and benzene not separable")]:
    add(source_key=S, location="Fig. 4 (p.218), filled circles = CCl4, phi = benzene",
        data_origin="figure read-off (900 dpi zoom); approximate", liquid=liq, wall_treatment="none",
        size_nm=rr, size_unc_nm=8.0, eta_ratio=1.0, eta_ratio_unc=0.02,
        unc_type="symbol overlaps the eta/eta0=1 line (centred value, estimate)",
        reported_quantity="eta/eta0", conversion="none", temperature_C="~20",
        notes=note + "; text: no change in viscosity down to r=0.05 um", **{**W, "electrolyte": "n/a"})
# Fig. 5 (PDF page 6; crop x 0.15-0.80, y 0.60-0.80 of page, 450 dpi): x 0 C -> 288 px, 25 C -> 744 px; y 1.0 -> 632 px, 1.5 -> 177 px
tx = lambda x: lin(x, 288, 0.0, 744, 25.0)
ty = lambda y: lin(y, 632, 1.0, 177, 1.5)
fig5 = {
    "filled": [(732, 197), (870, 200), (1018, 275), (1135, 334), (1272, 350), (1384, 462), (1545, 614)],
    "half-filled": [(640, 255), (698, 284), (820, 253), (1020, 228)],
    "open": [(436, 507), (624, 458), (641, 497), (679, 471), (835, 421), (1028, 486), (1225, 536), (1495, 617)],
}
rad = {"filled": ("0.046 or 0.050 um (symbol-radius assignment ambiguous)", 48.0, 3.0),
       "half-filled": ("0.046 or 0.050 um (symbol-radius assignment ambiguous)", 48.0, 3.0),
       "open": ("0.17 um", 170.0, 5.0)}
for sym, pts in fig5.items():
    for (x, y) in pts:
        add(source_key=S, location="Fig. 5 (p.218), %s symbol" % sym, data_origin="figure read-off (450 dpi crop)",
            liquid="water", wall_treatment="none", size_nm=rad[sym][1], size_unc_nm=rad[sym][2],
            eta_ratio=ty(y), eta_ratio_unc=0.015, unc_type="read-off; baseline eta/eta0=1 drifts ~0.01 across the scan",
            reported_quantity="eta/eta0 at the same T", conversion="none", temperature_C=round(tx(x), 1),
            notes="radius class %s; caption '(1) 0.17, (2) 0.046, (3) 0.050' with symbols 'filled,1; half,2; open,3' is "
                  "inconsistent with Fig.4 (see veri_tablosu_notu.md); mapping used: open = 0.17 um, filled/half = 0.046-0.050 um" % rad[sym][0],
            **W)

# ------------------------------------------------------------------ Li 2012
S = "Li_2012"
MU0 = 0.868  # mPa s, bulk water at 26.1 C as stated in the paper
# Fig. 4a (PDF page 4; 400 dpi crop): y 2.5 -> 77 px, 0.5 -> 797 px
ly = lambda y: lin(y, 797, 0.5, 77, 2.5)
sq = [(200, 8, 205, 143, 271), (393, 16, 426, 373, 480), (491, 13, 387, 342, 432),
      (838, 16, 627, 605, 650), (1112, 24, 710, 698, 722), (1530, 26, 672, 662, 683)]
for (D, Du, y, yt, yb) in sq:
    mu = ly(y)
    err = (ly(yt) - ly(yb)) / 2
    add(source_key=S, location="Fig. 4a (p.2450), red triangles", data_origin="figure read-off (400 dpi crop); size from Table 1",
        liquid="water", wall_material="fused silica", wall_treatment="acetone/ethanol/water/NaOH/water rinse; thermal bonding 1080 C",
        geometry="square channel (W~D)", size_symbol="Dh (hydraulic diameter 2DW/(D+W))", size_nm=float(D), size_unc_nm=float(Du),
        eta_ratio=mu / MU0, eta_ratio_unc=err / MU0, unc_type="error bar (authors: SD of size and fit), read off",
        reported_quantity="mu = %.3f x10^-3 Pa s (read)" % mu, conversion="eta_ratio = mu/%.3f mPa s (bulk value stated in paper, 26.1 C)" % MU0,
        temperature_C="26.1 +- 0.6", electrolyte="none (water >18 MOhm cm)",
        method="capillary filling under 0-2 MPa external pressure (Washburn, factor 4.5 for square)",
        notes="Dh = nominal size of Table 1; read-off uncertainty in mu ~+-0.02e-3 Pa s; text: 2.5x at 200 nm relative to 1559-nm channel (0.841e-3 Pa s)")
pl = [(25, 628, 593, 658), (50, 651, 640, 663), (100, 674, 656, 692), (189, 676, 667, 687), (415, 684, 672, 696)]
for (D, y, yt, yb) in pl:
    mu = ly(y)
    err = (ly(yt) - ly(yb)) / 2
    add(source_key=S, location="Fig. 4a (p.2450), blue inverted triangles", data_origin="figure read-off (400 dpi crop); size from Table 1",
        liquid="water", wall_material="fused silica", wall_treatment="as above", geometry="plate channel (W=20 um, W/D>20)",
        size_symbol="D (depth)", size_nm=float(D), size_unc_nm=1.0, eta_ratio=mu / MU0, eta_ratio_unc=max(err, 0.02) / MU0,
        unc_type="error bar read off (floor 0.02e-3 Pa s for small bars)", reported_quantity="mu = %.3f x10^-3 Pa s (read)" % mu,
        conversion="eta_ratio = mu/%.3f mPa s" % MU0, temperature_C="26.1 +- 0.6", electrolyte="none (water >18 MOhm cm)",
        method="capillary filling under 0-2 MPa external pressure",
        notes="text: 415 nm value 0.826+-0.036e-3 Pa s, 25 nm is 1.2x that; marker-centre read-off of inverted triangles ~+-0.02e-3")

# ------------------------------------------------------------------ Haneveld 2008
S = "Haneveld_2008"
H = dict(liquid="water", wall_material="thermal SiO2 spacer / native oxide on Si (bottom) + Borofloat glass (top)",
         wall_treatment="400 C bond anneal; cos(theta)=0.99+-0.01 (theta<10 deg)",
         geometry="slit (rectangular, w=20 um)", size_symbol="h0 (channel height)", electrolyte="none (DI water)",
         method="capillary filling (Washburn x=a sqrt(t) with correction C and loss term)",
         reported_quantity="C (dimensionless correction factor)",
         conversion="eq.(3): a = sqrt((1/C) gamma cos(theta) h0/(3 eta)) -> a ~ eta^-1/2, C = eta_app/eta_bulk "
                    "(abstract 'Washburn coefficient up to 1.6 smaller' = sqrt(2.5))")
hy = lambda y: lin(y, 845, 1.0, 90, 3.0)   # Fig. 5 (PDF page 4; 500 dpi crop): y 1.0 -> 845 px, 3.0 -> 90 px
add(source_key=S, location="text p.014309-4 and Fig. 5", data_origin="text (authors); Fig. 5 chip dots read C=2.55, 2.52",
    size_nm=5.0, size_unc_nm=0.8, eta_ratio=2.5, eta_ratio_unc=0.4, unc_type="authors (text)", temperature_C="22.7",
    notes="Authors: probable large wall deformation by capillary suction (~100 bar), 5 nm result not interpretable as a water property; "
          "excluded from their layer fit", **H)
add(source_key=S, location="text p.014309-5 (conclusions); Fig. 5", data_origin="text (authors); Fig. 5 chip dots read C=1.35, 1.30",
    size_nm=10.5, size_unc_nm=0.8, eta_ratio=1.32, eta_ratio_unc=0.11, unc_type="authors (text, '32+-11 %')", temperature_C="22.2",
    notes="Table I: d_channel=10.5+-0.8 nm (called '11 nm channels' in text); wall deformation up to ~5 nm possible (authors' FE estimate)", **H)
add(source_key=S, location="Fig. 5 (grey bar + dots)", data_origin="figure read-off (500 dpi crop)", size_nm=22.4, size_unc_nm=0.8,
    eta_ratio=hy(802), eta_ratio_unc=0.05, unc_type="grey bar half-width read off (C 1.07-1.17)", temperature_C="21.6",
    notes="Table I: 22.4+-0.8 nm ('23 nm'); value from dots at C~1.11; no number in text", **H)
add(source_key=S, location="text p.014309-4 and Fig. 5", data_origin="text (authors)", size_nm=46.0, size_unc_nm=1.3,
    eta_ratio=1.02, eta_ratio_unc=0.04, unc_type="authors (text)", temperature_C="20.8",
    notes="Table I: 46+-1.3 nm ('47 nm'); Fig. 5 chip dots C~1.02-1.05", **H)

# ------------------------------------------------------------------ Tas 2004
S = "Tas_2004"
T = dict(liquid="water", wall_material="thermal SiO2 20 nm on Si (bottom) + Borofloat glass (top)",
         wall_treatment="piranha (H2O2/H2SO4) clean; cos(theta)=0.94+-0.02 (low-speed advancing)",
         geometry="slit (rectangular, w=20 um, L=1 cm)", size_symbol="h (channel height)",
         method="capillary filling (Washburn x=a sqrt(t)); eta_app from slope vs handbook",
         reported_quantity="apparent viscosity / bulk viscosity (Fig. 4)", conversion="none")
# Fig. 4 (PDF page 2; 600 dpi crop, displayed coordinates): y 1.0 -> 932 px, 1.3 -> 228 px; x 50 nm -> 808 px, 150 nm -> 1507 px
ty4 = lambda y: lin(y, 932, 1.0, 228, 1.3)
tx4 = lambda x: lin(x, 808, 50.0, 1507, 150.0)
demi = {53: [(826, 309), (838, 369), (827, 452), (826, 485)], 111: [(1189, 610), (1174, 774), (1234, 699), (1236, 797)],
        152: [(1525, 753), (1541, 792), (1585, 730), (1526, 852)]}
nacl = {53: [(828, 827), (839, 844), (827, 925)], 111: [(1173, 869), (1190, 897), (1234, 935)],
        152: [(1524, 946), (1542, 965), (1584, 927)]}
nom = {53: 6.0, 111: 9.0, 152: 11.0}
for el, grp, mark in [("none (demineralised water 0.7 uS/cm, pH 5.8)", demi, "diamond"), ("0.1 M NaCl", nacl, "square")]:
    for h, pts in grp.items():
        for (x, y) in pts:
            add(source_key=S, location="Fig. 4 (p.3275), %s" % mark, data_origin="figure read-off (600 dpi crop)",
                size_nm=tx4(x), size_unc_nm=nom[h], eta_ratio=ty4(y), eta_ratio_unc=0.11,
                unc_type="authors' error bars ~+-0.11 (text: 24+-11 %, 7+-11 %); read-off itself +-0.01",
                temperature_C="21.5-22.0 +-0.5 (Fig.3 caption)", electrolyte=el,
                notes="nominal height group %d nm (+-%.0f nm random, +-5%% systematic); size_nm = plotted x read-off "
                      "(horizontal spread of repeated fills in the plot)" % (h, nom[h]), **T)
for h, val, unc, txt in [(53, 1.24, 0.11, "text: 24+-11 % after correcting the height systematic (tall NaCl channels set to bulk); 23+-16 % before"),
                         (152, 1.07, 0.06, "text: 7+-6 % after correction; 7+-11 % before (tallest channels)")]:
    add(source_key=S, location="text p.3275 and abstract", data_origin="text (authors)", size_nm=float(h), size_unc_nm=nom[h],
        eta_ratio=val, eta_ratio_unc=unc, unc_type="authors", temperature_C="21.5-22.0 +-0.5",
        electrolyte="none (demineralised water)", notes=txt + " (group mean)", **T)

# ------------------------------------------------------------------ Kazoe 2022 (Table 1)
S = "Kazoe_2022"
mu0_implied = 3.21 / 3.6   # mPa s: text says 3.6x bulk at 190 nm -> mu0 = 0.892 mPa s (value itself not given in paper)
K = dict(liquid="water", wall_material="fused silica", wall_treatment="e-beam lithography + dry etch; thermal bonding 1080 C",
         method="pressure-driven flow, mass flowmetry (microbalance), dP<=2 MPa; eta from fluid resistance "
                "C=128 mu L/(4.5 Dh^4) (square) or 12 mu L/(W D^3) (plate)",
         temperature_C="not stated in paper", reported_quantity="mu (Pa s), Table 1")
tab = [  # type, Dr, Dr_unc, W, D, N, R, mu(1e-3 Pa s), mu_unc, electrolyte
    ("square", 190, 8, 163, 230, 11100, 6.92e21, 3.21, 0.54, "none (pure water)"),
    ("square", 394, 6, 410, 380, 5500, 2.93e20, 2.50, 0.19, "none (pure water)"),
    ("square", 760, 11, 777, 745, 814, 9.16e18, 1.08, 0.09, "none (pure water)"),
    ("square", 1691, 10, 1572, 1830, 44, 3.52e17, 1.01, 0.02, "none (pure water)"),
    ("square", 190, 8, 163, 230, 11100, 6.62e21, 3.07, 0.51, "10 mM KCl"),
    ("plate", 234, 3, 10000, 234, 1022, 1.10e19, 1.17, 0.06, "none (pure water)"),
    ("plate", 417, 4, 10360, 417, 110, 1.28e18, 0.80, 0.09, "none (pure water)"),
    ("plate", 745, 12, 15400, 745, 22, 1.84e17, 0.98, 0.08, "none (pure water)"),
]
# Fig. 5 values read (400 dpi crop): y 1 -> 843 px, 2 -> 632 px, 3 -> 421 px
fig5v = {("square", 190, "none (pure water)"): 3.60, ("square", 394, "none (pure water)"): 2.45,
         ("square", 760, "none (pure water)"): 1.20, ("square", 1691, "none (pure water)"): 1.12,
         ("plate", 234, "none (pure water)"): 1.31, ("plate", 417, "none (pure water)"): 0.90,
         ("plate", 745, "none (pure water)"): 1.09}
kz_checks = []
for (typ, Dr, Dru, Wd, Dd, N, R, mu, muu, el) in tab:
    if typ == "square":
        Dh = 2 * Wd * Dd / (Wd + Dd)
        mu_calc = R * 4.5 * (Dh * 1e-9) ** 4 / (128 * 100e-6)      # L = 100 um (text)
        geom, sym = "square channel (W/D~1)", "Dh (hydraulic diameter)"
    else:
        mu_calc = R * Wd * 1e-9 * (Dd * 1e-9) ** 3 / (12 * 100e-6)
        geom, sym = "plate channel (W/D>20)", "D (depth)"
    kz_checks.append((typ, Dr, el, mu, mu_calc * 1e3))
    f5 = fig5v.get((typ, Dr, el))
    add(source_key=S, location="Table 1 (p.283)", data_origin="table (viscosity); ratio computed", geometry=geom, size_symbol=sym,
        size_nm=float(Dr), size_unc_nm=float(Dru), eta_ratio=mu / mu0_implied, eta_ratio_unc=muu / mu0_implied,
        unc_type="Table 1 (error propagation of R and size)", electrolyte=el,
        conversion="eta_ratio = mu/%.3f mPa s; mu0 NOT given in paper: inferred from text (3.21e-3 Pa s = 3.6 x mu0 at 190 nm)" % mu0_implied,
        notes="mu = %s +- %s x10^-3 Pa s; W=%d, D=%d nm, N=%d, R=%.3g Pa s/m3. Fig. 5 value (read) = %s" % (mu, muu, Wd, Dd, N, R, f5 if f5 else "n/a"),
        **K)

# ------------------------------------------------------------------ Kazoe 2023 (Fig. 4; 500 dpi crop)
S = "Kazoe_2023"
k3y = lambda y: lin(y, 1043, 0.0, 165, 6.0)          # y 0 -> 1043 px, 6 -> 165 px
k3x = lambda x: logc(x, 528, 100.0, 1213, 1000.0)    # x 100 nm -> 528 px, 1000 nm -> 1213 px
K3 = dict(wall_material="fused silica", wall_treatment="top-down e-beam + dry etch; thermal bonding",
          geometry="square channel (aspect ratio 1)", size_symbol="Dh (hydraulic diameter)",
          method="pressure-driven flow, mass flowmetry; eta from 4.5 Dh^4 dP/(128 mu L)",
          temperature_C="not stated in paper", electrolyte="none",
          reported_quantity="mu/mu0 (Fig. 4); mu0 from literature [19], value not given", conversion="none")
for liq, x, y, yt, yb, Dh_ref, note in [
        ("water", 428, 350, 273, 426, 73.0, "text: ~5x bulk for 70 and 100 nm; W=73+-3, H=72+-1 nm; 200,000 channels"),
        ("water", 581, 231, 158, 302, 119.0, "W=130+-3, H=110+-1 nm; 150,000 channels (called '100 nm channels' in text/Fig. 3)"),
        ("DMSO", 428, 918, 905, 937, 73.0, "text: DMSO similar to bulk"),
        ("DMSO", 549, 873, 853, 897, 107.0, "W=110+-3, H=105+-1 nm")]:
    add(source_key=S, location="Fig. 4 (p.782)", data_origin="figure read-off (500 dpi crop); Dh from text", liquid=liq,
        size_nm=Dh_ref, size_unc_nm=2.0, eta_ratio=k3y(y), eta_ratio_unc=(k3y(yt) - k3y(yb)) / 2, unc_type="error bar read off",
        notes=note + "; plotted x gives Dh=%.0f nm (consistency check)" % k3x(x), **K3)
for Dh_prev, y, yt, yb in [(190, 516, 447, 584), (394, 679, 651, 707), (760, 866, 852, 880), (1691, 877, 868, 886)]:
    add(source_key=S, location="Fig. 4 (p.782), open circles (= Kazoe 2022 data, ref [18])",
        data_origin="figure read-off (duplicate of Kazoe_2022)", liquid="water", size_nm=float(Dh_prev), size_unc_nm=0.0,
        eta_ratio=k3y(y), eta_ratio_unc=(k3y(yt) - k3y(yb)) / 2, unc_type="error bar read off",
        notes="DUPLICATE of Kazoe_2022 (re-plotted); do not count twice", **K3)

# ------------------------------------------------------------------ Cheng 2002 Fig. 4 (PDF page 4; 450 dpi crop)
S = "Cheng_2002"
cy = lambda y: lin(y, 1143, 0.0, 90, 6.0)           # y 0 -> 1143 px, 6 -> 90 px
cx = lambda x: logc(x, 545, 100.0, 1048, 1000.0)    # x 100 nm -> 545 px, 1000 nm -> 1048 px (axis 20 ... 5000 nm)
C = dict(wall_material="cover glass + photoresist layers (top and bottom)",
         wall_treatment="photoresist (top unbaked); roughness peak-to-peak ~0.5 nm", geometry="slit (rectangular, w=2-20 um, L=100-900 um)",
         size_symbol="h (channel height)", electrolyte="none stated",
         method="pressure-driven flow (syringe, ~1 atm); flow rate from meniscus in outlet capillary; Q=h^3 w P/(12 mu L)",
         temperature_C="25 (room temperature)", reported_quantity="alpha_expt/alpha_theory (flow-rate coefficient ratio, no-slip theory)",
         conversion="eta_app/eta_bulk = 1/(alpha_expt/alpha_theory) since alpha ~ 1/mu at fixed geometry")
series = {
    "water": [(354, 978), (401, 975), (440, 968), (496, 992), (544, 997), (696, 987), (823, 989), (895, 977), (946, 972), (958, 972), (1265, 949)],
    "decane (control; slip)": [(433, 644), (468, 708), (579, 817), (610, 855), (691, 997), (853, 1001), (1266, 985)],
    "hexadecane (control; slip)": [(393, 204), (535, 583), (568, 636), (581, 636), (689, 948), (886, 939), (975, 968), (1040, 960),
                                   (1096, 954), (1143, 919), (1163, 930), (1261, 914)],
    "silicone oil (control; slip)": [(441, 588)],
}
for liq, pts in series.items():
    for (x, y) in pts:
        a = cy(y)
        add(
            source_key=S, location="Fig. 4 (p.4)", data_origin="figure read-off (450 dpi crop)", liquid=liq, size_nm=cx(x),
            size_unc_nm=0.05 * cx(x), eta_ratio=1 / a, eta_ratio_unc=0.05 / a ** 2,
            unc_type="read-off; authors: error bars ~ symbol size (~+-0.05 in ratio); h systematic +-5% -> +-15% in Q",
            notes=("alpha ratio=%.3f; " % a) + ("authors: water equals no-slip theory within uncertainty at all h (40 nm - 2.7 um)"
                                               if liq == "water" else "alpha ratio>1 = slip; eta_app/eta_bulk<1 is an APPARENT value caused by slip"),
            **C)

# ------------------------------------------------------------------ Churaev 1984 (Tables I and II)
S = "Churaev_1984"
CH = dict(wall_material="fused quartz (>99.99% SiO2)",
          wall_treatment="hydrophobised: 5% trimethylchlorosilane in benzene, 80-100 C, 5 h (thetaA 72-91 deg or >90)",
          geometry="cylindrical capillary", size_symbol="r (radius)", electrolyte="none (distilled water ~1e-6 S/cm)",
          method="gas-pressure-driven flow (N2); flow rate via meniscus in a wide capillary; eta_app from slope vs CCl4 control",
          reported_quantity="Delta_eta (%) printed with minus sign; slip coefficient gamma",
          conversion="eta_app/eta0 = 1/(1+|Delta_eta|/100) from Eq.[1] eta=eta0/(1+4 eps/r) and Delta_eta=4 eps/r (printed sign = decrease). "
                     "Alternative reading (eta-eta0)/eta0 would give 1-|Delta_eta|/100 (see note)")
t1 = [(1, 0.88, 0.01, 36.8, 2.8, 8.8e-4, ">90"), (2, 0.88, 0.01, 12.6, 2.8, 3.1e-4, ">90"), (3, 1.5, 0.02, 6.5, 3.0, 2.8e-4, "91"),
      (4, 1.77, 0.06, 6.3, 2.2, 3.2e-4, "81"), (5, 1.77, 0.06, 13.9, 4.8, 7.1e-4, "81"), (6, 1.81, 0.08, 9.9, 3.2, 4.9e-4, "87"),
      (7, 1.83, 0.02, 40.4, 6.0, 20.7e-4, ">90"), (8, 2.4, 0.07, 5.4, 1.2, 3.6e-4, "89"), (9, 4.35, 0.35, 4.3, 1.2, 5.2e-4, "82"),
      (10, 4.6, 0.04, 2.0, 0.7, 2.5e-4, "83"), (11, 5.8, 0.18, 4.3, 1.3, 6.9e-4, "73"), (12, 7.2, 1.0, -0.3, 1.3, 0.0, "72")]
for (n, r, ru, d, du, g, th) in t1:
    q = 1 / (1 + d / 100)
    qu = du / 100 * q ** 2
    add(source_key=S, location="Table I exp. %d (p.575)" % n, data_origin="table",
        liquid="water (hydrophobised wall: SLIP, eta_app<eta0)", size_nm=r * 1000, size_unc_nm=ru * 1000,
        eta_ratio=q, eta_ratio_unc=qu, unc_type="Table I (Delta_eta error propagated)",
        temperature_C="room temperature (22-26 C, Fig. 2 captions)",
        notes="thetaA=%s deg; Delta_eta=%+.1f+-%s %% (printed %+.1f); gamma=%.1e cm3/(dyne s); exp 1 and 7 called 'anomalously high' by authors; "
              "alt. reading -> %.3f" % (th, -d, du, -d, g, 1 - d / 100), **CH)
t2 = [(4.35, 0.35, 8.75, 4.3, 1.5, 3.5e-4, ""), (4.35, 0.35, 24.4, 4.3, 1.2, 5.2e-4, "Table I exp 9"), (4.35, 0.35, 33.2, -0.6, 1.3, 0, ""),
      (4.35, 0.35, 33.4, 0.3, 1.6, 0, ""), (4.35, 0.35, 43.2, -2.0, 1.6, 0, ""),
      (4.6, 0.04, 2.1, 9.1, 0.7, 6.4e-4, ""), (4.6, 0.04, 23.4, 2.0, 0.7, 2.6e-4, "Table I exp 10"), (4.6, 0.04, 34.0, 0.5, 0.9, 0, ""),
      (4.6, 0.04, 34.2, -0.5, 1.2, 0, ""),
      (5.8, 0.18, 8.8, 15.0, 0.7, 16.1e-4, ""), (5.8, 0.18, 24.45, 4.3, 1.3, 6.9e-4, "Table I exp 11"), (5.8, 0.18, 35.4, -1.9, 0.7, 0, "")]
for (r, ru, t, d, du, g, dup) in t2:
    q = 1 / (1 + d / 100)
    qu = du / 100 * q ** 2
    add(source_key=S, location="Table II (p.579)", data_origin="table", liquid="water (hydrophobised wall: SLIP, eta_app<eta0)",
        size_nm=r * 1000, size_unc_nm=ru * 1000, eta_ratio=q, eta_ratio_unc=qu, unc_type="Table II (Delta_eta error propagated)",
        temperature_C=t, notes="Delta_eta (printed)=%+.1f+-%s %%; gamma=%.1e" % (-d, du, g) + ("; DUPLICATE of %s" % dup if dup else ""), **CH)

# ------------------------------------------------------------------ write
with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)
print("rows:", len(rows), "->", OUT)


def sel(src, cond=lambda r: True):
    return [r for r in rows if r["source_key"] == src and cond(r)]


print("\n-- Kazoe 2022: Table-1 mu vs mu recomputed from R (L=100 um; R assumed per single channel) --")
for t in kz_checks:
    print(t[0], t[1], t[2], "mu_table=%.3f  mu_from_R=%.3f (x1e-3 Pa s)" % (t[3], t[4]))
print("mu0 implied by '3.6x at 190 nm': %.3f mPa s" % mu0_implied)
print("\n-- Churaev 1970 Fig.4 water, r<70 nm --")
for r in sel("Churaev_1970", lambda r: r["location"].startswith("Fig. 4") and r["liquid"] == "water" and float(r["size_nm"]) < 70):
    print(r["size_nm"], r["eta_ratio"])
print("\n-- Tas 2004 group means (figure read) --")
for g, lo, hi in [(53, 40, 70), (111, 90, 120), (152, 140, 170)]:
    v = [float(r["eta_ratio"]) for r in sel("Tas_2004", lambda r: r["electrolyte"].startswith("none") and r["location"].startswith("Fig. 4") and lo <= float(r["size_nm"]) < hi)]
    n = [float(r["eta_ratio"]) for r in sel("Tas_2004", lambda r: r["electrolyte"] == "0.1 M NaCl" and lo <= float(r["size_nm"]) < hi)]
    print(g, "demi mean %.3f (n=%d); NaCl mean %.3f (n=%d)" % (st.mean(v), len(v), st.mean(n), len(n)))
print("\n-- Li 2012 --")
for r in sel("Li_2012"):
    print(r["geometry"][:6], r["size_nm"], r["eta_ratio"], r["eta_ratio_unc"])
print("\n-- Cheng 2002 water --")
for r in sel("Cheng_2002", lambda r: r["liquid"] == "water"):
    print(r["size_nm"], r["eta_ratio"])
print("\n-- Churaev 1984: implied bulk viscosity eta0 = Delta*r/(4*gamma) (internal consistency, mPa s) --")
for (n, r, ru, d, du, g, th) in t1:
    if g > 0:
        print("exp", n, "eta0_implied = %.2f mPa s" % (d / 100 * r * 1e-4 / (4 * g) * 100))
