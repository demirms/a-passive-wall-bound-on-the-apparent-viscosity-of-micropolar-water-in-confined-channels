"""Extended Table 3.

For every group of Table 3 (21 groups, hydrophilic water, 299 K coefficients, BC1 wall) the script prints
 - number of members, source, geometry, size, T (deg C), method class, uncertainty and its type,
 - measured range, BC1 prediction at 299 K (3 decimals; checked against the printed values),
 - measured / predicted, number of members above the 299 K bound 1 + eta_r/eta0 = 1.243,
 - excess over the bound and over the prediction in units of the member's own uncertainty ("error bar").
Then: (2) the 89 water rows / 24 rows above the bound (row ids, selection rule), (3) subsets by measurement method
(pressure-driven vs capillary filling vs filling under applied pressure), (4) the plate-channel size question of
Kazoe et al. 2022 (D_r = depth D = H = 234 nm; alternatives), (5) Fig. 4 / Table 3 membership.

Coefficients: Hansen et al. 2010 Table II (SPC/Fw), MD-MD normalisation, mu = eta0 - eta_r, kappa = 2 eta_r,
gamma = zeta0 + zeta_r (beta does not enter BC1). 299 K as in the paper; the
largest predictions occur at 293 K (<= +1.2e-3 above the 299 K values) while the bound is largest at 299 K.
Predictions: tube, slit/plate: closed forms (paper_common); square: finite differences N = 160 (stretched grid) for
D_h <= 491 nm and the large-size law (19) beyond, exactly as in Table 3.
Output: data_table3_extended_output.txt, and data_table3_members.csv (one line per member row, used by the sensitivity
script).
"""
import csv
import gc
import os
import re
import numpy as np
from paper_common import TABLE2, eringen, lam, water_tube, water_slit, water_slip
from micropolar_duct_2d import solve_duct, stretch_for

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []


def P(s=""):
    print(s, flush=True)
    OUT.append(s)


E0, ER, Z0, ZR = [x for x in TABLE2 if x[0] == 299][0][1:]
COEF299 = (E0, ER, Z0 + ZR)
BOUND299 = 1 + ER / E0

rows = list(csv.DictReader(open(os.path.join(HERE, "..", "01_arastirma", "veri_tablosu.csv"), encoding="utf-8-sig")))
byid = {r["id"]: r for r in rows}
ids = lambda a, b: [f"R{n:03d}" for n in range(a, b + 1)]

# method classes (from the 'method' column of the table, sources read in full text)
METHOD = {
    "Churaev_1970": "pressure-driven",              # gas-pressure-driven meniscus flow, eta from slope of v(P)
    "Kazoe_2022": "pressure-driven",                # mass flowmetry, dP <= 2 MPa, C = dP/Q
    "Kazoe_2023": "pressure-driven",
    "Cheng_2002": "pressure-driven",                # (photoresist walls: separate set, not in Table 3)
    "Li_2012": "filling under applied pressure",    # Washburn with 0-2 MPa external pressure; eta from slope
    "Tas_2004": "capillary filling",                # spontaneous filling, x = a sqrt(t)
    "Haneveld_2008": "capillary filling",           # spontaneous filling with loss term
}

# Table 3 groups: (label, kind, ids, printed prediction)
GROUPS = [
    ("Churaev [1] tube r 41-60", "tube", ids(1, 3), "1.025-1.036"),
    ("Churaev [1] tube r 181-255", "tube", ids(4, 8), "1.006-1.008"),
    ("Churaev [1] tube r 547-998", "tube", ids(9, 11), "1.002-1.003"),
    ("Haneveld [3] slit H 10.5", "slit", ["R046"], "1.162"),
    ("Haneveld [3] slit H 22.4", "slit", ["R047"], "1.092"),
    ("Haneveld [3] slit H 46", "slit", ["R048"], "1.048"),
    ("Tas [2] slit H 53", "slit", ids(49, 52) + ["R070"], "1.041-1.042"),
    ("Tas [2] slit H 102-161", "slit", ids(53, 60) + ["R071"], "1.014-1.022"),
    ("Tas [2] 0.1 M NaCl slit H 53-161", "slit", ids(61, 69), "1.014-1.042"),
    ("Li [6] plate H 25-415", "slit", ids(40, 44), "1.006-1.084"),
    ("Kazoe [4] plate H 234", "slit", ["R077"], "1.010"),
    ("Kazoe [4] plate H 417-745", "slit", ["R078", "R079"], "1.003-1.005"),
    ("Kazoe [5] square Dh 73", "sq", ["R080"], "1.040"),
    ("Kazoe [5] square Dh 119", "sq", ["R081"], "1.025"),
    ("Kazoe [4] square Dh 190", "sq", ["R072"], "1.016"),
    ("Li [6] square Dh 200", "sq", ["R034"], "1.015"),
    ("Li [6] square Dh 393", "sq", ["R035"], "1.008"),
    ("Kazoe [4] square Dh 394", "sq", ["R073"], "1.008"),
    ("Li [6] square Dh 491", "sq", ["R036"], "1.006"),
    ("Kazoe [4] square Dh 760-1691", "sq", ["R074", "R075"], "1.002-1.004"),
    ("Li [6] square Dh 838-1530", "sq", ids(37, 39), "1.002-1.004"),
]
EXTRA_FIG_VALUE = {"R073": 2.487}   # figure value of the 394 nm point (Kazoe 2023 Fig. 4, row R085); Table 1 of [4] gives 2.804

FD = {}


def pred(kind, s, coef=COEF299):
    e0, er, z = coef
    if kind == "tube":
        return water_tube(s, e0, er, z)
    if kind == "slit":
        return water_slit(s, e0, er, z)
    if s <= 491:
        key = (s, coef)
        if key not in FD:
            mu, k, g = eringen(e0, er, z)
            L = lam(mu, k, g)
            st = stretch_for(float(s * L))
            FD[key] = solve_duct(float(s), float(s), mu, k, g, 0.0, 160, 160, st, st)[0] / e0
            gc.collect()
        return FD[key]
    return 1 + water_slip(e0, er, z)[1] / s


def fmt(lo, hi, nd=3):
    a, b = f"{lo:.{nd}f}", f"{hi:.{nd}f}"
    return a if a == b else f"{a}-{b}"


def tnum(txt):
    m = re.match(r"\s*~?\s*([\d.]+)", txt)
    return float(m.group(1)) if m else None


P("(1) Extended Table 3 (hydrophilic water, BC1 wall, 299 K coefficients; bound 1 + eta_r/eta0 = %.4f)" % BOUND299)
P("    'sigma' = the member's own uncertainty in column eta_ratio_unc (author error bar if given, else read-off, whichever is larger);")
P("    excess = (measured - reference)/sigma. Method classes: PD pressure-driven; CF capillary filling; FP filling under applied pressure.")
P(f"    {'group':34s} {'n':>2s} {'cl':>2s} {'T(C)':>12s} {'meas':>11s} {'unc':>11s} {'pred':>11s} {'printed':>11s} {'ok':>3s} {'meas/pred':>10s} {'>bnd':>5s} {'max exc bnd':>11s} {'max exc pred':>12s}")
members = []
n_ok = 0
for lab, kind, mem, printed in GROUPS:
    meas = []
    unc = []
    sizes = []
    for i in mem:
        r = byid[i]
        m, u = float(r["eta_ratio"]), float(r["eta_ratio_unc"])
        meas.append(m)
        unc.append(u)
        sizes.append(float(r["size_nm"]))
    pr = [pred(kind, s) for s in sizes]
    cls = {METHOD[byid[i]["source_key"]] for i in mem}
    assert len(cls) == 1
    cl = {"pressure-driven": "PD", "capillary filling": "CF", "filling under applied pressure": "FP"}[cls.pop()]
    ts = [byid[i]["temperature_C"] for i in mem]
    tlist = sorted({t for t in ts})
    tt = ts[0] if len(tlist) == 1 else (str(min(tnum(t) for t in ts)) + "-" + str(max(tnum(t) for t in ts)))
    tt = tt.replace(" +- 0.5", "").strip()[:12]
    ratio = [m / p for m, p in zip(meas, pr)]
    nab = sum(m > BOUND299 for m in meas)
    ex_b = max((m - BOUND299) / u for m, u in zip(meas, unc))
    ex_p = max((m - p) / u for m, p, u in zip(meas, pr, unc))
    ok = fmt(min(pr), max(pr)) == printed
    n_ok += ok
    P(f"    {lab:34s} {len(mem):2d} {cl:>2s} {tt:>12s} {fmt(min(meas), max(meas), 2):>11s} {fmt(min(unc), max(unc), 3):>11s} {fmt(min(pr), max(pr)):>11s} {printed:>11s} {'yes' if ok else 'NO':>3s} {fmt(min(ratio), max(ratio), 2):>10s} {nab:>2d}/{len(mem):<2d} {ex_b:+11.2f} {ex_p:+12.2f}")
    for i, s, m, u, p in zip(mem, sizes, meas, unc, pr):
        members.append((lab, i, byid[i]["source_key"], kind, s, m, u, byid[i]["unc_type"], p, cl))
P(f"    prediction column reproduces Table 3: {n_ok}/{len(GROUPS)} groups")
P("    note: the 394 nm square of Kazoe [4] has Table 1 value 2.804 (R073) and figure value 2.487 (Kazoe 2023 Fig. 4, row R085); the paper prints 2.45-2.80 (figure read 2.45).")

# uncertainty type per group
P("\n    Uncertainty type per group (as recorded in veri_tablosu.csv, column unc_type):")
for lab, kind, mem, printed in GROUPS:
    types = sorted({byid[i]["unc_type"][:70] for i in mem})
    P(f"    {lab:34s} {' | '.join(types)}")

# ---------------------------------------------------------------------------- (2) 89 / 24
P("\n(2) Water rows counted as '89' and '24 above the bound' (same filter as compare_data_md_bound.py / paper_numbers.py (7))")
P("    selection rule: liquid == water, not a DUPLICATE row (4 re-plotted Kazoe 2023 points), no hydrophobised wall and not Churaev 1984;")
P("    rows with a numeric size and eta_ratio; every electrolyte and every wall material kept (so: salt solutions, photoresist-walled slits of Cheng 2002, the 5 nm slit R045).")
sel = []
for r in rows:
    if r["liquid"].strip().lower() != "water" or "DUPLICATE" in r["notes"]:
        continue
    if "hydrophob" in (r["wall_treatment"] + r["notes"]).lower() or r["source_key"] == "Churaev_1984":
        continue
    try:
        float(r["size_nm"]), float(r["eta_ratio"])
    except ValueError:
        continue
    sel.append(r)
above = [r for r in sel if float(r["eta_ratio"]) > BOUND299]
P(f"    selected rows: {len(sel)}; above the 299 K bound {BOUND299:.4f}: {len(above)}")
from collections import Counter
P("    selected rows per source: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r['source_key'] for r in sel).items())))
P("    rows above the bound per source: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r['source_key'] for r in above).items())))
P("    ids above the bound (24): " + " ".join(r["id"] for r in above))
inT3 = {i for _, _, mem, _ in GROUPS for i in mem}
P("    selected rows NOT in Table 3 / Fig. 4 (and why):")
why = {}
for r in sel:
    if r["id"] in inT3:
        continue
    i = r["id"]
    if r["source_key"] == "Cheng_2002":
        w = "photoresist-walled slit (not silica) [24]"
    elif i == "R045":
        w = "5 nm slit of [3], excluded (below validity limit; collapse suspected by its authors)"
    elif i == "R076":
        w = "10 mM KCl square 190 nm of [4] (salt solution; Table 3 lists the pure-water point)"
    elif r["source_key"] == "Churaev_1970":
        w = "Fig. 5 temperature series (r = 48 and 170 nm), shown in Fig. 5 instead"
    else:
        w = "other"
    why.setdefault(w, []).append(i)
for w, l in why.items():
    P(f"      {len(l):2d} rows: {w}: {' '.join(l)}")
P(f"    Table 3 member rows: {len(inT3)}; of these above the 299 K bound: {sum(float(byid[i]['eta_ratio']) > BOUND299 for i in inT3)}")

# ---------------------------------------------------------------------------- (3) method subsets
P("\n(3) Subsets by measurement method (Table 3 member rows only; bound %.3f; 'meas > pred' = measured above the BC1 prediction)" % BOUND299)
for cl, name in (("PD", "pressure-driven (Churaev [1], Kazoe [4, 5])"), ("CF", "capillary filling (Tas [2], Haneveld [3])"),
                 ("FP", "filling under applied pressure (Li [6])")):
    mm = [m for m in members if m[9] == cl]
    nab = [m for m in mm if m[5] > BOUND299]
    nab1 = [m for m in mm if m[5] - m[6] > BOUND299]
    npr = [m for m in mm if m[5] > m[8]]
    npr1 = [m for m in mm if m[5] - m[6] > m[8]]
    P(f"    {name}: {len(mm)} rows; above the bound {len(nab)} (by more than one error bar: {len(nab1)}); above the prediction {len(npr)} (by more than one error bar: {len(npr1)})")
    P("      above the bound: " + (", ".join(f"{m[1]} ({m[4]:g} nm, {m[5]:.2f}+-{m[6]:.2f})" for m in nab) or "none"))
pdm = [m for m in members if m[9] == "PD"]
P("\n    Pressure-driven subset alone, by geometry (measured / BC1 prediction, 299 K):")
for kind, nm in (("tube", "tubes (Churaev)"), ("slit", "plates (Kazoe [4])"), ("sq", "squares (Kazoe [4, 5])")):
    mm = [m for m in pdm if m[3] == kind]
    if not mm:
        continue
    rt = [m[5] / m[8] for m in mm]
    nab = sum(m[5] > BOUND299 for m in mm)
    P(f"      {nm}: {len(mm)} rows, measured {min(m[5] for m in mm):.3f}-{max(m[5] for m in mm):.3f}, prediction {min(m[8] for m in mm):.3f}-{max(m[8] for m in mm):.3f}, "
      f"measured/predicted {min(rt):.2f}-{max(rt):.2f}, above the bound: {nab}")
sq_small = [m for m in pdm if m[3] == "sq" and m[4] <= 491]
P(f"      squares with D_h <= 491 nm (pressure-driven): {len(sq_small)} rows: measured/predicted {min(m[5] / m[8] for m in sq_small):.2f}-{max(m[5] / m[8] for m in sq_small):.2f} "
  f"(all above the bound: {all(m[5] > BOUND299 for m in sq_small)})")
tubes_small = [m for m in pdm if m[3] == "tube" and m[4] <= 60]
P(f"      tubes r = 41-60 nm: measured {min(m[5] for m in tubes_small):.3f}-{max(m[5] for m in tubes_small):.3f} vs prediction {min(m[8] for m in tubes_small):.3f}-{max(m[8] for m in tubes_small):.3f} (all above the bound: {all(m[5] > BOUND299 for m in tubes_small)})")
fs = [m for m in members if m[9] != "PD" and m[3] == "sq"]
P(f"      (for comparison, filling-type squares (Li [6]) D_h <= 491: measured/predicted "
  f"{min(m[5] / m[8] for m in fs if m[4] <= 491):.2f}-{max(m[5] / m[8] for m in fs if m[4] <= 491):.2f})")
# whole-CSV count for the Churaev Fig. 5 temperature series (pressure-driven, not in Table 3)
ser = [r for r in sel if r["source_key"] == "Churaev_1970" and r["id"] not in inT3]
P(f"    Churaev Fig. 5 temperature series (pressure-driven, not in Table 3): {len(ser)} rows, above the 299 K bound {sum(float(r['eta_ratio']) > BOUND299 for r in ser)}")
pd_all = [r for r in sel if METHOD[r['source_key']] == 'pressure-driven' and r['source_key'] != 'Cheng_2002']
P(f"    All pressure-driven hydrophilic-silica/quartz water rows of the 89 (Churaev 1970, Kazoe 2022/2023): {len(pd_all)} rows, above the bound: {sum(float(r['eta_ratio']) > BOUND299 for r in pd_all)}")
P("    -> the conclusion (square channels 1.75-5.4 times the prediction, tubes 1.3 times, bound exceeded) holds on the pressure-driven sets alone.")

# ---------------------------------------------------------------------------- (4) plate channel size
P("\n(4) Plate channel of Kazoe et al. 2022 [4]: size convention")
W, D = 10000.0, 234.0
P("    Source (Kazoe 2022, J. Anal. Sci. 38, p. 283 and Table 1 p. 283): 'For plate nanochannels, ... D_h and D were used as the representative size D_r of the square- and plate-nanochannels, respectively.'")
P(f"    Table 1: D_r = 234 +- 3 nm, W = 10000 +- 15 nm, D = 234 +- 3 nm (so D_r = depth D = gap H); resistance C = 12 mu L/(W D^3) (Eq. 3).")
P(f"    -> in the slit model the gap is H = D = 234 nm (half-gap 117 nm), not 117 nm. The manuscript's D_h = 2H = 468 nm is the conventional hydraulic diameter of a plate (2WD/(W+D) = {2 * W * D / (W + D):.1f} nm for W = 10 um).")
for lab, Hn in (("H = D = 234 nm (source's D_r; used in Table 3)", 234.0), ("alternative: D_r read as D_h, i.e. H = 117 nm", 117.0),
                ("hydraulic diameter 2WD/(W+D) = 457 nm read as H (inconsistent; for completeness)", 2 * W * D / (W + D))):
    p = pred("slit", Hn)
    P(f"    {lab:84s} BC1 prediction {p:.4f} (299 K); measured 1.312 +- 0.067; bound {BOUND299:.4f}")
r77 = byid["R077"]
m77, u77 = float(r77["eta_ratio"]), float(r77["eta_ratio_unc"])
P(f"    R077 measured {m77:.3f} +- {u77:.3f}: above the 299 K bound {BOUND299:.4f} by {m77 - BOUND299:.3f} ({(m77 - BOUND299) / u77:.2f} sigma); the lower end {m77 - u77:.3f} is {'above' if m77 - u77 > BOUND299 else 'below'} the bound (299 K); 284 K bound {1 + 0.17 / 0.92:.3f}.")
P("    Source's interpretation (Kazoe 2022, p. 286): 'The 1.3 times higher water viscosity in the plate nanochannel with D_r = 234 nm ... is in agreement with the results of previous studies; therefore, this increased viscosity is considered to be due to electroviscous effects.'")
P("    Same paper, same page: for the square 190 nm channel the 10 mM KCl value (3.07e-3 Pa s; text 3.08e-3) is 'almost the same as that of pure water', 'therefore, the electroviscous effect is not a dominant factor' for the square channels.")
P("    Consequence for the manuscript: the plate point R077 is above the bound only marginally (0.07 above, about one error bar) and its source attributes it to electroviscous effects, which the model excludes; the abstract should not list plate channels among the geometries that exceed the bound.")

with open(os.path.join(HERE, "data_table3_members.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group", "id", "source_key", "kind", "size_nm", "eta_ratio", "unc", "unc_type", "pred_299K_BC1", "method_class"])
    for m in members:
        w.writerow([m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], f"{m[8]:.6f}", m[9]])

with open(os.path.join(HERE, "data_table3_extended_output.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(OUT) + "\n")
