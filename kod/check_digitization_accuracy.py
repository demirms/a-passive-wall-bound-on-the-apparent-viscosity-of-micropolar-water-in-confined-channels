"""Digitization accuracy check.

Purpose
-------
The viscosity data of the paper were partly read off published figures (digitize_capillary_data.py).
For every data point that the source ALSO gives as a number (table or text), this script re-reads
the point from a fresh figure crop and compares the re-read value with the published number.
The mean and the largest deviation are then compared with the measured enhancements and with the
margin to the model bound.

Procedure (order matters, reading comes first)
----------------------------------------------
Stage 1  Figure reading.  Axis calibration is done from tick marks / grid lines that are detected in
         the crop; marker centres are found from colour components (bounding-box centre), from ring
         templates (hollow circles) or from dark-blob centroids.  The published reference values are
         NOT touched in this stage (they live in data/digitization_reference_values.csv and are
         loaded in stage 2).  Result: data/digitization_figure_readings.csv.
Stage 2  Comparison with the published numbers (reference CSV), statistics, source inconsistencies
         listed separately, comparison with measured enhancements and the bound.

Limits stated in the output: the author had seen the table/text numbers while preparing the earlier
digitization (digitize_capillary_data.py), so the reading is "re-read with automatic marker
detection", not a strictly blind experiment.

Run: py -I check_digitization_accuracy.py     (needs numpy, scipy, Pillow, PyMuPDF)
"""
import csv
import hashlib
import math
import pathlib
import re
import statistics as st

import numpy as np
import pymupdf
from PIL import Image
from scipy import ndimage as ndi

ROOT = pathlib.Path(__file__).resolve().parents[1]
ACC = ROOT / "sekil" / "digitize" / "accuracy"
CODE = pathlib.Path(__file__).resolve().parent
DATA = CODE / "data"
LIT = ROOT / "literatur"
VERI = ROOT / "01_arastirma" / "veri_tablosu.csv"
OUT_LINES = []


def log(s=""):
    OUT_LINES.append(s)
    print(s)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


# Source PDFs (or the figure crops made from them in sekil/digitize/accuracy/) required, not distributed (copyrighted). Without it the script stops here and the saved
# output file of the original run is kept unchanged.
if not (LIT.is_dir() or ACC.is_dir()):
    print('SKIPPED: source PDFs required, not distributed: literatur/ (or the crops in sekil/digitize/accuracy/); '
          'saved outputs check_digitization_accuracy_output.txt and data/digitization_*.csv kept')
    raise SystemExit(0)
HASH_BEFORE = sha(VERI)
ACC.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------- crops (regenerated if missing)
# name: (pdf, page, dpi, x0, y0, x1, y1) in page-width/height fractions (page-width/height fractions)
CROPS = {
    "k22_fig5": ("Kazoe_2022_pressure_driven_water_flows_nanochannels.pdf", 6, 500, 0.06, 0.09, 0.49, 0.33),
    "c84_fig3": ("Churaev_1984_slippage_lyophobic_surfaces.pdf", 5, 450, 0.50, 0.04, 0.97, 0.43),
    "h08_fig5": ("Haneveld_2008_capillary_filling_sub10nm.pdf", 4, 400, 0.06, 0.04, 0.47, 0.29),
    "t04_fig4": ("Tas_2004_Capillary_filling_speed_of_water_in_nanochannels.pdf", 2, 450, 0.52, 0.03, 0.97, 0.27),
    "l12_fig4a": ("Li_2012_viscosity_wetting_water_nanospace.pdf", 4, 450, 0.06, 0.37, 0.50, 0.60),
    "k23_fig4": ("Kazoe_2023_liquids_sub100nm_nanochannels.pdf", 4, 450, 0.52, 0.58, 0.98, 0.86),
}


def load(name):
    path = ACC / (name + ".png")
    if not path.exists():
        pdf, pg, dpi, x0, y0, x1, y1 = CROPS[name]
        page = pymupdf.open(str(LIT / pdf))[pg - 1]
        r = page.rect
        clip = pymupdf.Rect(r.x0 + x0 * r.width, r.y0 + y0 * r.height, r.x0 + x1 * r.width, r.y0 + y1 * r.height)
        page.get_pixmap(dpi=dpi, clip=clip).save(str(path))
    return np.array(Image.open(path).convert("RGB")).astype(int)


def runs(mask1d):
    """Centres of connected True-runs of a 1-D boolean array."""
    lab, n = ndi.label(mask1d)
    return [float(np.where(lab == i)[0].mean()) for i in range(1, n + 1)]


def linfit(pix, val):
    a, b = np.polyfit(pix, val, 1)
    return lambda p: a * np.asarray(p, float) + b


def comps(mask, min_area):
    lab, n = ndi.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(ys) >= min_area:
            out.append(dict(x0=xs.min(), x1=xs.max(), y0=ys.min(), y1=ys.max(), n=len(ys), cx=xs.mean(), cy=ys.mean()))
    return out


def marker_centres(cs, up=True):
    """Bounding-box centre of triangle/circle components; partly hidden ones are completed from the median size."""
    W = st.median(c["x1"] - c["x0"] + 1 for c in cs)
    H = st.median(c["y1"] - c["y0"] + 1 for c in cs)
    res = []
    for c in cs:
        w, h = c["x1"] - c["x0"] + 1, c["y1"] - c["y0"] + 1
        xc = (c["x0"] + c["x1"]) / 2
        yc = (c["y0"] + c["y1"]) / 2
        flag = ""
        if w < W - 3:
            xc = c["x1"] - W / 2  # left side hidden (assumes the right edge is visible)
            flag = "partly hidden (x completed)"
        if h < H - 3:
            yc = (c["y0"] + H / 2) if up else (c["y1"] - H / 2)
            flag = (flag + "; " if flag else "") + "partly hidden (y completed)"
        res.append((xc, yc, flag))
    return res


def ring_refine(mask, xy, r_mid, half=14, search=14):
    """Refine the centre of a hollow circle (ring) starting from an approximate position xy=(x,y)."""
    yy, xx = np.mgrid[-24:25, -24:25]
    rr = np.hypot(yy, xx)
    ker = ((rr >= r_mid - 2.0) & (rr <= r_mid + 2.0)).astype(float) * 1.0
    ker -= ((rr > r_mid + 3.0) & (rr <= r_mid + 7.0)).astype(float) * 0.5
    ker -= (rr < r_mid - 4.0).astype(float) * 0.3
    best = (-1e9, xy)
    x0, y0 = int(round(xy[0])), int(round(xy[1]))
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            sub = mask[y0 + dy - 24:y0 + dy + 25, x0 + dx - 24:x0 + dx + 25]
            if sub.shape != ker.shape:
                continue
            s = float((sub * ker).sum())
            if s > best[0]:
                best = (s, (x0 + dx, y0 + dy))
    # sub-pixel: centroid of the ring pixels around the best position
    bx, by = best[1]
    sub = mask[by - 20:by + 21, bx - 20:bx + 21].astype(float)
    yy, xx = np.mgrid[-20:21, -20:21]
    rr = np.hypot(yy, xx)
    w = sub * ((rr >= r_mid - 2.5) & (rr <= r_mid + 2.5))
    return (bx + (w * xx).sum() / w.sum(), by + (w * yy).sum() / w.sum())


READ = []  # figure readings: dict(source, figure, item, kind, value, note)


def add_read(source, figure, item, kind, value, note=""):
    READ.append(dict(source=source, figure=figure, item=item, kind=kind, value=float(value), note=note))


# ============================================================================= STAGE 1: figure reading
log("=" * 100)
log("STAGE 1  Figure reading (reference numbers not loaded yet)")
log("=" * 100)

# ---------------------------------------------------------------- Kazoe 2022 Fig. 5 (linear axes, vector-free raster)
im = load("k22_fig5")
g = im.mean(axis=2)
ytick = runs((g[:, 313:330] < 100).mean(axis=1) > 0.8)  # ticks 4,3,2,1,0 (top to bottom)
yfun = linfit(ytick, [4, 3, 2, 1, 0])
xs_t = [t for t in runs((g[1130:1146, :] < 110).mean(axis=0) > 0.6) if 300 < t < 1700]  # 0 (axis), 500, 1000, 1500
xfun = linfit(xs_t[:4], [0, 500, 1000, 1500])
R, G, B = im[:, :, 0], im[:, :, 1], im[:, :, 2]
red = comps((R > 200) & (G < 80) & (B < 80), 500)
red = [c for c in red if not (c["y0"] < 120 and c["x0"] > 800)]  # drop legend symbol
blue = comps((B > 200) & (R < 80) & (G < 80), 500)
blue = [c for c in blue if not (c["y0"] < 200 and c["x0"] > 800)]
for tag, cs, up in (("sq", sorted(red, key=lambda c: c["x0"]), True), ("pl", sorted(blue, key=lambda c: c["x0"]), True)):
    for k, (xc, yc, flag) in enumerate(marker_centres(cs, up)):
        add_read("Kazoe_2022", "Fig. 5", f"{tag}{k + 1}", "ratio", float(yfun(yc)), flag)
        add_read("Kazoe_2022", "Fig. 5", f"{tag}{k + 1}", "size_nm", float(xfun(xc)), flag)
log(f"Kazoe 2022 Fig. 5: y ticks (px) {[round(v, 1) for v in ytick]}, x ticks (px) {[round(v, 1) for v in xs_t[:4]]}; "
    f"{len(red)} square and {len(blue)} plate markers found")

# ---------------------------------------------------------------- Kazoe 2023 Fig. 4 (log x), open circles = Kazoe 2022 data
im = load("k23_fig4")
g = im.mean(axis=2)
yt = runs((g[:, 293:305] < 110).mean(axis=1) > 0.8)  # picks tick rows near the axis; use the known pairs below
# tick rows detected: 6,5,4,3 and the x axis (0); the 2 and 1 ticks are hidden by thin marks -> fit with detected ones
yt_vals = {}
cand = [(347.0, 6), (478.0, 5), (610.0, 4), (742.0, 3), (1138.0, 0)]
tick_pix, tick_val = [], []
for p0, v in cand:
    near = [t for t in yt if abs(t - p0) < 4]
    if near:
        tick_pix.append(near[0])
        tick_val.append(v)
yfun = linfit(tick_pix, tick_val)
xt = runs((g[1120:1136, :] < 110).mean(axis=0) > 0.8)
x100 = [t for t in xt if abs(t - 475.5) < 4][0]
x1000 = [t for t in xt if abs(t - 1091.5) < 4][0]
xfun_log = lambda p: 10 ** (2 + (p - x100) / (x1000 - x100))
blue = (im[:, :, 2] > 180) & (im[:, :, 0] < 90) & (im[:, :, 1] < 90)
starts = [(647, 662), (843, 809), (1019, 978), (1233, 988)]
for k, s0 in enumerate(starts):
    cx, cy = ring_refine(blue, s0, r_mid=12.5)
    add_read("Kazoe_2023", "Fig. 4 (open circles = Kazoe 2022 data)", f"circ{k + 1}", "ratio", float(yfun(cy)))
    add_read("Kazoe_2023", "Fig. 4 (open circles = Kazoe 2022 data)", f"circ{k + 1}", "size_nm", float(xfun_log(cx)))
log(f"Kazoe 2023 Fig. 4: y tick px {[round(v, 1) for v in tick_pix]}, log-x ticks 100 -> {x100:.1f}, 1000 -> {x1000:.1f}")

# ---------------------------------------------------------------- Churaev 1984 Fig. 3 (scan, slightly tilted axes)
im = load("c84_fig3")
g = im.mean(axis=2)
dark = g < 130
# y ticks (5, 10, 15) are short strokes left of the axis; locate them in windows around the expected rows
ytk = []
for yc in (471, 789, 1106):
    rows = [y for y in range(yc - 14, yc + 15) if dark[y, 172:188].sum() >= 4]
    ytk.append(float(np.mean(rows)))
yfun = linfit(ytk, [15, 10, 5])
# x ticks below the horizontal axis (0.5, 1, 1.5)
xtk = [t for t in runs((g[1435:1455, :] < 130).mean(axis=0) > 0.5) if t > 400]
xfun = linfit(xtk[:3], [0.5, 1.0, 1.5])
# tilt of the y axis: x position of the axis line as a function of y
ax_y, ax_x = [], []
for y in range(380, 1300, 40):
    row = np.where(dark[y, 170:230])[0]
    if len(row):
        ax_y.append(y)
        ax_x.append(170 + float(np.median(row)))
tilt = np.polyfit(ax_y, ax_x, 1)  # axis x = tilt[0]*y + tilt[1]
x_at_tick_row = tilt[0] * 1445 + tilt[1]
x0_origin = float(xfun(x_at_tick_row))  # 1/r value at the axis foot (should be ~0)
log(f"Churaev 1984 Fig. 3: y ticks (px) {[round(v, 1) for v in ytk]}, x ticks (px) {[round(v, 1) for v in xtk[:3]]}, "
    f"axis tilt {tilt[0] * 1000:.2f} px per 1000 px, 1/r at axis foot {x0_origin:+.3f}")
starts = [(897, 625), (607, 1007), (549, 1021), (548, 541), (536, 794), (452, 1077), (336, 1147), (329, 1296),
          (295, 1151), (275, 1443)]
for k, s0 in enumerate(starts):
    cx, cy = ring_refine(dark, s0, r_mid=9.0, search=10)
    # remove the axis tilt: shift x by the axis displacement between the tick row and the point row
    cx_corr = cx - (tilt[0] * cy + tilt[1] - x_at_tick_row)
    add_read("Churaev_1984", "Fig. 3", f"pt{k + 1}", "inv_r_1e6", float(xfun(cx_corr)))
    add_read("Churaev_1984", "Fig. 3", f"pt{k + 1}", "delta_eta_pct", float(yfun(cy)))

# ---------------------------------------------------------------- Haneveld 2008 Fig. 5 (dots on grey bars; one dot = one chip)
im = load("h08_fig5")
g = im.mean(axis=2)
rows = runs((g[:, 540:640] < 90).mean(axis=1) > 0.9)
vals_by_row = {}
# grid lines every 0.2 from 3.0 (top frame) downwards
grid_pix, grid_val = [], []
for p in rows:
    v = 3.0 - round((p - 115.0) / 302.5 / 0.2) * 0.2
    if 0.79 < v < 3.01:
        grid_pix.append(p)
        grid_val.append(v)
yfun = linfit(grid_pix, grid_val)
cols = runs((g[130:760, :] < 90).mean(axis=0) > 0.95)  # left axis (0 nm) and right frame (60 nm)
xfun = linfit([cols[0], cols[1]], [0.0, 60.0])
dk = g < 70
for name, (xa, xb, ya, yb, flag) in {
    "h5": (395, 430, 243, 292, ""),
    "h10": (486, 512, 603, 645, ""),
    "h47": (1075, 1115, 690, 713, "dots merge with the thick C=1.0 line: only the part above the line is visible"),
}.items():
    sub = dk[ya:yb, xa:xb]
    lab, n = ndi.label(sub)
    pts = []
    for i in range(1, n + 1):
        yy, xx = np.where(lab == i)
        if len(yy) >= 15:
            pts.append((xa + xx.mean(), ya + yy.mean(), len(yy)))
    ys = [p[1] for p in pts]
    # mean of the dark-pixel centroid (union of dots): pixel-weighted
    cy = sum(p[1] * p[2] for p in pts) / sum(p[2] for p in pts)
    cx = sum(p[0] * p[2] for p in pts) / sum(p[2] for p in pts)
    add_read("Haneveld_2008", "Fig. 5", name, "C_dotmean", float(yfun(cy)), f"{len(pts)} dark blobs; {flag}")
    add_read("Haneveld_2008", "Fig. 5", name, "size_nm", float(xfun(cx)))
log(f"Haneveld 2008 Fig. 5: {len(grid_pix)} grid lines used; x axis px {[round(c, 1) for c in cols[:2]]}")

# ---------------------------------------------------------------- Tas 2004 Fig. 4 (navy diamonds)
im = load("t04_fig4")
g = im.mean(axis=2)
rows = runs((g[:, 300:560] < 90).mean(axis=1) > 0.9)
rows = [p for p in rows if 250 < p < 1060]  # drop the top frame; keep grid lines 1.3 ... 0.9
vals = [1.3 - 0.05 * k for k in range(len(rows))]
assert len(rows) == 9, rows
yfun = linfit(rows, vals)
xt = runs((g[1038:1052, :] < 90).mean(axis=0) > 0.8)
xt = [t for t in xt if t > 250]
xfun = linfit(xt[:5], [0, 50, 100, 150, 200])
R, G, B = im[:, :, 0], im[:, :, 1], im[:, :, 2]
navy = comps((B - R > 25) & (R < 90) & (B < 170) & (G < 80), 100)
dia = [c for c in navy if 10 <= (c["x1"] - c["x0"] + 1) <= 22 and 10 <= (c["y1"] - c["y0"] + 1) <= 20
       and not (c["x0"] > 900 and c["y0"] < 420)]
assert len(dia) == 12, len(dia)  # 4 diamonds in each of the three height groups
dia.sort(key=lambda c: (round(((c["x0"] + c["x1"]) / 2 - 575) / 280), (c["y0"] + c["y1"])))
for k, c in enumerate(dia):
    cx, cy = (c["x0"] + c["x1"]) / 2, (c["y0"] + c["y1"]) / 2
    grp = 0 if cx < 700 else (1 if cx < 1000 else 2)
    add_read("Tas_2004", "Fig. 4 (demi water diamonds)", f"dia{k + 1}_g{grp}", "ratio", float(yfun(cy)))
    add_read("Tas_2004", "Fig. 4 (demi water diamonds)", f"dia{k + 1}_g{grp}", "size_nm", float(xfun(cx)))
log(f"Tas 2004 Fig. 4: {len(dia)} diamonds found; grid rows (px) {[round(v, 1) for v in rows]}")

# ---------------------------------------------------------------- Li 2012 Fig. 4a (triangles, viscosity axis in 1e-3 Pa s)
im = load("l12_fig4a")
g = im.mean(axis=2)
yt = runs((g[:, 366:376] < 110).mean(axis=1) > 0.8)
pix = [t for t in yt if min(abs(t - c) < 4 for c in (387.5, 590.0, 792.5, 995.5)) is True or
       any(abs(t - c) < 4 for c in (387.5, 590.0, 792.5, 995.5))]
pix = sorted({round(t, 1) for t in pix if any(abs(t - c) < 4 for c in (387.5, 590.0, 792.5, 995.5))})
yfun = linfit(pix, [2.5, 2.0, 1.5, 1.0])
R, G, B = im[:, :, 0], im[:, :, 1], im[:, :, 2]
red = [c for c in comps((R > 200) & (G < 80) & (B < 80), 250) if not (c["y0"] < 480 and c["x0"] > 1000)]
blu = [c for c in comps((B > 200) & (R < 80) & (G < 80), 250) if not (c["y0"] < 540 and c["x0"] > 1000)]
for tag, cs, up in (("sq", sorted(red, key=lambda c: c["x0"]), True), ("pl", sorted(blu, key=lambda c: c["x0"]), False)):
    for k, (xc, yc, flag) in enumerate(marker_centres(cs, up)):
        add_read("Li_2012", "Fig. 4a", f"{tag}{k + 1}", "mu_1e-3Pas", float(yfun(yc)), flag)
log(f"Li 2012 Fig. 4a: y ticks (px) {pix}; {len(red)} square, {len(blu)} plate markers")

# ---------------------------------------------------------------- write the readings
with open(DATA / "digitization_figure_readings.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["source", "figure", "item", "kind", "value", "note"])
    w.writeheader()
    for r in READ:
        w.writerow({**r, "value": "%.5g" % r["value"]})
log(f"Stage 1 finished: {len(READ)} readings written to data/digitization_figure_readings.csv")


def reading(source, item, kind):
    for r in READ:
        if r["source"] == source and r["item"] == item and r["kind"] == kind:
            return r["value"]
    raise KeyError((source, item, kind))


# ============================================================================= STAGE 2: comparison with published numbers
log("")
log("=" * 100)
log("STAGE 2  Comparison with the published numbers (data/digitization_reference_values.csv)")
log("=" * 100)
REF = list(csv.DictReader(open(DATA / "digitization_reference_values.csv", encoding="utf-8")))
mu0_k22 = float([r for r in REF if r["pair_id"] == "MU0_KAZOE2022"][0]["ref_value"])
REF = [r for r in REF if not r["pair_id"].startswith("MU0_")]

pairs = []


def to_ratio_ref(r):
    v = float(r["ref_value"])
    t = r["value_type"]
    if t == "ratio":
        return v
    if t == "mu_kazoe":
        return v / mu0_k22
    if t == "mu_li":
        return v / 0.868  # bulk value stated by Li et al. (26.1 C)
    if t == "delta_eta_pct_plot":
        return v  # handled separately
    raise ValueError(t)


# match readings to reference rows
# Churaev 1984: greedy nearest-neighbour matching in (1/r, delta_eta) because the points carry no labels
c84_ref = [r for r in REF if r["source"] == "Churaev_1984"]
c84_read = [dict(item=f"pt{k + 1}", x=reading("Churaev_1984", f"pt{k + 1}", "inv_r_1e6"),
                 y=reading("Churaev_1984", f"pt{k + 1}", "delta_eta_pct")) for k in range(10)]
used = set()
c84_match = {}
cand = []
for rr in c84_ref:
    for rd in c84_read:
        d = math.hypot((rd["x"] - float(rr["ref_size"])) / 0.1, (rd["y"] - float(rr["ref_value"])) / 2.0)
        cand.append((d, rr["pair_id"], rd["item"]))
for d, pid, it in sorted(cand):
    if pid in c84_match or it in used:
        continue
    c84_match[pid] = (it, d)
    used.add(it)
assert len(c84_match) == len(c84_ref) == 10, (len(c84_match), len(c84_ref))

for r in REF:
    pid, src = r["pair_id"], r["source"]
    flag = r["flag"]
    if src == "Churaev_1984":
        it, dist = c84_match[pid]
        read_pct = reading(src, it, "delta_eta_pct")
        ref_pct = float(r["ref_value"])
        rd_ratio = 1.0 / (1.0 + read_pct / 100.0)
        rf_ratio = 1.0 / (1.0 + ref_pct / 100.0)
        pairs.append(dict(pair_id=pid, source=src, figure=r["figure"], item=it, quantity="eta_app/eta0 (from Delta eta)",
                          read=rd_ratio, ref=rf_ratio, ref_res=float(r["ref_resolution"]) / 100.0 * rf_ratio ** 2,
                          dev=rd_ratio - rf_ratio, flag=flag,
                          extra=f"read {read_pct:.2f} %, table {ref_pct:.2f} % (plot sign), match distance {dist:.2f}"))
        if r["size_ref_value"]:
            pass
        continue
    if r["read_kind"] == "C_dotmean":
        rd = reading(src, r["item"], "C_dotmean")
        rf = float(r["ref_value"])
    elif r["read_kind"] in ("ratio", "mu_1e-3Pas"):
        rd = reading(src, r["item"], r["read_kind"])
        rf = to_ratio_ref(r)
        if r["read_kind"] == "mu_1e-3Pas":
            rd = rd / 0.868
    elif r["read_kind"] == "ratio_group_mean":
        items = [x["item"] for x in READ if x["source"] == src and x["kind"] == "ratio" and x["item"].endswith("_g" + r["item"])]
        rd = st.mean(reading(src, i, "ratio") for i in items)
        rf = float(r["ref_value"])
    else:
        raise ValueError(r)
    res = float(r["ref_resolution"]) if r["ref_resolution"] else 0.0
    if r["value_type"] == "mu_kazoe":
        res = float(r["ref_resolution"]) / mu0_k22
    if r["value_type"] == "mu_li":
        res = float(r["ref_resolution"]) / 0.868
    pairs.append(dict(pair_id=pid, source=src, figure=r["figure"], item=r["item"], quantity="viscosity ratio",
                      read=rd, ref=rf, ref_res=res, dev=rd - rf, flag=flag, extra=r["note"]))

# --- printed pair table
log("")
log("Pairs (deviation = re-read minus published, in viscosity-ratio units):")
log(f"{'pair':<16}{'source':<14}{'item':<10}{'read':>8}{'published':>11}{'dev':>9}{'rel %':>8}{'ref.res':>9}  flag / note")
for p in pairs:
    log(f"{p['pair_id']:<16}{p['source']:<14}{p['item']:<10}{p['read']:>8.3f}{p['ref']:>11.3f}{p['dev']:>+9.3f}"
        f"{100 * p['dev'] / p['ref']:>+8.1f}{p['ref_res']:>9.3f}  {p['flag'] or ''}")

# --- statistics
main = [p for p in pairs if not p["flag"]]
flagged = [p for p in pairs if p["flag"]]


def stats(ps, label):
    d = [p["dev"] for p in ps]
    ad = [abs(x) for x in d]
    rel = [abs(p["dev"]) / p["ref"] for p in ps]
    worst = max(ps, key=lambda p: abs(p["dev"]))
    log(f"{label:<38} n={len(ps):<3} mean|dev|={st.mean(ad):.4f}  RMS={math.sqrt(st.mean([x * x for x in d])):.4f}  "
        f"max|dev|={max(ad):.4f} ({worst['pair_id']})  signed mean={st.mean(d):+.4f}  mean|rel|={100 * st.mean(rel):.2f} %  max|rel|={100 * max(rel):.2f} %")
    return dict(n=len(ps), mean=st.mean(ad), rms=math.sqrt(st.mean([x * x for x in d])), mx=max(ad), rel_mean=st.mean(rel),
                rel_max=max(rel))


log("")
log("Statistics (viscosity-ratio units).  Pairs with a source inconsistency or an unresolved reading are NOT in 'main'.")
enh_pairs = [p for p in main if p["source"] != "Churaev_1984"]
S = stats(enh_pairs, "HEADLINE enhancement data (no Churaev)")
S_all = stats(main, "all non-flagged incl. Churaev 1984")
distinct = {(p["source"] if p["source"] != "Kazoe_2023" else "Kazoe_2022", p["pair_id"].split("_")[1].replace("circ", "sq")) for p in enh_pairs}
log(f"  (headline pairs: {len(enh_pairs)} readings of {len(distinct)} distinct published values; Kazoe 2023 re-plots 3 Kazoe 2022 values)")
for src in sorted({p["source"] for p in main}):
    stats([p for p in main if p["source"] == src], f"  {src}")
c84 = [p for p in main if p["source"] == "Churaev_1984"]
c84_pp = [abs(float(re.search(r"read ([-\d.]+) %", p["extra"]).group(1)) - float(re.search(r"table ([-\d.]+) %", p["extra"]).group(1))) for p in c84]
log(f"  Churaev 1984 in percentage points of Delta eta: mean|dev| = {st.mean(c84_pp):.3f} pp, max = {max(c84_pp):.3f} pp (n={len(c84_pp)}); "
    f"the ratio-unit values above are tiny because Delta eta itself is small (<= 14 %)")
if flagged:
    log("")
    log("Flagged pairs (kept out of the main statistics):")
    for p in flagged:
        log(f"  {p['pair_id']}: read {p['read']:.3f} vs published {p['ref']:.3f} (dev {p['dev']:+.3f}, {100 * p['dev'] / p['ref']:+.1f} %): {p['flag']}")
    stats(pairs, "ALL pairs incl. flagged")

# source-consistency check for the Kazoe 394 nm square point: the two independent figures agree with each other
k22_394 = reading("Kazoe_2022", "sq2", "ratio")
k23_394 = reading("Kazoe_2023", "circ2", "ratio")
log("")
log(f"Kazoe 394 nm point: Kazoe 2022 Fig. 5 -> {k22_394:.3f}; Kazoe 2023 Fig. 4 (same data) -> {k23_394:.3f}; "
    f"Table 1 -> {2.50 / mu0_k22:.3f}.  The two figures agree to {abs(k22_394 - k23_394):.3f}; the table is "
    f"{(2.50 / mu0_k22 - 0.5 * (k22_394 + k23_394)):+.3f} above them (source inconsistency, not a reading error).")

# mu0 sensitivity of the Kazoe references: fit one mu0 to the figure points (mu0-independent check)
kz = [p for p in main if p["source"] == "Kazoe_2022" and p["pair_id"].startswith("K22")]
mu_tab = {r["pair_id"]: float(r["ref_value"]) for r in REF if r["value_type"] == "mu_kazoe"}
if kz:
    fit_mu0 = st.mean(mu_tab[p["pair_id"]] / p["read"] for p in kz)
    resid = [p["read"] - mu_tab[p["pair_id"]] / fit_mu0 for p in kz]
    log(f"Kazoe 2022 with ONE fitted mu0 = {fit_mu0:.3f} mPa s (text-derived value: {mu0_k22:.3f}): "
        f"mean|dev| = {st.mean(abs(x) for x in resid):.4f}, max|dev| = {max(abs(x) for x in resid):.4f} (n={len(resid)})")

# --- size-axis accuracy (x axis)
log("")
log("Size-axis accuracy (re-read size vs published size):")
szp = []
nominal = {r["item"]: float(r["size_ref_value"]) for r in REF if r["source"] == "Kazoe_2022" and r["size_ref_value"]}
order_sq = ["sq1", "sq2", "sq3", "sq4"]
order_pl = ["pl1", "pl2", "pl3"]
for it in order_sq + order_pl:
    if it in nominal:
        rd = reading("Kazoe_2022", it, "size_nm")
        szp.append(("Kazoe_2022 " + it, rd, nominal[it], "nm"))
for k, it in enumerate(["circ1", "circ2", "circ3", "circ4"]):
    szp.append(("Kazoe_2023 " + it, reading("Kazoe_2023", it, "size_nm"), nominal[order_sq[k]], "nm"))
ref_size_c84 = {r["pair_id"]: float(r["ref_size"]) for r in c84_ref}
for pid_, (it_, _d) in sorted(c84_match.items()):
    szp.append(("Churaev_1984 " + pid_[4:] + " (1/r)", reading("Churaev_1984", it_, "inv_r_1e6"), ref_size_c84[pid_], "1e-6/m"))
for lab_, rd, nm, un in szp:
    log(f"  {lab_:<26} read {rd:9.3f} {un}  published {nm:9.3f} {un}  rel dev {100 * (rd - nm) / nm:+.2f} %")
rel_sz = [abs(rd - nm) / nm for _, rd, nm, un in szp if un == "nm"]
rel_sz_c84 = [abs(rd - nm) / nm for _, rd, nm, un in szp if un != "nm"]
log(f"  Churaev 1984 1/r axis (scan with tilted axes): mean|rel dev| = {100 * st.mean(rel_sz_c84):.2f} %, max = {100 * max(rel_sz_c84):.2f} % (n={len(rel_sz_c84)})")
log(f"  size axis (Kazoe, nm): mean|rel dev| = {100 * st.mean(rel_sz):.2f} %, max = {100 * max(rel_sz):.2f} % (n={len(rel_sz)})")

# --- test-retest against the earlier digitization (rows of veri_tablosu.csv that were read from the same figure)
log("")
log("Test-retest against the earlier digitization (veri_tablosu.csv, same figures, informational):")
vt = {r["id"]: r for r in csv.DictReader(open(VERI, encoding="utf-8-sig"))}
rt = []
li_sq = ["R034", "R035", "R036", "R037", "R038", "R039"]
li_pl = ["R040", "R041", "R042", "R043", "R044"]
for k, rid in enumerate(li_sq):
    rt.append(("Li sq%d %s" % (k + 1, rid), reading("Li_2012", f"sq{k + 1}", "mu_1e-3Pas") / 0.868, float(vt[rid]["eta_ratio"])))
for k, rid in enumerate(li_pl):
    rt.append(("Li pl%d %s" % (k + 1, rid), reading("Li_2012", f"pl{k + 1}", "mu_1e-3Pas") / 0.868, float(vt[rid]["eta_ratio"])))
tas_ids = ["R049", "R050", "R051", "R052", "R053", "R054", "R055", "R056", "R057", "R058", "R059", "R060"]
tas_items = sorted([r for r in READ if r["source"] == "Tas_2004" and r["kind"] == "ratio"], key=lambda r: (r["item"].split("_g")[1], -r["value"]))
tas_old = sorted([(vt[i]["notes"], float(vt[i]["eta_ratio"]), i) for i in tas_ids], key=lambda t: t[1])
new_sorted = sorted(r["value"] for r in tas_items)
old_sorted = sorted(t[1] for t in tas_old)
for a, b, i in zip(new_sorted, old_sorted, ["rank%d" % k for k in range(1, 13)]):
    rt.append(("Tas (rank-matched) " + i, a, b))
for k, rid in enumerate(["R084", "R085", "R086", "R087"]):
    rt.append(("Kazoe 2023 circ%d %s" % (k + 1, rid), reading("Kazoe_2023", f"circ{k + 1}", "ratio"), float(vt[rid]["eta_ratio"])))
dd = [a - b for _, a, b in rt]
for lab_, a, b in rt:
    log(f"  {lab_:<28} new {a:7.3f}  old {b:7.3f}  diff {a - b:+.3f}")
log(f"  test-retest (n={len(dd)}): mean|diff| = {st.mean(abs(x) for x in dd):.4f}, RMS = {math.sqrt(st.mean([x * x for x in dd])):.4f}, "
    f"max|diff| = {max(abs(x) for x in dd):.4f}")
RT = dict(n=len(dd), mean=st.mean(abs(x) for x in dd), mx=max(abs(x) for x in dd))

# ============================================================================= comparison scales (measured enhancements, bound)
log("")
log("=" * 100)
log("Comparison with the measured enhancements and the bound (water rows of the paper's data table)")
log("=" * 100)
wtxt = open(CODE / "water_slip_length_output.txt", encoding="utf-8").read()
sec3 = wtxt.split("(3) Every data row")[1].split("(4) Headline")[0]
bm = re.search(r"bound: A = ([\d.]+) ; B\(299 K\) = ([\d.]+) ; B\(284 K\) = ([\d.]+)", sec3)
bA, bB299, bB284 = map(float, bm.groups())
rows = []
for line in sec3.splitlines():
    m = re.match(r"\s*(R\d{3})\s.*\|\s*([YNn])/([YNn])\s*$", line)
    if m:
        rows.append(dict(id=m.group(1), meas=float(line.split("|")[0].split()[-1]), above=m.group(3) == "Y"))
assert len(rows) == 89, len(rows)
orig = {r["id"]: r["data_origin"] for r in vt.values()}
enh = [r["meas"] - 1 for r in rows if r["meas"] > 1]
above = [r for r in rows if r["above"]]
marg = [r["meas"] - bB299 for r in above]
fig_above = [r for r in above if "figure" in orig[r["id"]]]
log(f"water rows: {len(rows)}; rows with measured ratio > 1: {len(enh)}; rows above the 299 K bound ({bB299}): {len(above)} "
    f"(of which read from a figure: {len(fig_above)})")
log(f"measured enhancements (ratio - 1), rows > 1: min {min(enh):.3f}, median {st.median(enh):.3f}, max {max(enh):.3f}")
enh_above = [r["meas"] - 1 for r in above]
log(f"measured enhancements of the above-bound rows: min {min(enh_above):.3f}, median {st.median(enh_above):.3f}")
log(f"margin (measured - bound) of the above-bound rows: min {min(marg):.3f}, median {st.median(marg):.3f}, max {max(marg):.3f}")
mean_dev, max_dev = S["mean"], S["mx"]
log(f"mean re-reading deviation {mean_dev:.4f}, largest {max_dev:.4f}")
log(f"  mean deviation / smallest margin above the bound = {mean_dev / min(marg):.3f}; largest deviation / smallest margin = {max_dev / min(marg):.3f}")
log(f"  mean deviation / median margin = {mean_dev / st.median(marg):.3f}; largest deviation / median margin = {max_dev / st.median(marg):.3f}")
log(f"  mean deviation / smallest enhancement (rows > 1) = {mean_dev / min(enh):.3f}; / median enhancement = {mean_dev / st.median(enh):.3f}")
log(f"  mean deviation / median enhancement of above-bound rows = {mean_dev / st.median(enh_above):.3f}")
still = [r for r in above if r["meas"] - bB299 > max_dev]
still_fig = [r for r in fig_above if r["meas"] - bB299 > max_dev]
still_mean = [r for r in above if r["meas"] - bB299 > mean_dev]
log(f"above-bound rows that stay above the bound if every reading is shifted down by the largest deviation: "
    f"{len(still)} of {len(above)}; by the mean deviation: {len(still_mean)} of {len(above)}")
log(f"  figure-read above-bound rows that stay above after a shift by the largest deviation: {len(still_fig)} of {len(fig_above)}")
smallest = sorted(above, key=lambda r: r["meas"] - bB299)[:6]
log("  smallest margins: " + "; ".join(f"{r['id']} {r['meas']:.3f} (+{r['meas'] - bB299:.3f}, {'figure' if 'figure' in orig[r['id']] else 'table/text'})" for r in smallest))
# additional scales
fine = [p for p in enh_pairs if p["ref_res"] <= 0.01]
mx_fine = max(abs(p["dev"]) for p in fine)
log(f"largest deviation among pairs whose published value is given with resolution <= 0.01: {mx_fine:.4f} "
    f"({max(fine, key=lambda p: abs(p['dev']))['pair_id']}, n={len(fine)}); the 5 nm Haneveld value (2.5) is only given to 0.05")
small_enh = [r for r in rows if 0 < r["meas"] - 1 <= max_dev]
log(f"water rows whose measured enhancement (ratio - 1) is not larger than the largest deviation {max_dev:.3f}: {len(small_enh)} of {len(rows)} "
    f"(these are rows that are practically bulk-like anyway)")
unc = []
for r in rows:
    u = vt[r["id"]]["eta_ratio_unc"]
    if u:
        unc.append(float(u))
log(f"stated measurement uncertainty of the water rows (eta_ratio_unc, authors' error bars or read-off floor): median {st.median(unc):.3f}, "
    f"10th percentile {sorted(unc)[len(unc) // 10]:.3f}, n = {len(unc)}; mean re-reading deviation / median stated uncertainty = {mean_dev / st.median(unc):.3f}")
# the headline rows of the paper: the largest measured value and the sub-100 nm rows
big = sorted(rows, key=lambda r: -r["meas"])[:3]
log("  largest measured ratios: " + "; ".join(f"{r['id']} {r['meas']:.3f}" for r in big))

# ============================================================================= pairs CSV and integrity
with open(DATA / "digitization_accuracy_pairs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["pair_id", "source", "figure", "item", "quantity", "reread", "published", "deviation", "relative_deviation",
                "reference_resolution", "excluded_from_main", "note"])
    for p in pairs:
        w.writerow([p["pair_id"], p["source"], p["figure"], p["item"], p["quantity"], "%.4f" % p["read"], "%.4f" % p["ref"],
                    "%+.4f" % p["dev"], "%+.4f" % (p["dev"] / p["ref"]), "%.4f" % p["ref_res"], "yes" if p["flag"] else "no",
                    (p["flag"] + " | " if p["flag"] else "") + p["extra"]])
HASH_AFTER = sha(VERI)
log("")
log(f"veri_tablosu.csv sha256 before {HASH_BEFORE[:16]}..., after {HASH_AFTER[:16]}...: {'UNCHANGED' if HASH_BEFORE == HASH_AFTER else 'CHANGED (ERROR)'}")
assert HASH_BEFORE == HASH_AFTER
srcs = sorted({p["source"] for p in main})
log(f"main pairs: {len(main)} from {len(srcs)} sources ({', '.join(srcs)}); flagged pairs: {len(flagged)}")
(CODE / "check_digitization_accuracy_output.txt").write_text("\n".join(OUT_LINES) + "\n", encoding="utf-8")
