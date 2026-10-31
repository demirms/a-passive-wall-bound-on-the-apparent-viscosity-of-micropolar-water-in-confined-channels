"""Number checks and the normalisation audit of the points above the bound.
(i)   measured/predicted for the pressure-driven square channels and tubes (must agree with data_table3_extended_output.txt)
(ii)  eta_r/eta0 = N^2/(1-N^2) for N = 0.65 and kappa/mu = 5.25
(iii) excess over the bound in units of sigma for the groups above the bound (299 K bound)
(iv)  the 24 points above the 299 K bound: id, source, reported quantity, conversion, T, bound at T,
      difference to the largest bound of Table 1 (1.333 at 319 K); how many stay above it.
Output: data_points_above_bound_output.txt
"""
import csv
import os
import re
from paper_common import TABLE2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = []


def P(s=""):
    print(s, flush=True)
    OUT.append(str(s))


mem = list(csv.DictReader(open(os.path.join(HERE, "data_table3_members.csv"), encoding="utf-8")))
BOUND299 = 1 + 0.17 / 0.70

# ----------------------------------------------------------------------------- (i)
P("(i) Measured / predicted, pressure-driven rows (BC1, 299 K)")
for lab, mv, pv in (("Kazoe [4] 394 nm, figure value", 2.45, 1.008), ("Kazoe [4] 394 nm, Table 1 value", 2.80, 1.008), ("Kazoe [5] 119 nm", 5.55, 1.025)):
    P(f"    {lab}: {mv}/{pv} = {mv/pv:.2f}")
sq = [m for m in mem if m["kind"] == "sq" and m["method_class"] == "PD" and float(m["size_nm"]) <= 491]
rat = {m["id"]: float(m["eta_ratio"]) / float(m["pred_299K_BC1"]) for m in sq}
P("    members (pressure-driven squares, D_h <= 491 nm): " + ", ".join(f"{k} ({[float(m['size_nm']) for m in sq if m['id']==k][0]:.0f} nm) {v:.2f}" for k, v in rat.items()))
P(f"    range {min(rat.values()):.2f}-{max(rat.values()):.2f}   (data_table3_extended_output.txt: 2.78-5.41)")
assert round(min(rat.values()), 2) == 2.78 and round(max(rat.values()), 2) == 5.41
allsq = [m for m in mem if m["kind"] == "sq" and float(m["size_nm"]) <= 491]
r_all = [float(m["eta_ratio"]) / float(m["pred_299K_BC1"]) for m in allsq]
P(f"    all squares D_h <= 491 nm (n = {len(allsq)}, incl. filling-type squares of Li [6]): measured/predicted {min(r_all):.2f}-{max(r_all):.2f}")
P("    manuscript range '2.4-5.4': lower end 2.43 = Li [6] 200 nm (filling under applied pressure) or 2.43 = 2.45/1.008 (figure value of the 394 nm square); upper end 5.41 (Kazoe [5], 119 nm).")
tb = [m for m in mem if m["kind"] == "tube" and float(m["size_nm"]) <= 60]
rt = [float(m["eta_ratio"]) / float(m["pred_299K_BC1"]) for m in tb]
P(f"    tubes r = 41-60 nm (n = {len(tb)}): measured {min(float(m['eta_ratio']) for m in tb):.3f}-{max(float(m['eta_ratio']) for m in tb):.3f}, measured/predicted {min(rt):.2f}-{max(rt):.2f} (about 1.3-1.4 in round numbers)")
assert (round(min(rt), 2), round(max(rt), 2)) == (1.29, 1.37)

# ----------------------------------------------------------------------------- (ii)
P("\n(ii) eta_r/eta0 from the coupling number N and from kappa/mu")
N = 0.65
P(f"    N = {N}: N^2/(1-N^2) = {N**2/(1-N**2):.4f}")
P(f"    kappa/mu = 5.25: (kappa/2)/(mu + kappa/2) = {2.625/(1+2.625):.4f}")
assert round(N**2 / (1 - N**2), 4) == 0.7316 and round(2.625 / (1 + 2.625), 4) == 0.7241

# ----------------------------------------------------------------------------- (iii)
P("\n(iii) excess over the 299 K bound 1.2429 in units of sigma (own uncertainty of the member), groups above the bound")
groups = {}
for m in mem:
    groups.setdefault((m["kind"], m["group"]), []).append(m)
ex = {}
for (kd, g), ms in groups.items():
    e = max((float(m["eta_ratio"]) - BOUND299) / float(m["unc"]) for m in ms)
    nab = sum(float(m["eta_ratio"]) > BOUND299 for m in ms)
    ex[(kd, g)] = (e, nab, len(ms))
for kd in ("sq", "tube"):
    sel = {g: v for (k, g), v in ex.items() if k == kd and v[1] > 0}
    P(f"    {kd}: groups above the bound: " + "; ".join(f"{g} {v[0]:+.2f} sigma ({v[1]}/{v[2]})" for g, v in sel.items()))
    P(f"      range {min(v[0] for v in sel.values()):+.1f} ... {max(v[0] for v in sel.values()):+.1f} sigma")
sqe = [v[0] for (k, g), v in ex.items() if k == "sq" and v[1] > 0]
tbe = [v[0] for (k, g), v in ex.items() if k == "tube" and v[1] > 0]
assert round(min(sqe), 1) == 3.0 and round(max(sqe), 1) == 8.8 and round(max(tbe), 1) == 10.9

# ----------------------------------------------------------------------------- (iv)
P("\n(iv) the points above the 299 K bound; normalisation and the bound at the measurement temperature")
Ts = [t[0] for t in TABLE2]
xr = [t[2] / t[1] for t in TABLE2]          # eta_r/eta0
BMAX = 1 + max(xr)
P("    Table 1 range 284-319 K (10.85-45.85 C); eta_r/eta0: " + ", ".join(f"{T} K {x:.4f}" for T, x in zip(Ts, xr)) + f"; largest bound {BMAX:.4f} (319 K)")


def bound_at(TC):
    if TC is None:
        return BMAX
    TK = TC + 273.15
    if TK <= Ts[0]:
        return 1 + xr[0]
    if TK >= Ts[-1]:
        return 1 + xr[-1]
    for i in range(len(Ts) - 1):
        if Ts[i] <= TK <= Ts[i + 1]:
            w = (TK - Ts[i]) / (Ts[i + 1] - Ts[i])
            return 1 + xr[i] + w * (xr[i + 1] - xr[i])


def parse_T(s):
    s = s.strip()
    if s.lower().startswith("not stated"):
        return None
    if s.startswith("room temperature"):
        return 24.0     # midpoint of 22-26 C
    mm = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(mm.group(1))


rows = list(csv.DictReader(open(os.path.join(HERE, "..", "01_arastirma", "veri_tablosu.csv"), encoding="utf-8-sig")))
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
P(f"    selected rows {len(sel)}, above the 299 K bound {len(above)}")
assert len(sel) == 89 and len(above) == 24
P(f"    {'id':5s} {'source':14s} {'T [C]':>6s} {'meas':>6s} {'unc':>6s} {'bound(T)':>8s} {'meas-bnd(T)':>11s} {'meas-1.333':>10s} {'>1.333':>6s} {'by>1sig':>7s}  reported_quantity | conversion")
n_above = n_above_sig = n_bound_T = 0
for r in above:
    T = parse_T(r["temperature_C"])
    b = bound_at(T)
    y, u = float(r["eta_ratio"]), float(r["eta_ratio_unc"] or 0)
    d = y - BMAX
    n_bound_T += y > b
    n_above += y > BMAX
    sg = (y - u) > BMAX
    n_above_sig += sg
    Ts_ = "n/s" if T is None else f"{T:.1f}"
    P(f"    {r['id']:5s} {r['source_key']:14s} {Ts_:>6s} {y:6.3f} {u:6.3f} {b:8.4f} {y-b:+11.3f} {d:+10.3f} {'yes' if y > BMAX else 'no':>6s} {'yes' if sg else 'no':>7s}  {r['reported_quantity'][:45]} | {r['conversion'][:70]}")
P(f"    above their own bound(T): {n_bound_T}/24; above the most favourable bound {BMAX:.3f} even so: {n_above}/24; also by more than one error bar: {n_above_sig}/24")
P("    (T = 'n/s': not stated, largest bound used; 'room temperature (22-26 C)': 24 C; outside 11-46 C the end value of Table 1, no extrapolation.)")
if n_above < 24:
    P(f"    WARNING: only {n_above} of the 24 points stay above the most favourable bound (1.333); {24-n_above} fall below it: " + ", ".join(r["id"] for r in above if float(r["eta_ratio"]) <= BMAX))
with open(os.path.join(HERE, "data_points_above_bound_output.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")
