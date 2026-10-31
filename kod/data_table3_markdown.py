"""Prints the proposed extended Table 3 as a Markdown table (from data_table3_members.csv written by data_table3_extended.py).
Rounding: half up. Columns: data set, method class, geometry, size, T, measured, uncertainty (+ type), BC1 prediction at
299 K, measured/predicted, members above the 299 K bound and the largest excess in units of the member's own uncertainty.
Output: data_table3_markdown_output.txt
"""
import csv
import os
from decimal import Decimal, ROUND_HALF_UP

HERE = os.path.dirname(os.path.abspath(__file__))
BOUND = 1 + 0.17 / 0.70
mem = list(csv.DictReader(open(os.path.join(HERE, "data_table3_members.csv"), encoding="utf-8")))


def r(x, nd):
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def rng(vals, nd):
    a, b = r(min(vals), nd), r(max(vals), nd)
    return a if a == b else f"{a}-{b}"


META = {   # group -> (data set, geometry label, size label, T, uncertainty type)
    "Churaev [1] tube r 41-60": ("Churaev *et al.* [1]", "tube, $r$", "41-60", "~20", "R"),
    "Churaev [1] tube r 181-255": ("Churaev *et al.* [1]", "tube, $r$", "181-255", "~20", "R"),
    "Churaev [1] tube r 547-998": ("Churaev *et al.* [1]", "tube, $r$", "547-998", "~20", "R"),
    "Haneveld [3] slit H 10.5": ("Haneveld *et al.* [3]", "slit, $H$", "10.5", "22", "A"),
    "Haneveld [3] slit H 22.4": ("Haneveld *et al.* [3]", "slit, $H$", "22.4", "22", "B"),
    "Haneveld [3] slit H 46": ("Haneveld *et al.* [3]", "slit, $H$", "46", "21", "A"),
    "Tas [2] slit H 53": ("Tas *et al.* [2]", "slit, $H$", "53", "21.5-22", "A"),
    "Tas [2] slit H 102-161": ("Tas *et al.* [2]", "slit, $H$", "102-161", "21.5-22", "A"),
    "Tas [2] 0.1 M NaCl slit H 53-161": ("Tas *et al.* [2], 0.1 M NaCl", "slit, $H$", "53-161", "21.5-22", "A"),
    "Li [6] plate H 25-415": ("Li *et al.* [6]", "plate, $H$", "25-415", "26.1", "B"),
    "Kazoe [4] plate H 234": ("Kazoe *et al.* [4]", "plate, $H$", "234", "n.s.", "A"),
    "Kazoe [4] plate H 417-745": ("Kazoe *et al.* [4]", "plate, $H$", "417-745", "n.s.", "A"),
    "Kazoe [5] square Dh 73": ("Kazoe *et al.* [5]", "square, $D_h$", "73", "n.s.", "B"),
    "Kazoe [5] square Dh 119": ("Kazoe *et al.* [5]", "square, $D_h$", "119", "n.s.", "B"),
    "Kazoe [4] square Dh 190": ("Kazoe *et al.* [4]", "square, $D_h$", "190", "n.s.", "A"),
    "Li [6] square Dh 200": ("Li *et al.* [6]", "square, $D_h$", "200", "26.1", "B"),
    "Li [6] square Dh 393": ("Li *et al.* [6]", "square, $D_h$", "393", "26.1", "B"),
    "Kazoe [4] square Dh 394": ("Kazoe *et al.* [4]", "square, $D_h$", "394", "n.s.", "A"),
    "Li [6] square Dh 491": ("Li *et al.* [6]", "square, $D_h$", "491", "26.1", "B"),
    "Kazoe [4] square Dh 760-1691": ("Kazoe *et al.* [4]", "square, $D_h$", "760-1691", "n.s.", "A"),
    "Li [6] square Dh 838-1530": ("Li *et al.* [6]", "square, $D_h$", "838-1530", "26.1", "B"),
}
CL = {"PD": "PD", "CF": "CF", "FP": "FP"}
groups = []
for m in mem:
    if not groups or groups[-1][0] != m["group"]:
        groups.append((m["group"], []))
    groups[-1][1].append(m)

out = []
out.append("| Data set | Method | Geometry | Size (nm) | $T$ (C) | Measured | Uncertainty | Predicted | Measured / predicted | Above bound | Largest excess ($\\sigma$) |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|")
for g, ms in groups:
    ds, geo, size, T, ut = META[g]
    meas = [float(m["eta_ratio"]) for m in ms]
    unc = [float(m["unc"]) for m in ms]
    pr = [float(m["pred_299K_BC1"]) for m in ms]
    rt = [a / b for a, b in zip(meas, pr)]
    k = sum(a > BOUND for a in meas)
    ex = max((a - BOUND) / u for a, u in zip(meas, unc))
    nd = 3 if max(meas) < 1.1 and max(meas) - min(meas) < 0.05 and False else 2
    out.append(f"| {ds} | {CL[ms[0]['method_class']]} | {geo} | {size} | {T} | {rng(meas, 2)} | {rng(unc, 2)} ({ut}) | {rng(pr, 3)} | {rng(rt, 2)} | {k} of {len(ms)} | {ex:+.1f} |")
out.append("")
out.append("Method: PD pressure-driven flow; CF capillary filling; FP filling under applied pressure. Uncertainty type: A stated by the authors (text or table), B error bar read off the figure, R reading uncertainty only. n.s.: not stated in the source.")
out.append("Excess: (measured - 1.243)/uncertainty of the same row, largest over the group; negative when the whole group lies below the bound.")
text = "\n".join(out)
print(text)
with open(os.path.join(HERE, "data_table3_markdown_output.txt"), "w", encoding="utf-8") as f:
    f.write(text + "\n")
