# BC1 predictions of Table 3 at 293 K against 299 K (the temperature used for the predictions):
# how much larger can the prediction be at another temperature up to 299 K?  Sizes of Table 3:
# tubes (radius), slits/plates (height), squares (side = D_h; finite differences N = 160 up to 491 nm,
# large-size law (19) beyond, as in data_table3_extended.py).
import gc
from paper_common import TABLE2, eringen, lam, water_tube, water_slit, water_slip
from micropolar_duct_2d import solve_duct, stretch_for

COEF = {T: (e0, er, z0 + zr) for (T, e0, er, z0, zr) in TABLE2 if T <= 299}
SIZES = {"tube": [41.25, 53.03, 59.76, 181, 255, 547, 998],
         "slit": [10.5, 22.4, 46, 53, 102, 161, 25, 234, 415, 417, 745],
         "sq": [73, 119, 190, 200, 393, 394, 491, 760, 838, 1530, 1691]}


def pred(kind, s, c):
    e0, er, z = c
    if kind == "tube":
        return water_tube(s, e0, er, z)
    if kind == "slit":
        return water_slit(s, e0, er, z)
    if s <= 491:
        mu, k, g = eringen(e0, er, z)
        st = stretch_for(float(s * lam(mu, k, g)))
        v = solve_duct(float(s), float(s), mu, k, g, 0.0, 160, 160, st, st)[0] / e0
        gc.collect()
        return v
    return 1 + water_slip(e0, er, z)[1] / s


worst = {}
for kind, sizes in SIZES.items():
    for s in sizes:
        p = {T: pred(kind, s, c) for T, c in COEF.items()}
        d = max(p.values()) - p[299]
        Tmax = max(p, key=p.get)
        worst[kind] = max(worst.get(kind, 0.0), d)
        print(f"{kind:5s} {s:7.2f} nm: " + "  ".join(f"{T}:{v:.4f}" for T, v in p.items())
              + f"  max-299 = {d:+.4f} (at {Tmax} K)")
print("largest excess over the 299 K prediction:", {k: round(v, 4) for k, v in worst.items()},
      "overall", round(max(worst.values()), 4))
