"""Digitize Fig. 12 of Bondarenko et al. 2024 (Phys. Fluids 36, 062004): micropolar parameters of distilled water.

Source image: ../literatur/Bondarenko_2024_figures/fig12_water_screenshot.jpg (browser screenshot, 1568x750 px;
the figure itself occupies about 475x200 px, i.e. the 640x269 publisher preview shown at ~0.74 scale).
Caption (verbatim): "Identified temperature-dependent values of micropolar model parameters (a) and shear
viscosity mu, rotational viscosity kappa, and spin viscosity gamma (b) of distilled water (pH 9.4)."

Method (no OCR; the image was viewed, the tick labels read by eye):
  1. Axis calibration. For each axis, the pixel row/column of each labelled tick mark was picked by eye
     (approximate value in TICKS_*), then refined automatically to the darkness-weighted centroid within
     +-2 px along a line one or two pixels inside the frame. A least-squares line (value or log10 value
     against pixel) gives the calibration; its rms residual is printed.
  2. Markers. For each series and plotted temperature a small window around the marker was picked by eye
     (WINDOWS_*). Inside it, a colour score (blue: b-(r+g)/2; orange: r-b with g>90; red: r-(g+b)/2 with
     g<170) is thresholded at 40 % of the window maximum, and the score-weighted centroid gives the marker
     position. The 40 % threshold keeps the filled marker and suppresses the thinner, paler line.
  3. Reading uncertainty: sigma = 1 px in the value direction (marker size ~5 px, JPEG blur ~1-2 px,
     calibration rms < 0.3 px). It is converted to value units with the local calibration slope.
     The temperature coordinate is checked against the nominal 15, 25, 35, 45, 55 C.

Then (light computation): mapping-consistency tests, the mappings to the Eringen variables of the paper
(Eq. (7), paper_common.py), the bound 1 + eta_r/eta_0, the internal length 1/lambda, and BC1 predictions for the
tube (Eq. (13)), the slit (M1 closed form) and the square (finite differences, micropolar_duct_2d.solve_duct)
at the sizes of the data (data_table3_members.csv).
Output: bondarenko2024_fig12_digitize_output.txt. Run with OPENBLAS_NUM_THREADS=1.
"""
import os
import sys
import csv
import time
import itertools
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paper_common import tube_muapp, lam, slit_muapp, TABLE2  # noqa: E402
from micropolar_duct_2d import solve_duct, stretch_for          # noqa: E402

IMG = os.path.join(HERE, "..", "literatur", "Bondarenko_2024_figures", "fig12_water_screenshot.jpg")
OUTF = os.path.join(HERE, "bondarenko2024_fig12_digitize_output.txt")
OUT = []


def P(s=""):
    print(s)
    sys.stdout.flush()
    OUT.append(s)


# Source PDF required, not distributed (copyrighted). Without it the script stops here and the saved
# output file of the original run is kept unchanged.
if not os.path.isfile(IMG):
    print("SKIPPED: source figure required, not distributed: literatur/Bondarenko_2024_figures/; "
          "saved output bondarenko2024_fig12_digitize_output.txt kept")
    sys.exit(0)
A = np.asarray(Image.open(IMG).convert("RGB")).astype(float)
R_, G_, B_ = A[..., 0], A[..., 1], A[..., 2]
DARK = 255.0 - A.mean(2)
SCORE = {"blue": np.clip(B_ - (R_ + G_) / 2, 0, None),
         "orange": np.clip(R_ - B_ - 20, 0, None) * (G_ > 90),
         "red": np.clip(R_ - (G_ + B_) / 2 - 20, 0, None) * (G_ < 170)}

# ---------------------------------------------------------------- 1. axis calibration
# (label value, approximate pixel picked by eye); line = fixed column (vertical axis) or row (horizontal axis)
TICKS_A_LEFT = dict(line=586, vertical=True, log=False,
                    t=[(0.932, 294.5), (0.930, 313.5), (0.928, 332.7), (0.925, 361.5), (0.922, 389.8),
                       (0.920, 409.3), (0.918, 428.5)])
TICKS_A_RIGHT = dict(line=741, vertical=True, log=False, t=[(24, 309), (22, 343), (20, 377.5), (18, 411.5)])
TICKS_A_X = dict(line=441, vertical=False, log=False, t=[(20, 609.5), (30, 646.5), (40, 683.7), (50, 721)])
TICKS_B_LEFT = dict(line=830, vertical=True, log=True,
                    t=[(9, 285.8), (8, 290.8), (7, 296.5), (6, 304), (5, 312), (4, 322.5), (3, 335.5), (2, 353.7),
                       (1, 385.3), (0.9, 390), (0.8, 395.5), (0.7, 401), (0.6, 408.2), (0.5, 416.5), (0.4, 426.8)])
TICKS_B_RIGHT = dict(line=986, vertical=True, log=True,
                     t=[(8, 291), (7, 296.5), (5, 312.5), (2, 353.5), (1, 385.5), (0.9, 390), (0.8, 395.5),
                        (0.7, 401.5), (0.6, 408.5), (0.5, 416.5), (0.4, 427)])
TICKS_B_X = dict(line=439, vertical=False, log=False, t=[(20, 860.8), (30, 892.6), (40, 924.2), (50, 956.0)])


def refine(line, vertical, p0, half=2):
    idx = np.arange(int(round(p0)) - half, int(round(p0)) + half + 1)
    prof = DARK[idx, line] if vertical else DARK[line, idx]
    w = np.clip(prof - np.median(DARK[int(p0) - 8:int(p0) + 9, line] if vertical
                                 else DARK[line, int(p0) - 8:int(p0) + 9]), 0, None)
    if w.sum() == 0:
        return float(p0)
    return float((w * idx).sum() / w.sum())


def calibrate(name, spec):
    vals = np.array([v for v, _ in spec["t"]], float)
    pix = np.array([refine(spec["line"], spec["vertical"], p) for _, p in spec["t"]])
    y = np.log10(vals) if spec["log"] else vals
    slope, icpt = np.polyfit(pix, y, 1)
    res_pix = (y - (slope * pix + icpt)) / slope
    P(f"  {name:10s}: {'log10 ' if spec['log'] else ''}value = {icpt:.6g} + ({slope:.6g}) * pixel;"
      f" {len(vals)} ticks, rms residual {np.sqrt(np.mean(res_pix**2)):.2f} px, max {np.max(np.abs(res_pix)):.2f} px")
    P("              ticks (value: eye -> refined px): " +
      ", ".join(f"{v:g}: {p0:.1f}->{p:.2f}" for (v, p0), p in zip(spec["t"], pix)))
    return slope, icpt, spec["log"]


def to_val(cal, p):
    s, c, lg = cal
    y = c + s * p
    return 10 ** y if lg else y


P("Bondarenko et al. 2024, Fig. 12 (water) - digitization, mapping test and predictions")
P(f"image: literatur/Bondarenko_2024_figures/fig12_water_screenshot.jpg, size {A.shape[1]}x{A.shape[0]} px")
P("\n1. Axis calibration (tick marks; pixel picked by eye, refined to the darkness centroid within +-2 px)")
CAL = {k: calibrate(k, s) for k, s in [("a-left N", TICKS_A_LEFT), ("a-right l", TICKS_A_RIGHT),
                                       ("a-x T", TICKS_A_X), ("b-left", TICKS_B_LEFT),
                                       ("b-right", TICKS_B_RIGHT), ("b-x T", TICKS_B_X)]}
P("  The two log axes of panel (b) have the same pixel positions (decades 10/1/0.1 at the same rows); left in mPa s,"
  " right in Pa s nm^2 = 1000 mPa s nm^2.")

# ---------------------------------------------------------------- 2. markers
TEMPS = [15, 25, 35, 45, 55]
# (x0, x1, y0, y1) windows picked by eye around each marker (legend arrows excluded)
WIN = {
    ("N", "blue", "a-left N", "a-x T"): [(585, 597, 407, 418), (621, 634, 420, 432), (659, 672, 423, 435),
                                         (695, 708, 287, 299), (733, 746, 287, 299)],
    ("l", "orange", "a-right l", "a-x T"): [(585, 597, 390, 404), (621, 634, 305, 320), (658, 671, 285, 300),
                                            (694, 707, 384, 399), (732, 746, 421, 435)],
    ("mu", "blue", "b-left", "b-x T"): [(839, 852, 373, 386), (870, 884, 384, 397), (902, 916, 394, 407),
                                        (934, 947, 402, 415), (965, 979, 410, 424)],
    ("kappa", "orange", "b-left", "b-x T"): [(839, 852, 295, 309), (870, 884, 307, 321), (902, 916, 317, 331),
                                             (934, 947, 316, 330), (965, 979, 324, 338)],
    ("gamma", "red", "b-right", "b-x T"): [(839, 852, 356, 370), (870, 884, 346, 360), (902, 916, 352, 366),
                                           (934, 947, 385, 399), (965, 979, 403, 417)],
}


def centroid(score, win, frac=0.4):
    x0, x1, y0, y1 = win
    s = score[y0:y1 + 1, x0:x1 + 1]
    s = np.where(s >= frac * s.max(), s, 0.0)
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    return float((s * xx).sum() / s.sum()), float((s * yy).sum() / s.sum()), float(score[y0:y1 + 1, x0:x1 + 1].max())


SIGMA_PX = 1.0
DIG = {}
P("\n2. Markers (score-weighted centroid, threshold 40 % of window maximum); sigma = 1 px in the value direction")
P("   series  T_nom  x_px     T_read  y_px     value       sigma      (peak score)")
for (nm, col, ycal, xcal), wins in WIN.items():
    vals = []
    for T, w in zip(TEMPS, wins):
        x, y, pk = centroid(SCORE[col], w)
        v = to_val(CAL[ycal], y)
        sig = abs(to_val(CAL[ycal], y + SIGMA_PX) - to_val(CAL[ycal], y - SIGMA_PX)) / 2
        Tr = to_val(CAL[xcal], x)
        vals.append((v, sig))
        P(f"   {nm:6s} {T:5d}  {x:7.2f}  {Tr:6.2f}  {y:7.2f}  {v:10.5g}  {sig:9.2g}   ({pk:.0f})")
    DIG[nm] = vals

# units: mu, kappa in mPa s; gamma read in Pa s nm^2 -> mPa s nm^2
mu_B = np.array([v for v, _ in DIG["mu"]])
ka_B = np.array([v for v, _ in DIG["kappa"]])
ga_B = np.array([v for v, _ in DIG["gamma"]]) * 1000.0
N_B = np.array([v for v, _ in DIG["N"]])
l_B = np.array([v for v, _ in DIG["l"]])
sN = np.array([s for _, s in DIG["N"]])
sl = np.array([s for _, s in DIG["l"]])
rel_b = np.array([s / v for v, s in DIG["mu"]])   # relative sigma on the log axis (same for all series)

P("\n   Digitized table (sigma from 1 px):")
P("   T (C)  T (K)   N                l (nm)          mu (mPa s)      kappa (mPa s)   gamma (Pa s nm^2)")
for i, T in enumerate(TEMPS):
    P(f"   {T:4d}  {T+273.15:6.2f}  {N_B[i]:.4f}+-{sN[i]:.4f}  {l_B[i]:5.2f}+-{sl[i]:.2f}"
      f"     {mu_B[i]:.3f}+-{mu_B[i]*rel_b[i]:.3f}     {ka_B[i]:.2f}+-{ka_B[i]*rel_b[i]:.2f}"
      f"      {ga_B[i]/1000:.3f}+-{ga_B[i]/1000*rel_b[i]:.3f}")

# ---------------------------------------------------------------- 3. mapping consistency
P("\n3. Internal consistency of the plotted quantities (which definitions of N and l reproduce panel (a) from panel (b))")
P("   Each candidate is evaluated with the panel-(b) values; z = (candidate - plotted)/sigma_combined,"
  " sigma_combined from 1 px on both panels.")
cands_N = {"N^2 = k/(mu+k)": lambda m, k: np.sqrt(k / (m + k)),
           "N^2 = k/(2mu+k)": lambda m, k: np.sqrt(k / (2 * m + k)),
           "N^2 = k/(2(mu+k))": lambda m, k: np.sqrt(k / (2 * (m + k))),
           "N^2 = 2k/(2mu+3k)": lambda m, k: np.sqrt(2 * k / (2 * m + 3 * k))}
for name, f in cands_N.items():
    c = f(mu_B, ka_B)
    # d ln N = 1/2 (d ln k - d ln(m+k)) -> conservative: rel sigma of k/mu ratio * N/2 * (1-N^2)
    sc = c * 0.5 * np.sqrt(2) * rel_b * (1 - c**2)
    z = (c - N_B) / np.sqrt(sc**2 + sN**2)
    P(f"   {name:20s}: " + "  ".join(f"{ci:.4f}(z={zi:+.1f})" for ci, zi in zip(c, z)))
cands_l = {"l^2 = g/(4mu)": lambda m, k, g: np.sqrt(g / (4 * m)),
           "l^2 = g/(2mu)": lambda m, k, g: np.sqrt(g / (2 * m)),
           "l^2 = g/mu": lambda m, k, g: np.sqrt(g / m),
           "l^2 = g/(4(mu+k))": lambda m, k, g: np.sqrt(g / (4 * (m + k))),
           "l^2 = g/(2(2mu+k))": lambda m, k, g: np.sqrt(g / (2 * (2 * m + k))),
           "l^2 = g/k": lambda m, k, g: np.sqrt(g / k),
           "l^2 = g/(4k)": lambda m, k, g: np.sqrt(g / (4 * k)),
           "l^2 = g(mu+k)/[k(2mu+k)]": lambda m, k, g: np.sqrt(g * (m + k) / (k * (2 * m + k)))}
for name, f in cands_l.items():
    c = f(mu_B, ka_B, ga_B)
    sc = c * 0.5 * np.sqrt(2) * rel_b
    z = (c - l_B) / np.sqrt(sc**2 + sl**2)
    P(f"   {name:26s}: " + "  ".join(f"{ci:6.2f}(z={zi:+.1f})" for ci, zi in zip(c, z)))

IAPWS = np.array([1.1375, 0.8900, 0.7194, 0.5963, 0.5040])  # bulk viscosity of water, mPa s, 0.1 MPa (IAPWS 2008), check only
P("   Plotted mu against the bulk viscosity of water (IAPWS 2008 values at 0.1 MPa; used only for this check):")
P("   " + "  ".join(f"{T} C: {m:.3f} vs {w:.4f} (z={(m-w)/(m*r):+.1f})" for T, m, w, r in zip(TEMPS, mu_B, IAPWS, rel_b)))
P("   -> panel (a) and panel (b) are related by N^2 = kappa/(mu+kappa) and l^2 = gamma/(4 mu) (see z above), and the"
  " plotted mu is the bulk viscosity of water. Only N and l are therefore independent identified parameters;"
  " kappa = mu N^2/(1-N^2) and gamma = 4 mu l^2 follow from them with mu = bulk viscosity.")

# ---------------------------------------------------------------- 4. mappings to the paper's Eringen variables
P("\n4. Mappings to the Eringen variables of the paper (mu_E, kappa_E, gamma_E; eta_0 = mu_E + kappa_E/2, eta_r = kappa_E/2)")
P("   E : Eringen variables directly: mu_E = mu, kappa_E = kappa, gamma_E = gamma. Bulk viscosity mu + kappa/2.")
P("   H1: mu = eta_0 (bulk), kappa = eta_r (Cosserat couple viscosity, Neff/Nowacki form 2 kappa skew),"
  " gamma_E = gamma: mu_E = mu - kappa, kappa_E = 2 kappa. Bulk viscosity mu.")
P("   H2: mu = eta_0 (bulk), kappa = kappa_E = 2 eta_r: mu_E = mu - kappa/2, kappa_E = kappa. Bulk viscosity mu.")
P("   Closed forms in terms of N and l (with l^2 = gamma/(4 mu), N^2 = kappa/(mu+kappa)):")
P("     E : eta_r/eta_0 = N^2/(2-N^2);       lambda^2 l^2 = N^2 (2-N^2) / [4 (1-N^2)]")
P("     H1: eta_r/eta_0 = N^2/(1-N^2);       lambda^2 l^2 = N^2   (1/lambda = l/N)")
P("     H2: eta_r/eta_0 = N^2/[2(1-N^2)];    lambda^2 l^2 = N^2/(2-N^2)")
P("   Bulk-viscosity check: E gives a model bulk viscosity mu + kappa/2 = " +
  ", ".join(f"{(m + k/2):.2f}" for m, k in zip(mu_B, ka_B)) + " mPa s = " +
  ", ".join(f"{(m + k/2)/w:.1f}x" for m, k, w in zip(mu_B, ka_B, IAPWS)) + " the water value; H1/H2 give mu = water value.")


def eringen_from(mapping, N, l, mu=1.0):
    k = mu * N**2 / (1 - N**2)
    g = 4 * mu * l**2
    if mapping == "E":
        return mu, k, g
    if mapping == "H1":
        return mu - k, 2 * k, g
    if mapping == "H2":
        return mu - k / 2, k, g
    raise ValueError(mapping)


def derived(mapping, N, l):
    mE, kE, gE = eringen_from(mapping, N, l)
    e0 = mE + kE / 2
    return mE, kE, gE, e0, (kE / 2) / e0, 1.0 / lam(mE, kE, gE)


MAPS = ["E", "H1", "H2"]
I15, I25 = 0, 1
MD = {T: (e0, er, z0 + zr) for (T, e0, er, z0, zr) in TABLE2}
MD[299] = (0.70, 0.17, 2.1)   # MD-MD normalisation of the paper 
MD[284] = (0.92, 0.17, 3.0)


def md_derived(T):
    e0, er, z = MD[T]
    mE, kE = e0 - er, 2 * er
    return er / e0, 1 / lam(mE, kE, z)


P("\n   Derived quantities at the overlap temperatures (15 C = 288 K, 25 C = 298 K); range = 1-px corners of (N, l)")
P("   map  T(C)   eta_r/eta_0           bound 1+eta_r/eta_0    1/lambda (nm)        ratio to MD (eta_r/eta_0; 1/lambda)")
for mp in MAPS:
    for i, Tmd in ((I15, 289), (I25, 299)):
        corners = [derived(mp, N_B[i] + a * sN[i], l_B[i] + b * sl[i]) for a, b in itertools.product((-1, 1), (-1, 1))]
        nom = derived(mp, N_B[i], l_B[i])
        r = [c[4] for c in corners]
        L = [c[5] for c in corners]
        rmd, lmd = md_derived(Tmd)
        P(f"   {mp:3s}  {TEMPS[i]:3d}   {nom[4]:.4f} [{min(r):.4f},{max(r):.4f}]   {1+nom[4]:.4f}"
          f" [{1+min(r):.4f},{1+max(r):.4f}]   {nom[5]:5.2f} [{min(L):5.2f},{max(L):5.2f}]"
          f"   x{nom[4]/rmd:.2f} vs MD {Tmd} K ({rmd:.3f}); x{nom[5]/lmd:.1f} ({lmd:.3f} nm)")
P("   MD (Table 1): 284 K " + "%.3f, %.3f nm" % md_derived(284) + "; 289 K " + "%.3f, %.3f nm" % md_derived(289)
  + "; 299 K " + "%.3f, %.3f nm" % md_derived(299))

# ---------------------------------------------------------------- 5. predictions vs data
rows = list(csv.DictReader(open(os.path.join(HERE, "data_table3_members.csv"), encoding="utf-8-sig")))


def grp(pred):
    out = {}
    for r in rows:
        if pred(r):
            out.setdefault(r["group"], []).append((float(r["size_nm"]), float(r["eta_ratio"])))
    return out


tubes = grp(lambda r: r["kind"] == "tube")
slits = grp(lambda r: r["kind"] == "slit" and "NaCl" not in r["group"])
squares = grp(lambda r: r["kind"] == "sq" and float(r["size_nm"]) <= 491)


def pred_tube(mp, i, R, dN=0, dl=0):
    mE, kE, gE, e0, _, _ = derived(mp, N_B[i] + dN, l_B[i] + dl)
    return tube_muapp(mE, kE, lam(mE, kE, gE) * R) / e0


def pred_slit(mp, i, H):
    mE, kE, gE, e0, _, _ = derived(mp, N_B[i], l_B[i])
    return slit_muapp(mE, kE, gE, H / 2.0) / e0


def pred_square(mp, i, Dh, Ngrid, dN=0, dl=0):
    mE, kE, gE, e0, _, _ = derived(mp, N_B[i] + dN, l_B[i] + dl)
    st = stretch_for(float(Dh * lam(mE, kE, gE)))
    return solve_duct(float(Dh), float(Dh), mE, kE, gE, 0.0, Ngrid, Ngrid, st, st)[0] / e0


P("\n5. BC1 predictions (mu_app/eta_0) with the digitized parameters vs measured values (data_table3_members.csv)")
P("   Tubes, radius R (Eq. (13)); data of Churaev [1] at about 20 C. Range in [] = 1-px corners of (N, l).")
for g, mem in tubes.items():
    P(f"   {g}")
    for R, y in sorted(mem):
        s = f"     R = {R:6.1f} nm  measured {y:.3f} |"
        for mp in MAPS:
            for i in (I15, I25):
                v = pred_tube(mp, i, R)
                c = [pred_tube(mp, i, R, a * sN[i], b * sl[i]) for a, b in itertools.product((-1, 1), (-1, 1))]
                s += f" {mp}{TEMPS[i]}: {v:.3f}[{min(c):.3f},{max(c):.3f}]"
        P(s)
P("   Slits, height H (M1 closed form, h = H/2), salt-free data:")
for g, mem in slits.items():
    for H, y in sorted(mem):
        P(f"     {g:28s} H = {H:6.1f} nm measured {y:.3f} | " +
          "  ".join(f"{mp}{TEMPS[i]}: {pred_slit(mp, i, H):.3f}" for mp in MAPS for i in (I15, I25)))

P("   Squares, hydraulic diameter D_h <= 491 nm (finite differences, M1 = BC1 walls, alpha+beta = 0).")
P("   Grid N = 80 and 160 (tanh-stretched); the N = 160 value is quoted; 1-px corners at N = 80 for D_h = 73 and 119 nm.")
t0 = time.time()
sq_sizes = sorted({(Dh, y) for mem in squares.values() for Dh, y in mem})
for Dh, y in sq_sizes:
    s = f"     D_h = {Dh:5.0f} nm measured {y:.3f} |"
    for mp in MAPS:
        for i in (I15, I25):
            v80 = pred_square(mp, i, Dh, 80)
            v160 = pred_square(mp, i, Dh, 160)
            extra = ""
            if Dh in (73.0, 119.0) and mp in ("E", "H1"):
                c = [pred_square(mp, i, Dh, 80, a * sN[i], b * sl[i]) for a, b in itertools.product((-1, 1), (-1, 1))]
                extra = f"[{min(c):.3f},{max(c):.3f}]"
            s += f" {mp}{TEMPS[i]}: {v160:.3f} (N80 {v80:.3f}){extra}"
    P(s)
P(f"   [square FD time {time.time()-t0:.0f} s]")

# compact summary for the paper's ranges
P("\n6. Summary against the paper's ranges")
for mp in MAPS:
    for i in (I15, I25):
        _, _, _, _, r, L = derived(mp, N_B[i], l_B[i])
        tub = [pred_tube(mp, i, R) for R in (41.25, 53.03, 59.76)]
        P(f"   {mp:3s} {TEMPS[i]} C: bound {1+r:.3f}; 1/lambda {L:.1f} nm; BC1 tube R = 41-60 nm: {min(tub):.3f}-{max(tub):.3f}"
          f" (measured 1.339-1.407); squares measured 1.76-5.55 for D_h 73-491 nm")

with open(OUTF, "w", encoding="utf-8") as fh:
    fh.write("\n".join(OUT) + "\n")
P(f"\nwritten: {os.path.basename(OUTF)}")
