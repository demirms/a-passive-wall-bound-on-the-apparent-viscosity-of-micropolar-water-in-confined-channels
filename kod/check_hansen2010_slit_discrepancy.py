"""Why do Fig. 3 and the Conclusions of Hansen et al. 2010 (5/15/50 % at slit heights 200/100/20 nm)
differ from their own Eq. (23) (0.89/1.73/7.17 %)?

Steps:
 1. Extract the plotted curve of Fig. 3 exactly from the vector path in the PDF (page 5) and calibrate
    it with the axis tick marks.
 2. Evaluate Eq. (23) under the candidate variants: (a) h = half height (walls at +-h, as stated) or
    h = full height; (b) zeta scaled by 1/4, 1/2, 1, 2, 4 and zeta = 2.5 / 2.6 / 2.1 / zeta0 only /
    zeta_r only; (c) eta0 = 1.0, 0.89 (experiment) and 0.70 (MD, 299 K); (d) the h -> 0 saturation;
    (e) other reduction definitions (Q_N/Q - 1; reference viscosity eta0 - eta_r).
 3. Least-squares fit of the Eq. (23) form, with free prefactor and 1/K, to the plotted curve.
 4. Evaluate Eq. (23) with zeta = 2.6 (0.90/1.77/7.27 %).
Eq. (23): 1 - Q/Q^P = -3 eta_r (tanh Kh - Kh) / [tanh(Kh) (eta0 + eta_r) (Kh)^2],
          K = sqrt(4 eta_r eta0 / ((eta0 + eta_r) zeta)).   Units: mPa s, nm.
"""
import os
import numpy as np
import pymupdf
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, '..', 'literatur', 'Hansen_2010_water_spin_viscosities.pdf')


def red23(H, e0, er, z, half=True):
    """Fractional flow-rate reduction of Eq. (23); H = channel height (nm)."""
    h = H / 2.0 if half else H
    K = np.sqrt(4 * er * e0 / ((e0 + er) * z))
    x = K * h
    return -3 * er * (np.tanh(x) - x) / (np.tanh(x) * (e0 + er) * x ** 2)


# ---------------------------------------------------------------- 1. Fig. 3 curve from the vector path
# Source PDF required, not distributed (copyrighted). Without it the script stops here and the saved
# output file of the original run is kept unchanged.
if not (os.path.isfile(PDF) and os.path.isfile(os.path.join(HERE, '..', 'literatur', 'Hansen_2015_Continuum_Nanofluidics.pdf'))):
    print('SKIPPED: source PDFs required, not distributed: literatur/Hansen_2010_*.pdf, Hansen_2015_*.pdf; '
          'saved output check_hansen2010_slit_discrepancy_output.txt kept')
    raise SystemExit(0)
page = pymupdf.open(PDF)[4]
dr = page.get_drawings()
fig_area = pymupdf.Rect(page.rect.width * 0.5, page.rect.height * 0.7, page.rect.width, page.rect.height * 0.9)
xt, yt = [], []
curve = None
for d in dr:
    r = d['rect']
    if not fig_area.contains(r):
        continue
    if len(d['items']) > 20:
        curve = d
    elif abs(r.x1 - r.x0) < 1e-6 and r.y0 > fig_area.y0 + 0.7 * fig_area.height:   # bottom x ticks
        xt.append(r.x0)
    elif abs(r.y1 - r.y0) < 1e-6 and r.x0 < fig_area.x0 + 0.25 * fig_area.width:    # left y ticks
        yt.append(r.y0)
xt = sorted(set(round(v, 3) for v in xt)); yt = sorted(set(round(v, 3) for v in yt), reverse=True)
print("x ticks (pt):", xt, "-> 0, 50, 100, 150, 200 nm")
print("y ticks (pt):", yt, "-> 0, 10, ..., 50 %")
xs = np.polyfit(xt, np.arange(len(xt)) * 50.0, 1)
ys = np.polyfit(yt, np.arange(len(yt)) * 10.0, 1)
pts = []
for it in curve['items']:
    for P in it[1:]:
        if isinstance(P, pymupdf.Point):
            pts.append((P.x, P.y))
pts = np.array(sorted(set(pts)))
Hc = np.polyval(xs, pts[:, 0]); Rc = np.polyval(ys, pts[:, 1])
o = np.argsort(Hc); Hc, Rc = Hc[o], Rc[o]
print(f"curve: {len(Hc)} vertices, H = {Hc.min():.2f}-{Hc.max():.2f} nm, reduction = {Rc.min():.2f}-{Rc.max():.2f} %")
print("plotted reduction [%] at H = 20, 50, 100, 150, 200 nm:",
      ", ".join(f"{np.interp(H, Hc, Rc):.2f}" for H in (20, 50, 100, 150, 200)))
print("plotted reduction x H (nm %), H = 50, 100, 150, 200:",
      ", ".join(f"{np.interp(H, Hc, Rc)*H:.0f}" for H in (50, 100, 150, 200)))

# ---------------------------------------------------------------- 2. variants of Eq. (23)
target = {200: 5.0, 100: 15.0, 20: 50.0}
print()
print("Eq. (23) variants: reduction [%] at H = 200 / 100 / 20 nm; target (Conclusions) 5 / 15 / 50;"
      " 'match' = all three within 1 point")
print(" h-def   eta0  eta_r  zeta    zeta-note        200nm   100nm    20nm   H->0 limit  match")
base_z = 2.5
zetas = [(2.5, 'Fig.3 caption'), (2.6, 'zeta = 2.6'), (2.1, 'Table II 299K'), (1.1, 'zeta0 only'),
         (1.4, 'zeta_r only')] + [(base_z * f, f'2.5 x {f}') for f in (0.25, 0.5, 2.0, 4.0)]
nmatch = 0
for half in (True, False):
    for e0 in (1.0, 0.89, 0.70):
        for z, note in zetas:
            v = {H: 100 * red23(H, e0, 0.17, z, half) for H in (200, 100, 20)}
            m = all(abs(v[H] - target[H]) <= 1.0 for H in target)
            nmatch += m
            print(f" {'h=H/2' if half else 'h=H  '}  {e0:4.2f}  0.17  {z:5.3f}  {note:15s}  {v[200]:6.2f}  {v[100]:6.2f}  {v[20]:6.2f}"
                  f"   {100*0.17/(e0+0.17):6.2f}      {'YES' if m else 'no'}")
print(f"variants reproducing 5/15/50 within 1 point: {nmatch} of {2*3*len(zetas)}")
print("(d) saturation: Kh -> 0 gives eta_r/(eta0+eta_r) for every zeta and every height definition:")
for e0 in (1.0, 0.89, 0.70):
    print(f"   eta0 = {e0}: {100*0.17/(e0+0.17):.2f} %  (zeta-independent); 50 % would need eta_r >= eta0")
print("(e) other definitions, H -> 0 limit: Q_N/Q - 1 -> eta_r/eta0 = "
      f"{100*0.17/1.0:.1f} %; reference viscosity eta0 - eta_r -> 2 eta_r/(eta0+eta_r) = {100*0.34/1.17:.1f} %")
print("    none of the definitions reaches 50 % with eta_r/eta0 = 0.17")

# ---------------------------------------------------------------- 3. fit to the plotted curve
def shape(H, ell, half=True):
    """(x - tanh x)/(x^2 tanh x), x = K h: the H-dependence of Eq. (23) without its prefactor."""
    x = (H / 2.0 if half else H) / ell
    return (x - np.tanh(x)) / (np.tanh(x) * x ** 2)


print()
print("fit of A * 3 * shape(K h) to the plotted curve (A free, 1/K free). Eq. (23) has A = eta_r/(eta0+eta_r):")
for half in (True, False):
    fit = least_squares(lambda p: 100 * 3 * p[0] * shape(Hc, p[1], half) - Rc, x0=[0.5, 5.0])
    A, ell = fit.x
    print(f"   {'h = H/2' if half else 'h = H  '}: A = {A:.4f}, 1/K = {ell:.3f} nm, rms = {np.sqrt(np.mean(fit.fun**2)):.4f}"
          f" points, max |dev| = {np.max(abs(fit.fun)):.4f}")
ellK = 1 / np.sqrt(4 * 0.17 * 1.0 / (1.17 * 2.5))
eps0 = 0.17 / 1.17
print(f"   stated coefficients (Fig. 3 caption): A = eta_r/(eta0+eta_r) = {eps0:.4f}, 1/K = {ellK:.3f} nm")
ratio = Rc / (100 * red23(Hc, 1.0, 0.17, 2.5))
print(f"   plotted / Eq.(23) at all {len(Hc)} vertices: {ratio.min():.3f}-{ratio.max():.3f}"
      f" (mean {ratio.mean():.3f}); (eta0+eta_r)/eta_r = {1/eps0:.3f}")
dev = Rc - 100 * red23(Hc, 1.0, 0.17, 2.5) / eps0
print(f"   plotted - Eq.(23)/[eta_r/(eta0+eta_r)]: max |dev| = {np.max(abs(dev)):.3f} points, rms = {np.sqrt(np.mean(dev**2)):.3f}")
print("   => Fig. 3 is Eq. (23) with the stated coefficients and h = H/2, but without the prefactor"
      " eta_r/(eta0+eta_r); equivalently, the reduction normalised by its H -> 0 limit.")
print("   Eq.(23)/[eta_r/(eta0+eta_r)] at H = 200 / 100 / 20 nm: " +
      " / ".join(f"{100*red23(H, 1.0, 0.17, 2.5)/eps0:.2f}" for H in (200, 100, 20)) + " % (Conclusions: 5 / 15 / 50)")
for name, e0, er, z in (("eta_r x 10 (1.7 mPa s)", 1.0, 1.7, 2.5),):
    print(f"   check {name}: " + " / ".join(f"{100*red23(H, e0, er, z):.2f}" for H in (200, 100, 20)) + " % (no match)")

# ---------------------------------------------------------------- 4. zeta = 2.6
print()
print("check, zeta = 2.6, eta0 = 1.0, eta_r = 0.17, h = H/2: " +
      " / ".join(f"{100*red23(H, 1.0, 0.17, 2.6):.2f}" for H in (200, 100, 20)) + " %")
print("manuscript values, zeta = 2.5 (Fig. 3 caption: zeta0 = 1.1, zeta_r = 1.4): " +
      " / ".join(f"{100*red23(H, 1.0, 0.17, 2.5):.2f}" for H in (200, 100, 20)) + " %")
print("H -> 0 limit with eta0 = 1.0: {:.2f} %".format(100 * 0.17 / 1.17))

# ---------------------------------------------------------------- 5. later paper of the same group
# Hansen et al. 2015 (Continuum Nanofluidics, arXiv:1506.03661v2), Eq. (50) = Eq. (23) with eta_t = eta0 + eta_r;
# SI Table I water: T = 298.7 K, eta0 = 0.7, eta_r = 0.17 mPa s, zeta0 + zeta_r = 2.1 mPa s nm^2.
print()
print("Hansen et al. 2015 Eq. (50) with SI Table I water (0.7, 0.17, 2.1), h = L/2:")
for L in (3.3, 9.0, 20.0, 100.0, 200.0):
    print(f"   L = {L:5.1f} nm: Delta Q_rel = {100*red23(L, 0.7, 0.17, 2.1):.2f} %")
print(f"   L -> 0 limit eta_r/eta_t = {100*0.17/0.87:.2f} % (their Fig. 6 axis ends at 0.2; text: 'about 10 %' at 9 nm)")
# their Fig. 6 water curve (blue dash-dot vector path, PDF page 23), x in sigma_water = 0.317 nm, frame 0-50 x 0-0.2
PDF15 = os.path.join(HERE, '..', 'literatur', 'Hansen_2015_Continuum_Nanofluidics.pdf')
pg = pymupdf.open(PDF15)[22]
dws = pg.get_drawings()
frame = [d['rect'] for d in dws if len(d['items']) == 1 and d['rect'].width > 100 and d['rect'].height > 100
         and d.get('color') == (0.0, 0.0, 0.0)][0]
blue = [d for d in dws if d.get('color') == (0.0, 0.0, 1.0) and len(d['items']) > 20][0]
pw = []
for it in blue['items']:
    for P_ in it[1:]:
        if isinstance(P_, pymupdf.Point):
            pw.append(((P_.x - frame.x0) / frame.width * 50.0, (frame.y1 - P_.y) / frame.height * 0.2))
pw = np.array(sorted(set(pw)))
Lnm = pw[:, 0] * 0.317
print(f"   Fig. 6 water curve: {len(pw)} vertices, L = {Lnm.min():.2f}-{Lnm.max():.2f} nm, Delta Q_rel = {pw[:,1].min():.4f}-{pw[:,1].max():.4f}")
eq50 = red23(Lnm, 0.7, 0.17, 2.1)
print(f"   plotted / Eq.(50)[h = L/2]: {np.min(pw[:,1]/eq50):.3f}-{np.max(pw[:,1]/eq50):.3f};"
      f" plotted / Eq.(50)[h = L]: {np.min(pw[:,1]/red23(Lnm, 0.7, 0.17, 2.1, half=False)):.3f}-"
      f"{np.max(pw[:,1]/red23(Lnm, 0.7, 0.17, 2.1, half=False)):.3f}")
print(f"   plotted value at L = 9 nm: {np.interp(9.0, Lnm, pw[:,1]):.4f}")
