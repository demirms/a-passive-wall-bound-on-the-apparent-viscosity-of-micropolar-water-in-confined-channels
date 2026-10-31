"""Khanukaeva & Troshkin 2023 (Phys. Fluids 35, 072005): wall conditions in our notation, passivity, reproduction.

Source: literatur/Khanukaeva_2023_Steady_nanocapillary_flow_micropolar_approach_vs_classical_models.pdf
(journal pages 072005-1 ... -9 = PDF pages 2 ... 10).

Parts
  A  Map of their Nowacki-form constants (mu_N, j, alpha, delta, eps) onto Eringen's (mu, kappa, alpha, beta, gamma)
     of Eqs. (1)-(4) of our manuscript, from the balance equations (their Eq. (9)-(10) vs our Eq. (3)-(4)), sympy.
     Their N^2 = j/(mu_N + j), L^2 = (delta + eps)/(4 mu_N R^2) expressed in our variables; N/L = lambda R.
  B  Their wall conditions: hyper-stick (u = 0, omega = 0) = BC1; spinning omega = n curl v (Kirwan-Newman)
     = BC2 with the same n. Their flow rates Q_NL, Q_NLn (p. 072005-7) against our Eq. (13) with
     s_hat = n/[mu + (1-n) kappa]; limits; their pole n < 1/(2N^2) vs our n_p = 1 + mu/kappa.
     Note on their Eq. (15): Q~ = Q/(pi R^2 u0) equals 2 * int u~ r~ dr~ (their printed factor 2 pi gives pi).
  C  Their constants: N = 0.65, L = 0.25 at R = 40 nm (Figs. 3-7, from Kolpashchikov et al. 1983) and the
     Kolpashchikov/Migun fit used in our Discussion (delta_K = kappa/(mu+kappa) = 0.84, 1/lambda = 14.22 nm).
     Bound 1 + eta_r/eta0 = 1/(1 - N^2) for both sets and for the MD set (Hansen 2010 Table II).
  D  Reproduction of their Fig. 8 (hyper-stick, Q_NL/Q_0 against r) and Fig. 9 (spinning, n = 0.068):
     curve and data points digitized from the raster images in the PDF (axis-tick calibration);
     least-squares fit of n to the digitized points (their value: n = 0.068, p. 072005-8).
  E  Passivity (our Eqs. (14)-(15)): s_hat, s*, n*, sign of the wall power P and P*(2 pi R)/(G Q) for their n values
     (0, 0.068, 0.25, 0.3, 0.5, 0.7), beta/gamma in {-1, 0, 1} (beta not reported by them), R = 40 nm ... 1 um,
     for their constants and for the MD constants.
Units: mPa s, nm. Run from the repository root:  py -u check_khanukaeva2023_walls.py > check_khanukaeva2023_walls_output.txt
"""
import os
import sys
import time

import numpy as np
import sympy as sp
from scipy import ndimage, optimize
from scipy.special import iv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_common import tube_muapp, s_star, X_, TABLE2  # noqa: E402

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
PDF = os.path.join(HERE, '..', 'literatur',
                   'Khanukaeva_2023_Steady_nanocapillary_flow_micropolar_approach_vs_classical_models.pdf')


def hdr(s):
    print('\n' + '=' * 100 + '\n' + s + '\n' + '=' * 100)


# ---------------------------------------------------------------------------------------------- A
hdr('A  Constant map Nowacki form (Khanukaeva & Troshkin, p. 072005-4) -> Eringen form (our Eqs. (1)-(4))')
muN, j, alN, de, ep = sp.symbols('mu_N j alpha_N delta epsilon', real=True)
mu, ka, al, be, ga = sp.symbols('mu kappa alpha beta gamma', real=True)
# Their Eq. (9): (mu_N + j) Lap v + 2 j curl w = grad p ;  (delta + eps) Lap w + (alpha + delta - eps) grad div w
#                + 2 j curl v - 4 j w = 0   (Eq. (8) with the alpha+delta-eps grad div term)
# Our Eqs. (3)-(4): (mu + kappa) Lap w + kappa curl nu + G = 0 ; gamma Lap nu + (alpha+beta) grad div nu
#                + kappa curl v - 2 kappa nu = 0
eqs = [sp.Eq(mu + ka, muN + j), sp.Eq(ka, 2 * j), sp.Eq(ga, de + ep), sp.Eq(al + be, alN + de - ep),
       sp.Eq(2 * ka, 4 * j)]
sol = sp.solve(eqs[:4], [mu, ka, ga, be], dict=True)[0]
sol_beta = sp.Eq(be, de - ep)          # couple-stress split: 2 delta chi^S + 2 eps chi^A = (delta+eps) chi + (delta-eps) chi^T
print('  from the balance equations: mu =', sol[mu], ', kappa =', sol[ka], ', gamma =', sol[ga],
      ', alpha + beta =', sol[be] + al)
print('  spin-equation check 2 kappa - 4 j =', sp.simplify(2 * sol[ka] - 4 * j))
print('  couple-stress split (constitutive): beta = delta - eps, alpha = alpha_N; residual of alpha+beta:',
      sp.simplify((alN + de - ep) - (alN + (de - ep))))
# Newtonian viscosity and the MD (Hansen 2010) map: mu + kappa/2 = mu_N = eta0, kappa/2 = j = eta_r
print('  mu + kappa/2 =', sp.simplify(sol[mu] + sol[ka] / 2), '  (= mu_N: their mu is the Newtonian viscosity eta0)')
print('  kappa/2 =', sol[ka] / 2, '  (= j: their rotational viscosity is eta_r of Hansen 2010)')
R, lam_s = sp.symbols('R lambda', positive=True)
N2 = j / (muN + j)
L2 = (de + ep) / (4 * muN * R**2)
lam2 = sol[ka] * (2 * sol[mu] + sol[ka]) / (sol[ga] * (sol[mu] + sol[ka]))   # our lambda^2
print('  N^2 = j/(mu_N+j) = kappa/[2(mu+kappa)] = eta_r/(eta0+eta_r);  check:',
      sp.simplify(N2 - sol[ka] / (2 * (sol[mu] + sol[ka]))))
print('  (N/L)^2 - (lambda R)^2 =', sp.simplify(N2 / L2 - lam2 * R**2))
print('  1/(2 N^2) - (1 + mu/kappa) =', sp.simplify(1 / (2 * N2) - (1 + sol[mu] / sol[ka])),
      ' (their bound on n = pole n_p of our BC2)')
print('  1/(1 - N^2) - (mu+kappa)/(mu+kappa/2) =',
      sp.simplify(1 / (1 - N2) - (sol[mu] + sol[ka]) / (sol[mu] + sol[ka] / 2)), ' (bound 1 + eta_r/eta0)')


def from_NL(Nv, a):
    """Eringen (mu, kappa) with eta0 = mu_N = 1 from N; a = lambda R = N/L."""
    jj = Nv**2 / (1 - Nv**2)          # j/mu_N
    return 1 - jj, 2 * jj, a


# ---------------------------------------------------------------------------------------------- B
hdr('B  Wall conditions and closed forms: their Q_NL, Q_NLn (p. 072005-7) vs our Eq. (13)')


def Q_their(Nv, Lv, n=0.0):
    x = Nv / Lv
    return 1 + 8 * (1 - 2 * n) / (1 - 2 * n * Nv**2) * (Lv**2 - Lv * Nv / 2 * iv(0, x) / iv(1, x))


def shat_of_n(m, k, n):
    return n / (m + (1 - n) * k)


def Q_ours(Nv, Lv, n=0.0):
    m, k, a = from_NL(Nv, Nv / Lv)
    return 1.0 / tube_muapp(m, k, a, shat_of_n(m, k, n))      # Q/Q_0 = eta0/mu_app with eta0 = 1


def u_their(r, Nv, Lv, n=0.0):         # their Eq. (13); n = 0 gives Eq. (12)
    x = Nv / Lv
    c = (2 * n - 1) / (2 * n * Nv**2 - 1)
    return 2 * (1 - r**2) - 4 * Lv * Nv * c * (iv(0, x) - iv(0, r * x)) / iv(1, x)


dmax = 0.0
for Nv in (0.1, 0.3, 0.5, 0.65, 0.8):
    for Lv in (0.02, 0.1, 0.25, 1.0, 3.0):
        for n in (0.0, 0.068, 0.25, 0.5, 0.7, -0.5):
            if 1 - 2 * n * Nv**2 <= 0:
                continue
            dmax = max(dmax, abs(Q_their(Nv, Lv, n) - Q_ours(Nv, Lv, n)) / abs(Q_ours(Nv, Lv, n)))
print(f'  max relative difference Q_NLn(theirs) vs 1/[mu_app(Eq. 13)/eta0], 5 N x 5 L x 6 n: {dmax:.2e}')
print('  -> spinning condition omega = n curl v at r = R is our BC2 with the same n; hyper-stick (n = 0) is BC1.')
rr = np.linspace(0, 1, 20001)
q2 = 2 * np.trapezoid(u_their(rr, 0.65, 0.25, 0.068) * rr, rr)
print(f'  their Eq. (15) normalisation: 2*int(u r dr) = {q2:.6f} vs Q_NLn = {Q_their(0.65, 0.25, 0.068):.6f};'
      f' Poiseuille 2*int = {2*np.trapezoid(2*(1-rr**2)*rr, rr):.6f}, 2 pi*int = {2*np.pi*np.trapezoid(2*(1-rr**2)*rr, rr):.6f}'
      ' -> printed factor 2 pi should read 2 (typographical)')
for Nv in (0.4, 0.65):
    print(f'  N = {Nv}: L -> inf  Q_NL = {Q_their(Nv, 200.0):.6f}, 1 - N^2 = {1 - Nv**2:.6f};'
          f'  L -> 0  Q_NL = {Q_their(Nv, 1e-3):.6f};  n = 1/2 gives Q = {Q_their(Nv, 0.25, 0.5):.6f}')
print('  -> hyper-stick: Q_NL/Q_0 -> 1 - N^2 = eta0/(eta0+eta_r) in the narrowest capillary, i.e. mu_app -> mu + kappa'
      '\n     (= our bound; in their paper only as "a certain constant determined by N", p. 072005-5, Fig. 6 asymptotes).')

# ---------------------------------------------------------------------------------------------- C
hdr('C  Coefficient sets')
NK = np.sqrt(0.84 / 2)                 # Kolpashchikov/Migun: kappa/(mu+kappa) = 0.84 -> N^2 = 0.42
ellK = 14.22                           # 1/lambda [nm] of the Migun/Kolpashchikov fit (our Discussion)
sets = {'KT2023 (N=0.65, L=0.25 at 40 nm)': (0.65, 0.25 * 40 / 0.65),
        'Kolpashchikov/Migun fit (N=0.648, 1/lam=14.22)': (NK, ellK)}
print('  set                                             N      eta_r/eta0  kappa/mu  1/lam[nm]  L(40nm)  bound 1/(1-N^2)')
for name, (Nv, ell) in sets.items():
    m, k, _ = from_NL(Nv, 1.0)
    print(f'  {name:46s} {Nv:.4f}   {Nv**2/(1-Nv**2):.4f}     {k/m:.3f}    {ell:6.2f}    {Nv*ell/40:.4f}   {1/(1-Nv**2):.4f}')
for T, e0, er, z0, zr in TABLE2[:4]:
    Nv = np.sqrt(er / (e0 + er))
    m, k, g = e0 - er, 2 * er, z0 + zr
    ell = 1 / np.sqrt(k * (2 * m + k) / (g * (m + k)))
    print(f'  MD Hansen 2010 Table II, {T} K                   {Nv:.4f}   {er/e0:.4f}     {k/m:.3f}    {ell:6.2f}    {Nv*ell/40:.4f}   {1/(1-Nv**2):.4f}')
print('  -> their N = 0.65 is the Kolpashchikov fit (N^2 = delta_K/2 = 0.42); their L = 0.25 at 40 nm implies'
      ' 1/lambda = 15.38 nm\n     (Kolpashchikov 1/lambda = 14.22 nm gives L = 0.230). The MD coupling N = 0.40-0.44 and L = 0.02.')

# ---------------------------------------------------------------------------------------------- D
hdr('D  Reproduction of Figs. 8 and 9 (raster images on PDF page 9 = p. 072005-8)')
import pymupdf  # noqa: E402

# Source PDF required, not distributed (copyrighted). Without it the script stops here (sections A-C above do not need it) and the saved
# output file of the original run is kept unchanged.
if not os.path.isfile(PDF):
    print('SKIPPED section D and later: source PDF required, not distributed: literatur/Khanukaeva_2023_*.pdf; '
          'saved output check_khanukaeva2023_walls_output.txt kept')
    sys.exit(0)
doc = pymupdf.open(PDF)
page = doc[8]
imgs = {}
for info in page.get_image_info(xrefs=True):
    pix = pymupdf.Pixmap(doc, info['xref'])
    a = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, 0].astype(int)
    imgs['Fig8' if info['bbox'][1] < 300 else 'Fig9'] = a
# axis calibration: tick pixels (detected as dark runs attached to the axes; labels 0.2 ... 1.0 and 0.7 ... 1.0)
CAL = {'Fig8': dict(xt=[202.5, 345.5, 489.5, 634.5, 779.0], yt=[(1.0, 184.5), (0.9, 305.5), (0.8, 424.0), (0.7, 542.5)],
                    xaxis=553, yaxis=58),
       'Fig9': dict(xt=[213.0, 355.5, 497.5, 642.0, 786.0], yt=[(1.0, 178.0), (0.9, 298.5), (0.8, 415.5), (0.7, 533.0)],
                    xaxis=543, yaxis=69)}
digit = {}
for fig, c in CAL.items():
    a = imgs[fig]
    px = np.polyfit([0.2, 0.4, 0.6, 0.8, 1.0], c['xt'], 1)
    yv, yp = zip(*c['yt'])
    pyc = np.polyfit(yv, yp, 1)
    res_y = np.max(np.abs(np.polyval(pyc, yv) - np.array(yp)))
    res_x = np.max(np.abs(np.polyval(px, [0.2, 0.4, 0.6, 0.8, 1.0]) - np.array(c['xt'])))
    dark = a < 140
    plot = np.zeros_like(dark)
    plot[100:c['xaxis'] - 3, c['yaxis'] + 4:800] = True
    dk = dark & plot
    # filled dots (diameter ~16 px) survive an erosion that removes the curve (width ~9 px)
    yy, xx = np.mgrid[-6:7, -6:7]
    disk = (xx**2 + yy**2) <= 36
    core = ndimage.binary_erosion(dk, structure=disk)
    lab, nlab = ndimage.label(core)
    cent = ndimage.center_of_mass(core, lab, range(1, nlab + 1))
    pts = sorted(((xc - px[1]) / px[0], (yc - pyc[1]) / pyc[0]) for yc, xc in cent)
    # curve: centre of the thin dark run (7-11 px) in each column, columns without dots
    cur = []
    for r in (0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0):
        x = int(round(np.polyval(px, r)))
        ys = np.where(dk[:, x])[0]
        runs = np.split(ys, np.where(np.diff(ys) > 1)[0] + 1)
        thin = [rn for rn in runs if 6 <= len(rn) <= 11]
        if thin:
            cur.append((r, ((thin[-1][0] + thin[-1][-1]) / 2 - pyc[1]) / pyc[0]))
    digit[fig] = (pts, cur)
    print(f'  {fig}: calibration x {px[0]:.2f} px/um (max tick residual {res_x:.2f} px), y {pyc[0]:.2f} px/unit'
          f' (max residual {res_y:.2f} px); 1 px = {1/px[0]*1000:.2f} nm, {1/abs(pyc[0]):.5f} in Q/Q0')
    print(f'    {len(pts)} dots (r [um], Q/Q0): ' + ', '.join(f'({p:.3f}, {q:.3f})' for p, q in pts))

for fig, n_used in (('Fig8', 0.0), ('Fig9', 0.068)):
    pts, cur = digit[fig]
    print(f'\n  {fig}: curve read-off vs closed form (n = {n_used})')
    print('    r[um]  read   KT2023 set  Kolpashchikov set')
    for r, q in cur:
        q1 = Q_their(0.65, 0.25 * 40 / (r * 1000), n_used)
        q2 = Q_their(NK, NK * ellK / (r * 1000), n_used)
        print(f'    {r:4.2f}  {q:.4f}   {q1:.4f}      {q2:.4f}')

# least-squares fit of n to the dots (both figures show the same data set)
pts = digit['Fig8'][0]
rp = np.array([p for p, _ in pts]) * 1000
qp = np.array([q for _, q in pts])
print(f'\n  Least-squares n from the {len(pts)} digitized dots of Fig. 8 (their result: n = 0.068):')
for name, (Nv, ell) in sets.items():
    def resid(nv):
        return np.array([Q_their(Nv, Nv * ell / r, nv) for r in rp]) - qp
    fit = optimize.least_squares(lambda p: resid(p[0]), [0.05])
    ss0 = np.sum(resid(0.0)**2)
    print(f'    {name:46s} n = {fit.x[0]:.4f}  (rms {np.sqrt(np.mean(fit.fun**2)):.4f}; n = 0 rms {np.sqrt(ss0/len(rp)):.4f};'
          f' n = 0.068 rms {np.sqrt(np.mean(resid(0.068)**2)):.4f})')
    # same fit on 1/Q (apparent viscosity), as an alternative objective
    fit2 = optimize.least_squares(lambda p: 1 / (resid(p[0]) + qp) - 1 / qp, [0.05])
    print(f'    {"":46s} fit on mu_app/eta0 = Q0/Q instead: n = {fit2.x[0]:.4f}')
pts9 = digit['Fig9'][0]
d89 = [abs(a[1] - b[1]) for a, b in zip(pts, pts9)] if len(pts) == len(pts9) else []
if d89:
    print(f'  Fig. 9 dots vs Fig. 8 dots (same data): max |dQ| = {max(d89):.4f}, count {len(pts9)}')
else:
    print(f'  Fig. 9 dot count {len(pts9)} vs Fig. 8 {len(pts)}')

# ---------------------------------------------------------------------------------------------- E
hdr('E  Passivity of their walls (our Eqs. (14)-(15)); P > 0 = active (wall delivers power)')


def wall_power_ratio(m, k, a, shv, bg):
    """2 pi R P / (G Q) with R = G = 1, gamma from lambda (gamma = kappa(2mu+kappa)/[(mu+kappa) a^2])."""
    g = k * (2 * m + k) / ((m + k) * a**2)
    e = 1 / (2 * m + k)
    X = X_(a)
    nr = shv / 2
    P = nr * (g * (e * (1 - X) / 2 + nr * X) - bg * g * nr)
    Q = np.pi / (8 * tube_muapp(m, k, a, shv))
    return 2 * np.pi * P / Q


def nstar(m, k, a, bg):
    ss = s_star(m, k, a, bg)
    return ss * (m + k) / (1 + ss * k)


print('  n* (upper end of the passive BC2 range), their constants, beta/gamma = -1 / 0 / +1:')
print('    set                                             R[nm]   lamR    n*(-1)   n*(0)    n*(+1)   n_pole')
for name, (Nv, ell) in sets.items():
    for Rn in (40, 60, 100, 300, 1000):
        m, k, a = from_NL(Nv, Rn / ell)
        print(f'    {name:46s} {Rn:5d}  {a:6.2f}   {nstar(m,k,a,-1):.4f}   {nstar(m,k,a,0):.4f}   {nstar(m,k,a,1):.4f}'
              f'   {1 + m/k:.4f}')
print('\n  Classification at R = 40 nm and 1000 nm (their constants), beta = 0 unless stated:')
print('    set       n       R[nm]  s_hat/s*(b=0)  P sign (b=-1/0/+1)     2piRP/(GQ) (b=0)   Q/Q0     mu_app/eta0')
for name, (Nv, ell) in sets.items():
    tag = 'KT2023' if name.startswith('KT') else 'Kolp.'
    for n in (0.0, 0.068, 0.25, 0.3, 0.5, 0.7):
        for Rn in (40, 1000):
            m, k, a = from_NL(Nv, Rn / ell)
            if 1 + shat_of_n(m, k, n) * k == 0 or n >= 1 + m / k:
                continue
            sh = shat_of_n(m, k, n)
            signs = []
            for bg in (-1.0, 0.0, 1.0):
                pr = wall_power_ratio(m, k, a, sh, bg)
                signs.append('0' if abs(pr) < 1e-12 else ('+' if pr > 0 else '-'))
            pr0 = wall_power_ratio(m, k, a, sh, 0.0)
            mua = tube_muapp(m, k, a, sh)
            print(f'    {tag:8s} {n:5.3f}   {Rn:5d}   {sh/s_star(m,k,a,0):8.4f}       {"/".join(signs):10s}        {pr0:+.3e}       {1/mua:.4f}   {mua:.4f}')
print('\n  Same n values with the MD constants (Hansen 2010 Table II, 299 K and 284 K), R = 40 nm, beta = 0:')
for T, e0, er, z0, zr in (TABLE2[0], TABLE2[3]):
    m, k, g = e0 - er, 2 * er, z0 + zr
    lm = np.sqrt(k * (2 * m + k) / (g * (m + k)))
    a = lm * 40
    m1, k1 = m / e0, k / e0
    out = []
    for n in (0.0, 0.068, 0.5, 0.7):
        sh = shat_of_n(m1, k1, n)
        out.append(f'n={n}: {tube_muapp(m1,k1,a,sh):.4f} ({"+" if wall_power_ratio(m1,k1,a,sh,0)>1e-14 else "-/0"})')
    print(f'    {T} K  n* = {nstar(m1,k1,a,0):.4f};  mu_app/eta0: ' + ';  '.join(out))
print('\n  Data comparison: Churaev 1970 water, r = 41-60 nm: mu_app/eta0 = 1.31-1.48 (our data table R001-R003, R015-R025);'
      '\n  BC1 with Kolpashchikov constants at 40 nm gives 1.485 (inside the fitted bound 1.724), with MD constants 1.029-1.038'
      '\n  (bound 1.18-1.24).')
print(f'\nruntime {time.time()-T0:.1f} s')
