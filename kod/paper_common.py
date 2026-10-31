"""Shared helpers for the manuscript figures and numbers (no computation at import time).

Model: linear isotropic micropolar fluid (Eringen), fully developed pressure-driven flow, no-slip wall.
Water map (MD coefficients of Hansen et al. 2010, Table II, SPC/Fw):
    mu = eta0 - eta_r,  kappa = 2 eta_r,  gamma = zeta0 + zeta_r,
    lambda^2 = kappa (2 mu + kappa) / [gamma (mu + kappa)] = 4 eta_r eta0 / [(eta0 + eta_r) zeta].
Units: mPa s, nm (gamma in mPa s nm^2).
Closed forms (same formulas as check_tube_m3_bound.py and water_slip_length.py):
    tube:  mu_app = (mu+k) / [1 + k e + 4 k (s_hat - e) J(a)],  e = 1/(2 mu + k),  J = I2(a)/(a I1(a)),  a = lam R
    slit (M1, half-gap h): Hansen 2010 Eq. (23) = exact closed form.
"""
import numpy as np
from scipy.special import ive

# Hansen et al. 2010 Table II: T [K], eta0, eta_r [mPa s], zeta0, zeta_r [mPa s nm^2]
TABLE2 = [(284, 0.92, 0.17, 1.4, 1.6), (289, 0.83, 0.17, 1.2, 1.5), (293, 0.76, 0.17, 1.2, 1.4),
          (299, 0.70, 0.17, 0.9, 1.2), (305, 0.61, 0.16, 0.93, 1.2), (309, 0.56, 0.164, 0.83, 1.0),
          (313, 0.54, 0.161, 0.8, 0.9), (319, 0.48, 0.16, 0.7, 0.92)]
WATER_299 = (0.70, 0.17, 2.1)     # eta0, eta_r, zeta at 299 K (MD-MD normalisation)
WATER_284 = (0.92, 0.17, 3.0)


def eringen(e0, er, z):
    return e0 - er, 2.0 * er, z


def lam(mu, k, g):
    return np.sqrt(k * (2 * mu + k) / (g * (mu + k)))


def J_(a):
    return ive(2, a) / (a * ive(1, a))


def X_(a):
    return a * ive(0, a) / ive(1, a) - 1.0


def tube_muapp(mu, k, a, s_hat=0.0):
    """Apparent viscosity in a tube of radius R, a = lam R, wall spin nu_R = (G R / 2) s_hat."""
    e = 1.0 / (2 * mu + k)
    return (mu + k) / (1 + k * e + 4 * k * (s_hat - e) * J_(a))


def s_star(mu, k, a, beta_over_gamma=0.0):
    """Upper end of the passive wall-spin range (M5)."""
    e = 1.0 / (2 * mu + k)
    X = X_(a)
    return e * (X - 1) / (X - beta_over_gamma)


def slit_muapp(mu, k, g, h):
    """M1 slit, half-gap h (Hansen 2010 Eq. (23) in Eringen variables)."""
    l = lam(mu, k, g)
    kh = l * h
    red = -1.5 * k * (np.tanh(kh) - kh) / (np.tanh(kh) * (mu + k) * kh ** 2)
    return (mu + k / 2) / (1 - red)


def water_tube(r, e0, er, z):
    mu, k, g = eringen(e0, er, z)
    return tube_muapp(mu, k, lam(mu, k, g) * r) / e0


def water_slit(H, e0, er, z):
    mu, k, g = eringen(e0, er, z)
    return slit_muapp(mu, k, g, H / 2.0) / e0


def water_slip(e0, er, z):
    """Negative slip length b and large-channel coefficient c = -8 b (square, circle)."""
    mu, k, g = eringen(e0, er, z)
    l = lam(mu, k, g)
    b = -k / (2 * (mu + k) * l)
    return b, -8.0 * b


def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11.5, "font.family": "DejaVu Sans", "axes.linewidth": 0.6,
                         "lines.linewidth": 1.0, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                         "legend.fontsize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "mathtext.fontset": "dejavusans"})
    return plt
