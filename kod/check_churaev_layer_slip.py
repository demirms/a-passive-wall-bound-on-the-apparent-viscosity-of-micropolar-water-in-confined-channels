"""Slip-length equivalent of the immobile layer of Churaev et al. [1].

Source: Ref. [1], p. 218, Eq. (6): eta/eta0 = [r/(r - h0)]^2 (meniscus-velocity method), h0 = 8 nm.
A slip length b changes the tube flow rate by the factor (1 + 4 b / r), so the apparent viscosity is
eta/eta0 = 1/(1 + 4 b/r). Equating the two gives b exactly; to first order b = -h0/2.
The micropolar wall layer of water corresponds to |b| = 0.36-0.40 nm (Sec. III, Eq. (18), Table 1).
"""
h0 = 8.0
print("first order: b = -h0/2 = %.2f nm" % (-h0 / 2))
for r in (40.0, 41.25, 46.0, 50.0, 59.76, 60.0):
    ratio = (r / (r - h0)) ** 2
    b = r / 4.0 * (1.0 / ratio - 1.0)
    print("r = %6.2f nm: eta/eta0 = %.4f, exact b = %.3f nm" % (r, ratio, b))
for bm in (0.36, 0.40):
    print("micropolar |b| = %.2f nm: first-order ratio %.1f; exact ratio %.1f-%.1f"
          % (bm, h0 / 2 / bm, 3.6 / bm, 3.73 / bm))
