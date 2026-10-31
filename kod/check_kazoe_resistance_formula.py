"""Newtonian flow rate of the actual rectangular cross-sections vs the square-duct formula
Q = 4.5 Dh^4 G/(128 eta) used by Kazoe et al. to convert resistance into viscosity.
Ratio > 1 means the formula underestimates Q, i.e. the inferred viscosity is too high by that factor."""
import numpy as np

def qn_rect(W, D, nmax=401):
    a, b = min(W, D), max(W, D)
    s = sum(np.tanh(n * np.pi * b / (2 * a)) / n**5 for n in range(1, nmax, 2))
    return a**3 * b / 12 * (1 - 192 * a / (np.pi**5 * b) * s)

lines = []
for W, D in [(1, 1), (73, 72), (130, 110), (163, 230), (410, 380), (777, 745)]:
    Dh = 2 * W * D / (W + D)
    q_formula = 4.5 * Dh**4 / 128
    r = qn_rect(W, D) / q_formula
    lines.append(f"W={W:4d} D={D:4d} Dh={Dh:7.2f}: Q_exact/Q_formula = {r:.4f}  (viscosity bias factor)")
with open(__file__.replace('.py', '_output.txt'), 'w') as f:
    for s in lines:
        print(s); f.write(s + "\n")
