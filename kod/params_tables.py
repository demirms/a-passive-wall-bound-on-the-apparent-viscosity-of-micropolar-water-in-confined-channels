"""Parameter files (params/) for the manuscript tables, whose values are printed by other scripts:
Table 1 <- water_slip_length.py (1) and check_hansen2010_table_eq23.py (Hansen et al. 2010 Table II, MD-MD map);
Table 3 <- water_slip_length.py (3)-(4) and micropolar_duct_2d.py (d) (data rows versus M1 prediction and bound);
Table 2 <- micropolar_duct_2d.py (b) and check_tube_m3_bound.py N1 (verification).
"""
from params_writer import save_params

save_params("Table 1", script="water_slip_length.py; check_hansen2010_table_eq23.py",
       parameters={"source": "Hansen et al. 2010 Table II (SPC/Fw), 284-299 K",
                     "map": "mu = eta0 - eta_r, kappa = 2 eta_r, gamma = zeta0 + zeta_r"},
       domain="not applicable", mesh="closed form", tolerances="exact arithmetic", initial_guess="not applicable")
save_params("Table 3", script="water_slip_length.py; micropolar_duct_2d.py",
       parameters={"coefficients": "Hansen 2010 Table II at 299 K (eta0 0.70, eta_r 0.17 mPa s, zeta 2.1 mPa s nm^2)",
                     "wall": "M1", "data": "01_arastirma/veri_tablosu.csv"},
       domain="tube, slit: closed forms; square: 2D cross-section of side D_h",
       mesh="square: stretched grid, 160 x 160 (120 x 120 check)",
       tolerances="N = 120/160 difference <= 2e-5", initial_guess="not applicable")
save_params("Table 2", script="micropolar_duct_2d.py; check_tube_m3_bound.py",
       parameters={"square": "mu = gamma = 1, alpha + beta = 0, kappa/mu = 1 and 10, lambda a = 1, 10, 100",
                     "tube": "mu = 1, kappa/mu = 0.1-10, lambda R = 0.3-100, n = -1 to 3"},
       domain="square side a; tube radius R",
       mesh="square: N = 40, 80, 160 per side; tube: N = 1000, 2000, 4000 radial intervals",
       tolerances="Richardson extrapolation; sparse direct solves", initial_guess="not applicable")
print("params written")
