# Reproducibility of the computed results

This repository holds the code and data behind the computed results of the paper *A passive-wall bound on the apparent viscosity of micropolar water in confined channels*. The paper has no appendices; it points here for the details. Wall-condition labels follow the paper: BC1 (zero wall spin), BC2 (spin proportional to vorticity), BC3 (spin slip), BC4 (zero couple traction). In the scripts and in the parameter files the wall conditions BC1 and BC2 may appear under their working names `M1` and `M3` (script and output file names containing `m3` refer to BC2), and `a` in the tube formulas is the paper's $q=\lambda R$.

No item needs an initial guess. All results were computed with Python 3.12.10, NumPy 2.5.3, SciPy 1.18.1, SymPy 1.14.0 (mpmath 1.3.0) and Matplotlib 3.11.2; the four scripts that read source PDFs also need PyMuPDF 1.28.2 and Pillow 12.3.0. The full parameter files are in `params/`, written by `params_writer.py` when each script runs. `run_all.py` runs every script listed in `run_all.txt`, in that order, and writes the screen output of each script to `cikti/`.

## Figure numbering

Script, output-file and parameter-file names carry a figure number that differs from the paper's by one, because the file names were kept when the paper's figures were renumbered. Some comments and printed texts inside the scripts and `*_output.txt` files use the numbering of the file names. The mapping is:

| Paper | Script | Figure files written to `../sekil/` | Parameter file | Number in file names |
|---|---|---|---|---|
| Fig. 1 | `fig_size_bounds.py` | `fig2_size_bounds.pdf/.png` | `params/Fig_2.json` | 2 |
| Fig. 2 | `fig_tube_wall_family.py` | `fig3_tube_wall_family.pdf/.png` | `params/Fig_3.json` | 3 |
| Fig. 3 | `fig_data_size.py` | `fig4_data_size.pdf/.png` | `params/Fig_4.json` | 4 |
| Fig. 4 | `fig_data_temperature.py` | `fig5_data_temperature.pdf/.png` | `params/Fig_5.json` | 5 |

Tables 1–3 keep their numbers (`params/Table_1.json` … `Table_3.json`).

## Parameters, domains, meshes and tolerances

The table uses the paper numbers.

| Item | Script | Parameters | Domain / mesh | Tolerances |
|---|---|---|---|---|
| Fig. 1 | `fig_size_bounds.py` | mu=1; kappa=1; gamma=1; alpha+beta=0; … (full list in `fig_size_bounds.py`) | slit and tube: closed forms; square: side a, 2D cross-section / square: stretched grid, 80 x 80 intervals, second order | sparse direct solve; grid error at most 0.00085 at N = 160 (micropolar_duct_2d.py (b)) |
| Fig. 2 | `fig_tube_wall_family.py` | mu=1; kappa/mu=[1, 1, 10]; lambda R=[1, 10, 10]; beta/gamma=0; … (full list in `fig_tube_wall_family.py`) | tube 0 <= r <= R (closed form, modified Bessel functions) / closed form; 1601 sigma points | closed form (agrees with finite differences to 1.1e-9, check_tube_m3_bound.py N1) |
| Fig. 3 | `fig_data_size.py` | eta0 [mPa s]=0.7; eta_r [mPa s]=0.17; zeta0+zeta_r [mPa s nm^2]=2.1; … (full list in `fig_data_size.py`) | tube, slit: closed forms; square: finite differences (160 x 160, stretched grid) for D_h <= 200 nm and the large-size law (19) beyond | junction at D_h = 200 nm: finite differences and law differ by 2.5e-4 (`fig4_data_size_output.txt`) |
| Fig. 4 | `fig_data_temperature.py` | coefficients=Hansen 2010 Table II, 284-319 K (MD-MD ratios); wall=BC1; … (full list in `fig_data_temperature.py`) | tube, closed form / closed form (no grid) | closed form |
| Table 1 | `water_slip_length.py, check_hansen2010_table_eq23.py` | source=Hansen et al. 2010 Table II (SPC/Fw), 284-299 K; … (full list in `water_slip_length.py; check_hansen2010_table_eq23.py`) | not applicable / closed form | exact arithmetic |
| Table 2 | `micropolar_duct_2d.py, check_tube_m3_bound.py` | square=mu = gamma = 1, alpha + beta = 0, kappa/mu = 1 and 10, lambda a = 1, 10, 100; … (full list in `micropolar_duct_2d.py; check_tube_m3_bound.py`) | square side a; tube radius R / square: N = 40, 80, 160 per side; tube: N = 1000, 2000, 4000 radial intervals | Richardson extrapolation; sparse direct solves |
| Table 3 | `water_slip_length.py, micropolar_duct_2d.py` | coefficients=Hansen 2010 Table II at 299 K (eta0 0.70, eta_r 0.17 mPa s, zeta 2.1 mPa s nm^2); … (full list in `water_slip_length.py; micropolar_duct_2d.py`) | tube, slit: closed forms; square: 2D cross-section of side D_h / square: stretched grid, 160 x 160 (120 x 120 check) | N = 120/160 difference <= 2e-5 |

## Further numbers of the paper

- Grid stretching of the duct solver (`micropolar_duct_2d.py`): x = (L/2)[1 + tanh(s xi)/tanh s], xi in [-1, 1], with s = min(3, 0.8 + 0.5 ln(lambda a)) for lambda a > 3 and s = 0 (uniform grid) otherwise; printed by `check_duct_converged_differences.py`, part (d).
- Converged comparisons (N = 80/160/320, Richardson extrapolation): `check_duct_converged_differences.py`.
- Data table for distribution: `data/water_viscosity_data_public.csv` (142 rows, ASCII) with `data/README_data.md`, produced by `data_public_table.py`.
- Digitization accuracy: `check_digitization_accuracy.py` writes `data/digitization_accuracy_pairs.csv`.
- Extended Table 3 and its sensitivity checks: `data_table3_extended.py`, `data_group_sensitivity.py`, `data_measurement_factors.py`, `data_points_above_bound.py`, `check_extended_tables.py`.
- Joint change of the rotational and spin viscosities, ceilings and finite-difference checks: `tube_joint_eta_r_zeta.py`, `square_ceiling_fd_check.py`.
- Wall power, passivity and energy-identity checks: `check_wall_power_sign_conventions.py`, `check_bc4_slip_order.py`, `check_tube_fourier_orthogonality.py`, `khanukaeva_n07_beta_power.py`.
- Other digitization and literature checks: `bondarenko2024_fig12_digitize.py`, `wu2017_interfacial_viscosity_200nm.py`, `check_hansen2010_slit_discrepancy.py`, `check_khanukaeva2023_walls.py`.
- Data preparation: `digitize_capillary_data.py` (figure readings and table conversions that build the data table), `data_table3_markdown.py` (prints the extended Table 3 from `data_table3_members.csv`).
- Comparison of the data with the bound and with the BC1 predictions: `compare_data_md_bound.py` (size dependence), `temperature_comparison.py` and `check_temperature_envelope.py` (temperature dependence), `hansen2010_md_vs_deryagin.py` (MD coefficients against the quartz-capillary data as fitted by Migun 1981), `migun_fit_corrected_check.py` (reading of the fit of Migun 1981 and Kolpashchikov et al. 1983).
- Model and wall-condition checks: `check_dissipation_unidirectional.py` (symbolic dissipation of unidirectional duct flow), `check_slit_wall_conditions.py` (slit closed forms for several wall conditions), `check_slit_asymptotics.py` (large-gap law and equivalent slip length of the slit), `check_tube_nonaxisymmetric_modes.py` (coefficient conditions of the tube bound), `check_hansen2010_mapping.py` (term-by-term mapping of Hansen et al. 2010 onto Eringen's model), `check_kazoe_resistance_formula.py` (Newtonian flow rate of the actual cross-sections against the square-duct formula used by Kazoe et al.).
- Checks added for the submitted version: `bound_md_uncertainty.py` (standard error of the bound 1 + eta_r/eta0 from the MD standard errors of Hansen et al. 2010, Table II, and the distance of the narrow-tube and square-channel data above it), `check_tube_bc2_n_data_radii.py` (BC2 constant n and wall power needed to reach the tube data at the data radii 41.25-59.76 nm), `check_prediction_293_vs_299.py` (BC1 predictions of Table 3 at 284-293 K against 299 K), `check_churaev_layer_slip.py` (slip-length equivalent of the 8 nm immobile layer of Churaev et al.).
- Numbers and parameter files of the text: `paper_numbers.py` (derived numbers that no other script prints; output `paper_numbers_output.txt`), `params_tables.py` (parameter files of Tables 1 and 3).

## Running everything

- Run from this folder (`kod/`): `python run_all.py`. Set `OPENBLAS_NUM_THREADS=1`. The slowest script (`check_duct_converged_differences.py`) takes about 20 min.
- Memory: three computations solve sparse systems with 320 x 320 intervals and need about 4-5 GB of RAM: `check_duct_converged_differences.py`, `square_ceiling_fd_check.py` and the finite-difference check in `tube_joint_eta_r_zeta.py`. On smaller machines they stop with a MemoryError and keep their saved outputs; all other scripts run in under 1.5 GB.
- On Windows, clone into a short path (e.g. `C:\src\`) or enable long paths; `run_all.py` writes `cikti/<script>.txt` and fails beyond the 260-character path limit.
- Rectangular sections (Results, real cross-sections of Refs. [4, 5] instead of squares): part (d) of `micropolar_duct_2d.py` uses an earlier coefficient set (eta0 = 1.0 mPa s, 298 K). With the 299 K coefficients of Table 1 the differences are also below 1e-3 (at most 4.2e-4); the statement in the paper holds for both sets.
- The data table is written to `../01_arastirma/veri_tablosu.csv` and the figures to `../sekil/` (folders next to `kod/`, created if missing). The distributed public copy of the data table is `data/water_viscosity_data_public.csv` (written by `data_public_table.py`).
- Source PDFs are copyrighted and not distributed. Four scripts read them: `check_hansen2010_slit_discrepancy.py` (vector data of the published figures of Hansen et al. 2010 and 2015), `check_khanukaeva2023_walls.py` (section D only: digitized figures of Khanukaeva and Troshkin 2023), `check_digitization_accuracy.py` (figure crops for the reading-accuracy check) and `bondarenko2024_fig12_digitize.py` (screenshot of Fig. 12 of Bondarenko et al. 2024). **Source PDF required, not distributed:** without the files these scripts print `SKIPPED: source PDF required, not distributed` and keep their saved output files (`*_output.txt`, `data/digitization_*.csv`), which are distributed.
- Helper modules: `paper_common.py` (closed forms and water coefficients) and `params_writer.py` (writes `params/*.json`).
