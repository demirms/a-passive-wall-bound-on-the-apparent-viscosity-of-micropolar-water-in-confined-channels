# Digitized apparent-viscosity data (water in fine channels)

`water_viscosity_data_public.csv` has 142 rows from eight published studies (Churaev et al. 1970 and 1984, Cheng and Giordano 2002,
Tas et al. 2004, Haneveld et al. 2008, Li et al. 2012, Kazoe et al. 2022 and 2023). Each row gives the source, the figure or table the value
was taken from, the geometry, the size and its symbol (tube radius r, slit or plate height, square hydraulic diameter D_h; all in nm), the
ratio of the apparent to the bulk viscosity with its uncertainty and the type of that uncertainty, the temperature, the electrolyte,
the measurement method and its class (pressure-driven flow, capillary filling, or filling under applied pressure), the normalization viscosity
where it is known, and a calibration note.

Values given in tables or in the text were taken directly (data_origin: table, text). The other values were read from the published figures
(data_origin: figure read-off) by pixel calibration against two axis ticks at 400-900 dpi; the reading uncertainty is 0.01-0.03 in the viscosity ratio
(about 1-2 % of the abscissa), and the column eta_ratio_unc holds the author error bar where one exists or the reading uncertainty, whichever is larger
(unc_type says which). Rows R012-R014 (CCl4 and benzene) are approximate because the markers overlap the ratio = 1 line.

The ratio is relative to the bulk viscosity used by each source. Li et al. state 0.868 mPa s (26.1 C). Kazoe et al. 2022 do not state their bulk
viscosity; 0.892 mPa s follows from the reported ratio 3.6 at 190 nm and the tabulated viscosity (normalization_note). Kazoe et al. 2023 give no
value, and the temperature is not stated in either paper. The columns in_table3, in_fig4 (the data-size figure, Fig. 3 of the paper) and in_89_water_rows mark the subsets used in the paper;
rows flagged duplicate_flag = yes re-plot data of another row (Kazoe 2023 Fig. 4 repeats the Kazoe 2022 values; Churaev 1984 Table II repeats three rows of Table I)
and must be counted once. Churaev 1984 and Cheng and Giordano (non-water liquids) are slip data (ratio below 1 is an apparent value) and are
not used as evidence for an increased viscosity. The photoresist-walled slits of Cheng and Giordano (water) are kept apart from the silica data.

Source of the numbers in the paper: `data_table3_extended.py`, `data_group_sensitivity.py`, `data_measurement_factors.py` and `data_public_table.py`
(outputs `*_output.txt`).
