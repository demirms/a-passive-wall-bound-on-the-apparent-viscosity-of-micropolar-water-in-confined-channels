"""Builds the public, deposit-ready data table of apparent viscosities.

Reads 01_arastirma/veri_tablosu.csv (unchanged) and data_table3_members.csv, writes
  data/water_viscosity_data_public.csv   (142 rows, UTF-8, ASCII only, English column names)
  data/README_data.md                    (paragraph for the repository README)
and prints a verification report (row count, id uniqueness, ASCII check, value identity with the source table,
counts of the subsets used in Table 3, Fig. 4 and the '89 water rows'). Output: data_public_table_output.txt
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "01_arastirma", "veri_tablosu.csv")
DATA = os.path.join(HERE, "data")
OUT = []


def P(s=""):
    print(s)
    OUT.append(s)


rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig")))
members = list(csv.DictReader(open(os.path.join(HERE, "data_table3_members.csv"), encoding="utf-8")))
in_t3 = {m["id"] for m in members}
in_fig4 = {m["id"] for m in members if not m["group"].startswith("Tas [2] 0.1 M NaCl")}

BOUND299 = 1 + 0.17 / 0.70

REF = {
    "Churaev_1970": "Churaev, Sobolev, Zorin (1970), Spec. Discuss. Faraday Soc. 1, 213-220",
    "Churaev_1984": "Churaev, Sobolev, Somov (1984), J. Colloid Interface Sci. 97, 574-581",
    "Cheng_2002": "Cheng, Giordano (2002), Phys. Rev. E 65, 031206",
    "Tas_2004": "Tas et al. (2004), Appl. Phys. Lett. 85, 3274",
    "Haneveld_2008": "Haneveld et al. (2008), J. Appl. Phys. 104, 014309",
    "Li_2012": "Li, Kazoe, Mawatari, Sugii, Kitamori (2012), J. Phys. Chem. Lett. 3, 2447-2452",
    "Kazoe_2022": "Kazoe, Kubori, Morikawa, Mawatari, Kitamori (2022), Anal. Sci. 38, 281-287",
    "Kazoe_2023": "Kazoe et al. (2023), Anal. Sci. 39, 779-784",
}
METHOD_CLASS = {
    "Churaev_1970": "pressure-driven", "Churaev_1984": "pressure-driven", "Cheng_2002": "pressure-driven",
    "Kazoe_2022": "pressure-driven", "Kazoe_2023": "pressure-driven",
    "Li_2012": "filling under applied pressure", "Tas_2004": "capillary filling", "Haneveld_2008": "capillary filling",
}


def norm_of(r):
    c = r["conversion"]
    if "0.868" in c:
        return "0.868", "bulk value stated in the source (26.1 C)"
    if "0.892" in c:
        return "0.892", "not stated in the source; inferred from the reported ratio 3.6 and the tabulated viscosity at 190 nm"
    if r["source_key"] == "Cheng_2002":
        return "", "ratio of the measured to the no-slip theoretical flow-rate coefficient (alpha_theory = h^3 w/(12 mu L)); no separate viscosity value"
    if r["source_key"] == "Churaev_1984":
        return "", "eta0 = bulk viscosity at the measurement temperature (derived from the tabulated values; see conversion)"
    return "", "bulk viscosity of the source at its measurement temperature (value not stated in the source)"


sel_ids = set()
for r in rows:
    if r["liquid"].strip().lower() != "water" or "DUPLICATE" in r["notes"]:
        continue
    if "hydrophob" in (r["wall_treatment"] + r["notes"]).lower() or r["source_key"] == "Churaev_1984":
        continue
    try:
        float(r["size_nm"]), float(r["eta_ratio"])
    except ValueError:
        continue
    sel_ids.add(r["id"])

COLS = ["id", "source", "source_reference", "figure_or_table", "liquid", "wall_material", "wall_treatment", "geometry",
        "size_symbol", "size_value_nm", "size_unc_nm", "eta_ratio", "eta_ratio_unc", "unc_type", "temperature_C", "electrolyte",
        "normalization_viscosity_mPa_s", "normalization_note", "method", "method_class", "data_origin", "calibration_note",
        "duplicate_flag", "in_89_water_rows", "in_table3", "in_fig4", "above_299K_bound", "notes"]
os.makedirs(DATA, exist_ok=True)
out_rows = []
for r in rows:
    nv, nn = norm_of(r)
    cal = r["data_origin"]
    if r["conversion"].strip() and r["conversion"].strip() != "none":
        cal += "; conversion: " + r["conversion"].strip()
    dup = "yes" if "DUPLICATE" in r["notes"] else "no"
    ab = ""
    if r["id"] in sel_ids:
        ab = "yes" if float(r["eta_ratio"]) > BOUND299 else "no"
    out_rows.append({
        "id": r["id"], "source": r["source_key"], "source_reference": REF[r["source_key"]], "figure_or_table": r["location"],
        "liquid": r["liquid"], "wall_material": r["wall_material"], "wall_treatment": r["wall_treatment"], "geometry": r["geometry"],
        "size_symbol": r["size_symbol"], "size_value_nm": r["size_nm"], "size_unc_nm": r["size_unc_nm"],
        "eta_ratio": r["eta_ratio"], "eta_ratio_unc": r["eta_ratio_unc"], "unc_type": r["unc_type"],
        "temperature_C": r["temperature_C"], "electrolyte": r["electrolyte"], "normalization_viscosity_mPa_s": nv,
        "normalization_note": nn, "method": r["method"], "method_class": METHOD_CLASS[r["source_key"]],
        "data_origin": r["data_origin"], "calibration_note": cal, "duplicate_flag": dup,
        "in_89_water_rows": "yes" if r["id"] in sel_ids else "no", "in_table3": "yes" if r["id"] in in_t3 else "no",
        "in_fig4": "yes" if r["id"] in in_fig4 else "no", "above_299K_bound": ab, "notes": r["notes"]})

path = os.path.join(DATA, "water_viscosity_data_public.csv")
with open(path, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(out_rows)

# ---------------------------------------------------------------- verification
back = list(csv.DictReader(open(path, encoding="utf-8")))
txt = open(path, encoding="utf-8").read()
nonascii = sorted({ch for ch in txt if ord(ch) > 127})
P(f"rows written: {len(back)} (source table: {len(rows)}); unique ids: {len({r['id'] for r in back}) == len(back)}")
P(f"non-ASCII characters in the public CSV: {nonascii if nonascii else 'none'}")
same = all(b["eta_ratio"] == r["eta_ratio"] and b["eta_ratio_unc"] == r["eta_ratio_unc"] and b["size_value_nm"] == r["size_nm"] and b["id"] == r["id"]
           for b, r in zip(back, rows))
P(f"values identical to veri_tablosu.csv (id, size, eta_ratio, uncertainty): {same}")
from collections import Counter
P("rows per source: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r['source'] for r in back).items())))
P("rows per method class: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r['method_class'] for r in back).items())))
P(f"duplicate rows: {sum(r['duplicate_flag'] == 'yes' for r in back)}; in the 89 water rows: {sum(r['in_89_water_rows'] == 'yes' for r in back)}; "
  f"above the 299 K bound (of the 89): {sum(r['above_299K_bound'] == 'yes' for r in back)}; in Table 3: {sum(r['in_table3'] == 'yes' for r in back)}; in Fig. 4: {sum(r['in_fig4'] == 'yes' for r in back)}")
P("rows with an empty uncertainty: " + str(sum(r["eta_ratio_unc"].strip() == "" for r in back)))

readme = """# Digitized apparent-viscosity data (water in fine channels)

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
"""
with open(os.path.join(DATA, "README_data.md"), "w", encoding="utf-8", newline="\n") as f:
    f.write(readme)
nonascii_readme = sorted({ch for ch in readme if ord(ch) > 127})
P(f"README_data.md written; non-ASCII characters: {nonascii_readme if nonascii_readme else 'none'}")
assert len(back) == 142 and not nonascii and same
with open(os.path.join(HERE, "data_public_table_output.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(OUT) + "\n")
