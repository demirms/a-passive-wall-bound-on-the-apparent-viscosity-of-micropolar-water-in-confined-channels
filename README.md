# A passive-wall bound on the apparent viscosity of micropolar water in confined channels

Code and data behind the paper *A passive-wall bound on the apparent viscosity of micropolar water in confined channels* by Mehmet Şirin Demir (Istanbul University-Cerrahpaşa, Faculty of Engineering, Department of Mechanical Engineering, Istanbul, Türkiye), submitted to *Physics of Fluids*.

## Contents

| Folder | What it holds |
|---|---|
| `kod/` | Python scripts that compute every number, table and figure of the paper, the solver modules they share, the parameter files (`kod/params/`), the saved screen output of each script (`*_output.txt`) and the distributed data table (`kod/data/water_viscosity_data_public.csv`, 142 rows, with `kod/data/README_data.md`). `kod/README.md` lists which script produces which figure, table and number, with parameters, meshes and tolerances. |

## Reproducing the results

```
cd kod
python run_all.py
```

`run_all.py` runs the scripts listed in `kod/run_all.txt` in order. Python 3.12 with NumPy, SciPy, SymPy (with mpmath) and Matplotlib is required; the four scripts that read source PDFs also import PyMuPDF and Pillow at start-up, so these two packages must be installed even to reach the SKIPPED message. The exact versions are in `kod/README.md`. Set `OPENBLAS_NUM_THREADS=1` on small machines; the slowest script takes about 20 min. Figures are written to `sekil/` and the working data table to `01_arastirma/` next to `kod/` (both created if missing).

Four scripts read figures or tables from copyrighted source papers, which are not distributed. Without those files they print `SKIPPED: source PDF required, not distributed` and keep their saved outputs, which are included.

## Licence and citation

Please cite the paper when using the code or the data. The data table was compiled from the published measurements cited in the paper; the original sources should be cited for the measurements themselves.
