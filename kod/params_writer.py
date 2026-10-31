# -*- coding: utf-8 -*-
"""Writes params/<label>.json with the parameters, domain, mesh and tolerances of a figure or table.

Usage in a script:
    from params_writer import save_params
    save_params("Fig. 6", parameters={"kappa": 1.0}, domain="eta in [0, 40]", mesh="Chebyshev N=200",
                tolerances="RelTol 1e-10", initial_guess="linear profile")
The file is written to params/ next to the calling script ("Fig. 6" -> Fig_6.json). The keys `script` (name of the
calling file), `software` (Python, NumPy, SciPy and, if imported, SymPy, mpmath, Matplotlib versions) and `date`
(YYYY-MM-DD) are filled in automatically.

params JSON keys: label, script, parameters (dict), domain, mesh, tolerances, initial_guess, software (dict), date.
"""
import datetime
import json
import os
import platform
import re
import sys


def _versions():
    ver = {"python": platform.python_version()}
    try:
        from importlib import metadata
    except Exception:
        return ver
    for name in ("numpy", "scipy", "sympy", "mpmath", "matplotlib"):
        if name in ("numpy", "scipy") or name in sys.modules:
            try:
                ver[name] = metadata.version(name)
            except Exception:
                pass
    return ver


def _json_item(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def label_file_name(label):
    s = re.sub(r"[^\w\-]+", "_", label.strip(), flags=re.U).strip("_")
    return s or "item"


def save_params(label, parameters=None, domain=None, mesh=None, tolerances=None, initial_guess=None, script=None,
                params_dir=None):
    """Writes params/<label>.json and returns the path written.
    If params_dir is not given, the params/ folder next to the calling script is used."""
    if script is None or params_dir is None:
        try:
            frame = sys._getframe(1)
            caller = frame.f_globals.get("__file__") or sys.argv[0]
        except Exception:
            caller = sys.argv[0]
        caller = os.path.abspath(caller)
        if script is None:
            script = os.path.basename(caller)
        if params_dir is None:
            params_dir = os.path.join(os.path.dirname(caller), "params")
    os.makedirs(params_dir, exist_ok=True)
    record = {"label": label, "script": script, "parameters": dict(parameters or {}), "domain": domain, "mesh": mesh,
              "tolerances": tolerances, "initial_guess": initial_guess, "software": _versions(),
              "date": datetime.date.today().isoformat()}
    path = os.path.join(params_dir, label_file_name(label) + ".json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, ensure_ascii=False, indent=1, default=_json_item)
        f.write("\n")
    return path
