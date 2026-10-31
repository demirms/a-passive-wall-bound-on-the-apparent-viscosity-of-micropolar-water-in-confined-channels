# -*- coding: utf-8 -*-
"""
run_all.py -- runs all scripts of this folder in the order of run_all.txt (from this folder, kod/: python -u run_all.py).
The screen output of each script is written to cikti/<script>.txt. Standard library only.
Lines "cikti: <folder>" in run_all.txt name output folders that are created if missing; lines "girdi: <folder>" name
optional input folders; "after: a.py, b.py" lists the scripts that must have finished successfully before a script runs.
"""
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def read_list(path="run_all.txt"):
    scripts, out_dirs = [], ["cikti"]
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            s = raw.split(" #")[0].strip() if not raw.lstrip().startswith("#") else ""
            if not s:
                continue
            if s.lower().startswith("cikti:"):
                out_dirs.append(s.split(":", 1)[1].strip())
            elif s.lower().startswith("girdi:"):
                continue
            else:
                body, _, before = s.partition(" after:")
                scripts.append((shlex.split(body), [x.strip() for x in before.split(",") if x.strip()]))
    return scripts, out_dirs


def main():
    scripts, out_dirs = read_list()
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
    succeeded, done, failed = 0, set(), set()
    total, count = {}, {}
    for args, _ in scripts:
        total[args[0]] = total.get(args[0], 0) + 1
    for args, before in scripts:
        name = args[0]
        count[name] = count.get(name, 0) + 1
        if any(b in failed or (b != name and b not in done) for b in before):
            print(f"SKIPPED {' '.join(args)}: prerequisite missing or failed {before}")
            continue
        log = os.path.splitext(name)[0] + (f"_{count[name]}" if total[name] > 1 else "") + ".txt"
        t0 = time.time()
        with open(os.path.join("cikti", log), "w", encoding="utf-8", newline="\n") as f:
            r = subprocess.run([sys.executable, "-u"] + args, stdout=f, stderr=subprocess.STDOUT)
        print(f"{'OK   ' if r.returncode == 0 else 'ERROR'} {' '.join(args)} ({time.time() - t0:.1f} s, exit code {r.returncode})")
        if r.returncode == 0:
            succeeded += 1
            done.add(name)
        else:
            failed.add(name)
    print(f"RESULT: {succeeded}/{len(scripts)} scripts succeeded")
    return 0 if succeeded == len(scripts) else 1


if __name__ == "__main__":
    sys.exit(main())
