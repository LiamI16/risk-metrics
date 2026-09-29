"""Regenerate every figure: the paper figures, and with --examples the example scripts too.

The paper figure scripts are not in figures/ yet, so only --examples does any work; list them
in FIGURES when they land.

Run from the repository root:  python main.py [--examples]
Outputs go to output/figures/ and output/examples/.
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIGURES = []
EXAMPLES = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "examples").glob("[0-9]*.py"))

if __name__ == "__main__":
    scripts = FIGURES + (EXAMPLES if "--examples" in sys.argv[1:] else [])
    sys.argv = sys.argv[:1]                        # the scripts parse their own flags
    sys.path.insert(0, str(ROOT / "figures"))      # figure scripts import figures/export.py
    for script in scripts:
        print(f"\n=== {script}")
        runpy.run_path(str(ROOT / script), run_name="__main__")
    if not FIGURES:
        raise NotImplementedError("no paper figure scripts in figures/ yet")
