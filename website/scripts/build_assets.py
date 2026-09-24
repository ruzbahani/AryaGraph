# =============================================================================
#
#      _                     ____                 _
#     / \   _ __ _   _  __ _ / ___|_ __ __ _ _ __ | |__
#    / _ \ | '__| | | |/ _` | |  _| '__/ _` | '_ \| '_ \
#   / ___ \| |  | |_| | (_| | |_| | | | (_| | |_) | | | |
#  /_/   \_\_|   \__, |\__,_|\____|_|  \__,_| .__/|_| |_|
#               |___/                     |_|
#
#          Graph & DAG visualization, analysis and simulation.
#
# -----------------------------------------------------------------------------
#  Copyright (c) 2026 Ali Mohammadi Ruzbahani
#  SPDX-License-Identifier: MIT
#
#  https://ruzbahani.com/aryagraph
# =============================================================================

"""Generate every figure and interactive embed used by the website.

Each module in ``scripts/assets/`` defines ``build(out: Path) -> None`` and
writes its files into ``source/_static/generated/<module name>/``. All figures
on the site are produced by running AryaGraph here, never drawn by hand.

    python scripts/build_assets.py            # all sections
    python scripts/build_assets.py landing    # selected sections
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
ROOT = SITE.parent
GENERATED = SITE / "source" / "_static" / "generated"
sys.path.insert(0, str(ROOT / "src"))


def load(path: Path):
    spec = importlib.util.spec_from_file_location(f"assets_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str]) -> int:
    scripts = sorted(p for p in (HERE / "assets").glob("*.py") if not p.name.startswith("_"))
    if argv:
        wanted = set(argv)
        scripts = [p for p in scripts if p.stem in wanted]
        missing = wanted - {p.stem for p in scripts}
        if missing:
            print(f"unknown asset sections: {sorted(missing)}", file=sys.stderr)
            return 2
    failures = 0
    for path in scripts:
        out = GENERATED / path.stem
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        t0 = time.perf_counter()
        try:
            load(path).build(out)
        except Exception as exc:  # report every broken section, then fail
            failures += 1
            print(f"FAILED {path.stem}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        files = sum(1 for _ in out.rglob("*") if _.is_file())
        print(f"{path.stem:<24} {files:3d} files  {time.perf_counter() - t0:5.1f} s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
