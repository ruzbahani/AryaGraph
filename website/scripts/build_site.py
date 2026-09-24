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

"""Build the whole website into ``_build/html`` (ready to upload to /aryagraph/).

    python scripts/build_site.py              # reference + figures + code checks + strict HTML build
    python scripts/build_site.py --no-assets  # reuse the generated figures
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parent


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=SITE)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-assets", action="store_true", help="skip regenerating figures")
    ap.add_argument("--no-check", action="store_true", help="skip executing the pages' code examples")
    ap.add_argument("--no-strict", action="store_true", help="do not turn warnings into errors")
    args = ap.parse_args()
    py = sys.executable
    run([py, str(HERE / "gen_reference.py")])
    if not args.no_assets:
        run([py, str(HERE / "build_assets.py")])
    if not args.no_check:
        run([py, str(HERE / "check_docs.py")])  # every python block on every page must run
    cmd = [py, "-m", "sphinx", "-b", "html", "-j", "auto", "source", "_build/html"]
    if not args.no_strict:
        cmd[3:3] = ["-W", "--keep-going"]
    run(cmd)
    print(f"\nsite ready: {SITE / '_build' / 'html' / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
