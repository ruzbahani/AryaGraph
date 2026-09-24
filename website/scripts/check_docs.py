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

"""Check the website pages: run their code and enforce the writing rules.

For every Markdown page under ``source/``:

* every fenced ```` ```python ```` block is executed, top to bottom, in one
  namespace per page and in a fresh temporary working directory (so files the
  examples save land there). Use ```` ```pycon ```` or ```` ```text ```` for code
  that is shown but not meant to run;
* the em dash character (U+2014) is an error: the project's writing style uses
  colons, commas, parentheses or a new sentence instead;
* absolute wording ("always", "never", "exactly", "best", "perfect",
  "guaranteed", "blazing") is reported as a warning to review.

    python scripts/check_docs.py                     # every page
    python scripts/check_docs.py source/tutorials    # a folder or single pages
    python scripts/check_docs.py --no-run            # style checks only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SITE = HERE.parent
ROOT = SITE.parent
SOURCE = SITE / "source"

FENCE = re.compile(r"^```python[ \t]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
ABSOLUTE = re.compile(r"\b(always|never|exactly|best|perfect(?:ly)?|guarantee[sd]?|blazing(?:ly)?)\b", re.IGNORECASE)

RUNNER = r"""
import json, sys, traceback
blocks = json.load(open(sys.argv[1], encoding="utf-8"))
ns = {"__name__": "__main__"}
for i, (line, code) in enumerate(blocks):
    try:
        exec(compile(code, f"<block {i + 1} (line {line})>", "exec"), ns)
    except BaseException:
        print(f"BLOCK {i + 1} starting at line {line} failed:")
        traceback.print_exc(limit=6, file=sys.stdout)
        sys.exit(1)
print(f"OK {len(blocks)} blocks")
"""


def pages(targets: list[str]) -> list[Path]:
    if not targets:
        targets = [str(SOURCE)]
    out: list[Path] = []
    for t in targets:
        p = Path(t)
        if not p.is_absolute():
            p = (Path.cwd() / p) if (Path.cwd() / p).exists() else SITE / p
        if p.is_dir():
            out += sorted(q for q in p.rglob("*.md") if "_build" not in q.parts and "generated" not in q.parts)
        elif p.suffix == ".md":
            out.append(p)
    return out


def run_page(page: Path, timeout: int) -> tuple[bool, str]:
    text = page.read_text(encoding="utf-8")
    blocks = [(text.count("\n", 0, m.start()) + 2, m.group(1)) for m in FENCE.finditer(text)]
    if not blocks:
        return True, "no code"
    with tempfile.TemporaryDirectory(prefix="agdoc-") as tmp:
        spec = Path(tmp) / "blocks.json"
        spec.write_text(json.dumps(blocks), encoding="utf-8")
        runner = Path(tmp) / "runner.py"
        runner.write_text(RUNNER, encoding="utf-8")
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
        try:
            proc = subprocess.run(
                [sys.executable, str(runner), str(spec)],
                cwd=tmp, env=env, capture_output=True, text=True, encoding="utf-8", timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, f"timed out after {timeout} s"
    ok = proc.returncode == 0
    detail = (proc.stdout + proc.stderr).strip()
    return ok, (detail.splitlines()[-1] if ok else detail[-3000:])


def style(page: Path) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    in_code = False
    for n, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if "—" in line:
            errors.append(f"{n}: em dash: {line.strip()[:100]}")
        if not in_code:
            for m in ABSOLUTE.finditer(line):
                warnings.append(f"{n}: '{m.group(0)}': {line.strip()[:100]}")
    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("targets", nargs="*")
    ap.add_argument("--no-run", action="store_true", help="skip executing code blocks")
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()
    failed = 0
    warned = 0
    for page in pages(args.targets):
        rel = page.relative_to(SITE)
        errors, warnings = style(page)
        ok, detail = (True, "not run") if args.no_run else run_page(page, args.timeout)
        status = "ok  " if ok and not errors else "FAIL"
        print(f"{status} {rel}  [{detail if ok else 'code failed'}]")
        if not ok:
            print("     " + detail.replace("\n", "\n     "))
        for e in errors:
            print(f"     error   {e}")
        for w in warnings:
            print(f"     review  {w}")
        failed += (not ok) or bool(errors)
        warned += len(warnings)
    print(f"\n{failed} page(s) failing, {warned} wording note(s) to review")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
