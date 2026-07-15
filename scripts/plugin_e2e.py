#!/usr/bin/env python3
"""CI wrapper for the mock-Figma plugin compiler E2E harness.

Runs figma-plugin/test/run_e2e.js under Node and forwards its exit code:
0 = every fixture passed, non-zero = at least one failure (or Node missing).

Usage:
    python scripts/plugin_e2e.py                 # all fixtures
    python scripts/plugin_e2e.py --fixture kitchen-sink --verbose
    python scripts/plugin_e2e.py --json out.json
Extra arguments are passed straight through to run_e2e.js.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNNER = PROJECT_ROOT / "figma-plugin" / "test" / "run_e2e.js"


def main() -> int:
    node = shutil.which("node")
    if not node:
        print("plugin_e2e: node executable not found on PATH", file=sys.stderr)
        return 2
    if not RUNNER.exists():
        print(f"plugin_e2e: runner not found at {RUNNER}", file=sys.stderr)
        return 2

    args = sys.argv[1:]
    if not any(a.startswith("--fixture") for a in args) and "--all" not in args:
        args = ["--all", *args]

    proc = subprocess.run([node, str(RUNNER), *args], cwd=PROJECT_ROOT)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
