#!/usr/bin/env python3
"""Run-directory management for ad-decompiler-v2.

The ``runs/`` directory accumulates dozens of pipeline output folders (PNGs,
masks, layer assets, qa.json). Some are precious (golden benchmarks, A/B
evidence cited in docs); most are stale iteration debris (crashfix-*, fixtest-*).
This tool reports on them and safely prunes/archives the debris.

Subcommands
-----------
report   Print a table of every run: size, last-modified, has-qa.json,
         qa pass/fail, and whether the run name is referenced in any *.md doc.
prune    Delete stale runs.  Never touches docs-referenced or pattern-matched
         runs.  Dry-run by default; ``--apply`` is required to actually delete.
archive  Same selection as prune, but *moves* runs into ``--to <dir>`` instead
         of deleting them.  Dry-run by default; ``--apply`` required.

Safety
------
* Refuses to run unless it can find a project root containing ``run_pipeline.py``.
* Destruction/movement requires an explicit ``--apply`` flag (dry-run default).
* Docs-referenced runs and ``--keep-pattern`` matches are always protected.
* Recently-modified runs (within ``--keep-days``) are always protected.
* Every run that would be / was removed is printed with its size.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

# Runs matching these glob patterns are protected from prune/archive by default.
# golden-* runs are the precious benchmark suites; keep them safe out of the box.
DEFAULT_KEEP_PATTERNS = ("golden*",)

# Characters that count as part of a run "word" for docs-reference boundary
# matching, so that referencing ``inspo_test`` does not accidentally protect
# ``inspo_test2``.
_WORD = r"A-Za-z0-9_\-"


# --------------------------------------------------------------------------- #
# Project-root discovery (safety guard)
# --------------------------------------------------------------------------- #
def find_project_root(start: Optional[Path] = None) -> Optional[Path]:
    """Walk upward from *start* looking for a dir containing run_pipeline.py.

    Returns the first matching directory, or ``None`` if none is found before
    reaching the filesystem root.  This is the guard that prevents the tool
    from operating anywhere except an ad-decompiler checkout.
    """
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "run_pipeline.py").is_file():
            return candidate
    return None


# --------------------------------------------------------------------------- #
# Filesystem measurement
# --------------------------------------------------------------------------- #
def scan_tree(path: Path) -> tuple[int, float, int]:
    """Return (total_size_bytes, max_mtime, file_count) for a directory tree.

    ``max_mtime`` reflects the newest file in the tree (falling back to the
    directory's own mtime for empty trees), so "last modified" tracks real
    activity rather than just when the folder was created.
    """
    total = 0
    files = 0
    try:
        newest = path.stat().st_mtime
    except OSError:
        newest = 0.0
    for root, _dirs, names in os.walk(path):
        for name in names:
            fp = os.path.join(root, name)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            total += st.st_size
            files += 1
            if st.st_mtime > newest:
                newest = st.st_mtime
    return total, newest, files


def human_mb(num_bytes: int) -> float:
    return num_bytes / (1024.0 * 1024.0)


# --------------------------------------------------------------------------- #
# QA status
# --------------------------------------------------------------------------- #
def find_qa_files(run_dir: Path) -> list[Path]:
    """Locate qa.json files for a run.

    Single runs keep qa.json at the run root; benchmark *suites* keep one
    qa.json per image in immediate subdirectories.  Handle both layouts.
    """
    root_qa = run_dir / "qa.json"
    if root_qa.is_file():
        return [root_qa]
    return sorted(run_dir.glob("*/qa.json"))


def qa_status(run_dir: Path) -> tuple[bool, str]:
    """Return (has_qa, status_string).

    status_string is one of: ``n/a`` (no qa.json), ``pass``/``fail`` (single
    run), ``pass (n/n)`` / ``fail (0/n)`` / ``mixed (k/n)`` (suite), or a value
    suffixed with ``err`` when a qa.json could not be parsed.
    """
    qa_files = find_qa_files(run_dir)
    if not qa_files:
        return False, "n/a"

    results: list[bool] = []
    errors = 0
    for qa in qa_files:
        try:
            data = json.loads(qa.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            errors += 1
            continue
        results.append(bool(data.get("ok")))

    total = len(results)
    if total == 0:
        return True, "err"

    passed = sum(results)
    if total == 1:
        status = "pass" if results[0] else "fail"
    elif passed == total:
        status = f"pass ({passed}/{total})"
    elif passed == 0:
        status = f"fail (0/{total})"
    else:
        status = f"mixed ({passed}/{total})"
    if errors:
        status += " +err"
    return True, status


# --------------------------------------------------------------------------- #
# Docs-reference detection
# --------------------------------------------------------------------------- #
def collect_doc_texts(root: Path) -> list[tuple[Path, str]]:
    """Read every *.md at the project root and under docs/ (recursively)."""
    md_files: list[Path] = sorted(root.glob("*.md"))
    docs_dir = root / "docs"
    if docs_dir.is_dir():
        md_files += sorted(docs_dir.rglob("*.md"))
    out: list[tuple[Path, str]] = []
    for md in md_files:
        try:
            out.append((md, md.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    return out


def referenced_run_names(root: Path, names: Iterable[str]) -> set[str]:
    """Return the subset of *names* that appear as a whole token in any doc.

    Uses boundary matching so ``inspo_test`` does not match ``inspo_test2``.
    """
    texts = [text for _path, text in collect_doc_texts(root)]
    blob = "\n".join(texts)
    hits: set[str] = set()
    for name in names:
        pattern = re.compile(rf"(?<![{_WORD}]){re.escape(name)}(?![{_WORD}])")
        if pattern.search(blob):
            hits.add(name)
    return hits


# --------------------------------------------------------------------------- #
# Run scanning
# --------------------------------------------------------------------------- #
@dataclass
class RunInfo:
    name: str
    path: Path
    size_bytes: int
    mtime: float
    file_count: int
    has_qa: bool
    qa_state: str
    referenced: bool = False

    @property
    def size_mb(self) -> float:
        return human_mb(self.size_bytes)

    @property
    def age_days(self) -> float:
        return max(0.0, (time.time() - self.mtime) / 86400.0)


def iter_run_dirs(runs_dir: Path) -> list[Path]:
    """Immediate subdirectories of runs/ (ignoring stray files like .gitkeep)."""
    if not runs_dir.is_dir():
        return []
    return sorted(p for p in runs_dir.iterdir() if p.is_dir())


def scan_runs(runs_dir: Path, project_root: Optional[Path] = None) -> list[RunInfo]:
    """Build a RunInfo for every run directory, with docs-reference flags."""
    run_dirs = iter_run_dirs(runs_dir)
    infos: list[RunInfo] = []
    for rd in run_dirs:
        size, mtime, files = scan_tree(rd)
        has_qa, state = qa_status(rd)
        infos.append(
            RunInfo(
                name=rd.name,
                path=rd,
                size_bytes=size,
                mtime=mtime,
                file_count=files,
                has_qa=has_qa,
                qa_state=state,
            )
        )
    root = project_root or runs_dir.parent
    refs = referenced_run_names(root, [i.name for i in infos])
    for info in infos:
        info.referenced = info.name in refs
    return infos


# --------------------------------------------------------------------------- #
# Prune / archive selection
# --------------------------------------------------------------------------- #
@dataclass
class Decision:
    run: RunInfo
    protected: bool
    reasons: list[str] = field(default_factory=list)


def classify_run(run: RunInfo, keep_days: float, keep_patterns: Iterable[str]) -> Decision:
    """Decide whether *run* is protected, and why.

    Protected if: referenced in docs, matches a keep-pattern, or modified
    within ``keep_days`` days.  Anything else is a removal candidate.
    """
    reasons: list[str] = []
    if run.referenced:
        reasons.append("docs-referenced")
    for pat in keep_patterns:
        if fnmatch.fnmatch(run.name, pat):
            reasons.append(f"matches '{pat}'")
            break
    if run.age_days <= keep_days:
        reasons.append(f"recent ({run.age_days:.0f}d <= {keep_days:.0f}d)")
    return Decision(run=run, protected=bool(reasons), reasons=reasons)


def plan(
    runs: Iterable[RunInfo], keep_days: float, keep_patterns: Iterable[str]
) -> tuple[list[Decision], list[Decision]]:
    """Split runs into (to_remove, to_keep) decision lists."""
    keep_patterns = list(keep_patterns)
    decisions = [classify_run(r, keep_days, keep_patterns) for r in runs]
    remove = [d for d in decisions if not d.protected]
    keep = [d for d in decisions if d.protected]
    return remove, keep


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _fmt_date(mtime: float) -> str:
    if not mtime:
        return "?"
    return time.strftime("%Y-%m-%d", time.localtime(mtime))


def format_report(runs: list[RunInfo], sort: str = "size", limit: Optional[int] = None) -> str:
    if sort == "name":
        runs = sorted(runs, key=lambda r: r.name.lower())
    elif sort == "age":
        runs = sorted(runs, key=lambda r: r.mtime)
    else:  # size
        runs = sorted(runs, key=lambda r: r.size_bytes, reverse=True)

    shown = runs[:limit] if limit else runs

    headers = ["RUN", "SIZE MB", "MODIFIED", "AGE", "QA", "STATUS", "DOCS"]
    rows: list[list[str]] = []
    for r in shown:
        rows.append(
            [
                r.name,
                f"{r.size_mb:,.1f}",
                _fmt_date(r.mtime),
                f"{r.age_days:.0f}d",
                "yes" if r.has_qa else "no",
                r.qa_state,
                "yes" if r.referenced else "-",
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render(cells: list[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            # left-align name/status/date/qa/docs; right-align size/age
            if i in (1, 3):
                out.append(cell.rjust(widths[i]))
            else:
                out.append(cell.ljust(widths[i]))
        return "  ".join(out).rstrip()

    lines = [render(headers), "  ".join("-" * w for w in widths)]
    lines += [render(row) for row in rows]

    total_bytes = sum(r.size_bytes for r in runs)
    ref_count = sum(1 for r in runs if r.referenced)
    qa_count = sum(1 for r in runs if r.has_qa)
    footer = (
        f"\n{len(runs)} run(s), {human_mb(total_bytes):,.1f} MB total  |  "
        f"{qa_count} with qa.json  |  {ref_count} referenced in docs"
    )
    if limit and len(runs) > limit:
        footer = f"\n(showing top {limit} of {len(runs)} by {sort})" + footer
    return "\n".join(lines) + "\n" + footer


def format_plan(
    remove: list[Decision],
    keep: list[Decision],
    *,
    action: str,
    apply: bool,
    destination: Optional[Path] = None,
) -> str:
    remove = sorted(remove, key=lambda d: d.run.size_bytes, reverse=True)
    keep = sorted(keep, key=lambda d: d.run.name.lower())

    verb_future = "archive" if action == "archive" else "delete"
    verb_past = "Archived" if action == "archive" else "Deleted"
    lines: list[str] = []

    lines.append("KEEP (protected):")
    if keep:
        for d in keep:
            lines.append(
                f"  [keep]   {d.run.name}  ({d.run.size_mb:,.1f} MB)  "
                f"-> {', '.join(d.reasons)}"
            )
    else:
        lines.append("  (none)")

    lines.append("")
    header = f"{verb_past} :" if apply else f"WOULD {verb_future.upper()}:"
    lines.append(header)
    removed_bytes = 0
    if remove:
        for d in remove:
            removed_bytes += d.run.size_bytes
            lines.append(f"  [{verb_future}] {d.run.name}  ({d.run.size_mb:,.1f} MB)")
    else:
        lines.append("  (nothing selected)")

    lines.append("")
    dest = f" -> {destination}" if (action == "archive" and destination) else ""
    lines.append(
        f"{len(remove)} run(s) {'reclaimed' if apply else 'selected'}, "
        f"{human_mb(removed_bytes):,.1f} MB{dest}"
    )
    if not apply:
        lines.append(
            f"DRY RUN -- nothing was {('moved' if action == 'archive' else 'deleted')}. "
            f"Re-run with --apply to {verb_future}."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Destructive actions
# --------------------------------------------------------------------------- #
def _on_rm_error(func, path, exc_info):  # pragma: no cover - platform dependent
    """Best-effort: clear read-only bit and retry (Windows)."""
    try:
        os.chmod(path, 0o700)
        func(path)
    except OSError:
        pass


def do_delete(decisions: list[Decision]) -> list[str]:
    errors: list[str] = []
    for d in decisions:
        try:
            shutil.rmtree(d.run.path, onerror=_on_rm_error)
        except OSError as exc:
            errors.append(f"{d.run.name}: {exc}")
    return errors


def do_archive(decisions: list[Decision], destination: Path) -> list[str]:
    errors: list[str] = []
    destination.mkdir(parents=True, exist_ok=True)
    for d in decisions:
        target = destination / d.run.name
        if target.exists():
            errors.append(f"{d.run.name}: already exists in archive, skipped")
            continue
        try:
            shutil.move(str(d.run.path), str(target))
        except OSError as exc:
            errors.append(f"{d.run.name}: {exc}")
    return errors


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _resolve_root(args, parser: argparse.ArgumentParser) -> Path:
    if args.root:
        root = Path(args.root).resolve()
        if not (root / "run_pipeline.py").is_file():
            parser.error(
                f"--root {root} does not contain run_pipeline.py; refusing to run."
            )
        return root
    root = find_project_root()
    if root is None:
        parser.error(
            "could not find run_pipeline.py in the current directory or any parent; "
            "refusing to operate on an unknown tree. Pass --root <project> to override."
        )
    return root


def _add_selection_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--keep-days",
        type=float,
        default=14,
        help="protect runs modified within the last N days (default: 14).",
    )
    sub.add_argument(
        "--keep-pattern",
        action="append",
        default=None,
        metavar="GLOB",
        help="glob of run names to always protect (repeatable). "
        "Defaults to 'golden*' when omitted.",
    )
    grp = sub.add_mutually_exclusive_group()
    grp.add_argument(
        "--dry-run",
        dest="apply",
        action="store_false",
        help="show what would happen without changing anything (default).",
    )
    grp.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        help="actually delete/move the selected runs.",
    )
    sub.set_defaults(apply=False)


def _keep_patterns(args) -> list[str]:
    if args.keep_pattern is None:
        return list(DEFAULT_KEEP_PATTERNS)
    return list(args.keep_pattern)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prune_runs.py",
        description="Report on, prune, or archive ad-decompiler run directories.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="project root containing run_pipeline.py (default: search upward from cwd).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="print a table of all runs.")
    p_report.add_argument(
        "--sort", choices=("size", "name", "age"), default="size",
        help="sort order (default: size, largest first).",
    )
    p_report.add_argument(
        "--limit", type=int, default=None,
        help="only show the top N runs after sorting.",
    )

    p_prune = sub.add_parser("prune", help="delete stale runs (dry-run by default).")
    _add_selection_args(p_prune)

    p_archive = sub.add_parser(
        "archive", help="move stale runs into a directory (dry-run by default)."
    )
    p_archive.add_argument(
        "--to", required=True, metavar="DIR",
        help="destination directory to move runs into.",
    )
    _add_selection_args(p_archive)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    root = _resolve_root(args, parser)
    runs_dir = root / "runs"

    print(f"project root: {root}")
    print(f"runs dir:     {runs_dir}")
    if not runs_dir.is_dir():
        print(f"error: no runs/ directory under {root}", file=sys.stderr)
        return 2

    runs = scan_runs(runs_dir, project_root=root)
    print(f"scanned {len(runs)} run(s)\n")

    if args.command == "report":
        print(format_report(runs, sort=args.sort, limit=args.limit))
        return 0

    keep_patterns = _keep_patterns(args)
    remove, keep = plan(runs, keep_days=args.keep_days, keep_patterns=keep_patterns)

    print(f"keep-days: {args.keep_days:g}   keep-pattern(s): {keep_patterns}\n")

    if args.command == "prune":
        errors: list[str] = []
        if args.apply and remove:
            errors = do_delete(remove)
        print(format_plan(remove, keep, action="prune", apply=args.apply))
        if errors:
            print("\nERRORS:", file=sys.stderr)
            for e in errors:
                print(f"  {e}", file=sys.stderr)
            return 1
        return 0

    if args.command == "archive":
        destination = Path(args.to).resolve()
        errors = []
        if args.apply and remove:
            errors = do_archive(remove, destination)
        print(
            format_plan(
                remove, keep, action="archive", apply=args.apply, destination=destination
            )
        )
        if errors:
            print("\nERRORS:", file=sys.stderr)
            for e in errors:
                print(f"  {e}", file=sys.stderr)
            return 1
        return 0

    parser.error(f"unknown command {args.command!r}")
    return 2  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
