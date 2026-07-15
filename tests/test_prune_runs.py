"""Tests for scripts/prune_runs.py using a synthetic temp runs/ tree."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prune_runs  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic tree helpers
# --------------------------------------------------------------------------- #
def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _set_age(path: Path, days: float) -> None:
    """Backdate mtime of a path (and its files) by *days* days."""
    when = time.time() - days * 86400
    for root, _dirs, names in os.walk(path):
        for name in names:
            os.utime(os.path.join(root, name), (when, when))
    os.utime(path, (when, when))


def make_run(runs: Path, name: str, *, size_kb: int = 4, ok=None, nested_ok=None,
             age_days: float = 0.0) -> Path:
    """Create a synthetic run directory.

    ok         -> write a root qa.json with this bool.
    nested_ok  -> iterable of bools, each a qa.json in a subfolder (suite layout).
    """
    rd = runs / name
    rd.mkdir(parents=True, exist_ok=True)
    _write(rd / "normalized.png", "P" * (size_kb * 1024))
    (rd / "assets").mkdir(exist_ok=True)
    _write(rd / "assets" / "layer_0.png", "L" * 512)
    if ok is not None:
        _write(rd / "qa.json", json.dumps({"ok": ok, "ssim": 0.5}))
    if nested_ok is not None:
        for i, val in enumerate(nested_ok):
            _write(rd / f"{i:03d}_img" / "qa.json", json.dumps({"ok": val}))
    if age_days:
        _set_age(rd, age_days)
    return rd


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A fake project root with run_pipeline.py, runs/, and docs/."""
    (tmp_path / "run_pipeline.py").write_text("# stub\n", encoding="utf-8")
    (tmp_path / "runs").mkdir()
    (tmp_path / "docs").mkdir()
    return tmp_path


# --------------------------------------------------------------------------- #
# find_project_root
# --------------------------------------------------------------------------- #
def test_find_project_root_walks_upward(project: Path):
    deep = project / "runs" / "some-run" / "assets"
    deep.mkdir(parents=True)
    assert prune_runs.find_project_root(deep) == project.resolve()


def test_find_project_root_returns_none_outside_project(tmp_path: Path):
    assert prune_runs.find_project_root(tmp_path) is None


# --------------------------------------------------------------------------- #
# scanning / qa status / docs references
# --------------------------------------------------------------------------- #
def test_scan_reports_size_qa_and_references(project: Path):
    runs = project / "runs"
    make_run(runs, "golden-final", ok=True, size_kb=8)
    make_run(runs, "crashfix-021", ok=False, size_kb=4)
    make_run(runs, "no-qa-run", size_kb=2)
    (project / "docs" / "EVIDENCE.md").write_text(
        "Benchmark evidence lives in runs/golden-final.\n", encoding="utf-8"
    )

    infos = {i.name: i for i in prune_runs.scan_runs(runs, project_root=project)}

    assert infos["golden-final"].has_qa is True
    assert infos["golden-final"].qa_state == "pass"
    assert infos["golden-final"].referenced is True
    assert infos["golden-final"].size_bytes > infos["crashfix-021"].size_bytes

    assert infos["crashfix-021"].qa_state == "fail"
    assert infos["crashfix-021"].referenced is False

    assert infos["no-qa-run"].has_qa is False
    assert infos["no-qa-run"].qa_state == "n/a"


def test_qa_status_suite_aggregation(project: Path):
    runs = project / "runs"
    make_run(runs, "all-pass", nested_ok=[True, True, True])
    make_run(runs, "all-fail", nested_ok=[False, False])
    make_run(runs, "mixed", nested_ok=[True, False, True])

    infos = {i.name: i for i in prune_runs.scan_runs(runs, project_root=project)}
    assert infos["all-pass"].qa_state == "pass (3/3)"
    assert infos["all-fail"].qa_state == "fail (0/2)"
    assert infos["mixed"].qa_state == "mixed (2/3)"


def test_reference_boundary_does_not_match_substring(project: Path):
    runs = project / "runs"
    make_run(runs, "inspo_test", ok=True)
    make_run(runs, "inspo_test2", ok=True)
    (project / "docs" / "HANDOFF.md").write_text(
        "Validate against runs/inspo_test (the whey pack case).\n", encoding="utf-8"
    )

    infos = {i.name: i for i in prune_runs.scan_runs(runs, project_root=project)}
    assert infos["inspo_test"].referenced is True
    # inspo_test2 must NOT be protected just because inspo_test is a prefix.
    assert infos["inspo_test2"].referenced is False


def test_corrupt_qa_json_is_flagged_not_fatal(project: Path):
    runs = project / "runs"
    rd = make_run(runs, "broken")
    (rd / "qa.json").write_text("{not valid json", encoding="utf-8")
    infos = {i.name: i for i in prune_runs.scan_runs(runs, project_root=project)}
    assert infos["broken"].has_qa is True
    assert "err" in infos["broken"].qa_state


# --------------------------------------------------------------------------- #
# selection logic
# --------------------------------------------------------------------------- #
def test_plan_protects_docs_pattern_and_recent(project: Path):
    runs = project / "runs"
    make_run(runs, "golden-v2", ok=True, age_days=90)        # pattern
    make_run(runs, "cited-run", ok=True, age_days=90)        # docs
    make_run(runs, "fresh-run", ok=False, age_days=1)        # recent
    make_run(runs, "crashfix-067", ok=False, age_days=90)    # stale -> remove
    make_run(runs, "fixtest-016", ok=False, age_days=45)     # stale -> remove
    (project / "docs" / "A.md").write_text("see runs/cited-run\n", encoding="utf-8")

    infos = prune_runs.scan_runs(runs, project_root=project)
    remove, keep = prune_runs.plan(infos, keep_days=14, keep_patterns=["golden*"])

    removed_names = {d.run.name for d in remove}
    kept_names = {d.run.name for d in keep}

    assert removed_names == {"crashfix-067", "fixtest-016"}
    assert kept_names == {"golden-v2", "cited-run", "fresh-run"}

    reasons = {d.run.name: d.reasons for d in keep}
    assert any("matches" in r for r in reasons["golden-v2"])
    assert "docs-referenced" in reasons["cited-run"]
    assert any("recent" in r for r in reasons["fresh-run"])


# --------------------------------------------------------------------------- #
# CLI: safety guard
# --------------------------------------------------------------------------- #
def test_cli_refuses_without_project_root(tmp_path: Path, capsys):
    with pytest.raises(SystemExit) as exc:
        prune_runs.main(["--root", str(tmp_path), "report"])
    assert exc.value.code != 0
    assert "run_pipeline.py" in capsys.readouterr().err


def test_cli_report_runs(project: Path, capsys):
    runs = project / "runs"
    make_run(runs, "golden-final", ok=True, size_kb=16)
    make_run(runs, "crashfix-021", ok=False, size_kb=2, age_days=60)

    rc = prune_runs.main(["--root", str(project), "report"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "golden-final" in out
    assert "crashfix-021" in out
    assert "SIZE MB" in out
    # golden-final is larger, should be listed first with size sort
    assert out.index("golden-final") < out.index("crashfix-021")


# --------------------------------------------------------------------------- #
# CLI: prune dry-run vs apply
# --------------------------------------------------------------------------- #
def test_prune_dry_run_default_deletes_nothing(project: Path, capsys):
    runs = project / "runs"
    make_run(runs, "crashfix-131", ok=False, age_days=90)

    rc = prune_runs.main(["--root", str(project), "prune", "--keep-days", "14"])
    out = capsys.readouterr().out

    assert rc == 0
    assert (runs / "crashfix-131").exists()  # nothing deleted
    assert "DRY RUN" in out
    assert "crashfix-131" in out


def test_prune_apply_deletes_only_unprotected(project: Path, capsys):
    runs = project / "runs"
    make_run(runs, "golden-final", ok=True, age_days=90)      # protected pattern
    make_run(runs, "cited-run", ok=True, age_days=90)         # protected docs
    make_run(runs, "crashfix-131", ok=False, age_days=90)     # removed
    make_run(runs, "fixtest-016d", ok=False, age_days=90)     # removed
    (project / "docs" / "R.md").write_text("runs/cited-run\n", encoding="utf-8")

    rc = prune_runs.main(["--root", str(project), "prune", "--keep-days", "14", "--apply"])
    out = capsys.readouterr().out

    assert rc == 0
    assert not (runs / "crashfix-131").exists()
    assert not (runs / "fixtest-016d").exists()
    assert (runs / "golden-final").exists()
    assert (runs / "cited-run").exists()
    assert "Deleted" in out


def test_prune_custom_keep_pattern_protects(project: Path):
    runs = project / "runs"
    make_run(runs, "benchmark-001", ok=True, age_days=90)
    make_run(runs, "crashfix-021", ok=False, age_days=90)

    rc = prune_runs.main(
        [
            "--root", str(project), "prune",
            "--keep-days", "14",
            "--keep-pattern", "benchmark*",
            "--apply",
        ]
    )
    assert rc == 0
    assert (runs / "benchmark-001").exists()      # protected by custom pattern
    assert not (runs / "crashfix-021").exists()   # golden* default not applied here


# --------------------------------------------------------------------------- #
# CLI: archive
# --------------------------------------------------------------------------- #
def test_archive_moves_instead_of_deleting(project: Path, tmp_path: Path, capsys):
    runs = project / "runs"
    make_run(runs, "crashfix-021", ok=False, age_days=90)
    make_run(runs, "golden-final", ok=True, age_days=90)
    archive_dir = tmp_path / "run_archive"

    rc = prune_runs.main(
        [
            "--root", str(project), "archive",
            "--to", str(archive_dir),
            "--keep-days", "14",
            "--apply",
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert not (runs / "crashfix-021").exists()             # moved out
    assert (archive_dir / "crashfix-021").exists()          # moved in
    assert (archive_dir / "crashfix-021" / "qa.json").exists()
    assert (runs / "golden-final").exists()                 # protected, stays
    assert "Archived" in out


def test_archive_dry_run_moves_nothing(project: Path, tmp_path: Path):
    runs = project / "runs"
    make_run(runs, "crashfix-021", ok=False, age_days=90)
    archive_dir = tmp_path / "run_archive"

    rc = prune_runs.main(
        ["--root", str(project), "archive", "--to", str(archive_dir), "--keep-days", "14"]
    )
    assert rc == 0
    assert (runs / "crashfix-021").exists()
    assert not archive_dir.exists()
