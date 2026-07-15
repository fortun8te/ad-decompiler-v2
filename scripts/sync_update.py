#!/usr/bin/env python3
"""Safely compare this checkout with a remote branch and fast-forward when clean.

This intentionally never stashes, resets, merges, or pulls a dirty/diverged worktree. It is
safe to call from a Windows launcher or a macOS launchd job: the JSON/terminal result always
says whether it checked, updated, or deliberately left local work alone.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable


Runner = Callable[[Path, str], tuple[int, str, str]]


def _runner(repo: Path, *args: str) -> tuple[int, str, str]:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=45,
        check=False,
    )
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _run_or_raise(run: Callable[..., tuple[int, str, str]], repo: Path, *args: str) -> str:
    code, stdout, stderr = run(repo, *args)
    if code:
        detail = stderr or stdout or f"exit {code}"
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return stdout


def _base_status(repo: Path, action: str, message: str, **extra) -> dict:
    return {"ok": action not in {"error", "not_repo"}, "repo": str(repo), "action": action,
            "message": message, **extra}


def sync(repo: str | Path, *, update: bool = False, remote: str = "origin", branch: str = "main",
         runner: Callable[..., tuple[int, str, str]] = _runner) -> dict:
    """Return remote/local revision status and only fast-forward a clean checkout.

    ``runner`` is injectable so tests exercise all safety decisions without invoking git or a
    network. A dirty tree is detected before fetch/pull and is never changed by this function.
    """
    repo = Path(repo).expanduser().resolve()
    try:
        inside = _run_or_raise(runner, repo, "rev-parse", "--is-inside-work-tree")
    except (RuntimeError, OSError) as exc:
        return _base_status(repo, "not_repo", str(exc))
    if inside.strip().lower() != "true":
        return _base_status(repo, "not_repo", "path is not inside a git worktree")

    try:
        dirty_entries = _run_or_raise(runner, repo, "status", "--porcelain=v1")
        local_revision = _run_or_raise(runner, repo, "rev-parse", "--short", "HEAD")
    except (RuntimeError, OSError) as exc:
        return _base_status(repo, "error", str(exc))
    dirty = bool(dirty_entries)
    if dirty:
        return _base_status(
            repo, "dirty", "local changes detected; skipped fetch and pull to preserve them",
            local_revision=local_revision, dirty=True, changed_paths=dirty_entries.splitlines(),
        )

    try:
        _run_or_raise(runner, repo, "fetch", "--quiet", remote, branch)
        remote_ref = f"{remote}/{branch}"
        remote_revision = _run_or_raise(runner, repo, "rev-parse", "--short", remote_ref)
        counts = _run_or_raise(runner, repo, "rev-list", "--left-right", "--count", f"HEAD...{remote_ref}")
        ahead, behind = (int(part) for part in counts.split())
    except (RuntimeError, OSError, ValueError) as exc:
        return _base_status(repo, "error", str(exc), local_revision=local_revision, dirty=False)

    status = {
        "local_revision": local_revision,
        "remote_revision": remote_revision,
        "remote": remote,
        "branch": branch,
        "ahead": ahead,
        "behind": behind,
        "dirty": False,
    }
    if behind == 0 and ahead == 0:
        return _base_status(repo, "current", "already matches remote", **status)
    if not update:
        return _base_status(repo, "update_available", "remote revision differs; rerun with --update", **status)
    if ahead:
        return _base_status(
            repo, "diverged", "local revision is ahead/diverged; skipped pull to preserve local history", **status,
        )

    try:
        _run_or_raise(runner, repo, "pull", "--ff-only", remote, branch)
        new_revision = _run_or_raise(runner, repo, "rev-parse", "--short", "HEAD")
    except (RuntimeError, OSError) as exc:
        return _base_status(repo, "error", str(exc), **status)
    return _base_status(
        repo, "updated", f"fast-forwarded {local_revision} -> {new_revision}",
        **{**status, "local_revision": new_revision},
    )


def _notify(message: str) -> None:
    """Best-effort desktop notification; terminal status is always printed regardless."""
    system = platform.system()
    try:
        if system == "Darwin":
            escaped = message.replace("\\", "\\\\").replace('"', '\\"')
            subprocess.run(
                ["osascript", "-e", f'display notification "{escaped}" with title "Ad Decompiler update"'],
                check=False, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        elif system == "Windows":
            # msg.exe is built into normal interactive Windows installs. Do not wait for it:
            # scheduled tasks, service sessions, and systems without msg.exe must never delay
            # or change the safe-update result.
            subprocess.Popen(
                ["msg.exe", "*", f"Ad Decompiler update: {message}"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
    except (OSError, subprocess.SubprocessError):
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Safely check/pull one clean git checkout")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--update", action="store_true", help="fast-forward only when the worktree is clean")
    parser.add_argument("--notify", action="store_true", help="show a best-effort desktop notification")
    parser.add_argument("--json", action="store_true", help="emit the machine-readable status only")
    args = parser.parse_args(argv)

    result = sync(args.repo, update=args.update, remote=args.remote, branch=args.branch)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"[update] {result['action'].upper()}: {result['message']}")
        if result.get("local_revision"):
            print(f"[update] local={result['local_revision']} remote={result.get('remote_revision', 'unknown')}")
    if args.notify:
        _notify(f"{result['action']}: {result['message']}")
    return 0 if result.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
