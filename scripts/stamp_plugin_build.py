#!/usr/bin/env python3
"""Stamp the Figma plugin with an automatic build label.

Build number = git commit count (monotonic per repo).
Falls back to a local counter when git is unavailable.

Writes figma-plugin/build-info.json and updates PLUGIN_BUILD constants in
code.js + ui.html.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "figma-plugin"
VERSION_FILE = PLUGIN / "VERSION"
COUNTER_FILE = PLUGIN / ".build-counter"
BUILD_INFO = PLUGIN / "build-info.json"
CODE_JS = PLUGIN / "code.js"
UI_HTML = PLUGIN / "ui.html"
MANIFEST = PLUGIN / "manifest.json"

MARKER_JS = re.compile(
    r"^const PLUGIN_BUILD = \{.*?\};$",
    re.MULTILINE | re.DOTALL,
)
MARKER_HTML = re.compile(
    r"^    const PLUGIN_BUILD = \{.*?\};$",
    re.MULTILINE | re.DOTALL,
)
EMBED_JS = re.compile(
    r"^const PLUGIN_BUILD = (\{.*?\});$",
    re.MULTILINE | re.DOTALL,
)
EMBED_HTML = re.compile(
    r"^    const PLUGIN_BUILD = (\{.*?\});$",
    re.MULTILINE | re.DOTALL,
)
# Stamp-managed paths must not affect the dirty flag — after git pull the stamp
# itself updates these files, which would otherwise keep dirty=true forever.
STAMP_MANAGED = (
    "figma-plugin/build-info.json",
    "figma-plugin/code.js",
    "figma-plugin/ui.html",
)


def _read_version() -> str:
    if VERSION_FILE.exists():
        line = VERSION_FILE.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        if line:
            return line
    return "0.0.0"


def _git_info() -> dict:
    try:
        count = subprocess.check_output(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = _repo_dirty_excluding_stamp()
        return {
            "build": int(count),
            "commit": commit,
            "dirty": dirty,
            "source": "git",
        }
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return {}


def _repo_dirty_excluding_stamp() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True
    excluded = set(STAMP_MANAGED)
    for line in out.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        if path not in excluded:
            return True
    return False


def _fallback_build() -> int:
    current = 0
    if COUNTER_FILE.exists():
        try:
            current = int(COUNTER_FILE.read_text(encoding="utf-8").strip() or "0")
        except ValueError:
            current = 0
    current += 1
    COUNTER_FILE.write_text(str(current) + "\n", encoding="utf-8")
    return current


def compute_build_info() -> dict:
    version = _read_version()
    git = _git_info()
    built_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    if git:
        build = git["build"]
        commit = git["commit"]
        dirty = git["dirty"]
        source = "git"
    else:
        build = _fallback_build()
        commit = "local"
        dirty = True
        source = "counter"
    label = f"v{version}+b{build}"
    if commit and commit != "local":
        label += f".{commit}"
    if dirty:
        label += "-dirty"
    return {
        "version": version,
        "build": build,
        "commit": commit,
        "dirty": dirty,
        "built_at": built_at,
        "label": label,
        "source": source,
    }


def _plugin_build_js(info: dict) -> str:
    payload = json.dumps(info, ensure_ascii=False, separators=(",", ":"))
    return f"const PLUGIN_BUILD = {payload};"


def _replace_or_insert(text: str, pattern: re.Pattern[str], replacement: str, insert_after: str) -> str:
    if pattern.search(text):
        return pattern.sub(replacement, text, count=1)
    idx = text.find(insert_after)
    if idx < 0:
        raise RuntimeError(f"insert anchor not found: {insert_after!r}")
    end = idx + len(insert_after)
    return text[:end] + "\n\n" + replacement + text[end:]


def _stable_key(info: dict) -> tuple:
    return (info.get("version"), info.get("build"), info.get("commit"), info.get("dirty"))


def _embedded_build_info(text: str, pattern: re.Pattern[str]) -> dict | None:
    match = pattern.search(text)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _embedded_matches_info(info: dict) -> bool:
    code_embed = _embedded_build_info(CODE_JS.read_text(encoding="utf-8"), EMBED_JS)
    html_embed = _embedded_build_info(UI_HTML.read_text(encoding="utf-8"), EMBED_HTML)
    if code_embed is None or html_embed is None:
        return False
    return _stable_key(code_embed) == _stable_key(info) and _stable_key(html_embed) == _stable_key(info)


def stamp_files(info: dict | None = None, *, force: bool = False) -> dict:
    info = info or compute_build_info()
    existing = None
    if BUILD_INFO.exists():
        try:
            existing = json.loads(BUILD_INFO.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            existing = None
    if (
        not force
        and isinstance(existing, dict)
        and _stable_key(existing) == _stable_key(info)
        and _embedded_matches_info(info)
    ):
        info["built_at"] = existing.get("built_at", info["built_at"])
        info["label"] = existing.get("label", info["label"])
        return info
    js_line = _plugin_build_js(info)
    BUILD_INFO.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    code = CODE_JS.read_text(encoding="utf-8")
    code = _replace_or_insert(
        code,
        MARKER_JS,
        js_line,
        "// It accepts the legacy flat design.json contract and scene-graph v2 documents.",
    )
    CODE_JS.write_text(code, encoding="utf-8")

    html = UI_HTML.read_text(encoding="utf-8")
    html = _replace_or_insert(
        html,
        MARKER_HTML,
        "    " + js_line,
        "const $ = function (id) { return document.getElementById(id); };",
    )
    UI_HTML.write_text(html, encoding="utf-8")

    # Figma's manifest validator only allows documented keys — keep build metadata in
    # build-info.json + PLUGIN_BUILD constants, not manifest.json.
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for key in ("version", "build", "commit", "built_at", "label", "icon"):
            manifest.pop(key, None)
        MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return info


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Stamp Figma plugin build metadata")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    info = stamp_files()
    if args.json:
        print(json.dumps(info, indent=2))
    elif not args.quiet:
        print(f"plugin build stamped: {info['label']} ({info['built_at']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
