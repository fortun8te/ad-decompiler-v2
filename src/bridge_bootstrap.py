"""Bootstrap config + inbox before the Figma bridge starts."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - doctor/setup should have installed pyyaml
    yaml = None  # type: ignore


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_config_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "config.yaml"


def default_example_path(root: Path | None = None) -> Path:
    return (root or repo_root()) / "config.example.yaml"


def default_inbox() -> str:
    return os.path.expanduser(os.environ.get("FIGMA_INBOX", "~/figma-inbox"))


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required — pip install pyyaml")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("PyYAML is required — pip install pyyaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, default_flow_style=False)


def prepare(
    config_path: str | os.PathLike[str] | None = None,
    inbox: str | os.PathLike[str] | None = None,
    *,
    root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Ensure config.yaml + figma inbox exist. Returns a small status dict."""
    root_path = Path(root).resolve() if root else repo_root()
    cfg_path = Path(config_path) if config_path else default_config_path(root_path)
    if not cfg_path.is_absolute():
        cfg_path = (root_path / cfg_path).resolve()
    else:
        cfg_path = cfg_path.resolve()
    example_path = default_example_path(root_path)
    inbox_path = os.path.expanduser(str(inbox or default_inbox()))

    created_config = False
    patched_figma = False
    if not cfg_path.exists() and example_path.exists():
        shutil.copyfile(example_path, cfg_path)
        created_config = True

    if cfg_path.exists():
        cfg = _load_yaml(cfg_path)
        figma = cfg.setdefault("figma", {})
        if not isinstance(figma, dict):
            figma = {}
            cfg["figma"] = figma
        if not figma.get("enabled"):
            figma["enabled"] = True
            figma.setdefault("mode", "plugin")
            figma.setdefault("inbox", inbox_path.replace("\\", "/"))
            figma.setdefault("bridge_port", 8790)
            _save_yaml(cfg_path, cfg)
            patched_figma = True
    else:
        cfg = {}

    os.makedirs(inbox_path, exist_ok=True)
    text_cfg = cfg.get("text_analysis") if isinstance(cfg.get("text_analysis"), dict) else {}
    font_raw = text_cfg.get("font_matching", False)
    if isinstance(font_raw, dict):
        font_enabled = bool(font_raw.get("enabled", True))
    elif isinstance(font_raw, bool):
        font_enabled = font_raw
    else:
        font_enabled = False

    return {
        "root": str(root_path),
        "config_path": str(cfg_path),
        "config_exists": cfg_path.exists(),
        "created_config": created_config,
        "patched_figma": patched_figma,
        "inbox": inbox_path,
        "font_matching_enabled": font_enabled,
    }


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Bootstrap ad-decompiler bridge files")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--inbox", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    status = prepare(config_path=args.config, inbox=args.inbox)
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        if status["created_config"]:
            print(f"Created {status['config_path']}")
        if status["patched_figma"]:
            print(f"Enabled figma plugin staging in {status['config_path']}")
        print(f"Inbox: {status['inbox']}")
        if status["config_exists"] and not status["font_matching_enabled"]:
            print("Note: text_analysis.font_matching is disabled — edit config.yaml for real font recognition.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
