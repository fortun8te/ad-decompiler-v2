"""figma_import.py — bridge design.json into Figma and export a screenshot back.

Figma has no fully-headless "create arbitrary nodes" API (REST is read-only for node
creation). The reliable path is the companion plugin in figma-plugin/, which reads a
design.json + assets from a shared inbox folder and builds real, editable nodes.

Two modes (cfg.figma.mode):
  'plugin'    — stage design.json + assets into FIGMA_INBOX; the plugin's "Import latest"
                builds nodes and writes figma_export.png back to the run dir (one click in
                Figma desktop). This is the recommended, highest-fidelity path.
  'clipboard' — reuse the Mac harness's proven kiwi clipboard encoder
                (studio/src/components/design/figmaClipboard.ts, 80/80 roundtrip) via a
                small Node bridge to produce a paste payload. ⌘V into Figma. No plugin needed.

export_screenshot() collects the PNG the plugin exported (plugin mode), or is a no-op the
agent flags for a manual export (clipboard mode).
"""
from __future__ import annotations
import hashlib, os, shutil, json, time, tempfile

DEFAULT_INBOX = os.environ.get("FIGMA_INBOX", os.path.expanduser("~/figma-inbox"))


def import_design(design_path: str, run_dir: str, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    mode = (cfg.get("figma") or {}).get("mode", "plugin")
    try:
        if mode == "clipboard":
            return _clipboard(design_path, run_dir, cfg)
        if mode != "plugin":
            return {"ok": False, "mode": mode, "error": f"unsupported Figma mode: {mode}"}
        return _stage_for_plugin(design_path, run_dir, cfg)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"ok": False, "mode": mode, "error": str(exc),
                "exception": type(exc).__name__}


def _stage_for_plugin(design_path, run_dir, cfg) -> dict:
    inbox = (cfg.get("figma") or {}).get("inbox", DEFAULT_INBOX)
    os.makedirs(inbox, exist_ok=True)
    if not os.path.isfile(design_path):
        raise FileNotFoundError(f"design.json not found: {design_path}")
    with open(design_path, encoding="utf-8") as fh:
        design = json.load(fh)
    if not isinstance(design, dict) or not isinstance(design.get("layers", []), list):
        raise ValueError("design.json must be an object with a layers list")
    doc_id = "".join(c if c.isalnum() or c in "-_" else "-"
                     for c in str(design.get("id") or os.path.basename(run_dir)))[:80] or "run"
    staged_root = os.path.join(inbox, "runs", doc_id)
    runs_root = os.path.join(inbox, "runs")
    os.makedirs(runs_root, exist_ok=True)
    temp_root = tempfile.mkdtemp(prefix=f".{doc_id}-", dir=runs_root)
    shutil.copyfile(design_path, os.path.join(temp_root, "design.json"))
    assets = os.path.join(run_dir, "assets")
    if os.path.isdir(assets):
        shutil.copytree(assets, os.path.join(temp_root, "assets"))
    for filename in ("preview.png", "design_preflight.json", "qa.json"):
        source = os.path.join(run_dir, filename)
        if os.path.exists(source):
            shutil.copyfile(source, os.path.join(temp_root, filename))
    shutil.rmtree(staged_root, ignore_errors=True)
    os.replace(temp_root, staged_root)

    files = []
    for root, _, names in os.walk(staged_root):
        for filename in sorted(names):
            path = os.path.join(root, filename)
            rel = os.path.relpath(path, staged_root).replace(os.sep, "/")
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
            files.append({"path": rel, "sha256": digest, "bytes": os.path.getsize(path)})
    preflight = {}
    preflight_path = os.path.join(run_dir, "design_preflight.json")
    if os.path.exists(preflight_path):
        with open(preflight_path, encoding="utf-8") as fh:
            preflight = json.load(fh)
    manifest = {
        "schema_version": design.get("schema_version", design.get("schemaVersion", 1)),
        "doc_id": doc_id,
        "design": "design.json",
        "staged_dir": os.path.relpath(staged_root, inbox).replace(os.sep, "/"),
        "assets": "assets",
        "files": files,
        "preview": "preview.png" if os.path.exists(os.path.join(staged_root, "preview.png")) else None,
        "export_to": os.path.abspath(os.path.join(run_dir, "figma_export.png")),
        "run_dir": os.path.abspath(run_dir),
        "staged_at": int(time.time()),
        "summary": {
            "name": design.get("name"),
            "canvas": design.get("canvas"),
            "layers": (design.get("meta") or {}).get("layer_count", len(design.get("layers") or [])),
            "editable_ratio": (design.get("meta") or {}).get("editable_ratio"),
            "warnings": preflight.get("warnings") or (design.get("meta") or {}).get("warnings") or [],
        },
    }
    manifest_path = os.path.join(inbox, "inbox.json")
    temp_manifest = manifest_path + ".tmp"
    with open(temp_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(temp_manifest, manifest_path)
    return {"ok": True, "mode": "plugin", "inbox": inbox,
            "doc_id": doc_id, "files": len(files),
            "action": "In Figma desktop: run the ad-decompiler plugin → Import latest."}


def _clipboard(design_path, run_dir, cfg) -> dict:
    """Convert design.json → Figma kiwi clipboard payload via the Node bridge in the Mac
    harness. Requires node + the studio repo path (cfg.figma.studio_path)."""
    import subprocess
    studio = (cfg.get("figma") or {}).get("studio_path")
    bridge = os.path.join(os.path.dirname(__file__), "..", "figma-plugin", "kiwi_bridge.mjs")
    if not studio or not os.path.exists(bridge):
        return {"ok": False, "mode": "clipboard",
                "error": "set cfg.figma.studio_path to the NEUEGEN/studio repo and ensure figma-plugin/kiwi_bridge.mjs exists"}
    out = os.path.join(run_dir, "figma_clipboard.bin")
    try:
        subprocess.run(["node", bridge, design_path, out, studio], check=True, timeout=120)
        return {"ok": True, "mode": "clipboard", "payload": out,
                "action": "Load the payload into the clipboard helper, then ⌘V/Ctrl+V into Figma."}
    except Exception as e:
        return {"ok": False, "mode": "clipboard", "error": str(e)}


def export_screenshot(run_dir: str, cfg: dict | None = None, wait_s: int = 0) -> dict:
    """Return path to figma_export.png once the plugin has written it. In plugin mode this may
    poll briefly; the pipeline can also run --resume after the manual import click."""
    target = os.path.join(run_dir, "figma_export.png")
    deadline = time.time() + wait_s
    while True:
        if os.path.exists(target):
            return {"ok": True, "path": target}
        if time.time() >= deadline:
            return {"ok": False, "path": target,
                    "note": "figma_export.png not found yet — run the plugin's Import+Export, then re-run QA with --resume"}
        time.sleep(1)
