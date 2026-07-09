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
import os, shutil, json, time

DEFAULT_INBOX = os.environ.get("FIGMA_INBOX", os.path.expanduser("~/figma-inbox"))


def import_design(design_path: str, run_dir: str, cfg: dict | None = None) -> dict:
    cfg = cfg or {}
    mode = (cfg.get("figma") or {}).get("mode", "plugin")
    if mode == "clipboard":
        return _clipboard(design_path, run_dir, cfg)
    return _stage_for_plugin(design_path, run_dir, cfg)


def _stage_for_plugin(design_path, run_dir, cfg) -> dict:
    inbox = (cfg.get("figma") or {}).get("inbox", DEFAULT_INBOX)
    os.makedirs(inbox, exist_ok=True)
    # copy design.json + the whole assets/ dir so the plugin can resolve layer.src
    shutil.copyfile(design_path, os.path.join(inbox, "design.json"))
    assets = os.path.join(run_dir, "assets")
    if os.path.isdir(assets):
        dst = os.path.join(inbox, "assets")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(assets, dst)
    # a manifest the plugin polls; also records where the export should land
    manifest = {"design": "design.json", "assets": "assets",
                "export_to": os.path.abspath(os.path.join(run_dir, "figma_export.png")),
                "run_dir": os.path.abspath(run_dir), "staged_at": int(time.time())}
    with open(os.path.join(inbox, "inbox.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return {"ok": True, "mode": "plugin", "inbox": inbox,
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
