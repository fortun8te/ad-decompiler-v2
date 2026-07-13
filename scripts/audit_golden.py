#!/usr/bin/env python3
"""Critical audit of golden benchmark runs — structural + visual evidence."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
IDS = ("009", "041", "042", "050", "052")
SUITES = (
    "golden-optimized-check",
    "golden-final",
    "golden-v4",
    "golden-v3",
)


def walk_layers(layers: list, out: dict) -> None:
    for ly in layers or []:
        t = str(ly.get("type") or ly.get("layer_type") or "unknown")
        out[t] = out.get(t, 0) + 1
        walk_layers(ly.get("children") or ly.get("layers") or [], out)


def audit_run(run_dir: Path) -> dict | None:
    qa_path = run_dir / "qa.json"
    if not qa_path.exists():
        return None
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    struct = qa.get("structural") or {}
    bg = struct.get("background") or {}
    preflight = {}
    pre = run_dir / "design_preflight.json"
    if pre.exists():
        preflight = json.loads(pre.read_text(encoding="utf-8"))
    layer_types: dict[str, int] = {}
    design_path = run_dir / "design.json"
    if design_path.exists():
        design = json.loads(design_path.read_text(encoding="utf-8"))
        walk_layers(design.get("layers") or [], layer_types)
    recon = {}
    recon_path = run_dir / "reconstruction.json"
    if recon_path.exists():
        recon = json.loads(recon_path.read_text(encoding="utf-8"))
    return {
        "ok": bool(qa.get("ok")),
        "ssim": qa.get("ssim"),
        "edge_f1": qa.get("edge_f1"),
        "text_recall": qa.get("text_recall"),
        "composite": qa.get("composite"),
        "hard_fails": [f.get("rule") for f in (qa.get("hard_fails") or [])],
        "repairs": len(qa.get("repairs") or []),
        "editable_ratio": struct.get("editable_ratio"),
        "element_recall": struct.get("element_recall"),
        "duplicates": struct.get("duplicate_ownership") or [],
        "bg_outside_changed": bg.get("outside_changed_ratio"),
        "bg_changed": bg.get("changed_ratio"),
        "bg_mean_change": bg.get("mean_change"),
        "figma_report_ok": (struct.get("figma_report") or {}).get("ok"),
        "preflight_warnings": len(preflight.get("warnings") or []),
        "preflight_errors": len(preflight.get("errors") or []),
        "layer_types": layer_types,
        "inpaint_backend": (recon.get("stats") or {}).get("inpaint", {}).get("backend"),
        "vector_fallbacks": (recon.get("stats") or {}).get("vector_fallback_count"),
    }


def find_run(suite: str, ad_id: str) -> Path | None:
    base = RUNS / suite
    if not base.exists():
        return None
    matches = sorted(base.glob(f"{ad_id}_*"))
    if not matches:
        return None
    return matches[-1]


def main() -> int:
    suites = sys.argv[1:] or list(SUITES)
    blockers: list[str] = []
    print("CRITICAL GOLDEN AUDIT")
    print("=" * 72)
    for suite in suites:
        print(f"\n[{suite}]")
        for ad_id in IDS:
            run_dir = find_run(suite, ad_id)
            if not run_dir:
                print(f"  {ad_id}  MISSING")
                blockers.append(f"{suite}/{ad_id}: missing run")
                continue
            a = audit_run(run_dir)
            if not a:
                print(f"  {ad_id}  NO qa.json")
                blockers.append(f"{suite}/{ad_id}: no qa.json")
                continue
            tag = "PASS" if a["ok"] else "FAIL"
            print(
                f"  {ad_id}  {tag}  ssim={a['ssim']}  edge={a['edge_f1']}  "
                f"text={a['text_recall']}  composite={a['composite']}  "
                f"editable={a['editable_ratio']}  bg_out={a['bg_outside_changed']}  "
                f"repairs={a['repairs']}  hard={a['hard_fails']}  "
                f"figma={a['figma_report_ok']}  types={a['layer_types']}"
            )
            if not a["ok"]:
                blockers.append(f"{suite}/{ad_id}: qa.ok=false ({a['hard_fails']})")
            if a["hard_fails"]:
                blockers.append(f"{suite}/{ad_id}: hard_fails={a['hard_fails']}")
            if a["repairs"] and a["ok"]:
                blockers.append(f"{suite}/{ad_id}: qa pass but {a['repairs']} repairs remain")
            if a["figma_report_ok"] is None:
                blockers.append(f"{suite}/{ad_id}: no confirmed Figma plugin round-trip")
            if a["bg_outside_changed"] not in (0, 0.0, None) and a["bg_outside_changed"] > 0.001:
                blockers.append(f"{suite}/{ad_id}: background leakage outside mask")
            if a["editable_ratio"] is not None and a["editable_ratio"] < 0.5:
                blockers.append(f"{suite}/{ad_id}: low editable ratio {a['editable_ratio']}")

    print("\n" + "=" * 72)
    if blockers:
        print(f"NOT READY TO PUSH — {len(blockers)} blocker(s):")
        for b in blockers[:40]:
            print(f"  - {b}")
        if len(blockers) > 40:
            print(f"  ... and {len(blockers) - 40} more")
        return 1
    print("All checks passed (automated gate only — still verify visuals manually).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
