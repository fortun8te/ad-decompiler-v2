#!/usr/bin/env python
"""
Evaluation spike: assess the Lens font-recognition model (github.com/mixfont/lens)
against the ad-decompiler pipeline's current local-font shape-matcher.

INTEL-ONLY. This script does NOT touch src/ and does NOT integrate anything. It:
  1. Loads the bundled Lens ResNet18 model (work/lens/model/font_classifier.pt) on CPU.
  2. Reads OCR geometry from the golden runs (runs/golden-optimized-check/*/ocr.json).
  3. Crops text lines (and the largest word per line) from normalized.png.
  4. Runs Lens top-K font classification on each crop, bypassing Lens's own
     Tesseract OCR (we already have word/line boxes).
  5. Compares Lens predictions vs. the font our pipeline chose (from ocr.json styles)
     vs. hand-labeled ground-truth typeface CLASS (serif / sans-serif / ...).
  6. Emits a console table + JSON + a Markdown table (for docs/FONT-MATCHER-EVAL.md),
     plus latency stats.

Run from repo root:
    .venv\\Scripts\\python.exe scripts\\eval_lens_fonts.py

CPU is forced (CUDA hidden + pick_device monkeypatched) to avoid GPU contention
with the rest of the pipeline / other agents.
"""

from __future__ import annotations

# --- Force CPU BEFORE torch initializes CUDA -------------------------------
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LENS_DIR = REPO_ROOT / "work" / "lens"
MODEL_PATH = LENS_DIR / "model" / "font_classifier.pt"
METADATA_PATH = LENS_DIR / "model" / "font_metadata.json"
DEFAULT_RUNS_DIR = REPO_ROOT / "runs" / "golden-optimized-check"
DEFAULT_OUT_DIR = LENS_DIR / "eval_out"

# Make the vendored Lens package importable.
sys.path.insert(0, str(LENS_DIR))


# --- Hand-labeled ground truth (eyeballed from original.png) ----------------
# See docs/FONT-MATCHER-EVAL.md for the assumptions behind these labels.
GROUND_TRUTH: dict[str, dict] = {
    "009": {
        "source": (
            "X/Twitter post (Dutch). The entire UI chrome and tweet body are set in "
            "Chirp, X's proprietary grotesque SANS-SERIF (Helvetica/Arial/Inter-like). "
            "There is no serif, script, or monospace anywhere in this image."
        ),
        "default_class": "sans-serif",
        "lines": {
            "L0": ("sans-serif", '"Post" nav header (bold Chirp)'),
            "L2": ("sans-serif", "tweet body headline line"),
            "L3": ("sans-serif", "tweet body paragraph"),
            "L4": ("sans-serif", "tweet body -- pipeline MISMATCHED to Gabriola (swash script)"),
            "L5": ("sans-serif", "tweet body paragraph"),
            "L7": ("sans-serif", "tweet body paragraph"),
            "L8": ("sans-serif", "timestamp + view-count row"),
            "L14": ("sans-serif", "tweet body -- pipeline MISMATCHED to Cascadia Code (mono)"),
        },
    },
    "052": {
        "source": (
            "Hair-product ad. The headline and sub-headline are a high-contrast "
            "transitional/Didone SERIF; the 'Before'/'After' pill chips are a bold "
            "grotesque SANS."
        ),
        "default_class": "serif",
        "lines": {
            "L0": ("serif", "headline line 1 (pipeline OK: Cambria=serif)"),
            "L1": ("serif", "headline line 2 (pipeline OK: Georgia=serif)"),
            "L2": ("sans-serif", '"Before" chip'),
            "L3": ("sans-serif", '"After" chip'),
            "L8": ("serif", "sub-headline line 1"),
            "L9": ("serif", "sub-headline line 2"),
        },
    },
}

# Class of the local Windows fonts the current pipeline selects, so we can score
# "did the current matcher at least get the right TYPE of font?".
WINFONT_CLASS = {
    "arial": "sans-serif", "calibri": "sans-serif", "candara": "sans-serif",
    "cambria": "serif", "georgia": "serif", "constantia": "serif",
    "bahnschrift": "sans-serif", "leelawadee ui": "sans-serif", "gadugi": "sans-serif",
    "ebrima": "sans-serif", "franklin gothic medium": "sans-serif",
    "consolas": "monospace", "cascadia code": "monospace", "cascadia mono": "monospace",
    "courier new": "monospace",
    "gabriola": "handwriting", "ink free": "handwriting", "comic sans ms": "handwriting",
    "dodo variable": "display", "javanese text": "display", "segoe ui": "sans-serif",
}


def load_family_categories(metadata_path: Path) -> dict[str, str]:
    """Family name -> Google Fonts category (SANS_SERIF/SERIF/DISPLAY/...)."""
    cats: dict[str, str] = {}
    if not metadata_path.exists():
        return cats
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    fonts = data.get("fonts", {})
    cat_re = re.compile(r'category:\s*"([^"]+)"')
    for name, entry in fonts.items():
        pb = entry.get("metadata_pb", "") if isinstance(entry, dict) else ""
        m = cat_re.search(pb) if isinstance(pb, str) else None
        cats[name] = (m.group(1) if m else "UNKNOWN")
    return cats


# Google Fonts category -> our coarse class buckets used for "class match".
GF_CAT_TO_CLASS = {
    "SANS_SERIF": "sans-serif",
    "SERIF": "serif",
    "DISPLAY": "display",
    "HANDWRITING": "handwriting",
    "MONOSPACE": "monospace",
    "UNKNOWN": "unknown",
}


def gf_class(family: str, cats: dict[str, str]) -> str:
    return GF_CAT_TO_CLASS.get(cats.get(family, "UNKNOWN"), "unknown")


def find_run_dir(runs_dir: Path, prefix: str) -> Path | None:
    for p in sorted(runs_dir.glob(f"{prefix}_*")):
        if p.is_dir() and (p / "ocr.json").exists():
            return p
    return None


def clamp_box(x, y, w, h, iw, ih, pad_frac):
    pad = max(2.0, h * pad_frac)
    x0 = max(0, int(round(x - pad)))
    y0 = max(0, int(round(y - pad)))
    x1 = min(iw, int(round(x + w + pad)))
    y1 = min(ih, int(round(y + h + pad)))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def largest_word(line: dict):
    best = None
    best_area = 0.0
    for wd in line.get("words") or []:
        txt = (wd.get("text") or "").strip()
        b = wd.get("box") or {}
        area = float(b.get("w", 0)) * float(b.get("h", 0))
        if len(txt) >= 2 and area > best_area:
            best_area, best = area, wd
    return best


def fmt_preds(preds, cats):
    parts = []
    for p in preds:
        parts.append(f"{p['name']} {p['score']:.2f} [{gf_class(p['name'], cats)}]")
    return " | ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate Lens font recognition on golden runs.")
    ap.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    ap.add_argument("--model-path", type=Path, default=MODEL_PATH)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--pad-frac", type=float, default=0.15)
    ap.add_argument("--save-crops", action="store_true", help="Write each crop PNG to out-dir/crops")
    args = ap.parse_args()

    from PIL import Image  # noqa: E402
    import lens_inference as L  # noqa: E402

    # Hard-force CPU regardless of what torch discovers.
    import torch
    L.pick_device = lambda: torch.device("cpu")

    cats = load_family_categories(METADATA_PATH)
    print(f"[setup] Lens families with categories: {len(cats)}")

    t_load = time.time()
    bundle = L.get_model_bundle(args.model_path)
    load_s = time.time() - t_load
    print(f"[setup] model loaded on {bundle.device} in {load_s:.2f}s "
          f"({len(bundle.idx_to_class)} classes, input {bundle.image_height}x{bundle.image_width})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = args.out_dir / "crops"
    if args.save_crops:
        crops_dir.mkdir(exist_ok=True)

    rows: list[dict] = []
    latencies: list[float] = []

    for prefix, gt in GROUND_TRUTH.items():
        run_dir = find_run_dir(args.runs_dir, prefix)
        if run_dir is None:
            print(f"[warn] no run dir for prefix {prefix}")
            continue
        ocr = json.loads((run_dir / "ocr.json").read_text(encoding="utf-8"))
        img_path = run_dir / "normalized.png"
        if not img_path.exists():
            img_path = run_dir / "original.png"
        image = Image.open(img_path).convert("RGB")
        iw, ih = image.size
        lines_by_id = {ln["id"]: ln for ln in ocr.get("lines", [])}

        for line_id, (expected_class, label) in gt["lines"].items():
            line = lines_by_id.get(line_id)
            if line is None:
                print(f"[warn] {prefix}:{line_id} not found in ocr.json")
                continue
            text = (line.get("text") or "").strip()
            style = line.get("style") or {}
            our_font = style.get("fontFamily") or "?"
            our_class = WINFONT_CLASS.get(our_font.lower(), "unknown")

            # --- full-line crop and largest-word crop ---
            variants = []
            b = line.get("box") or {}
            box = clamp_box(b.get("x", 0), b.get("y", 0), b.get("w", 0), b.get("h", 0),
                            iw, ih, args.pad_frac)
            if box:
                variants.append(("line", box, text))
            wd = largest_word(line)
            if wd:
                wb = wd.get("box") or {}
                wbox = clamp_box(wb.get("x", 0), wb.get("y", 0), wb.get("w", 0), wb.get("h", 0),
                                 iw, ih, args.pad_frac)
                if wbox:
                    variants.append(("word", wbox, (wd.get("text") or "").strip()))

            for kind, box, crop_text in variants:
                crop = image.crop(box)
                t0 = time.time()
                preds = L.run_model(crop, bundle, top_k=args.top_k)
                dt_ms = (time.time() - t0) * 1000.0
                latencies.append(dt_ms)
                top1 = preds[0]["name"] if preds else "?"
                lens_class = gf_class(top1, cats)
                # class match: does Lens top-1 category equal expected class?
                lens_ok = (lens_class == expected_class)
                our_ok = (our_class == expected_class)
                if args.save_crops:
                    crop.save(crops_dir / f"{prefix}_{line_id}_{kind}.png")
                rows.append({
                    "run": prefix, "line": line_id, "kind": kind,
                    "text": crop_text[:48], "label": label,
                    "expected_class": expected_class,
                    "our_font": our_font, "our_class": our_class, "our_class_ok": our_ok,
                    "lens_top1": top1, "lens_top1_class": lens_class,
                    "lens_class_ok": lens_ok,
                    "lens_top5": [{"name": p["name"], "score": p["score"],
                                   "class": gf_class(p["name"], cats)} for p in preds],
                    "latency_ms": round(dt_ms, 1),
                })

    # ---- Console report ----
    print("\n" + "=" * 118)
    print("LENS FONT-RECOGNITION EVAL  (crop -> our pipeline font vs. Lens top-5)")
    print("=" * 118)
    for r in rows:
        flag_lens = "OK " if r["lens_class_ok"] else "XX "
        flag_our = "OK " if r["our_class_ok"] else "XX "
        print(f"\n[{r['run']}:{r['line']}/{r['kind']:4}] {r['text']!r}  ({r['label']})")
        print(f"   ground-truth class : {r['expected_class']}")
        print(f"   OUR pipeline       : {flag_our}{r['our_font']} [{r['our_class']}]")
        print(f"   LENS top-1         : {flag_lens}{r['lens_top1']} [{r['lens_top1_class']}]  ({r['latency_ms']} ms)")
        print(f"   LENS top-5         : {fmt_preds([{'name': p['name'], 'score': p['score']} for p in r['lens_top5']], cats)}")

    # ---- Aggregate accuracy (class-level) ----
    line_rows = [r for r in rows if r["kind"] == "line"]
    word_rows = [r for r in rows if r["kind"] == "word"]

    def acc(rs, key):
        if not rs:
            return (0, 0, 0.0)
        n = sum(1 for r in rs if r[key])
        return (n, len(rs), 100.0 * n / len(rs))

    lens_line = acc(line_rows, "lens_class_ok")
    lens_word = acc(word_rows, "lens_class_ok")
    our_line = acc(line_rows, "our_class_ok")

    print("\n" + "=" * 118)
    print("CLASS-LEVEL ACCURACY (did the matcher pick the right typeface CLASS: serif/sans/...)")
    print(f"  Lens (line crops) : {lens_line[0]}/{lens_line[1]} = {lens_line[2]:.0f}%")
    print(f"  Lens (word crops) : {lens_word[0]}/{lens_word[1]} = {lens_word[2]:.0f}%")
    print(f"  Our  (line crops) : {our_line[0]}/{our_line[1]} = {our_line[2]:.0f}%")
    if latencies:
        print(f"\nLATENCY per crop (CPU): mean {statistics.mean(latencies):.1f} ms | "
              f"median {statistics.median(latencies):.1f} ms | "
              f"min {min(latencies):.1f} | max {max(latencies):.1f} | n={len(latencies)}")
        print(f"MODEL LOAD (one-time): {load_s:.2f} s")

    # ---- Persist JSON + Markdown ----
    out_json = args.out_dir / "lens_eval_results.json"
    out_json.write_text(json.dumps({
        "meta": {
            "device": str(bundle.device),
            "model_load_s": round(load_s, 2),
            "classes": len(bundle.idx_to_class),
            "input_hw": [bundle.image_height, bundle.image_width],
            "latency_ms": {
                "mean": round(statistics.mean(latencies), 1) if latencies else None,
                "median": round(statistics.median(latencies), 1) if latencies else None,
                "min": round(min(latencies), 1) if latencies else None,
                "max": round(max(latencies), 1) if latencies else None,
                "n": len(latencies),
            },
            "accuracy_class": {
                "lens_line": lens_line, "lens_word": lens_word, "our_line": our_line,
            },
        },
        "ground_truth_notes": {k: v["source"] for k, v in GROUND_TRUTH.items()},
        "rows": rows,
    }, indent=2), encoding="utf-8")

    # Markdown table (line crops) for the eval doc.
    md = []
    md.append("| Run:Line | Text | GT class | Our pipeline | Lens top-1 | Lens top-5 (score, class) |")
    md.append("|---|---|---|---|---|---|")
    for r in line_rows:
        our = f"{r['our_font']} `{r['our_class']}`" + (" OK" if r["our_class_ok"] else " **X**")
        lens = f"{r['lens_top1']} `{r['lens_top1_class']}`" + (" OK" if r["lens_class_ok"] else " **X**")
        top5 = "; ".join(f"{p['name']} {p['score']:.2f} ({p['class']})" for p in r["lens_top5"])
        text = r["text"].replace("|", "/")
        md.append(f"| {r['run']}:{r['line']} | {text} | {r['expected_class']} | {our} | {lens} | {top5} |")
    (args.out_dir / "lens_eval_table.md").write_text("\n".join(md), encoding="utf-8")

    print(f"\n[out] {out_json}")
    print(f"[out] {args.out_dir / 'lens_eval_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
