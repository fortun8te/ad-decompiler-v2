#!/usr/bin/env python3
"""Generate a self-contained visual HTML report for a benchmark run directory.

A benchmark run directory looks like ``runs/<name>/`` and contains ``benchmark.json``
plus one subdirectory per image with ``original.png`` / ``preview.png`` / ``diff.png``
and a ``qa.json``. This script renders a single ``report.html`` with, per image, the
original / reconstruction / diff triptych, the QA metric row, a pass/fail chip and an
expandable per-layer breakdown when ``qa.json`` carries ``per_layer`` scores.

The report is plain HTML with inline CSS -- no external fonts, scripts or stylesheets --
so it renders offline straight from the run directory. Images are referenced by relative
path because ``report.html`` is written inside the run directory.

Usage::

    python scripts/report_html.py runs/golden-optimized-check
    python scripts/report_html.py runs/after --baseline runs/before
"""
from __future__ import annotations

import argparse
import html
import json
from datetime import datetime, timezone
from pathlib import Path

# (basename candidates, label) for the per-image triptych. First existing wins.
TRIPTYCH = (
    (("original.png", "normalized.png"), "Original"),
    (("preview.png",), "Reconstruction"),
    (("diff.png",), "Diff"),
)

# (row-key, label) rendered in the per-image metric strip and summary tiles.
METRIC_FIELDS = (
    ("visual_score", "Visual"),
    ("ssim", "SSIM"),
    ("text_recall", "Text"),
    ("editable_text_recall", "Editable"),
    ("edge_f1", "Edge"),
    ("element_recall", "Element recall"),
)

# (summary-key, label, is_ratio) for the summary metric tiles.
SUMMARY_METRICS = (
    ("mean_visual_score", "Mean visual", True),
    ("mean_ssim", "Mean SSIM", True),
    ("mean_text_recall", "Mean text", True),
    ("mean_editable_text_recall", "Mean editable", True),
    ("mean_edge_f1", "Mean edge", True),
    ("mean_element_recall", "Mean element recall", True),
)

# Preferred column order for the per-layer table; any extra scalar keys are appended.
PER_LAYER_PREFERRED = (
    "id", "type", "role", "score", "ssim", "recall",
    "region_ssim", "ink_iou", "ink_excess", "region_px", "fallback",
)

# qa.json artifacts that mark a run subdirectory as worth reporting even when it is not
# listed in benchmark.json (a partially-failed / orphan run).
DISCOVERY_MARKERS = ("qa.json", "preview.png", "original.png", "diff.png")


def _read_json(path: Path, fallback=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _num(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value, digits: int = 3) -> str:
    number = _num(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _band(value) -> str:
    """Coarse quality band for a 0..1 metric, used only for colour."""
    number = _num(value)
    if number is None:
        return "na"
    if number >= 0.8:
        return "good"
    if number >= 0.6:
        return "warn"
    return "bad"


def _find_image(run_dir: Path, run_id: str, candidates: tuple[str, ...]) -> str | None:
    """Return a relative POSIX path to the first existing candidate image, else None."""
    for name in candidates:
        if (run_dir / run_id / name).is_file():
            return f"{run_id}/{name}"
    return None


def _normalize_from_benchmark(entry: dict, run_dir: Path) -> dict:
    """Build a render row from a benchmark.json ``runs`` entry (metrics pre-extracted)."""
    run_id = str(entry.get("id") or "")
    qa = _read_json(run_dir / run_id / "qa.json", {}) or {}
    missing = list(entry.get("missing_artifacts") or [])
    return {
        "id": run_id,
        "qa_ok": bool(entry.get("qa_ok")),
        "complete": bool(entry.get("complete", not missing)),
        "missing_artifacts": missing,
        "orphan": False,
        "runtime_status": entry.get("runtime_status"),
        "duration_s": entry.get("duration_s"),
        "archetype": entry.get("archetype"),
        "preset": entry.get("preset"),
        "visual_score": entry.get("visual_score"),
        "ssim": entry.get("ssim"),
        "text_recall": entry.get("text_recall"),
        "editable_text_recall": entry.get("editable_text_recall"),
        "edge_f1": entry.get("edge_f1"),
        "element_recall": entry.get("element_recall"),
        "regional_routes": entry.get("regional_inpaint_routes") or {},
        "hard_fails": entry.get("hard_fails") or [],
        "per_layer": qa.get("per_layer") or [],
        "images": {
            label: _find_image(run_dir, run_id, names) for names, label in TRIPTYCH
        },
    }


def _normalize_from_qa(run_id: str, run_dir: Path, *, orphan: bool) -> dict:
    """Best-effort render row from a subdirectory's qa.json (no benchmark.json entry)."""
    qa = _read_json(run_dir / run_id / "qa.json", {}) or {}
    structural = qa.get("structural") or {}
    hard_fails = qa.get("hard_fails")
    if not isinstance(hard_fails, list):
        hard_fails = structural.get("hard_fails") if isinstance(structural.get("hard_fails"), list) else []
    return {
        "id": run_id,
        "qa_ok": bool(qa.get("ok")),
        "complete": bool(qa),
        "missing_artifacts": [] if qa else ["qa.json"],
        "orphan": orphan,
        "runtime_status": None,
        "duration_s": None,
        "archetype": qa.get("archetype"),
        "preset": qa.get("preset"),
        "visual_score": qa.get("visual_score"),
        "ssim": qa.get("ssim"),
        "text_recall": qa.get("text_recall"),
        "editable_text_recall": qa.get("editable_text_recall") or structural.get("editable_text_recall"),
        "edge_f1": qa.get("edge_f1"),
        "element_recall": structural.get("element_recall"),
        "regional_routes": {},
        "hard_fails": hard_fails,
        "per_layer": qa.get("per_layer") or [],
        "images": {
            label: _find_image(run_dir, run_id, names) for names, label in TRIPTYCH
        },
    }


def _discover_subruns(run_dir: Path, known: set[str]) -> list[str]:
    """Subdirectories that carry pipeline artifacts but are absent from benchmark.json."""
    found = []
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir() or child.name in known:
            continue
        if any((child / marker).is_file() for marker in DISCOVERY_MARKERS):
            found.append(child.name)
    return found


def collect_rows(run_dir: Path, benchmark: dict | None) -> list[dict]:
    """Assemble render rows from benchmark.json entries plus any orphan subdirectories."""
    rows: list[dict] = []
    known: set[str] = set()
    for entry in (benchmark or {}).get("runs") or []:
        if isinstance(entry, dict) and entry.get("id"):
            rows.append(_normalize_from_benchmark(entry, run_dir))
            known.add(str(entry["id"]))
    orphan = benchmark is not None  # if no benchmark.json, these are the primary rows
    for run_id in _discover_subruns(run_dir, known):
        rows.append(_normalize_from_qa(run_id, run_dir, orphan=orphan))
    return rows


def _summarize(rows: list[dict]) -> dict:
    """Fallback summary computed from rows when benchmark.json has no summary block."""
    def mean(key):
        values = [_num(row.get(key)) for row in rows]
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 4) if values else None

    return {
        "images": len(rows),
        "qa_passing": sum(1 for row in rows if row["qa_ok"]),
        "complete_runs": sum(1 for row in rows if row["complete"]),
        "runs_with_hard_fails": sum(1 for row in rows if row["hard_fails"]),
        "mean_visual_score": mean("visual_score"),
        "mean_ssim": mean("ssim"),
        "mean_text_recall": mean("text_recall"),
        "mean_editable_text_recall": mean("editable_text_recall"),
        "mean_edge_f1": mean("edge_f1"),
        "mean_element_recall": mean("element_recall"),
    }


# --------------------------------------------------------------------------- render


def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _chip(text: str, kind: str) -> str:
    return f'<span class="chip chip-{kind}">{_esc(text)}</span>'


def _delta_html(cur, base, *, higher_better: bool = True, digits: int = 3) -> str:
    cur_n, base_n = _num(cur), _num(base)
    if cur_n is None or base_n is None:
        return ""
    diff = cur_n - base_n
    if abs(diff) < 10 ** (-digits) / 2:
        return '<span class="delta delta-flat">±0</span>'
    improved = diff > 0 if higher_better else diff < 0
    arrow = "▲" if diff > 0 else "▼"
    kind = "up" if improved else "down"
    return f'<span class="delta delta-{kind}">{arrow}{diff:+.{digits}f}</span>'


def _metric_cell(label: str, value, delta: str) -> str:
    return (
        f'<div class="metric metric-{_band(value)}">'
        f'<div class="metric-val">{_esc(_fmt(value))}{delta}</div>'
        f'<div class="metric-label">{_esc(label)}</div>'
        f"</div>"
    )


def _routes_html(routes: dict) -> str:
    if not routes:
        return '<span class="muted">—</span>'
    return "".join(
        f'<span class="route">{_esc(name)} <b>×{_esc(count)}</b></span>'
        for name, count in routes.items()
    )


def _hard_fails_html(hard_fails: list) -> str:
    if not hard_fails:
        return ""
    pills = []
    for item in hard_fails:
        if isinstance(item, dict):
            rule = item.get("rule", "unknown")
            detail = item.get("detail")
            text = f"{rule} — {detail}" if detail else str(rule)
        else:
            text = str(item)
        pills.append(f'<span class="fail-pill">{_esc(text)}</span>')
    return '<div class="fails">' + "".join(pills) + "</div>"


def _per_layer_html(per_layer: list) -> str:
    rows = [row for row in per_layer if isinstance(row, dict)]
    if not rows:
        return '<div class="no-layers">No per-layer scores recorded for this run.</div>'

    def is_scalar(value):
        return isinstance(value, (str, int, float, bool)) or value is None

    present: list[str] = []
    for key in PER_LAYER_PREFERRED:
        if any(key in row for row in rows):
            present.append(key)
    extra = sorted({
        key for row in rows for key, value in row.items()
        if key not in present and is_scalar(value)
    })
    columns = present + extra

    head = "".join(f"<th>{_esc(col)}</th>" for col in columns)
    body = []
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, float):
                text = f"{value:.4f}"
            elif value is None and col in row:
                text = "—"
            elif value is None:
                text = ""
            elif isinstance(value, bool):
                text = "yes" if value else "no"
            else:
                text = str(value)
            cells.append(f"<td>{_esc(text)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        '<details class="layers"><summary>Per-layer detail '
        f"({len(rows)} layer{'s' if len(rows) != 1 else ''})</summary>"
        '<div class="layers-scroll"><table class="layer-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody>"
        "</table></div></details>"
    )


def _image_cell(label: str, rel_path: str | None) -> str:
    if rel_path:
        media = f'<img loading="lazy" src="{_esc(rel_path)}" alt="{_esc(label)}">'
    else:
        media = '<div class="img-missing">missing</div>'
    return f'<figure class="shot"><figcaption>{_esc(label)}</figcaption>{media}</figure>'


def _run_card(row: dict, baseline_row: dict | None) -> str:
    chips = [_chip("PASS", "pass") if row["qa_ok"] else _chip("FAIL", "fail")]
    if row.get("orphan"):
        chips.append(_chip("orphan run", "warn"))
    if row["missing_artifacts"]:
        preview = ", ".join(row["missing_artifacts"][:4])
        if len(row["missing_artifacts"]) > 4:
            preview += f", +{len(row['missing_artifacts']) - 4} more"
        chips.append(_chip(f"incomplete: {preview}", "warn"))
    elif not row["complete"]:
        chips.append(_chip("incomplete", "warn"))

    meta_bits = []
    for label, value in (
        ("archetype", row.get("archetype")),
        ("preset", row.get("preset")),
        ("runtime", row.get("runtime_status")),
    ):
        if value:
            meta_bits.append(f'<span class="tag">{_esc(label)}: {_esc(value)}</span>')
    if _num(row.get("duration_s")) is not None:
        meta_bits.append(f'<span class="tag">{_fmt(row["duration_s"], 1)}s</span>')

    shots = "".join(_image_cell(label, path) for label, path in row["images"].items())

    metric_cells = []
    for key, label in METRIC_FIELDS:
        delta = _delta_html(row.get(key), (baseline_row or {}).get(key)) if baseline_row else ""
        metric_cells.append(_metric_cell(label, row.get(key), delta))
    route_block = (
        '<div class="metric metric-routes"><div class="metric-val routes-val">'
        f'{_routes_html(row.get("regional_routes") or {})}</div>'
        '<div class="metric-label">Regional routes</div></div>'
    )
    fails_count = len(row["hard_fails"])
    fails_band = "good" if fails_count == 0 else "bad"
    fail_metric = (
        f'<div class="metric metric-{fails_band}"><div class="metric-val">{fails_count}</div>'
        '<div class="metric-label">Hard fails</div></div>'
    )

    return (
        '<section class="card">'
        '<div class="card-head">'
        f'<h3>{_esc(row["id"])}</h3>'
        f'<div class="chips">{"".join(chips)}</div>'
        "</div>"
        f'<div class="meta">{"".join(meta_bits)}</div>'
        f'<div class="shots">{shots}</div>'
        f'<div class="metrics">{"".join(metric_cells)}{route_block}{fail_metric}</div>'
        f'{_hard_fails_html(row["hard_fails"])}'
        f'{_per_layer_html(row.get("per_layer") or [])}'
        "</section>"
    )


def _summary_html(summary: dict, baseline_summary: dict | None, baseline_name: str | None) -> str:
    def count_tile(key, label, denom_key="images"):
        cur = summary.get(key)
        denom = summary.get(denom_key)
        value = "—" if cur is None else str(cur)
        if denom is not None and cur is not None:
            value = f"{cur}<span class='denom'>/{denom}</span>"
        delta = ""
        if baseline_summary:
            delta = _delta_html(cur, baseline_summary.get(key), digits=0)
        return (
            f'<div class="tile"><div class="tile-val">{value}{delta}</div>'
            f'<div class="tile-label">{_esc(label)}</div></div>'
        )

    count_tiles = "".join((
        count_tile("qa_passing", "QA passing"),
        count_tile("complete_runs", "Complete runs"),
        count_tile("runtime_accepted", "Runtime accepted")
        if "runtime_accepted" in summary else count_tile("runs_with_hard_fails", "Runs w/ hard fails"),
    ))

    metric_tiles = []
    for key, label, _ratio in SUMMARY_METRICS:
        if key not in summary:
            continue
        delta = _delta_html(summary.get(key), (baseline_summary or {}).get(key)) if baseline_summary else ""
        metric_tiles.append(
            f'<div class="tile tile-{_band(summary.get(key))}">'
            f'<div class="tile-val">{_esc(_fmt(summary.get(key)))}{delta}</div>'
            f'<div class="tile-label">{_esc(label)}</div></div>'
        )

    baseline_note = ""
    if baseline_summary is not None:
        baseline_note = (
            f'<p class="baseline-note">Deltas shown vs baseline '
            f'<code>{_esc(baseline_name)}</code>. ▲ improvement, ▼ regression.</p>'
        )
    return (
        '<div class="summary">'
        f'<div class="tiles tiles-count">{count_tiles}</div>'
        f'<div class="tiles tiles-metric">{"".join(metric_tiles)}</div>'
        f"{baseline_note}"
        "</div>"
    )


CSS = """
:root{color-scheme:dark;}
*{box-sizing:border-box;}
body{margin:0;background:#0d1117;color:#e6edf3;
 font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;}
a{color:#58a6ff;}
.wrap{max-width:1200px;margin:0 auto;padding:28px 22px 64px;}
header h1{margin:0 0 4px;font-size:24px;}
header .sub{color:#8b949e;font-size:13px;margin:0;}
code{background:#161b22;border:1px solid #30363d;border-radius:5px;padding:1px 5px;font-size:12px;}
.muted{color:#8b949e;}
.summary{margin:22px 0 34px;}
.tiles{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:12px;}
.tile{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px 16px;min-width:130px;flex:1 1 130px;}
.tile-val{font-size:24px;font-weight:650;letter-spacing:-.02em;}
.tile-val .denom{color:#8b949e;font-size:15px;font-weight:500;}
.tile-label{color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:.04em;margin-top:2px;}
.tile-good{border-color:rgba(63,185,80,.5);} .tile-warn{border-color:rgba(210,153,34,.5);} .tile-bad{border-color:rgba(248,81,73,.5);}
.baseline-note{color:#8b949e;font-size:12px;margin:6px 0 0;}
.delta{font-size:12px;font-weight:600;margin-left:6px;vertical-align:middle;}
.delta-up{color:#3fb950;} .delta-down{color:#f85149;} .delta-flat{color:#8b949e;}
.card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:18px;margin:0 0 20px;}
.card-head{display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;}
.card-head h3{margin:0;font-size:16px;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;word-break:break-all;}
.chips{display:flex;gap:6px;flex-wrap:wrap;}
.chip{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;padding:3px 9px;border-radius:999px;border:1px solid transparent;}
.chip-pass{background:rgba(63,185,80,.15);color:#3fb950;border-color:rgba(63,185,80,.4);}
.chip-fail{background:rgba(248,81,73,.15);color:#f85149;border-color:rgba(248,81,73,.4);}
.chip-warn{background:rgba(210,153,34,.15);color:#e3b341;border-color:rgba(210,153,34,.4);}
.meta{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0 14px;}
.tag{font-size:11px;color:#adbac7;background:#21262d;border:1px solid #30363d;border-radius:6px;padding:2px 8px;}
.shots{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
.shot{margin:0;background:#0d1117;border:1px solid #30363d;border-radius:8px;overflow:hidden;display:flex;flex-direction:column;}
.shot figcaption{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.05em;padding:6px 8px;border-bottom:1px solid #21262d;}
.shot img{display:block;width:100%;height:auto;background:#fff;}
.img-missing{display:flex;align-items:center;justify-content:center;min-height:120px;color:#f85149;font-size:12px;text-transform:uppercase;letter-spacing:.05em;background:repeating-linear-gradient(45deg,#161b22,#161b22 10px,#1b2028 10px,#1b2028 20px);}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin:14px 0 0;}
.metric{background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:9px 11px;}
.metric-val{font-size:18px;font-weight:650;}
.metric-label{font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.04em;margin-top:1px;}
.metric-good .metric-val{color:#3fb950;} .metric-warn .metric-val{color:#e3b341;} .metric-bad .metric-val{color:#f85149;}
.metric-routes{grid-column:1/-1;} .routes-val{font-size:12px;display:flex;flex-wrap:wrap;gap:6px;}
.route{background:#21262d;border:1px solid #30363d;border-radius:6px;padding:2px 7px;font-weight:500;}
.route b{color:#58a6ff;}
.fails{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0 0;}
.fail-pill{font-size:12px;background:rgba(248,81,73,.12);color:#ff9d96;border:1px solid rgba(248,81,73,.4);border-radius:6px;padding:3px 9px;}
.layers{margin-top:14px;border-top:1px solid #21262d;padding-top:10px;}
.layers summary{cursor:pointer;color:#adbac7;font-size:13px;font-weight:600;}
.layers summary:hover{color:#58a6ff;}
.layers-scroll{overflow-x:auto;margin-top:10px;}
.layer-table{border-collapse:collapse;width:100%;font-size:12px;}
.layer-table th,.layer-table td{border:1px solid #30363d;padding:5px 8px;text-align:left;white-space:nowrap;}
.layer-table th{background:#21262d;color:#adbac7;position:sticky;top:0;}
.no-layers{margin-top:12px;color:#8b949e;font-size:12px;}
footer{color:#8b949e;font-size:12px;margin-top:36px;border-top:1px solid #21262d;padding-top:14px;}
"""


def build_html(run_dir: Path, baseline_dir: Path | None = None) -> str:
    """Render the full report HTML for ``run_dir`` (self-contained, dark theme)."""
    run_dir = Path(run_dir)
    benchmark = _read_json(run_dir / "benchmark.json")
    rows = collect_rows(run_dir, benchmark)
    summary = (benchmark or {}).get("summary") or _summarize(rows)

    baseline_summary = None
    baseline_rows_by_id: dict[str, dict] = {}
    baseline_name = None
    if baseline_dir is not None:
        baseline_dir = Path(baseline_dir)
        baseline_name = baseline_dir.name
        baseline_bench = _read_json(baseline_dir / "benchmark.json")
        b_rows = collect_rows(baseline_dir, baseline_bench)
        baseline_rows_by_id = {row["id"]: row for row in b_rows}
        baseline_summary = (baseline_bench or {}).get("summary") or _summarize(b_rows)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    input_dir = (benchmark or {}).get("input_dir")
    sub_bits = [f"Generated {generated}"]
    if input_dir:
        sub_bits.append(f"inputs: {input_dir}")
    if baseline_name:
        sub_bits.append(f"baseline: {baseline_name}")

    cards = "".join(
        _run_card(row, baseline_rows_by_id.get(row["id"])) for row in rows
    ) or '<p class="muted">No image runs found in this directory.</p>'

    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>Benchmark report — {_esc(run_dir.name)}</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n"
        '<div class="wrap">\n'
        "<header>\n"
        f"<h1>Benchmark report — {_esc(run_dir.name)}</h1>\n"
        f'<p class="sub">{_esc(" · ".join(sub_bits))}</p>\n'
        "</header>\n"
        f"{_summary_html(summary, baseline_summary, baseline_name)}\n"
        f"{cards}\n"
        "<footer>Each image shows original → reconstruction → diff. "
        "Metrics and pass/fail come from qa.json; a run is not accepted until every hard "
        "fail has a deliberate disposition.</footer>\n"
        "</div>\n</body>\n</html>\n"
    )


def generate_report(run_dir: str | Path, output: str | Path | None = None,
                    baseline_dir: str | Path | None = None) -> Path:
    """Write ``report.html`` for ``run_dir`` and return the written path."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise NotADirectoryError(f"not a benchmark run directory: {run_dir}")
    out_path = Path(output) if output else run_dir / "report.html"
    html_text = build_html(run_dir, Path(baseline_dir) if baseline_dir else None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_text, encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a visual HTML benchmark report")
    parser.add_argument("run_dir", help="benchmark run directory (contains benchmark.json)")
    parser.add_argument("--baseline", default=None,
                        help="another run directory to diff metrics against (before/after)")
    parser.add_argument("--output", default=None,
                        help="output HTML path (default: <run_dir>/report.html)")
    args = parser.parse_args(argv)
    path = generate_report(args.run_dir, args.output, args.baseline)
    print(f"Wrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
