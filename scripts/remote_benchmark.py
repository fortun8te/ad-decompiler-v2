#!/usr/bin/env python3
"""Drive a remote Windows RTX bridge over Tailscale and benchmark ad images.

POST /process uploads one image at a time (bridge returns 409 while busy).
Poll GET /process?job_id= until done/failed, then GET /run-summary for harness QA.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_BRIDGE = os.environ.get("AD_DECOMPILER_BRIDGE", "http://localhost:8790")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_POLL_INTERVAL_S = 2.0
DEFAULT_BUSY_RETRY_S = 3.0
DEFAULT_TIMEOUT_S = 900.0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_bridge(url: str) -> str:
    return str(url or DEFAULT_BRIDGE).strip().rstrip("/")


def discover_images(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        return []
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def generate_synthetic_images(out_dir: Path) -> list[Path]:
    """Create five diverse PNG fixtures: text, button, icon, gradient, photo composite."""
    from PIL import Image, ImageDraw, ImageFont

    out_dir.mkdir(parents=True, exist_ok=True)
    size = (1080, 1350)
    paths: list[Path] = []

    def save(name: str, image: Image.Image) -> Path:
        path = out_dir / name
        image.save(path, format="PNG")
        paths.append(path)
        return path

    def load_font(size_px: int):
        for candidate in ("Helvetica", "Arial", "DejaVuSans"):
            try:
                return ImageFont.truetype(candidate, size_px)
            except OSError:
                continue
        return ImageFont.load_default()

    # 1) Typography sample
    text_img = Image.new("RGB", size, "#f7f7f5")
    draw = ImageDraw.Draw(text_img)
    title_font = load_font(72)
    body_font = load_font(40)
    draw.text((80, 120), "Benchmark Sample", fill="#111111", font=title_font)
    draw.text(
        (80, 260),
        "Typography, spacing, and multi-line copy for OCR + layout.",
        fill="#333333",
        font=body_font,
    )
    draw.text((80, 360), "Shop now · Free shipping · 30-day returns", fill="#666666", font=body_font)
    save("synthetic_text.png", text_img)

    # 2) CTA button
    button_img = Image.new("RGB", size, "#eef2ff")
    draw = ImageDraw.Draw(button_img)
    draw.rounded_rectangle((340, 560, 740, 700), radius=36, fill="#2563eb")
    label_font = load_font(48)
    draw.text((430, 600), "Get Started", fill="#ffffff", font=label_font)
    draw.text((220, 220), "Primary action surface", fill="#1e293b", font=load_font(56))
    save("synthetic_button.png", button_img)

    # 3) Icon mark
    icon_img = Image.new("RGB", size, "#0f172a")
    draw = ImageDraw.Draw(icon_img)
    draw.ellipse((390, 420, 690, 720), fill="#22d3ee")
    draw.polygon([(540, 500), (620, 640), (460, 640)], fill="#0f172a")
    draw.text((300, 180), "Icon + glyph", fill="#e2e8f0", font=load_font(60))
    save("synthetic_icon.png", icon_img)

    # 4) Gradient background
    gradient = Image.new("RGB", size)
    px = gradient.load()
    for y in range(size[1]):
        ratio = y / max(size[1] - 1, 1)
        r = int(30 + 180 * ratio)
        g = int(64 + 90 * (1 - ratio))
        b = int(175 + 50 * ratio)
        for x in range(size[0]):
            px[x, y] = (r, g, b)
    draw = ImageDraw.Draw(gradient)
    draw.text((120, 120), "Gradient hero", fill="#ffffff", font=load_font(64))
    save("synthetic_gradient.png", gradient)

    # 5) Photo-style composite
    composite = Image.new("RGB", size, "#fafaf9")
    draw = ImageDraw.Draw(composite)
    draw.rectangle((80, 180, 1000, 980), fill="#d4d4d8")  # faux photo block
    for idx, color in enumerate(("#fb7185", "#fbbf24", "#34d399", "#60a5fa")):
        draw.rectangle((120 + idx * 180, 240, 260 + idx * 180, 420), fill=color)
    draw.text((80, 1040), "Composite ad layout", fill="#18181b", font=load_font(56))
    draw.text((80, 1140), "Photo block + headline + supporting copy", fill="#52525b", font=load_font(40))
    save("synthetic_photo_composite.png", composite)

    return paths


class BridgeHttp:
    """Thin urllib wrapper so tests can inject a fake transport."""

    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.timeout_s = timeout_s

    def get_json(self, url: str) -> tuple[int, dict[str, Any]]:
        try:
            with urlopen(url, timeout=self.timeout_s) as response:
                body = response.read()
                status = getattr(response, "status", 200)
        except HTTPError as error:
            body = error.read()
            status = error.code
        payload = json.loads(body.decode("utf-8")) if body else {}
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object from {url}")
        return status, payload

    def post_bytes(self, url: str, data: bytes) -> tuple[int, dict[str, Any] | None]:
        request = Request(url, data=data, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                body = response.read()
                status = getattr(response, "status", response.getcode())
        except HTTPError as error:
            body = error.read()
            status = error.code
        if not body:
            return status, None
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return status, None
        return status, payload if isinstance(payload, dict) else None


def submit_process(
    bridge: str,
    image_path: Path,
    *,
    http: BridgeHttp | None = None,
    busy_retry_s: float = DEFAULT_BUSY_RETRY_S,
) -> dict[str, Any]:
    http = http or BridgeHttp()
    data = image_path.read_bytes()
    url = f"{bridge}/process?{urlencode({'filename': image_path.name})}"
    while True:
        status, payload = http.post_bytes(url, data)
        if status == 409:
            time.sleep(busy_retry_s)
            continue
        if status not in (200, 202) or not payload or not payload.get("ok"):
            detail = (payload or {}).get("error") or f"upload failed with HTTP {status}"
            raise RuntimeError(detail)
        return payload


def poll_process(
    bridge: str,
    job_id: str,
    *,
    http: BridgeHttp | None = None,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    http = http or BridgeHttp()
    deadline = time.time() + timeout_s
    url = f"{bridge}/process?{urlencode({'job_id': job_id})}"
    last: dict[str, Any] = {}
    while time.time() < deadline:
        status_code, payload = http.get_json(url)
        if status_code == 404:
            raise RuntimeError(payload.get("error") or f"unknown job_id {job_id}")
        last = payload
        state = payload.get("status")
        if state in ("done", "failed", "cancelled"):
            return payload
        time.sleep(poll_interval_s)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_s:.0f}s (last={last.get('status')})")


def fetch_run_summary(
    bridge: str,
    job_id: str,
    *,
    http: BridgeHttp | None = None,
) -> dict[str, Any] | None:
    http = http or BridgeHttp()
    url = f"{bridge}/run-summary?{urlencode({'job_id': job_id})}"
    status_code, payload = http.get_json(url)
    if status_code == 404:
        return None
    if status_code != 200 or not payload.get("ok"):
        return None
    return payload


def _duration_from_status(status: dict[str, Any], *, wall_s: float | None) -> float | None:
    result = status.get("result") or {}
    for key in ("duration_s", "elapsed_s"):
        value = status.get(key)
        if value is None:
            value = result.get(key)
        if value is not None:
            return round(float(value), 3)
    if wall_s is not None:
        return round(wall_s, 3)
    return None


def _qa_ok_from_payload(status: dict[str, Any], summary: dict[str, Any] | None) -> bool | None:
    if summary:
        # Remote GPU evidence is fail-closed. A bridge saying "QA passed" is insufficient
        # when the actual mask/inpaint/layer failure list was omitted.
        if summary.get("qa_evidence_complete") is not True:
            return False
        if summary.get("hard_fails"):
            return False
        if summary.get("runtime_acceptable") is not True:
            return False
        if summary.get("final_qa_ok") is not None:
            return bool(summary["final_qa_ok"])
    if status.get("final_qa_ok") is not None:
        return bool(status["final_qa_ok"])
    result = status.get("result") or {}
    if result.get("qa_ok") is not None:
        return bool(result["qa_ok"])
    return None


def _harness_fields(status: dict[str, Any], summary: dict[str, Any] | None) -> dict[str, Any]:
    source = summary or status
    return {
        "harness_rounds": source.get("harness_rounds"),
        "harness_stopped": source.get("harness_stopped"),
        "final_qa_ok": source.get("final_qa_ok"),
    }


def benchmark_image(
    bridge: str,
    image_path: Path,
    *,
    http: BridgeHttp | None = None,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    busy_retry_s: float = DEFAULT_BUSY_RETRY_S,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    http = http or BridgeHttp(timeout_s=timeout_s)
    started = time.time()
    queued = submit_process(bridge, image_path, http=http, busy_retry_s=busy_retry_s)
    job_id = queued["job_id"]
    status = poll_process(
        bridge,
        job_id,
        http=http,
        poll_interval_s=poll_interval_s,
        timeout_s=timeout_s,
    )
    wall_s = time.time() - started
    summary = fetch_run_summary(bridge, job_id, http=http)
    harness = _harness_fields(status, summary)
    qa_ok = _qa_ok_from_payload(status, summary)
    row = {
        "id": image_path.stem,
        "filename": image_path.name,
        "job_id": job_id,
        "status": status.get("status"),
        "qa_ok": qa_ok,
        "duration_s": _duration_from_status(status, wall_s=wall_s),
        "staged": (summary or status).get("staged"),
        "doc_id": (summary or status).get("doc_id"),
        "bridge": bridge,
        "qa_evidence_complete": bool((summary or {}).get("qa_evidence_complete")),
        "hard_fails": (summary or {}).get("hard_fails") or [],
        "visual_score": (summary or {}).get("visual_score"),
        "ssim": (summary or {}).get("ssim"),
        "element_recall": (summary or {}).get("element_recall"),
        "background_audit": (summary or {}).get("background_audit"),
        "layer_alpha_audit": (summary or {}).get("layer_alpha_audit") or [],
        "runtime_status": (summary or {}).get("runtime_status"),
        "runtime_acceptable": (summary or {}).get("runtime_acceptable") is True,
        "runtime_violations": (summary or {}).get("runtime_violations") or [],
        **harness,
    }
    if status.get("status") == "failed":
        row["error"] = status.get("error") or status.get("user_detail")
        row["error_code"] = status.get("error_code")
        row["failed_stage"] = status.get("failed_stage")
    if summary and summary.get("manifest"):
        row["manifest_doc_id"] = (summary["manifest"] or {}).get("doc_id")
    return row


def build_report(
    *,
    bridge: str,
    input_dir: Path,
    runs: list[dict[str, Any]],
    synthetic: bool,
) -> dict[str, Any]:
    qa_passing = sum(1 for row in runs if row.get("qa_ok") is True)
    final_qa_passing = sum(1 for row in runs if row.get("final_qa_ok") is True)
    done = sum(1 for row in runs if row.get("status") == "done")
    durations = [float(row["duration_s"]) for row in runs if row.get("duration_s") is not None]
    mean_duration = round(sum(durations) / len(durations), 3) if durations else None
    return {
        "version": 1,
        "mode": "remote_bridge",
        "bridge": bridge,
        "input_dir": str(input_dir.resolve()),
        "synthetic": synthetic,
        "runs": runs,
        "summary": {
            "images": len(runs),
            "done": done,
            "failed": sum(1 for row in runs if row.get("status") == "failed"),
            "qa_passing": qa_passing,
            "qa_evidence_complete": sum(1 for row in runs if row.get("qa_evidence_complete")),
            "runtime_accepted": sum(1 for row in runs if row.get("runtime_acceptable")),
            "runs_with_hard_fails": sum(1 for row in runs if row.get("hard_fails")),
            "mask_inpaint_failure_runs": sum(1 for row in runs if any(
                item.get("rule") in {
                    "background-leakage", "inpaint-outside-mask", "layer-alpha-holes",
                    "empty-layer-alpha", "low-element-recall",
                }
                for item in (row.get("hard_fails") or []) if isinstance(item, dict)
            )),
            "final_qa_passing": final_qa_passing,
            "harness_rounds_total": sum(int(row.get("harness_rounds") or 0) for row in runs),
            "mean_duration_s": mean_duration,
        },
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Remote bridge benchmark",
        "",
        f"Bridge: `{report['bridge']}`",
        f"Input: `{report['input_dir']}`{' (synthetic)' if report.get('synthetic') else ''}",
        "",
        (
            f"Images: {summary['images']}  |  Done: {summary['done']}  |  "
            f"QA passing: {summary['qa_passing']}  |  Final QA: {summary['final_qa_passing']}"
        ),
        "",
        "| image | status | QA | evidence | runtime | final QA | seconds | hard fails | staged |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in report["runs"]:
        duration = row.get("duration_s")
        duration_cell = "—" if duration is None else f"{float(duration):.1f}"
        qa = "—" if row.get("qa_ok") is None else ("pass" if row["qa_ok"] else "fail")
        final_qa = (
            "—"
            if row.get("final_qa_ok") is None
            else ("pass" if row["final_qa_ok"] else "fail")
        )
        evidence = "complete" if row.get("qa_evidence_complete") else "missing"
        runtime = row.get("runtime_status") or ("ok" if row.get("runtime_acceptable") else "fail")
        fails = ", ".join(
            item.get("rule", "unknown") for item in (row.get("hard_fails") or [])
            if isinstance(item, dict)
        ) or "—"
        staged = "—" if row.get("staged") is None else ("yes" if row["staged"] else "no")
        lines.append(
            f"| {row['filename']} | {row.get('status', '—')} | {qa} | {evidence} | {runtime} | {final_qa} | "
            f"{duration_cell} | {fails} | {staged} |"
        )
    if summary.get("mean_duration_s") is not None:
        lines.extend(["", f"Mean duration: **{summary['mean_duration_s']:.1f}s**"])
    lines.append("")
    return "\n".join(lines)


def run_benchmark(
    *,
    bridge: str,
    input_dir: Path,
    output_dir: Path,
    generate_synthetic: bool = False,
    http: BridgeHttp | None = None,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    busy_retry_s: float = DEFAULT_BUSY_RETRY_S,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    bridge = normalize_bridge(bridge)
    images = discover_images(input_dir)
    synthetic = False
    if not images and generate_synthetic:
        synthetic_dir = Path("/tmp/ad-decompiler-remote-benchmark")
        images = generate_synthetic_images(synthetic_dir)
        input_dir = synthetic_dir
        synthetic = True
    if not images:
        raise SystemExit(f"No images found in {input_dir}")

    runs: list[dict[str, Any]] = []
    for image in images:
        print(f"→ {image.name}", flush=True)
        row = benchmark_image(
            bridge,
            image,
            http=http,
            poll_interval_s=poll_interval_s,
            busy_retry_s=busy_retry_s,
            timeout_s=timeout_s,
        )
        runs.append(row)
        print(
            f"  {row['status']}  qa_ok={row.get('qa_ok')}  "
            f"harness_rounds={row.get('harness_rounds')}  duration_s={row.get('duration_s')}",
            flush=True,
        )

    report = build_report(bridge=bridge, input_dir=input_dir, runs=runs, synthetic=synthetic)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "benchmark.md").write_text(markdown_report(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark images against a remote RTX bridge")
    parser.add_argument("--bridge", default=DEFAULT_BRIDGE, help="bridge base URL")
    parser.add_argument("--input-dir", required=True, type=Path, help="directory with png/jpg/webp images")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("."),
        help="where to write benchmark.json and benchmark.md",
    )
    parser.add_argument(
        "--generate-synthetic",
        action="store_true",
        help="if input-dir has no images, create five PNG fixtures under /tmp",
    )
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL_S)
    parser.add_argument("--busy-retry", type=float, default=DEFAULT_BUSY_RETRY_S)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    args = parser.parse_args(argv)

    if str(_repo_root()) not in sys.path:
        sys.path.insert(0, str(_repo_root()))

    report = run_benchmark(
        bridge=args.bridge,
        input_dir=args.input_dir,
        output_dir=args.output,
        generate_synthetic=args.generate_synthetic,
        poll_interval_s=args.poll_interval,
        busy_retry_s=args.busy_retry,
        timeout_s=args.timeout,
    )
    print(json.dumps(report["summary"], indent=2))
    summary = report["summary"]
    total = summary["images"]
    accepted = (
        total > 0
        and summary["done"] == total
        and summary["qa_passing"] == total
        and summary["final_qa_passing"] == total
        and summary["qa_evidence_complete"] == total
        and summary["runtime_accepted"] == total
        and summary["runs_with_hard_fails"] == 0
        and all(row.get("staged") is True for row in report["runs"])
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
