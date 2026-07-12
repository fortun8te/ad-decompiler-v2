"""User-facing error classification for bridge + plugin surfaces."""
from __future__ import annotations

import os
import re
from typing import Any

STAGE_LABELS = {
    "normalize": "Reading image",
    "ocr": "Reading text",
    "text": "Analyzing typography",
    "residual": "Detecting elements",
    "qwen": "Layer proposals",
    "sam": "Segmenting",
    "elements": "Merging elements",
    "merge": "Merging layers",
    "reconstruct": "Rebuilding background",
    "layout": "Building layout",
    "design": "Compiling scene",
    "preview": "Rendering preview",
    "figma": "Staging for Figma",
    "export": "Awaiting export",
    "qa": "Quality check",
}

_STAGE_MARKERS = [
    ("normalize", "normalize →"),
    ("ocr", "ocr["),
    ("text", "text analysis →"),
    ("residual", "residual proposals →"),
    ("qwen", "qwen →"),
    ("sam", "sam3["),
    ("elements", "element fusion →"),
    ("merge", "merge →"),
    ("reconstruct", "reconstruct →"),
    ("layout", "layout →"),
    ("design", "design.json →"),
    ("preview", "preview →"),
    ("figma", "figma import:"),
    ("export", "export:"),
    ("qa", "qa →"),
]

STAGE_ORDER = [name for name, _ in _STAGE_MARKERS]
_STAGE_ORDER = STAGE_ORDER

_OCR_BLOB = re.compile(
    r"ocr|paddle|ppocr|paddleocr|tesseract|surya|doctr|no configured OCR backend",
    re.I,
)
_CUDNN_BLOB = re.compile(r"cudnn|CUDNN", re.I)
_CUDA_BLOB = re.compile(r"cuda|CUDA|torch\.cuda|out of memory", re.I)
_CHARMAP_BLOB = re.compile(r"charmap|codec can't encode|UnicodeEncodeError", re.I)
# Container is also ordinary layout vocabulary in this project (for example the
# ``tighten-containers`` repair). Only classify an environment problem when the text
# contains a real container-runtime signature.
_DOCKER_BLOB = re.compile(
    r"\bdocker(?:file)?\b|docker daemon|\bcontainerd\b|/\.dockerenv|"
    r"(?:inside|within|running in|started in) (?:an? )?(?:docker )?container\b|"
    r"unsupported container (?:runtime|setup)",
    re.I,
)
_MISSING_DEP_BLOB = re.compile(r"ModuleNotFoundError|No module named", re.I)


def _tail_stage_from_log(lines: list[str]) -> str | None:
    current = None
    for line in lines:
        for name, marker in _STAGE_MARKERS:
            if marker in line:
                current = name
    return current


def _stage_after(last_completed: str | None) -> str | None:
    if not last_completed:
        return _STAGE_ORDER[0] if _STAGE_ORDER else None
    try:
        idx = _STAGE_ORDER.index(last_completed)
    except ValueError:
        return None
    return _STAGE_ORDER[idx + 1] if idx + 1 < len(_STAGE_ORDER) else last_completed


def _failed_stage_from_log(lines: list[str], error_text: str = "") -> str | None:
    """Infer the stage that failed when pipeline.log has ERROR but no success marker for it."""
    last_success = None
    saw_error = False
    for line in lines:
        if "ERROR:" in line:
            saw_error = True
            break
        for name, marker in _STAGE_MARKERS:
            if marker in line and "→" in line:
                last_success = name
    if not saw_error:
        return None
    if _OCR_BLOB.search(error_text):
        return "ocr"
    if last_success is not None:
        nxt = _stage_after(last_success)
        if nxt:
            return nxt
    return None


def _failed_stage_from_agent_debug(agent_debug: list[dict[str, Any]] | None) -> str | None:
    if not agent_debug:
        return None
    stage_by_location = (
        ("ocr.py", "ocr"),
        ("text_analysis", "text"),
        ("sam3_detect", "sam"),
        ("element_detect", "residual"),
        ("qwen_worker", "qwen"),
        ("normalize", "normalize"),
        ("merge_layers", "merge"),
        ("reconstruct", "reconstruct"),
        ("layout", "layout"),
        ("build_design_json", "design"),
        ("render_preview", "preview"),
        ("figma_import", "figma"),
        ("pixel_diff", "qa"),
    )
    for entry in reversed(agent_debug):
        location = str(entry.get("location") or "")
        message = str(entry.get("message") or "")
        if "failed" not in message.lower() and "error" not in message.lower():
            continue
        for needle, stage in stage_by_location:
            if needle in location:
                return stage
    return None


def tail_running_stage(run_dir: str | None) -> str | None:
    """Last pipeline stage seen in pipeline.log — for live progress, not failure inference."""
    if not run_dir:
        return None
    try:
        with open(os.path.join(run_dir, "pipeline.log"), encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return None
    return _tail_stage_from_log(lines)


def detect_failed_stage(
    run_dir: str | None,
    *,
    error_text: str = "",
    explicit_stage: str | None = None,
    agent_debug: list[dict[str, Any]] | None = None,
) -> str | None:
    if explicit_stage:
        return explicit_stage
    from_debug = _failed_stage_from_agent_debug(agent_debug)
    if from_debug:
        return from_debug
    lines: list[str] = []
    if run_dir:
        try:
            with open(os.path.join(run_dir, "pipeline.log"), encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            lines = []
    from_log = _failed_stage_from_log(lines, error_text=error_text)
    if from_log:
        return from_log
    if _OCR_BLOB.search(error_text):
        return "ocr"
    tail = _tail_stage_from_log(lines)
    if tail == "normalize" and _OCR_BLOB.search(error_text):
        return "ocr"
    return tail


def _debug_blob(agent_debug: list[dict[str, Any]] | None) -> str:
    if not agent_debug:
        return ""
    parts: list[str] = []
    for entry in agent_debug[-12:]:
        parts.append(str(entry.get("location") or ""))
        parts.append(str(entry.get("message") or ""))
        data = entry.get("data") or {}
        if isinstance(data, dict):
            for key in ("error", "error_type", "engine"):
                if data.get(key):
                    parts.append(str(data[key]))
    return "\n".join(parts)


def _sanitize_technical_detail(error: str, traceback_text: str = "", *, limit: int = 280) -> str:
    raw = "\n".join(part.strip() for part in (error, traceback_text) if part and part.strip())
    if not raw:
        return ""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    # Drop noisy Python frames; keep the exception line(s).
    kept = [ln for ln in lines if not ln.startswith("File ") and not ln.startswith('  File "')]
    summary = kept[-3:] if kept else lines[-3:]
    text = " ".join(summary)
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def classify_processing_error(
    *,
    error: str = "",
    traceback_text: str = "",
    failed_stage: str | None = None,
    agent_debug: list[dict[str, Any]] | None = None,
) -> dict[str, str | None]:
    blob = "\n".join(
        part for part in (error, traceback_text, _debug_blob(agent_debug)) if part
    )
    stage = failed_stage or detect_failed_stage(
        None, error_text=blob, agent_debug=agent_debug
    )
    if stage == "normalize" and _OCR_BLOB.search(blob):
        stage = "ocr"
    stage_label = STAGE_LABELS.get(stage or "", stage or "Processing")

    if _MISSING_DEP_BLOB.search(blob):
        return {
            "error_code": "dependency_missing",
            "error_hint": (
                "A Python package is missing on the bridge PC. "
                "Close the bridge, open PowerShell in the ad-decompiler folder, run .\\setup_rtx.ps1 "
                "(or pip install -r requirements-gpu.txt), then double-click Start Bridge.bat again."
            ),
            "user_title": "Setup incomplete",
            "user_detail": f"Missing dependency while {stage_label.lower()}.",
            "failed_stage": stage,
        }

    if _CHARMAP_BLOB.search(blob):
        return {
            "error_code": "windows_encoding",
            "error_hint": (
                "Windows console encoding blocked a log message. "
                "Always start the bridge with Start Bridge.bat (it sets UTF-8). "
                "If you start it manually, run in PowerShell first: $env:PYTHONUTF8=1"
            ),
            "user_title": "Windows encoding issue",
            "user_detail": f"The bridge hit a text-encoding error during {stage_label.lower()}.",
            "failed_stage": stage,
        }

    if _DOCKER_BLOB.search(blob) and not _MISSING_DEP_BLOB.search(blob):
        return {
            "error_code": "docker_not_supported",
            "error_hint": (
                "The bridge must run directly on your Windows GPU PC, not inside Docker. "
                "Double-click Start Bridge.bat on the machine that has the NVIDIA GPU."
            ),
            "user_title": "Wrong runtime",
            "user_detail": "Processing failed because the bridge was started in an unsupported container setup.",
            "failed_stage": stage,
        }

    if _CUDNN_BLOB.search(blob):
        return {
            "error_code": "cudnn_unavailable",
            "error_hint": (
                "cuDNN is missing or mismatched for your GPU PyTorch build. "
                "On the bridge PC: update NVIDIA drivers, install the CUDA toolkit version shown by "
                "python doctor.py, rerun .\\setup_rtx.ps1, then restart Start Bridge.bat."
            ),
            "user_title": "GPU library (cuDNN) issue",
            "user_detail": f"Text recognition needs cuDNN but it is not available ({stage_label}).",
            "failed_stage": stage or "ocr",
        }

    if _CUDA_BLOB.search(blob):
        return {
            "error_code": "cuda_unavailable",
            "error_hint": (
                "The bridge could not use your NVIDIA GPU. "
                "Confirm the driver is installed, set device: cuda in config.yaml, run python doctor.py, "
                "and restart Start Bridge.bat. To test on CPU only, set device: cpu in config.yaml."
            ),
            "user_title": "GPU not available",
            "user_detail": f"Processing stopped during {stage_label.lower()} because CUDA/GPU access failed.",
            "failed_stage": stage,
        }

    if _OCR_BLOB.search(blob) or stage == "ocr":
        if "paddleocr" in blob.lower() or "paddlepaddle" in blob.lower():
            hint = (
                "PaddleOCR is not installed or mismatched. On the bridge PC run .\\setup_rtx.ps1, "
                "then python doctor.py and confirm OCR shows ready before uploading again."
            )
        elif "tesseract" in blob.lower():
            hint = (
                "Tesseract OCR is missing. Install Tesseract for Windows, add it to PATH, "
                "then restart Start Bridge.bat."
            )
        else:
            hint = (
                "Text recognition failed on the bridge PC. Run python doctor.py, fix any OCR blockers, "
                "then restart Start Bridge.bat. If GPU OCR keeps failing, try device: cpu in config.yaml."
            )
        return {
            "error_code": "ocr_failed",
            "error_hint": hint,
            "user_title": "Couldn't read text in the image",
            "user_detail": "The image loaded, but the OCR step could not extract text.",
            "failed_stage": "ocr",
        }

    if re.search(r"config\.yaml|config file", blob, re.I):
        return {
            "error_code": "config_missing",
            "error_hint": (
                "Bridge config is missing or invalid. Double-click Start Bridge.bat to auto-create config.yaml, "
                "or copy config.example.yaml and edit device plus model paths."
            ),
            "user_title": "Bridge config problem",
            "user_detail": "The bridge could not read its configuration file.",
            "failed_stage": stage,
        }

    if re.search(r"sam3|checkpoint", blob, re.I):
        return {
            "error_code": "sam_checkpoint",
            "error_hint": (
                "SAM segmentation model is missing. Set the local checkpoint path in config.yaml on the bridge PC "
                "and confirm the file exists."
            ),
            "user_title": "Segmentation model missing",
            "user_detail": f"Processing stopped during {stage_label.lower()}.",
            "failed_stage": stage or "sam",
        }

    if re.search(r"ECONNREFUSED|Failed to fetch|NetworkError|health check", blob, re.I):
        return {
            "error_code": "bridge_unreachable",
            "error_hint": (
                "Figma can't reach the bridge. On the processing PC, double-click Start Bridge.bat "
                "and confirm http://localhost:8790/health opens in a browser."
            ),
            "user_title": "Bridge offline",
            "user_detail": "The plugin could not contact the local bridge.",
            "failed_stage": stage,
        }

    if re.search(r"409|already processing", blob, re.I):
        return {
            "error_code": "job_busy",
            "error_hint": "Wait for the current image to finish, or restart Start Bridge.bat if a job looks stuck.",
            "user_title": "Bridge is busy",
            "user_detail": "Another image is still processing.",
            "failed_stage": stage,
        }

    technical = _sanitize_technical_detail(error, traceback_text)
    return {
        "error_code": "pipeline_failed",
        "error_hint": (
            "Check the bridge terminal or run folder pipeline.log on the processing PC. "
            "Run python doctor.py, fix reported blockers, then restart Start Bridge.bat."
        ),
        "user_title": f"Stopped during {stage_label.lower()}" if stage else "Processing failed",
        "user_detail": technical or "The pipeline reported a failure with no extra detail.",
        "failed_stage": stage,
    }
