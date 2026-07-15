"""CPU-only clean-background benchmark with known synthetic ground truth.

This module deliberately measures only a background generator's contract:

* inside the supplied removal mask, compare its generated pixels with a known
  clean background;
* outside that mask, require byte-exact preservation of the composite input.

The cases are synthetic so the expected background is available without a
second model or a GPU.  They are useful for selecting and regression-testing
background removal routes; they are not a claim that a route will work on all
real advertising artwork.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


BACKGROUND_FAMILIES = ("flat", "gradient", "texture", "photo_like")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
BAKEOFF_MODES = ("big-lama", "flux_comfy", "powerpaint", "opencv")


@dataclass(frozen=True)
class AcceptanceThresholds:
    """Explicit, per-case acceptance limits for generated clean backgrounds."""

    max_inside_mae: float = 8.0
    min_inside_psnr_db: float = 30.0
    min_inside_ssim: float = 0.90
    max_outside_changed_pixels: int = 0


@dataclass(frozen=True)
class SyntheticCase:
    """One paired fixture: clean scene, composite input, and true foreground mask."""

    case_id: str
    family: str
    seed: int
    foreground: str
    clean_background: np.ndarray
    composite_input: np.ndarray
    removal_mask: np.ndarray


def _as_rgb_array(value: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    """Load an RGB uint8 image while rejecting ambiguous array shapes."""
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if isinstance(value, Image.Image):
        return np.asarray(value.convert("RGB"), dtype=np.uint8).copy()
    array = np.asarray(value)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise ValueError("image must be HxW, HxWx3, or HxWx4")
    if array.shape[2] == 4:
        array = array[..., :3]
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("image arrays must use integer pixel values")
    return np.clip(array, 0, 255).astype(np.uint8, copy=True)


def _as_mask(mask: str | Path | Image.Image | np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if isinstance(mask, (str, Path)):
        with Image.open(mask) as image:
            value = np.asarray(image.convert("L"), dtype=np.uint8)
    elif isinstance(mask, Image.Image):
        value = np.asarray(mask.convert("L"), dtype=np.uint8)
    else:
        value = np.asarray(mask)
        if value.ndim == 3:
            value = value[..., 0]
    if value.ndim != 2 or value.shape != shape:
        raise ValueError(f"mask must have shape {shape}, got {value.shape}")
    return value > 0


def _save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_as_rgb_array(image)).save(path)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(path)


def _resize_noise(rng: np.random.Generator, size: tuple[int, int], grid: tuple[int, int]) -> np.ndarray:
    """Smooth deterministic RGB noise without a vision-model dependency."""
    width, height = size
    tiny = rng.integers(0, 256, size=(grid[1], grid[0], 3), dtype=np.uint8)
    return np.asarray(
        Image.fromarray(tiny).resize((width, height), Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )


def _background(family: str, size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Create backgrounds with distinct reconstruction failure modes."""
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    fx = xx / max(1, width - 1)
    fy = yy / max(1, height - 1)

    if family == "flat":
        color = rng.integers(32, 224, size=3, dtype=np.uint8)
        return np.broadcast_to(color, (height, width, 3)).copy()

    if family == "gradient":
        start = rng.integers(16, 150, size=3)
        end = rng.integers(100, 240, size=3)
        blend = (0.62 * fx + 0.38 * fy)[..., None]
        return np.clip(start * (1.0 - blend) + end * blend, 0, 255).astype(np.uint8)

    if family == "texture":
        base = _background("gradient", size, rng).astype(np.float64)
        coarse = _resize_noise(rng, size, (max(3, width // 56), max(3, height // 56))).astype(np.float64)
        fine = rng.normal(0.0, 8.0, size=(height, width, 1))
        return np.clip(0.70 * base + 0.30 * coarse + fine, 0, 255).astype(np.uint8)

    if family == "photo_like":
        base = Image.fromarray(_background("gradient", size, rng))
        paint = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(paint)
        for _ in range(14):
            radius_x = int(rng.integers(max(14, width // 18), max(20, width // 4)))
            radius_y = int(rng.integers(max(14, height // 16), max(20, height // 3)))
            x = int(rng.integers(-radius_x, width))
            y = int(rng.integers(-radius_y, height))
            color = tuple(int(value) for value in rng.integers(0, 256, size=3)) + (int(rng.integers(28, 105)),)
            draw.ellipse((x, y, x + radius_x * 2, y + radius_y * 2), fill=color)
        paint = paint.filter(ImageFilter.GaussianBlur(radius=max(4, min(width, height) // 28)))
        mixed = Image.alpha_composite(base.convert("RGBA"), paint).convert("RGB")
        array = np.asarray(mixed, dtype=np.float64)
        array += rng.normal(0.0, 2.5, size=array.shape)
        return np.clip(array, 0, 255).astype(np.uint8)

    raise ValueError(f"unknown background family: {family}")


def _foreground(kind: str, size: tuple[int, int], rng: np.random.Generator) -> Image.Image:
    """Draw a layered, opaque-enough pasted foreground with a real alpha matte."""
    width, height = size
    foreground = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(foreground)
    font = ImageFont.load_default()
    accent = tuple(int(value) for value in rng.integers(35, 220, size=3)) + (255,)
    dark = (18, 24, 38, 255)

    if kind == "button":
        radius = max(10, min(width, height) // 8)
        draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=accent)
        text = "SHOP NOW"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        draw.text(((width - text_width) / 2, (height - text_height) / 2 - 1), text, font=font, fill=(255, 255, 255, 255))
        return foreground

    if kind == "card":
        radius = max(12, min(width, height) // 10)
        draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=radius, fill=(248, 249, 252, 248))
        draw.rounded_rectangle((8, 8, width - 9, height - 9), radius=max(8, radius - 4), outline=accent, width=3)
        badge = max(16, min(width, height) // 4)
        draw.ellipse((16, 16, 16 + badge, 16 + badge), fill=accent)
        draw.text((16, 22 + badge), "New drop", font=font, fill=dark)
        draw.rectangle((16, height - 34, max(17, width - 18), height - 20), fill=(45, 55, 72, 210))
        return foreground

    if kind == "badge":
        radius = min(width, height) // 2 - 2
        cx, cy = width // 2, height // 2
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=accent)
        draw.ellipse((cx - radius + 7, cy - radius + 7, cx + radius - 7, cy + radius - 7), outline=(255, 255, 255, 255), width=3)
        draw.text((cx - 13, cy - 5), "20%", font=font, fill=(255, 255, 255, 255))
        return foreground

    raise ValueError(f"unknown foreground kind: {kind}")


def _paste_foreground(clean_background: np.ndarray, foreground: Image.Image, position: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Alpha-composite a foreground and retain every nonzero-alpha foreground pixel."""
    height, width = clean_background.shape[:2]
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    canvas.alpha_composite(foreground, dest=position)
    alpha = np.asarray(canvas.getchannel("A"), dtype=np.uint8)
    composite = Image.alpha_composite(Image.fromarray(clean_background).convert("RGBA"), canvas)
    return np.asarray(composite.convert("RGB"), dtype=np.uint8), alpha > 0


def generate_synthetic_cases(
    *,
    cases_per_family: int = 2,
    size: tuple[int, int] = (384, 256),
    seed: int = 20260715,
    families: Sequence[str] = BACKGROUND_FAMILIES,
) -> list[SyntheticCase]:
    """Create deterministic paired fixtures; no downloaded assets or GPU is required."""
    if cases_per_family < 1:
        raise ValueError("cases_per_family must be at least 1")
    width, height = size
    if width < 96 or height < 96:
        raise ValueError("synthetic case size must be at least 96x96")
    invalid = [family for family in families if family not in BACKGROUND_FAMILIES]
    if invalid:
        raise ValueError(f"unknown background family: {', '.join(invalid)}")

    master = np.random.default_rng(seed)
    cases: list[SyntheticCase] = []
    foregrounds = ("button", "card", "badge")
    for family in families:
        for index in range(cases_per_family):
            case_seed = int(master.integers(0, np.iinfo(np.uint32).max, endpoint=True))
            rng = np.random.default_rng(case_seed)
            clean = _background(family, size, rng)
            kind = foregrounds[(index + BACKGROUND_FAMILIES.index(family)) % len(foregrounds)]
            if kind == "button":
                fg_size = (int(width * 0.38), int(height * 0.18))
            elif kind == "card":
                fg_size = (int(width * 0.40), int(height * 0.46))
            else:
                side = int(min(width, height) * 0.28)
                fg_size = (side, side)
            foreground = _foreground(kind, fg_size, rng)
            max_x = max(1, width - fg_size[0] - 12)
            max_y = max(1, height - fg_size[1] - 12)
            position = (int(rng.integers(8, max_x + 1)), int(rng.integers(8, max_y + 1)))
            composite, mask = _paste_foreground(clean, foreground, position)
            cases.append(
                SyntheticCase(
                    case_id=f"{family}-{index + 1:02d}",
                    family=family,
                    seed=case_seed,
                    foreground=kind,
                    clean_background=clean,
                    composite_input=composite,
                    removal_mask=mask,
                )
            )
    return cases


def _masked_ssim(expected: np.ndarray, actual: np.ndarray, mask: np.ndarray) -> float:
    """Global SSIM calculated exclusively from masked pixels and RGB channels."""
    a = expected[mask].astype(np.float64).reshape(-1)
    b = actual[mask].astype(np.float64).reshape(-1)
    if a.size == 0:
        return 1.0
    mu_a, mu_b = float(a.mean()), float(b.mean())
    var_a, var_b = float(a.var()), float(b.var())
    covariance = float(((a - mu_a) * (b - mu_b)).mean())
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    if denominator == 0:
        return 1.0
    return float(max(-1.0, min(1.0, ((2 * mu_a * mu_b + c1) * (2 * covariance + c2)) / denominator)))


def evaluate_background(
    clean_background: str | Path | Image.Image | np.ndarray,
    composite_input: str | Path | Image.Image | np.ndarray,
    removal_mask: str | Path | Image.Image | np.ndarray,
    generated_background: str | Path | Image.Image | np.ndarray,
    thresholds: AcceptanceThresholds = AcceptanceThresholds(),
    acceptance_failure_reasons: Sequence[str] = (),
) -> dict:
    """Score one generated background against known ground truth.

    The inside score never includes preserved pixels.  The separate outside gate
    is deliberately byte exact: a candidate cannot trade exterior damage for a
    better inside score.
    """
    expected = _as_rgb_array(clean_background)
    composite = _as_rgb_array(composite_input)
    generated = _as_rgb_array(generated_background)
    if composite.shape != expected.shape or generated.shape != expected.shape:
        raise ValueError("clean, composite, and generated images must have identical dimensions")
    mask = _as_mask(removal_mask, expected.shape[:2])
    if not bool(mask.any()):
        raise ValueError("removal mask cannot be empty")

    inside_delta = np.abs(generated.astype(np.int16) - expected.astype(np.int16))
    values = inside_delta[mask].astype(np.float64)
    mae = float(values.mean())
    rmse = float(math.sqrt(float(np.square(values).mean())))
    psnr = 99.0 if rmse == 0.0 else float(min(99.0, 20.0 * math.log10(255.0 / rmse)))
    ssim = _masked_ssim(expected, generated, mask)

    outside = ~mask
    outside_delta = np.abs(generated.astype(np.int16) - composite.astype(np.int16))
    outside_changed = np.any(outside_delta != 0, axis=2) & outside
    outside_pixels = int(outside.sum())
    outside_changed_pixels = int(outside_changed.sum())
    outside_changed_ratio = float(outside_changed_pixels / outside_pixels) if outside_pixels else 0.0
    max_outside_delta = int(outside_delta[outside].max()) if outside_pixels else 0

    reasons: list[str] = []
    if mae > thresholds.max_inside_mae:
        reasons.append("inside-mae")
    if psnr < thresholds.min_inside_psnr_db:
        reasons.append("inside-psnr")
    if ssim < thresholds.min_inside_ssim:
        reasons.append("inside-ssim")
    if outside_changed_pixels > thresholds.max_outside_changed_pixels:
        reasons.append("outside-mask")
    for reason in acceptance_failure_reasons:
        if reason and reason not in reasons:
            reasons.append(str(reason))
    return {
        "mask_pixels": int(mask.sum()),
        "inside": {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
            "psnr_db": round(psnr, 6),
            "ssim": round(ssim, 6),
        },
        "outside": {
            "pixels": outside_pixels,
            "changed_pixels": outside_changed_pixels,
            "changed_ratio": round(outside_changed_ratio, 8),
            "max_channel_delta": max_outside_delta,
            "exact": outside_changed_pixels == 0,
        },
        "inside_quality_ok": not any(reason.startswith("inside-") for reason in reasons),
        "outside_mask_ok": outside_changed_pixels <= thresholds.max_outside_changed_pixels,
        "accepted": not reasons,
        "failure_reasons": reasons,
    }


def _difference_image(expected: np.ndarray, actual: np.ndarray, active: np.ndarray) -> np.ndarray:
    difference = np.abs(expected.astype(np.int16) - actual.astype(np.int16)).astype(np.uint8)
    return np.where(active[..., None], difference, 0).astype(np.uint8)


def baseline_inpaint(
    composite_input: np.ndarray,
    removal_mask: np.ndarray,
    clean_background: np.ndarray,
    method: str = "telea",
) -> np.ndarray:
    """Provide CPU baselines and an oracle smoke-test mode.

    ``oracle`` exists only to verify the benchmark itself.  It must never be
    reported as an inpainting result.  The OpenCV baselines explicitly restore
    exterior pixels so they exercise the intended in-mask quality comparison.
    """
    composite = _as_rgb_array(composite_input)
    clean = _as_rgb_array(clean_background)
    mask = _as_mask(removal_mask, composite.shape[:2])
    if method == "oracle":
        return clean.copy()
    if method == "copy-input":
        return composite.copy()
    if method not in {"telea", "ns"}:
        raise ValueError("method must be one of oracle, copy-input, telea, or ns")
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - requirements enforce this in normal runs
        raise RuntimeError("OpenCV is required for the telea/ns CPU baseline") from exc
    cv_method = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
    generated = cv2.inpaint(composite, np.where(mask, 255, 0).astype(np.uint8), 3, cv_method)
    generated = _as_rgb_array(generated)
    generated[~mask] = composite[~mask]
    return generated


def _case_metadata(case: SyntheticCase) -> dict:
    return {
        "id": case.case_id,
        "family": case.family,
        "seed": case.seed,
        "foreground": case.foreground,
        "size": {"width": int(case.clean_background.shape[1]), "height": int(case.clean_background.shape[0])},
        "mask_pixels": int(case.removal_mask.sum()),
    }


def write_case_inputs(case: SyntheticCase, case_dir: str | Path) -> dict:
    """Save canonical inputs so another backend can be run on identical pairs."""
    directory = Path(case_dir)
    directory.mkdir(parents=True, exist_ok=True)
    _save_rgb(directory / "clean_background.png", case.clean_background)
    _save_rgb(directory / "composite_input.png", case.composite_input)
    _save_mask(directory / "removal_mask.png", case.removal_mask)
    metadata = _case_metadata(case)
    (directory / "input.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def _external_candidate(candidate_dir: Path, case_id: str) -> Path:
    matches = [candidate_dir / f"{case_id}{suffix}" for suffix in IMAGE_EXTENSIONS]
    existing = [path for path in matches if path.is_file()]
    if not existing:
        raise FileNotFoundError(
            f"missing generated background for {case_id}; expected one of "
            + ", ".join(str(path) for path in matches)
        )
    if len(existing) > 1:
        raise ValueError(f"ambiguous generated backgrounds for {case_id}: {', '.join(str(path) for path in existing)}")
    return existing[0]


def load_bakeoff_config(path: str | Path) -> dict:
    """Load a JSON or YAML config without importing the full pipeline.

    The bakeoff deliberately only overrides ``inpaint.mode``.  Everything else
    (Comfy endpoint, PowerPaint adapter, fallback policy, model paths) remains
    the actual RTX configuration supplied by the operator.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"bakeoff config does not exist: {config_path}")
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        value = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - pyyaml is a CPU requirement
            raise RuntimeError("PyYAML is required to load YAML bakeoff config") from exc
        value = yaml.safe_load(text)
    else:
        raise ValueError("bakeoff config must be a .json, .yaml, or .yml file")
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("bakeoff config must contain a top-level object/mapping")
    return value


def _requested_mode(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {"big_lama": "big-lama", "flux_comfy": "flux_comfy", "powerpaint": "powerpaint", "opencv": "opencv"}
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"requested backend must be one of: {', '.join(BAKEOFF_MODES)}") from exc


def _canonical_backend(value: object) -> str:
    """Normalize inpaint diagnostics while retaining the raw selected value in reports."""
    raw = str(value or "").strip().lower().replace("_", "-")
    if raw in {"big-lama", "simple-lama", "lama"}:
        return "big-lama"
    if raw in {"flux", "flux-comfy"}:
        return "flux-comfy"
    if raw in {"powerpaint", "power-paint"}:
        return "powerpaint"
    if raw in {"opencv", "cv2"} or raw.startswith("opencv-"):
        return "opencv"
    return raw or "unknown"


def _backend_matches(requested_mode: str, selected_backend: object) -> bool:
    return _canonical_backend(requested_mode) == _canonical_backend(selected_backend)


def _bakeoff_cfg(base_cfg: Mapping, requested_mode: str) -> dict:
    """Clone an operator config and make the requested inpaint mode unambiguous."""
    cfg = deepcopy(dict(base_cfg))
    existing = cfg.get("inpaint")
    if existing is None:
        inpaint_cfg: dict = {}
    elif isinstance(existing, Mapping):
        inpaint_cfg = dict(existing)
    else:
        raise ValueError("config.inpaint must be a mapping when present")
    inpaint_cfg["mode"] = _requested_mode(requested_mode)
    cfg["inpaint"] = inpaint_cfg
    return cfg


def _mean(rows: Iterable[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return round(sum(values) / len(values), 6) if values else None


def _summary_markdown(report: Mapping) -> str:
    summary = report["summary"]
    bakeoff = bool(report.get("bakeoff"))
    lines = [
        "# Clean-background benchmark",
        "",
        f"Acceptance: **{'PASS' if summary['accepted'] else 'FAIL'}**  |  "
        f"Cases: {summary['cases']}  |  Accepted: {summary['accepted_cases']}  |  "
        f"Outside-mask gate: {summary['outside_mask_passing_cases']}/{summary['cases']}",
        "",
        (
            "| case | background | foreground | requested backend | selected backend | inside MAE | inside PSNR | inside SSIM | outside changed | acceptance | reasons |"
            if bakeoff else
            "| case | background | foreground | inside MAE | inside PSNR | inside SSIM | outside changed | acceptance | reasons |"
        ),
        (
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |"
            if bakeoff else
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |"
        ),
    ]
    for case in report["cases"]:
        metrics = case["metrics"]
        inside = metrics["inside"]
        outside = metrics["outside"]
        prefix = f"| {case['id']} | {case['family']} | {case['foreground']}"
        if bakeoff:
            backend = case["backend"]
            prefix += f" | {backend['requested_mode']} | {backend['selected_raw'] or backend['selected']}"
        lines.append(
            f"{prefix} | {inside['mae']:.3f} | {inside['psnr_db']:.3f} | {inside['ssim']:.4f} | "
            f"{outside['changed_pixels']} | {'pass' if metrics['accepted'] else 'fail'} | "
            f"{', '.join(metrics['failure_reasons']) or '—'} |"
        )
    lines.extend([
        "",
        "Inside metrics compare only known removed foreground pixels against the saved clean background.",
        "The outside-mask gate is byte exact against `composite_input.png`; it is not included in inside scores.",
        *( ["A requested backend must match the backend selected by `src.inpaint`; substitutions fail unless explicitly allowed."] if bakeoff else [] ),
        "",
    ])
    return "\n".join(lines)


def run_synthetic_benchmark(
    output_dir: str | Path,
    *,
    cases_per_family: int = 2,
    size: tuple[int, int] = (384, 256),
    seed: int = 20260715,
    method: str = "telea",
    candidate_dir: str | Path | None = None,
    thresholds: AcceptanceThresholds = AcceptanceThresholds(),
    families: Sequence[str] = BACKGROUND_FAMILIES,
) -> dict:
    """Generate paired cases, score a backend, and save all review artifacts.

    When ``candidate_dir`` is supplied it must contain one generated file named
    ``<case-id>.png`` (or jpg/jpeg/webp) for every synthetic case.  Those files
    are evaluated untouched, so exterior damage cannot be hidden by the harness.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = generate_synthetic_cases(
        cases_per_family=cases_per_family,
        size=size,
        seed=seed,
        families=families,
    )
    external = Path(candidate_dir) if candidate_dir is not None else None
    case_reports: list[dict] = []
    for case in cases:
        case_dir = output / "cases" / case.case_id
        metadata = write_case_inputs(case, case_dir)
        candidate_source = None
        if external is None:
            generated = baseline_inpaint(
                case.composite_input, case.removal_mask, case.clean_background, method=method
            )
        else:
            candidate_source = _external_candidate(external, case.case_id)
            generated = _as_rgb_array(candidate_source)
        metrics = evaluate_background(
            case.clean_background,
            case.composite_input,
            case.removal_mask,
            generated,
            thresholds,
        )
        _save_rgb(case_dir / "generated_background.png", generated)
        _save_rgb(
            case_dir / "inside_diff.png",
            _difference_image(case.clean_background, generated, case.removal_mask),
        )
        _save_rgb(
            case_dir / "outside_diff.png",
            _difference_image(case.composite_input, generated, ~case.removal_mask),
        )
        case_report = {
            **metadata,
            "candidate": {
                "kind": "external" if candidate_source else "cpu-baseline",
                "method": None if candidate_source else method,
                "source": str(candidate_source.resolve()) if candidate_source else None,
            },
            "metrics": metrics,
            "artifacts": {
                "clean_background": "clean_background.png",
                "composite_input": "composite_input.png",
                "removal_mask": "removal_mask.png",
                "generated_background": "generated_background.png",
                "inside_diff": "inside_diff.png",
                "outside_diff": "outside_diff.png",
            },
        }
        (case_dir / "metrics.json").write_text(json.dumps(case_report, indent=2) + "\n", encoding="utf-8")
        case_reports.append(case_report)

    inside_rows = [item["metrics"]["inside"] for item in case_reports]
    failures: dict[str, int] = {}
    for item in case_reports:
        for reason in item["metrics"]["failure_reasons"]:
            failures[reason] = failures.get(reason, 0) + 1
    summary = {
        "cases": len(case_reports),
        "accepted_cases": sum(1 for item in case_reports if item["metrics"]["accepted"]),
        "rejected_cases": sum(1 for item in case_reports if not item["metrics"]["accepted"]),
        "outside_mask_passing_cases": sum(1 for item in case_reports if item["metrics"]["outside_mask_ok"]),
        "inside_quality_passing_cases": sum(1 for item in case_reports if item["metrics"]["inside_quality_ok"]),
        "mean_inside_mae": _mean(inside_rows, "mae"),
        "mean_inside_psnr_db": _mean(inside_rows, "psnr_db"),
        "mean_inside_ssim": _mean(inside_rows, "ssim"),
        "worst_inside_mae": round(max(row["mae"] for row in inside_rows), 6),
        "accepted": bool(case_reports) and all(item["metrics"]["accepted"] for item in case_reports),
        "failure_counts": failures,
    }
    report = {
        "version": 1,
        "kind": "synthetic-known-ground-truth-clean-background",
        "output": str(output.resolve()),
        "generator": {
            "families": list(families),
            "cases_per_family": cases_per_family,
            "size": {"width": size[0], "height": size[1]},
            "seed": seed,
        },
        "candidate": {
            "kind": "external" if external is not None else "cpu-baseline",
            "method": None if external is not None else method,
            "directory": str(external.resolve()) if external is not None else None,
        },
        "thresholds": asdict(thresholds),
        "cases": case_reports,
        "summary": summary,
    }
    (output / "manifest.json").write_text(
        json.dumps({"version": 1, "cases": [_case_metadata(case) for case in cases]}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(_summary_markdown(report), encoding="utf-8")
    return report


def run_inpaint_bakeoff(
    output_dir: str | Path,
    *,
    config: Mapping | str | Path,
    requested_backend: str,
    cases_per_family: int = 2,
    size: tuple[int, int] = (384, 256),
    seed: int = 20260715,
    thresholds: AcceptanceThresholds = AcceptanceThresholds(),
    families: Sequence[str] = BACKGROUND_FAMILIES,
    allow_backend_substitution: bool = False,
) -> dict:
    """Bake the repo's real ``src.inpaint`` backends against known targets.

    One requested mode is forced into a cloned operator config for every case.
    ``src.inpaint.inpaint_array`` remains responsible for actual model selection,
    including any fallback.  The returned selection is recorded and turns an
    otherwise-good result into a failure if it is not the requested backend,
    unless the caller passed ``allow_backend_substitution=True`` deliberately.
    """
    if isinstance(config, (str, Path)):
        base_cfg = load_bakeoff_config(config)
        config_path = Path(config)
    elif isinstance(config, Mapping):
        base_cfg = dict(config)
        config_path = None
    else:
        raise TypeError("config must be a mapping or a YAML/JSON path")
    mode = _requested_mode(requested_backend)
    active_cfg = _bakeoff_cfg(base_cfg, mode)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = generate_synthetic_cases(
        cases_per_family=cases_per_family,
        size=size,
        seed=seed,
        families=families,
    )

    # Import here rather than at module load so the ordinary CPU fixture path
    # never imports any GPU/runtime adapter. It also makes backend tests able to
    # monkeypatch the exact repository seam.
    from src.inpaint import inpaint_array

    case_reports: list[dict] = []
    for case in cases:
        case_dir = output / "cases" / case.case_id
        metadata = write_case_inputs(case, case_dir)
        selected_backend = "error"
        diagnostics: dict = {}
        backend_error = None
        generated = case.composite_input.copy()
        try:
            generated, selected_backend, diagnostics = inpaint_array(
                case.composite_input,
                np.where(case.removal_mask, 255, 0).astype(np.uint8),
                active_cfg,
                return_diagnostics=True,
            )
            generated = _as_rgb_array(generated)
        except Exception as exc:  # A real backend outage is benchmark evidence, not a crash.
            backend_error = f"{type(exc).__name__}: {exc}"
        if not isinstance(diagnostics, Mapping):
            diagnostics = {"raw_diagnostics": str(diagnostics)}
        route = diagnostics.get("backend_route") if isinstance(diagnostics, Mapping) else None
        route = route if isinstance(route, Mapping) else {}
        selected_raw = str(route.get("selected") or selected_backend)
        requested_raw = str(route.get("requested") or mode)
        selection_matches = backend_error is None and _backend_matches(mode, selected_raw)
        substituted = backend_error is None and not selection_matches
        extra_failures = ["backend-error"] if backend_error is not None else []
        if substituted and not allow_backend_substitution:
            extra_failures.append("backend-substitution")
        metrics = evaluate_background(
            case.clean_background,
            case.composite_input,
            case.removal_mask,
            generated,
            thresholds,
            acceptance_failure_reasons=extra_failures,
        )
        _save_rgb(case_dir / "generated_background.png", generated)
        _save_rgb(
            case_dir / "inside_diff.png",
            _difference_image(case.clean_background, generated, case.removal_mask),
        )
        _save_rgb(
            case_dir / "outside_diff.png",
            _difference_image(case.composite_input, generated, ~case.removal_mask),
        )
        backend = {
            "requested": _canonical_backend(mode),
            "requested_mode": requested_raw,
            "selected": _canonical_backend(selected_raw) if backend_error is None else "error",
            "selected_raw": selected_raw if backend_error is None else None,
            "selected_class": route.get("selected_class") or diagnostics.get("backend_class"),
            "matches_requested": selection_matches,
            "substituted": substituted,
            "substitution_allowed": bool(allow_backend_substitution),
            "error": backend_error,
            "diagnostics": diagnostics,
        }
        case_report = {
            **metadata,
            "backend": backend,
            "metrics": metrics,
            "artifacts": {
                "clean_background": "clean_background.png",
                "composite_input": "composite_input.png",
                "removal_mask": "removal_mask.png",
                "generated_background": "generated_background.png",
                "inside_diff": "inside_diff.png",
                "outside_diff": "outside_diff.png",
            },
        }
        (case_dir / "metrics.json").write_text(json.dumps(case_report, indent=2) + "\n", encoding="utf-8")
        case_reports.append(case_report)

    inside_rows = [item["metrics"]["inside"] for item in case_reports]
    failures: dict[str, int] = {}
    for item in case_reports:
        for reason in item["metrics"]["failure_reasons"]:
            failures[reason] = failures.get(reason, 0) + 1
    summary = {
        "cases": len(case_reports),
        "accepted_cases": sum(1 for item in case_reports if item["metrics"]["accepted"]),
        "rejected_cases": sum(1 for item in case_reports if not item["metrics"]["accepted"]),
        "outside_mask_passing_cases": sum(1 for item in case_reports if item["metrics"]["outside_mask_ok"]),
        "inside_quality_passing_cases": sum(1 for item in case_reports if item["metrics"]["inside_quality_ok"]),
        "backend_match_cases": sum(1 for item in case_reports if item["backend"]["matches_requested"]),
        "backend_substitution_cases": sum(1 for item in case_reports if item["backend"]["substituted"]),
        "backend_error_cases": sum(1 for item in case_reports if item["backend"]["error"] is not None),
        "mean_inside_mae": _mean(inside_rows, "mae"),
        "mean_inside_psnr_db": _mean(inside_rows, "psnr_db"),
        "mean_inside_ssim": _mean(inside_rows, "ssim"),
        "worst_inside_mae": round(max(row["mae"] for row in inside_rows), 6),
        "accepted": bool(case_reports) and all(item["metrics"]["accepted"] for item in case_reports),
        "failure_counts": failures,
    }
    config_digest = hashlib.sha256(json.dumps(base_cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    report = {
        "version": 1,
        "kind": "synthetic-known-ground-truth-inpaint-bakeoff",
        "output": str(output.resolve()),
        "generator": {
            "families": list(families),
            "cases_per_family": cases_per_family,
            "size": {"width": size[0], "height": size[1]},
            "seed": seed,
        },
        "bakeoff": {
            "requested_backend": _canonical_backend(mode),
            "requested_mode": mode,
            "allow_backend_substitution": bool(allow_backend_substitution),
            "config_path": str(config_path.resolve()) if config_path is not None else None,
            "config_sha256": config_digest,
        },
        "thresholds": asdict(thresholds),
        "cases": case_reports,
        "summary": summary,
    }
    (output / "manifest.json").write_text(
        json.dumps({"version": 1, "cases": [_case_metadata(case) for case in cases]}, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "summary.md").write_text(_summary_markdown(report), encoding="utf-8")
    return report
