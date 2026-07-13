#!/usr/bin/env python3
"""Download the FLUX.1 Fill (quantized GGUF) inpaint stack into a ComfyUI install.

Fetches, via huggingface_hub, the models needed by workflows/flux_fill_inpaint_api.json:

  * FLUX.1 Fill dev, quantized GGUF   -> models/unet/       (Q4_K_M ~6.8GB by default)
  * t5xxl_fp8_e4m3fn text encoder     -> models/clip/       (~4.9GB, fp8 = 16GB-VRAM safe)
  * clip_l text encoder               -> models/clip/       (~246MB)
  * Flux VAE (ae.safetensors)         -> models/vae/        (~335MB)
  * FLUX.1 Turbo-Alpha LoRA (8-step)  -> models/loras/      (~694MB)

No tokens are embedded: huggingface_hub uses your cached login (`huggingface-cli login`)
only if a repo is gated.  All defaults below are ungated public repos.

Usage:
    python scripts/setup_flux_inpaint.py --comfy-dir "C:/ComfyUI" [--quant Q4_K_M]
    python scripts/setup_flux_inpaint.py --list        # print the plan, download nothing

If --comfy-dir is omitted the script prints the exact target subfolders and exits so you
can wire it up by hand.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

# key -> (repo_id, filename-template, [target subdir(s)], local filename, approx size, gated)
# The GGUF filename embeds the quant, filled in from --quant. GGUF loaders scan both
# models/unet and models/diffusion_models; CLIP loaders scan models/clip and
# models/text_encoders — we write the first of each and note the alternatives.
MODELS = [
    {
        "key": "unet_gguf",
        "repo": "YarvixPA/FLUX.1-Fill-dev-gguf",
        "remote": "flux1-fill-dev-{quant}.gguf",
        "subdir": "unet",
        "alt_subdir": "diffusion_models",
        "local": "flux1-fill-dev-{quant}.gguf",
        "size": "~6.8 GB (Q4_K_M) / ~8.4 GB (Q5_K_M)",
        "gated": False,
    },
    {
        "key": "t5xxl",
        "repo": "comfyanonymous/flux_text_encoders",
        "remote": "t5xxl_fp8_e4m3fn.safetensors",
        "subdir": "clip",
        "alt_subdir": "text_encoders",
        "local": "t5xxl_fp8_e4m3fn.safetensors",
        "size": "~4.9 GB",
        "gated": False,
    },
    {
        "key": "clip_l",
        "repo": "comfyanonymous/flux_text_encoders",
        "remote": "clip_l.safetensors",
        "subdir": "clip",
        "alt_subdir": "text_encoders",
        "local": "clip_l.safetensors",
        "size": "~246 MB",
        "gated": False,
    },
    {
        "key": "vae",
        # FLUX.1-schnell is Apache-2.0 and ungated; its ae.safetensors is the shared Flux VAE.
        "repo": "black-forest-labs/FLUX.1-schnell",
        "remote": "ae.safetensors",
        "subdir": "vae",
        "alt_subdir": None,
        "local": "ae.safetensors",
        "size": "~335 MB",
        "gated": False,
    },
    {
        "key": "lora",
        "repo": "alimama-creative/FLUX.1-Turbo-Alpha",
        "remote": "diffusion_pytorch_model.safetensors",
        "subdir": "loras",
        "alt_subdir": None,
        "local": "flux1-turbo-alpha.safetensors",
        "size": "~694 MB",
        "gated": False,
    },
]

GGUF_NODE_HINT = (
    "The UnetLoaderGGUF node needs the ComfyUI-GGUF custom node pack:\n"
    "    cd <ComfyUI>/custom_nodes\n"
    "    git clone https://github.com/city96/ComfyUI-GGUF\n"
    "    <ComfyUI-python> -m pip install gguf\n"
)


def _plan(quant: str):
    for m in MODELS:
        yield {
            **m,
            "remote": m["remote"].format(quant=quant),
            "local": m["local"].format(quant=quant),
        }


def print_plan(quant: str, comfy_dir: str | None):
    print("FLUX.1 Fill inpaint stack - download plan")
    print("=" * 64)
    for m in _plan(quant):
        dest = f"models/{m['subdir']}/{m['local']}"
        alt = f" (or models/{m['alt_subdir']}/)" if m["alt_subdir"] else ""
        print(f"  [{m['key']:9}] {m['repo']}")
        print(f"              {m['remote']}  {m['size']}")
        print(f"           -> {dest}{alt}")
    print("=" * 64)
    print(GGUF_NODE_HINT)
    if not comfy_dir:
        print("No --comfy-dir given: nothing downloaded.")
        print("Re-run with, e.g.:")
        print('    python scripts/setup_flux_inpaint.py --comfy-dir "C:/ComfyUI"')
        print("Target subfolders (create if missing):")
        print("    <ComfyUI>/models/unet   (GGUF; models/diffusion_models also works)")
        print("    <ComfyUI>/models/clip   (t5xxl + clip_l; models/text_encoders also works)")
        print("    <ComfyUI>/models/vae")
        print("    <ComfyUI>/models/loras")


def download(comfy_dir: str, quant: str) -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run:")
        print("    .venv/Scripts/python -m pip install huggingface_hub")
        return 2

    failures = 0
    for m in _plan(quant):
        target_dir = os.path.join(comfy_dir, "models", m["subdir"])
        os.makedirs(target_dir, exist_ok=True)
        final_path = os.path.join(target_dir, m["local"])
        if os.path.isfile(final_path) and os.path.getsize(final_path) > 0:
            print(f"[skip] {m['local']} already present in {target_dir}")
            continue
        print(f"[get ] {m['repo']} :: {m['remote']}  ({m['size']}) -> {target_dir}")
        try:
            fetched = hf_hub_download(
                repo_id=m["repo"],
                filename=m["remote"],
                local_dir=target_dir,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue with the rest
            failures += 1
            print(f"       FAILED: {exc}")
            low = str(exc).lower()
            if "gated" in low or "401" in low or "403" in low or "authentication" in low:
                print(f"       {m['repo']} looks gated. Accept its license on huggingface.co,")
                print("       then run: huggingface-cli login   (no token is stored by this script)")
            continue
        # hf_hub_download preserves the remote filename; rename to the local target if needed.
        if os.path.basename(fetched) != m["local"]:
            try:
                shutil.copyfile(fetched, final_path)
                print(f"       saved as {m['local']}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"       rename FAILED: {exc}")

    print("-" * 64)
    print(GGUF_NODE_HINT)
    if failures:
        print(f"Done with {failures} failure(s). Re-run to retry the missing files.")
    else:
        print("All Flux Fill inpaint models are in place.")
        print("Point config at them:")
        print("    inpaint.mode: flux_comfy   # or keep 'auto' + inpaint.comfy.enabled: true")
        print(f"    inpaint.comfy.comfy_dir: {comfy_dir}")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfy-dir", default=os.environ.get("COMFYUI_DIR", ""),
                    help="Path to the ComfyUI install (contains a models/ folder).")
    ap.add_argument("--quant", default="Q6_K",
                    help="GGUF quant of Flux Fill (Q6_K best quality; Q5_K_S/Q4_K_S for tighter VRAM).")
    ap.add_argument("--list", action="store_true", help="Print the download plan and exit.")
    args = ap.parse_args()

    if args.list or not args.comfy_dir:
        print_plan(args.quant, args.comfy_dir or None)
        return 0

    comfy_dir = os.path.expandvars(os.path.expanduser(args.comfy_dir))
    if not os.path.isdir(comfy_dir):
        print(f"ERROR: --comfy-dir does not exist: {comfy_dir}")
        return 2
    print_plan(args.quant, comfy_dir)
    print()
    return download(comfy_dir, args.quant)


if __name__ == "__main__":
    sys.exit(main())
