import json, os, sys, tempfile
from PIL import Image
sys.path.insert(0, os.getcwd())
from src import figma_import as fi

R = r"runs/codex-targeted-002a/002_attached_5885519ba4359843"

# 1. original.png ICC
with Image.open(os.path.join(R, "original.png")) as im:
    icc = im.info.get("icc_profile")
    print("original.png mode=", im.mode, "icc_bytes=", len(icc) if icc else 0,
          "nonsrgb=", bool(icc) and (b"srgb" not in bytes(icc).lower()))

# 2. Stage into a temp inbox
tmp_inbox = tempfile.mkdtemp(prefix="inbox_test_")
cfg = {"figma": {"enabled": True, "mode": "plugin", "inbox": tmp_inbox,
                 "bridge_port": 8790, "stage_screenshot_sibling": True}}
res = fi.import_design(os.path.join(R, "design.json"), R, cfg)
print("STAGE ok=", res.get("ok"), "preflight=", res.get("preflight"),
      "pruned=", res.get("pruned_assets"))

manifest = json.load(open(os.path.join(tmp_inbox, "inbox.json"), encoding="utf-8"))
print("MANIFEST warnings:", len(manifest["summary"]["warnings"]))
for w in manifest["summary"]["warnings"]:
    print("   -", w.get("code"), w.get("layer_id"), "|", w.get("detail")[:70])
print("MANIFEST files (staged):")
for f in manifest["files"]:
    print("   ", f["path"])
print("PRUNED:", manifest.get("pruned_assets"))

# 3. verify screenshot proof is now sRGB
proof = os.path.join(tmp_inbox, "runs", manifest["doc_id"], "assets", "_screenshot_proof.png")
with Image.open(proof) as im:
    icc = im.info.get("icc_profile")
    print("staged _screenshot_proof.png icc_bytes=", len(icc) if icc else 0,
          "nonsrgb=", bool(icc) and (b"srgb" not in bytes(icc).lower()))
print("TMP_INBOX=", tmp_inbox)
