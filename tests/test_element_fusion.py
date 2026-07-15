"""Focused mask-aware fusion tests; no model/GPU required."""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import element_fusion


CANVAS = {"w": 100, "h": 100}


def _mask(x, y, w, h):
    out = np.zeros((100, 100), dtype=bool)
    out[y : y + h, x : x + w] = True
    return out


def _box(x, y, w, h):
    return {"x": x, "y": y, "w": w, "h": h}


def test_same_instance_from_three_sources_becomes_one_canonical_element(tmp_path):
    mask = _mask(20, 20, 30, 30)
    sam3 = {
        "elements": [
            {
                "id": "S0",
                "box": _box(20, 20, 30, 30),
                "role": "product",
                "kind": "photo-fragment",
                "score": 0.94,
                "_mask": mask,
                "provenance": {"mode": "box-refine", "residual_id": "R0"},
            }
        ]
    }
    residual = [
        {"id": "R0", "box": _box(20, 20, 30, 30), "kind": "photo-fragment", "_mask": mask}
    ]
    qwen = [
        {"id": "Q0", "box": _box(20, 20, 30, 30), "kind_hint": "product", "_mask": mask}
    ]

    fused = element_fusion.fuse(sam3, residual, qwen, CANVAS, run_dir=str(tmp_path))

    assert len(fused) == 1
    element = fused[0]
    assert element["id"] == "E000"
    assert element["role"] == "product"
    observations = element["provenance"]["observations"]
    assert len(observations) == 3
    assert len({o["key"] for o in observations}) == 3
    assert {o["source"] for o in observations} == {"sam3", "residual", "qwen"}
    assert element["provenance"]["nms"]["merged_count"] == 2
    assert os.path.exists(element["mask_path"])
    assert Image.open(element["mask_path"]).size == (30, 30)


def test_agreed_structural_group_metadata_survives_fusion():
    mask = _mask(20, 20, 30, 30)
    sam = {"id": "S0", "box": _box(20, 20, 30, 30), "role": "panel",
           "kind": "photo-fragment", "score": .94, "_mask": mask,
           "grid_group_id": "features", "row_index": 0, "column_index": 1}
    residual = {"id": "R0", "box": _box(20, 20, 30, 30), "role": "panel",
                "kind": "photo-fragment", "_mask": mask,
                "grid_group_id": "features", "row_index": 0, "column_index": 1}

    fused = element_fusion.fuse({"elements": [sam]}, [residual], [], CANVAS)

    assert fused[0]["grid_group_id"] == "features"
    assert fused[0]["row_index"] == 0
    assert fused[0]["column_index"] == 1


def test_conflicting_structural_group_metadata_is_discarded():
    mask = _mask(20, 20, 30, 30)
    sam = {"id": "S0", "box": _box(20, 20, 30, 30), "role": "panel",
           "kind": "photo-fragment", "score": .94, "_mask": mask,
           "grid_group_id": "left"}
    residual = {"id": "R0", "box": _box(20, 20, 30, 30), "role": "panel",
                "kind": "photo-fragment", "_mask": mask,
                "grid_group_id": "right"}

    fused = element_fusion.fuse({"elements": [sam]}, [residual], [], CANVAS)

    assert "grid_group_id" not in fused[0]


def test_meaningful_nested_icon_is_preserved_as_child(tmp_path):
    container = _mask(10, 10, 70, 50)
    icon = _mask(25, 22, 12, 12)
    sam3 = {
        "elements": [
            {
                "id": "S-container",
                "box": _box(10, 10, 70, 50),
                "role": "button",
                "kind": "shape",
                "score": 0.90,
                "_mask": container,
                "provenance": {"mode": "text-prompt", "prompt": "button"},
            },
            {
                "id": "S-icon",
                "box": _box(25, 22, 12, 12),
                "role": "icon",
                "kind": "icon",
                "score": 0.91,
                "_mask": icon,
                "provenance": {"mode": "text-prompt", "prompt": "icon"},
            },
        ]
    }

    fused = element_fusion.fuse(sam3, [], [], CANVAS, run_dir=str(tmp_path))

    assert len(fused) == 2  # containment is not treated as duplication
    parent = next(e for e in fused if e["role"] == "button")
    child = next(e for e in fused if e["role"] == "icon")
    assert child["parent_id"] == parent["id"]
    assert child["relationships"][0]["type"] == "nested-in"
    assert child["relationships"][0]["containment"] == 1.0


def test_duplicate_input_observation_key_is_counted_once():
    mask = _mask(5, 5, 10, 10)
    repeated = {
        "id": "S0",
        "box": _box(5, 5, 10, 10),
        "role": "logo",
        "kind": "icon",
        "score": 0.8,
        "_mask": mask,
        "provenance": {"mode": "text-prompt", "prompt": "logo"},
    }
    fused = element_fusion.fuse({"elements": [repeated, dict(repeated)]}, [], [], CANVAS)
    assert len(fused) == 1
    observations = fused[0]["provenance"]["observations"]
    assert [o["key"] for o in observations] == ["sam3:S0"]


def test_gap_band_area_ratio_still_links_shape_parent_to_child(tmp_path):
    """Regression for the 0.62-0.70 area_ratio gap: a fully-contained child whose area_ratio
    sits strictly between the old nested_max_area_ratio (0.62) and similar_area_ratio (0.70)
    must still be linked as a parent/child pair, not shipped as two unrelated top-level
    elements."""
    # Parent area = 50*20 = 1000. Child area = 32*20 = 640, fully inside parent's footprint.
    # area_ratio = 640/1000 = 0.64, containment = 1.0 -- squarely inside the former gap.
    parent_mask = _mask(10, 10, 50, 20)
    child_mask = _mask(10, 10, 32, 20)
    assert child_mask.sum() / parent_mask.sum() == 0.64

    sam3 = {
        "elements": [
            {
                "id": "S-card",
                "box": _box(10, 10, 50, 20),
                "role": "card",
                "kind": "shape",
                "score": 0.9,
                "_mask": parent_mask,
                "provenance": {"mode": "text-prompt", "prompt": "card"},
            },
            {
                "id": "S-icon",
                "box": _box(10, 10, 32, 20),
                "role": "icon",
                "kind": "icon",
                "score": 0.9,
                "_mask": child_mask,
                "provenance": {"mode": "text-prompt", "prompt": "icon"},
            },
        ]
    }

    fused = element_fusion.fuse(sam3, [], [], CANVAS, run_dir=str(tmp_path))

    assert len(fused) == 2  # kept separate, not collapsed as a duplicate
    parent = next(e for e in fused if e["role"] == "card")
    child = next(e for e in fused if e["role"] == "icon")
    assert child["parent_id"] == parent["id"]
    assert child["relationships"][0]["type"] == "nested-in"
    assert child["relationships"][0]["area_ratio"] == 0.64


def test_raster_role_parent_links_nested_product_child(tmp_path):
    """Regression: a photo (raster role) that fully contains a nested product must be linked
    as parent/child, not shipped as two unrelated overlapping top-level elements (duplicate
    ownership). _meaningful_parent previously only accepted kind=='shape' parents."""
    parent_mask = _mask(10, 10, 50, 20)
    child_mask = _mask(10, 10, 32, 20)

    sam3 = {
        "elements": [
            {
                "id": "S-photo",
                "box": _box(10, 10, 50, 20),
                "role": "photo",
                "kind": "photo-fragment",
                "score": 0.9,
                "_mask": parent_mask,
                "provenance": {"mode": "text-prompt", "prompt": "photo"},
            },
            {
                "id": "S-product",
                "box": _box(10, 10, 32, 20),
                "role": "product",
                "kind": "photo-fragment",
                "score": 0.9,
                "_mask": child_mask,
                "provenance": {"mode": "text-prompt", "prompt": "product"},
            },
        ]
    }

    fused = element_fusion.fuse(sam3, [], [], CANVAS, run_dir=str(tmp_path))

    assert len(fused) == 2
    parent = next(e for e in fused if e["role"] == "photo")
    child = next(e for e in fused if e["role"] == "product")
    assert child["parent_id"] == parent["id"]
    assert child["relationships"][0]["type"] == "nested-in"


def test_underscore_cluster_role_normalizes_before_parent_linking(tmp_path):
    parent_mask = _mask(8, 8, 64, 48)
    child_mask = _mask(20, 20, 18, 14)
    fused = element_fusion.fuse({"elements": [
        {"id": "panel", "box": _box(8, 8, 64, 48), "role": "ui_panel",
         "kind": "photo-fragment", "score": .95, "_mask": parent_mask,
         "provenance": {"mode": "text-prompt"}},
        {"id": "icon", "box": _box(20, 20, 18, 14), "role": "icon",
         "kind": "icon", "score": .95, "_mask": child_mask,
         "provenance": {"mode": "text-prompt"}},
    ]}, [], [], CANVAS, run_dir=str(tmp_path))
    parent = next(item for item in fused if item["role"] == "ui-panel")
    child = next(item for item in fused if item["role"] == "icon")
    assert child["parent_id"] == parent["id"]


def test_overlapping_but_semantically_distinct_masks_survive():
    photo = _mask(10, 10, 60, 60)
    badge = _mask(45, 45, 25, 25)
    sam3 = {
        "elements": [
            {
                "id": "photo",
                "box": _box(10, 10, 60, 60),
                "role": "photo",
                "kind": "photo-fragment",
                "score": 0.9,
                "_mask": photo,
                "provenance": {"mode": "text-prompt"},
            },
            {
                "id": "badge",
                "box": _box(45, 45, 25, 25),
                "role": "badge",
                "kind": "icon",
                "score": 0.9,
                "_mask": badge,
                "provenance": {"mode": "text-prompt"},
            },
        ]
    }
    fused = element_fusion.fuse(sam3, [], [], CANVAS)
    assert {e["role"] for e in fused} == {"photo", "badge"}


def test_residual_stream_unions_with_sam_residual_fallback(tmp_path):
    mask = _mask(12, 12, 20, 20)
    residual = [
        {
            "id": "R0",
            "box": _box(12, 12, 20, 20),
            "kind": "icon",
            "_mask": mask,
        }
    ]
    sam3 = {
        "status": "fallback",
        "elements": [
            {
                "id": "S0",
                "box": _box(12, 12, 20, 20),
                "role": "icon",
                "kind": "icon",
                "score": 0.35,
                "_mask": mask,
                "source": "residual-fallback",
                "provenance": {
                    "mode": "residual-fallback",
                    "residual_id": "R0",
                    "reason": "checkpoint missing",
                },
            }
        ],
    }
    fused = element_fusion.fuse(sam3, residual, [], CANVAS, run_dir=str(tmp_path))
    assert len(fused) == 1
    sources = fused[0]["provenance"]["sources"]
    assert "sam3:residual-fallback" in sources
    assert "residual" in sources
