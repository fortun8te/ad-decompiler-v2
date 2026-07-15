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


def test_residual_obs_merges_into_its_own_box_refine_cluster(tmp_path):
    """Regression for benchmark 009 E015/E016: a sparse residual CC (anti-aliased ring)
    against the solid SAM mask box-refined FROM it lands in the nested IoU band and used to
    ship twice. Provenance identity (residual_id) must merge the pair even with absorption
    disabled, so the link path itself is what is under test."""
    solid = _mask(20, 20, 30, 30)
    ring = solid & ~_mask(23, 23, 24, 24)  # 3px border ring: iou 0.36, containment 1.0
    assert 0.30 < ring.sum() / solid.sum() < 0.70

    sam3 = {
        "elements": [
            {
                "id": "S0",
                "box": _box(20, 20, 30, 30),
                "role": "icon",
                "kind": "icon",
                "score": 0.93,
                "_mask": solid,
                "provenance": {"mode": "box-refine", "residual_id": "R0"},
            }
        ]
    }
    residual = [{"id": "R0", "box": _box(20, 20, 30, 30), "kind": "icon", "_mask": ring}]
    cfg = {"element_fusion": {"absorb_fragments": False, "sliver_absorb": False}}

    fused = element_fusion.fuse(sam3, residual, [], CANVAS, cfg=cfg, run_dir=str(tmp_path))

    assert len(fused) == 1
    element = fused[0]
    assert element["box"] == _box(20, 20, 30, 30)  # solid SAM mask wins the cluster
    merges = element["provenance"]["nms"]["merges"]
    assert any(m.get("reason") == "residual-refine-link" for m in merges)
    keys = {o["key"] for o in element["provenance"]["observations"]}
    assert keys == {"sam3:S0", "residual:R0"}


def test_residual_refine_link_can_be_disabled():
    solid = _mask(20, 20, 30, 30)
    ring = solid & ~_mask(23, 23, 24, 24)
    sam3 = {"elements": [{"id": "S0", "box": _box(20, 20, 30, 30), "role": "icon",
                          "kind": "icon", "score": 0.93, "_mask": solid,
                          "provenance": {"mode": "box-refine", "residual_id": "R0"}}]}
    residual = [{"id": "R0", "box": _box(20, 20, 30, 30), "kind": "icon", "_mask": ring}]
    cfg = {"element_fusion": {"link_residual_refine": False,
                              "absorb_fragments": False, "sliver_absorb": False}}
    fused = element_fusion.fuse(sam3, residual, [], CANVAS, cfg=cfg)
    assert len(fused) == 2  # the historical duplicate leak, now opt-in only


def test_fragmented_icon_pieces_absorb_into_whole_icon(tmp_path):
    """Regression for benchmark 009 E017/E018/E023: box-refine snapped two residual CCs to
    glyph pieces of one share icon while the text prompt found the whole icon. Same-role
    contained pieces must collapse into the whole instead of shipping as three overlapping
    top-level icons (icon parents are not 'meaningful' so the parent-link pass never fires)."""
    whole = _mask(40, 40, 40, 40)
    piece_a = _mask(48, 42, 20, 18)  # fully inside, 22% of whole
    piece_b = _mask(44, 66, 30, 12)  # fully inside, 22% of whole
    sam3 = {
        "elements": [
            {
                "id": "S-whole",
                "box": _box(40, 40, 40, 40),
                "role": "icon",
                "kind": "icon",
                "score": 0.89,
                "_mask": whole,
                "provenance": {"mode": "text-prompt", "prompt": "icon"},
            },
            {
                "id": "S-a",
                "box": _box(48, 42, 20, 18),
                "role": "icon",
                "kind": "icon",
                "score": 0.93,
                "_mask": piece_a,
                "provenance": {"mode": "box-refine", "residual_id": "R1"},
            },
            {
                "id": "S-b",
                "box": _box(44, 66, 30, 12),
                "role": "icon",
                "kind": "icon",
                "score": 0.94,
                "_mask": piece_b,
                "provenance": {"mode": "box-refine", "residual_id": "R2"},
            },
        ]
    }
    residual = [
        {"id": "R1", "box": _box(48, 42, 20, 18), "kind": "icon", "_mask": piece_a},
        {"id": "R2", "box": _box(44, 66, 30, 12), "kind": "icon", "_mask": piece_b},
    ]

    fused = element_fusion.fuse(sam3, residual, [], CANVAS, run_dir=str(tmp_path))

    assert len(fused) == 1
    element = fused[0]
    assert element["role"] == "icon"
    # The whole-icon mask wins; the crop stays tight so downstream corner-radius/shape
    # inference (reconstruct._corner_radius) sees the full clean silhouette.
    assert element["box"] == _box(40, 40, 40, 40)
    assert Image.open(element["mask_path"]).size == (40, 40)
    assert len(element["provenance"]["observations"]) == 5
    reasons = {m.get("reason") for m in element["provenance"]["nms"]["merges"]}
    assert "absorbed-fragment" in reasons


def test_button_fill_slivers_absorb_into_model_button(tmp_path):
    """Regression for benchmark 009 'Volgend': OCR text dilation splits the button plate's
    residual CC into thin slivers hugging the pill border (some poking 1-2 px past the SAM
    mask). They are fill fragments, not children — absorbing them removes the white
    rectangle artifact behind the button."""
    button = _mask(10, 10, 60, 20)
    sliver = _mask(20, 8, 40, 4)  # top strip: 2 of 4 rows above the button mask
    sam3 = {
        "elements": [
            {
                "id": "S-btn",
                "box": _box(10, 10, 60, 20),
                "role": "button",
                "kind": "shape",
                "score": 0.73,
                "_mask": button,
                "provenance": {"mode": "text-prompt", "prompt": "button"},
            }
        ]
    }
    residual = [{"id": "R0", "box": _box(20, 8, 40, 4), "kind": "shape", "_mask": sliver}]

    fused = element_fusion.fuse(sam3, residual, [], CANVAS, run_dir=str(tmp_path))

    assert len(fused) == 1
    element = fused[0]
    assert element["role"] == "button"
    assert element["box"] == _box(10, 10, 60, 20)  # winner mask untouched (radius evidence)
    reasons = {m.get("reason") for m in element["provenance"]["nms"]["merges"]}
    assert "absorbed-sliver" in reasons


def test_sliver_absorb_can_be_disabled():
    button = _mask(10, 10, 60, 20)
    sliver = _mask(20, 8, 40, 4)
    sam3 = {"elements": [{"id": "S-btn", "box": _box(10, 10, 60, 20), "role": "button",
                          "kind": "shape", "score": 0.73, "_mask": button,
                          "provenance": {"mode": "text-prompt", "prompt": "button"}}]}
    residual = [{"id": "R0", "box": _box(20, 8, 40, 4), "kind": "shape", "_mask": sliver}]
    cfg = {"element_fusion": {"sliver_absorb": False}}
    fused = element_fusion.fuse(sam3, residual, [], CANVAS, cfg=cfg)
    assert len(fused) == 2


def test_sliver_inside_card_is_preserved_as_real_child():
    """A thin divider fully inside a card is a real element: card is not a sliver-parent
    role, so the sliver must survive and be linked as a nested child instead."""
    card = _mask(10, 10, 60, 40)
    divider = _mask(16, 28, 48, 3)
    sam3 = {
        "elements": [
            {"id": "S-card", "box": _box(10, 10, 60, 40), "role": "card", "kind": "shape",
             "score": 0.8, "_mask": card, "provenance": {"mode": "text-prompt"}},
            {"id": "S-div", "box": _box(16, 28, 48, 3), "role": "shape", "kind": "shape",
             "score": 0.7, "_mask": divider, "provenance": {"mode": "text-prompt"}},
        ]
    }
    fused = element_fusion.fuse(sam3, [], [], CANVAS)
    assert len(fused) == 2
    parent = next(e for e in fused if e["role"] == "card")
    child = next(e for e in fused if e["role"] == "shape")
    assert child["parent_id"] == parent["id"]


def test_residual_photo_fragment_label_does_not_block_shape_dedup():
    """Task: an element must never ship as both photo-fragment and shape. The residual-CC
    kind is a threshold heuristic; when its mask geometrically duplicates a model shape
    (iou >= mask_iou but < 0.88), the family conflict must not keep both."""
    button = _mask(10, 10, 40, 20)
    partial = _mask(10, 10, 30, 20)  # iou 0.75, area_ratio 0.75 -> duplicate band
    sam3 = {"elements": [{"id": "S-btn", "box": _box(10, 10, 40, 20), "role": "button",
                          "kind": "shape", "score": 0.8, "_mask": button,
                          "provenance": {"mode": "text-prompt", "prompt": "button"}}]}
    residual = [{"id": "R0", "box": _box(10, 10, 30, 20), "kind": "photo-fragment",
                 "_mask": partial}]

    fused = element_fusion.fuse(sam3, residual, [], CANVAS)

    assert len(fused) == 1
    assert fused[0]["role"] == "button"
    assert fused[0]["kind"] == "shape"


def test_model_vs_model_family_conflict_still_kept_separate():
    """The residual-family advisory must not weaken model-vs-model semantics: two SAM
    text-prompt observations with conflicting families and iou < 0.88 stay separate."""
    photo = _mask(10, 10, 40, 20)
    partial = _mask(10, 10, 30, 20)
    sam3 = {"elements": [
        {"id": "S-photo", "box": _box(10, 10, 40, 20), "role": "photo",
         "kind": "photo-fragment", "score": 0.8, "_mask": photo,
         "provenance": {"mode": "text-prompt"}},
        {"id": "S-btn", "box": _box(10, 10, 30, 20), "role": "button", "kind": "shape",
         "score": 0.8, "_mask": partial, "provenance": {"mode": "text-prompt"}},
    ]}
    fused = element_fusion.fuse(sam3, [], [], CANVAS)
    assert len(fused) == 2


def test_antialiased_fringe_child_still_links_to_parent():
    """A child mask whose anti-aliased fringe pokes 2px past the model parent mask used to
    fail the 0.90 containment bar (009 E007 vs the 'Volgend' button at 0.899); the dilated
    containment test must still record the icon-in-button relationship."""
    button = _mask(10, 10, 40, 20)
    icon = _mask(12, 8, 10, 10)  # rows 8-9 sit above the button mask: raw containment 0.8
    sam3 = {"elements": [
        {"id": "S-btn", "box": _box(10, 10, 40, 20), "role": "button", "kind": "shape",
         "score": 0.8, "_mask": button, "provenance": {"mode": "text-prompt"}},
        {"id": "S-icon", "box": _box(12, 8, 10, 10), "role": "icon", "kind": "icon",
         "score": 0.9, "_mask": icon, "provenance": {"mode": "text-prompt"}},
    ]}

    fused = element_fusion.fuse(sam3, [], [], CANVAS)

    assert len(fused) == 2
    parent = next(e for e in fused if e["role"] == "button")
    child = next(e for e in fused if e["role"] == "icon")
    assert child["parent_id"] == parent["id"]
    assert child["relationships"][0]["type"] == "nested-in"


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
