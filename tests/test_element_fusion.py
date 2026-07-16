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


def test_sibling_product_cutouts_do_not_merge_on_moderate_iou(tmp_path):
    """Hears packshot: box + earplugs stay two products even when masks partially overlap."""
    box = _mask(10, 20, 40, 50)
    plugs = _mask(35, 30, 35, 40)  # partial overlap with box, IoU well below 0.85
    metrics = element_fusion._mask_metrics(box, plugs)
    assert metrics["iou"] < 0.85
    assert metrics["iou"] > 0.05
    sam3 = {
        "elements": [
            {
                "id": "S-box",
                "box": _box(10, 20, 40, 50),
                "role": "product",
                "kind": "photo-fragment",
                "score": 0.93,
                "_mask": box,
                "provenance": {"mode": "text-prompt", "prompt": "box"},
            },
            {
                "id": "S-plugs",
                "box": _box(35, 30, 35, 40),
                "role": "product",
                "kind": "photo-fragment",
                "score": 0.91,
                "_mask": plugs,
                "provenance": {"mode": "text-prompt", "prompt": "product"},
            },
        ]
    }
    fused = element_fusion.fuse(sam3, [], [], CANVAS, run_dir=str(tmp_path))
    assert len(fused) == 2
    assert {e["role"] for e in fused} == {"product"}


def test_biomel_vs_comparison_two_products_and_vs_chip_stay_separate(tmp_path):
    """Biomel VS panel: coffee + bag products must not merge; middle VS badge stays."""
    coffee = _mask(5, 10, 35, 55)
    bag = _mask(55, 8, 35, 58)
    vs = _mask(42, 30, 12, 12)
    assert element_fusion._mask_metrics(coffee, bag)["iou"] < 0.85
    sam3 = {
        "elements": [
            {
                "id": "S-coffee",
                "box": _box(5, 10, 35, 55),
                "role": "product",
                "kind": "photo-fragment",
                "score": 0.94,
                "_mask": coffee,
                "provenance": {"mode": "text-prompt", "prompt": "product"},
            },
            {
                "id": "S-bag",
                "box": _box(55, 8, 35, 58),
                "role": "package",
                "kind": "photo-fragment",
                "score": 0.92,
                "_mask": bag,
                "provenance": {"mode": "text-prompt", "prompt": "package"},
            },
            {
                "id": "S-vs",
                "box": _box(42, 30, 12, 12),
                "role": "badge",
                "kind": "icon",
                "score": 0.88,
                "_mask": vs,
                "provenance": {"mode": "text-prompt", "prompt": "badge"},
            },
        ]
    }
    fused = element_fusion.fuse(sam3, [], [], CANVAS, run_dir=str(tmp_path))
    assert len(fused) == 3
    roles = {e["role"] for e in fused}
    assert "product" in roles or "package" in roles
    assert "badge" in roles
    assert sum(1 for e in fused if e["role"] in {"product", "package"}) == 2


def test_wavy_tube_across_photo_panel_stays_top_level_cutout(tmp_path):
    """Product tube straddling a photo panel must not nest under / merge into the panel."""
    photo = _mask(5, 5, 90, 55)       # ends at y=60
    tube = _mask(30, 20, 40, 75)      # extends to y=95 → spills past the panel
    metrics = element_fusion._mask_metrics(photo, tube)
    assert metrics["containment"] < 0.85
    assert metrics["iou"] < 0.68
    sam3 = {
        "elements": [
            {
                "id": "S-panel",
                "box": _box(5, 5, 90, 55),
                "role": "photo",
                "kind": "photo-fragment",
                "score": 0.9,
                "_mask": photo,
                "provenance": {"mode": "text-prompt", "prompt": "photo"},
            },
            {
                "id": "S-tube",
                "box": _box(30, 20, 40, 75),
                "role": "product",
                "kind": "photo-fragment",
                "score": 0.95,
                "_mask": tube,
                "provenance": {"mode": "text-prompt", "prompt": "product"},
            },
        ]
    }
    fused = element_fusion.fuse(sam3, [], [], CANVAS, run_dir=str(tmp_path))
    assert len(fused) == 2
    by_role = {e["role"]: e for e in fused}
    assert "product" in by_role and "photo" in by_role
    assert by_role["product"].get("parent_id") is None
    assert by_role["photo"].get("parent_id") is None


# ── canonical product geometry (013 / 135 / 067) ───────────────────────────────────


def _sam_el(sid, x, y, w, h, role, score, mode, prompt=None, residual_id=None):
    prov = {"mode": mode}
    if prompt:
        prov["prompt"] = prompt
    if residual_id is not None:
        prov["residual_id"] = residual_id
    return {
        "id": sid, "box": _box(x, y, w, h), "role": role,
        "kind": "photo-fragment", "score": score, "_mask": _mask(x, y, w, h),
        "provenance": prov,
    }


def test_near_identical_masks_merge_across_product_photo_labels():
    """135: box-refine 'photo' + text-prompt 'product' with IoU≈1 are one object."""
    photo = _sam_el("S0", 20, 20, 40, 40, "photo", 0.94, "box-refine", residual_id="R9")
    product = _sam_el("S1", 20, 20, 40, 40, "product", 0.88, "text-prompt", prompt="product")

    fused = element_fusion.fuse({"elements": [photo, product]}, [], [], CANVAS)

    assert len(fused) == 1
    # heuristic box-refine label loses the identity election to the semantic prompt
    assert fused[0]["role"] == "product"


def test_disjoint_products_never_merge():
    """067: distinct jars (zero overlap) stay separate canonical products."""
    jars = [
        _sam_el(f"S{i}", 5 + i * 32, 40, 26, 30, "product", 0.8, "text-prompt", prompt="product")
        for i in range(3)
    ]

    fused = element_fusion.fuse({"elements": jars}, [], [], CANVAS)

    assert len(fused) == 3
    assert all(e["role"] == "product" for e in fused)


def test_union_blob_photo_absorbed_into_products():
    """067: a no-evidence 'photo' equal to the union of jar masks is a duplicate."""
    jars = [
        _sam_el(f"S{i}", 5 + i * 32, 40, 26, 30, "product", 0.8, "text-prompt", prompt="product")
        for i in range(3)
    ]
    blob_mask = np.zeros((100, 100), dtype=bool)
    for i in range(3):
        blob_mask |= _mask(5 + i * 32, 40, 26, 30)
    blob = {
        "id": "S9", "box": _box(5, 40, 90, 30), "role": "photo",
        "kind": "photo-fragment", "score": 0.9, "_mask": blob_mask,
        "provenance": {"mode": "box-refine", "residual_id": "R1"},
    }

    fused = element_fusion.fuse({"elements": jars + [blob]}, [], [], CANVAS)

    assert len(fused) == 3
    assert all(e["role"] == "product" for e in fused)
    reasons = [m.get("reason") for e in fused for m in e["provenance"]["nms"]["merges"]]
    assert "absorbed-product-shadow" in reasons


def test_full_bleed_low_score_photo_band_is_suppressed(tmp_path):
    """013: residual-backed full-width gradient band must not ship as an element."""
    import json

    band = {
        "id": "S0", "box": _box(0, 30, 100, 30), "role": "photo",
        "kind": "photo-fragment", "score": 0.35, "_mask": _mask(0, 30, 100, 30),
        "provenance": {"mode": "box-refine", "residual_id": "R0"},
    }
    residual = [{"id": "R0", "box": _box(0, 30, 100, 30), "kind": "photo-fragment",
                 "_mask": _mask(0, 30, 100, 30)}]
    product = _sam_el("S1", 40, 40, 20, 50, "product", 0.9, "text-prompt", prompt="product")

    fused = element_fusion.fuse({"elements": [band, product]}, residual, [], CANVAS,
                                run_dir=str(tmp_path))

    assert [e["role"] for e in fused] == ["product"]
    report = json.load(open(tmp_path / "fusion_report.json"))
    assert report["counts"]["suppressed_junk_bands"] == 1


def test_high_score_full_bleed_photo_band_is_kept():
    """A real full-bleed photo (strong model score) keeps its element."""
    band = {
        "id": "S0", "box": _box(0, 30, 100, 40), "role": "photo",
        "kind": "photo-fragment", "score": 0.96, "_mask": _mask(0, 30, 100, 40),
        "provenance": {"mode": "box-refine", "residual_id": "R0"},
    }

    fused = element_fusion.fuse({"elements": [band]}, [], [], CANVAS)

    assert len(fused) == 1
    assert fused[0]["role"] == "photo"


# ── printed-on-product artwork absorption (013 grüns bag) ─────────────────────────


def test_logo_nested_in_product_absorbs_and_flags_printed_lockup(tmp_path):
    """013: SAM's 'logo' hits on the bag's printed lockup fold into the product."""
    import json

    product = _sam_el("S0", 20, 20, 60, 60, "product", 0.88, "text-prompt",
                      prompt="product")
    logo = _sam_el("S1", 30, 30, 20, 10, "logo", 0.83, "text-prompt", prompt="logo")
    logo["kind"] = "icon"

    fused = element_fusion.fuse({"elements": [product, logo]}, [], [], CANVAS,
                                run_dir=str(tmp_path))

    assert [e["role"] for e in fused] == ["product"]
    parent = fused[0]
    assert (parent.get("meta") or {}).get("printed_lockup") is True
    decorations = parent["meta"]["absorbed_decorations"]
    assert len(decorations) == 1 and decorations[0]["role"] == "logo"
    report = json.load(open(tmp_path / "fusion_report.json"))
    assert report["counts"]["absorbed_printed_artwork"] == 1
    assert report["absorbed_printed_artwork"][0]["reason"] == "printed-on-product-artwork"


def test_logo_over_photo_panel_stays_a_separate_element():
    """A brand mark overlaid on a photo/background is a genuine layer — untouched."""
    photo = _sam_el("S0", 10, 10, 80, 80, "photo", 0.9, "box-refine", residual_id="R0")
    logo = _sam_el("S1", 30, 30, 20, 10, "logo", 0.83, "text-prompt", prompt="logo")
    logo["kind"] = "icon"

    fused = element_fusion.fuse({"elements": [photo, logo]}, [], [], CANVAS)

    assert sorted(e["role"] for e in fused) == ["logo", "photo"]
    for e in fused:
        assert not (e.get("meta") or {}).get("printed_lockup")


def test_partially_overlapping_logo_is_not_absorbed():
    """A logo straddling the product edge is not printed ink — keep it separate."""
    product = _sam_el("S0", 20, 20, 40, 40, "product", 0.88, "text-prompt",
                      prompt="product")
    logo = _sam_el("S1", 50, 30, 30, 10, "logo", 0.83, "text-prompt", prompt="logo")
    logo["kind"] = "icon"

    fused = element_fusion.fuse({"elements": [product, logo]}, [], [], CANVAS)

    assert sorted(e["role"] for e in fused) == ["logo", "product"]


# ── generalized invariant: every non-product decoration inside a product folds ───────


def test_icon_glyph_nested_in_product_absorbs_as_decoration(tmp_path):
    """135/094: a generic on-pack icon (not a logo, not a wide flavor bar) rides the
    product cutout instead of shipping as a punchable element."""
    import json

    product = _sam_el("S0", 20, 20, 60, 60, "product", 0.9, "text-prompt", prompt="product")
    glyph = _sam_el("S1", 34, 34, 12, 12, "icon", 0.8, "text-prompt", prompt="icon")
    glyph["kind"] = "icon"

    fused = element_fusion.fuse({"elements": [product, glyph]}, [], [], CANVAS,
                                run_dir=str(tmp_path))

    assert [e["role"] for e in fused] == ["product"]
    parent = fused[0]
    assert (parent.get("meta") or {}).get("printed_lockup") is True
    assert [d["role"] for d in parent["meta"]["absorbed_decorations"]] == ["icon"]
    report = json.load(open(tmp_path / "fusion_report.json"))
    assert report["counts"]["absorbed_product_decorations"] == 1
    assert report["absorbed_product_decorations"][0]["reason"] == "printed-on-product-decoration"


def test_checkmark_and_cross_glyphs_on_pack_absorb():
    """On-pack ✓/✗ list glyphs (135 nutrition panel) never punch the packaging."""
    product = _sam_el("S0", 15, 15, 70, 70, "product", 0.9, "text-prompt", prompt="product")
    check = _sam_el("S1", 30, 30, 10, 10, "verified", 0.8, "text-prompt", prompt="check")
    check["kind"] = "icon"
    cross = _sam_el("S2", 30, 50, 10, 10, "cross", 0.8, "text-prompt", prompt="cross")
    cross["kind"] = "icon"

    fused = element_fusion.fuse({"elements": [product, check, cross]}, [], [], CANVAS)

    assert [e["role"] for e in fused] == ["product"]
    roles = {d["role"] for d in fused[0]["meta"]["absorbed_decorations"]}
    assert roles == {"verified", "cross"}


def test_transitive_glyph_under_panel_under_product_folds_to_product():
    """135: ✓/✗ chips nested in a nutrition panel that is itself nested in the bar all
    fold into the bar once the intermediate panel is absorbed (fixed-point)."""
    product = _sam_el("S0", 10, 10, 80, 80, "product", 0.9, "text-prompt", prompt="product")
    panel = _sam_el("S1", 28, 28, 30, 30, "shape", 0.8, "text-prompt", prompt="panel")
    panel["kind"] = "shape"
    glyph = _sam_el("S2", 32, 32, 8, 8, "verified", 0.8, "text-prompt", prompt="check")
    glyph["kind"] = "icon"

    fused = element_fusion.fuse({"elements": [product, panel, glyph]}, [], [], CANVAS)

    assert [e["role"] for e in fused] == ["product"]
    roles = sorted(d["role"] for d in fused[0]["meta"]["absorbed_decorations"])
    assert roles == ["shape", "verified"]


def test_edge_straddling_label_folds_by_box_containment():
    """A label sticking ~20% over the product edge (mask containment below the nesting
    threshold, so no nested-in link) still folds by box containment — it must not punch
    the bag."""
    product = _sam_el("S0", 20, 20, 40, 40, "product", 0.9, "text-prompt", prompt="product")
    # x44..64 vs product x20..60 → 16/20 = 0.80 box-contained, ~0.80 mask (< 0.90 link).
    label = _sam_el("S1", 44, 30, 20, 10, "icon", 0.8, "text-prompt", prompt="label")
    label["kind"] = "icon"

    fused = element_fusion.fuse({"elements": [product, label]}, [], [], CANVAS)

    assert [e["role"] for e in fused] == ["product"]
    assert (fused[0].get("meta") or {}).get("printed_lockup") is True


def test_low_overlap_decoration_stays_a_separate_element():
    """A glyph only ~1/3 over the product edge is a genuine overlay layer — untouched."""
    product = _sam_el("S0", 20, 20, 40, 40, "product", 0.9, "text-prompt", prompt="product")
    icon = _sam_el("S1", 50, 30, 30, 10, "icon", 0.8, "text-prompt", prompt="icon")
    icon["kind"] = "icon"

    fused = element_fusion.fuse({"elements": [product, icon]}, [], [], CANVAS)

    assert sorted(e["role"] for e in fused) == ["icon", "product"]


def test_distinct_product_inside_product_is_not_absorbed():
    """A distinct SKU is never a decoration — a bottle nested in a box stays its own
    alpha (product-instance roles are excluded from decoration absorption)."""
    box = _sam_el("S0", 10, 10, 80, 80, "product", 0.9, "text-prompt", prompt="box")
    bottle = _sam_el("S1", 30, 30, 20, 20, "bottle", 0.88, "text-prompt", prompt="bottle")

    fused = element_fusion.fuse({"elements": [box, bottle]}, [], [], CANVAS)

    assert sorted(e["role"] for e in fused) == ["bottle", "product"]
    for e in fused:
        assert not (e.get("meta") or {}).get("absorbed_decorations")


def test_product_decoration_absorb_can_be_disabled():
    product = _sam_el("S0", 20, 20, 60, 60, "product", 0.9, "text-prompt", prompt="product")
    glyph = _sam_el("S1", 34, 34, 12, 12, "icon", 0.8, "text-prompt", prompt="icon")
    glyph["kind"] = "icon"

    fused = element_fusion.fuse(
        {"elements": [product, glyph]}, [], [], CANVAS,
        cfg={"element_fusion": {"absorb_product_decorations": False}})

    assert sorted(e["role"] for e in fused) == ["icon", "product"]


# ── product cluster hull merge + alpha sealing (135 / packaging) ────────────────────


def test_adjacent_same_prompt_products_merge_into_one_owner():
    """135: dual bars / jar+lid with shared product prompt + hull proximity → 1 owner."""
    # Two bars 2px apart (gap < budget); same text-prompt "product".
    left = _sam_el("S0", 10, 30, 28, 40, "product", 0.9, "text-prompt", prompt="product")
    right = _sam_el("S1", 40, 30, 28, 40, "product", 0.88, "text-prompt", prompt="product")

    fused = element_fusion.fuse({"elements": [left, right]}, [], [], CANVAS)

    assert len(fused) == 1
    assert fused[0]["role"] == "product"
    reasons = [m.get("reason") for m in fused[0]["provenance"]["nms"]["merges"]]
    assert "product-cluster-hull-merge" in reasons
    # Union bbox covers both bars.
    box = fused[0]["box"]
    assert box["x"] <= 10 and box["x"] + box["w"] >= 68


def test_jar_and_lid_same_prompt_merge():
    """Jar body + lid stacked with a tiny gap share one product owner."""
    jar = _sam_el("S0", 30, 40, 30, 40, "product", 0.9, "text-prompt", prompt="product")
    lid = _sam_el("S1", 32, 28, 26, 14, "product", 0.85, "text-prompt", prompt="product")

    fused = element_fusion.fuse({"elements": [jar, lid]}, [], [], CANVAS)

    assert len(fused) == 1
    assert fused[0]["role"] == "product"
    reasons = [m.get("reason") for m in fused[0]["provenance"]["nms"]["merges"]]
    assert "product-cluster-hull-merge" in reasons


def test_spaced_same_prompt_skus_stay_separate():
    """067 regression: three jars with a larger gap must not hull-merge."""
    jars = [
        _sam_el(f"S{i}", 5 + i * 32, 40, 26, 30, "product", 0.8, "text-prompt",
                prompt="product")
        for i in range(3)
    ]

    fused = element_fusion.fuse({"elements": jars}, [], [], CANVAS)

    assert len(fused) == 3


def test_different_product_prompts_do_not_hull_merge():
    """Adjacent cutouts with distinct prompts are different objects."""
    a = _sam_el("S0", 10, 30, 28, 40, "product", 0.9, "text-prompt", prompt="chocolate bar")
    b = _sam_el("S1", 40, 30, 28, 40, "product", 0.88, "text-prompt", prompt="earplug")

    fused = element_fusion.fuse({"elements": [a, b]}, [], [], CANVAS)

    assert len(fused) == 2


def test_product_mask_swiss_cheese_holes_are_sealed():
    """Interior SAM holes on packaging fill so the product alpha stays solid."""
    solid = _mask(20, 20, 40, 40)
    # Punch two interior holes (Swiss cheese).
    solid[30:34, 30:34] = False
    solid[40:45, 40:44] = False
    product = {
        "id": "S0", "box": _box(20, 20, 40, 40), "role": "product",
        "kind": "photo-fragment", "score": 0.9, "_mask": solid,
        "provenance": {"mode": "text-prompt", "prompt": "product"},
    }

    fused = element_fusion.fuse({"elements": [product]}, [], [], CANVAS)

    assert len(fused) == 1
    # Winner mask is not exported as ndarray on the result — re-run seal helper on a
    # reconstructed observation via fuse internals: area must grow past the holed mask.
    assert fused[0]["area"] >= float(solid.sum())
    # Stronger check: materialize via a private seal on a copy of the input mask.
    sealed = element_fusion._fill_mask_holes(solid.copy())
    assert int(sealed[30:34, 30:34].sum()) == 16
    assert int(sealed[40:45, 40:44].sum()) == 20


def test_overlapping_gapped_product_masks_merge_via_box_iou():
    """135: dual bars with Swiss-cheese mask gap still merge when boxes overlap."""
    # Non-touching masks whose declared boxes still heavily overlap.
    left = _mask(10, 20, 30, 50)
    right = _mask(55, 30, 30, 50)
    clusters = [
        {
            "winner": {
                "key": "a", "role": "product", "kind": "photo-fragment",
                "mask": left, "box": _box(10, 20, 55, 50), "score": 0.9,
                "source": "sam3", "mask_quality": "mask",
            },
            "members": [{
                "key": "a", "role": "product", "mode": "text-prompt",
                "prompt": "product", "score": 0.9, "source": "sam3",
            }],
            "merges": [],
        },
        {
            "winner": {
                "key": "b", "role": "product", "kind": "photo-fragment",
                "mask": right, "box": _box(40, 30, 45, 50), "score": 0.88,
                "source": "sam3", "mask_quality": "mask",
            },
            "members": [{
                "key": "b", "role": "product", "mode": "text-prompt",
                "prompt": "product", "score": 0.88, "source": "sam3",
            }],
            "merges": [],
        },
    ]
    opts = dict(element_fusion.DEFAULTS)
    merged = element_fusion._merge_product_clusters(clusters, opts)
    assert len(merged) == 1
    assert merged[0]["winner"]["box"]["w"] >= 55


def test_packaging_flavor_bar_button_absorbs_into_product(tmp_path):
    """002: wide on-pack 'button' (VANILLE SMAAK bar) folds into the product."""
    import json

    product = _sam_el("S0", 10, 20, 70, 60, "product", 0.9, "text-prompt", prompt="product")
    bar = _sam_el("S1", 15, 50, 50, 8, "button", 0.8, "text-prompt", prompt="button")
    bar["kind"] = "shape"

    fused = element_fusion.fuse({"elements": [product, bar]}, [], [], CANVAS,
                                run_dir=str(tmp_path))

    assert [e["role"] for e in fused] == ["product"]
    assert (fused[0].get("meta") or {}).get("printed_lockup") is True
    report = json.load(open(tmp_path / "fusion_report.json"))
    assert report["counts"]["absorbed_packaging_shells"] == 1
    assert report["absorbed_packaging_shells"][0]["reason"] == "printed-on-product-shell"
