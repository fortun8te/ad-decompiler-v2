// Ad Decompiler Figma compiler.
// No build step on purpose: this file runs directly in Figma's plugin sandbox.
// It accepts the legacy flat design.json contract and scene-graph v2 documents.

const PLUGIN_BUILD = {"version":"2.0.0","build":21,"commit":"649d673","dirty":true,"built_at":"2026-07-12T01:42:29Z","label":"v2.0.0+b21.649d673-dirty","source":"git"};

figma.showUI(__html__, {
  width: 388,
  height: 620,
  title: "Ad Decompiler · b" + PLUGIN_BUILD.build,
  themeColors: true,
});

const SETTINGS_KEY = "ad-decompiler.settings.v2";
const ROOT_KEY = "adDecompilerRoot";
const DOC_KEY = "adDecompilerDocId";
const LAYER_KEY = "adDecompilerLayerId";
const STYLE_KEY = "adDecompilerStyleKey";
const DEFAULT_SETTINGS = {
  bridge: "http://localhost:8790",
  importMode: "replace",
};

let activeJob = null;

function post(type, payload) {
  figma.ui.postMessage(Object.assign({ type }, payload || {}));
}

function pick(object) {
  if (!object) return undefined;
  for (let i = 1; i < arguments.length; i += 1) {
    const key = arguments[i];
    if (object[key] !== undefined && object[key] !== null) return object[key];
  }
  return undefined;
}

function finite(value, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function opacityValue(value, fallback) {
  let n = finite(value, fallback);
  if (n > 1) n /= 100;
  return clamp(n, 0, 1);
}

function normalizedToken(value) {
  return String(value || "")
    .trim()
    .replace(/[\s-]+/g, "_")
    .toUpperCase();
}

function layerId(layer, fallback) {
  return String(pick(layer, "id", "source_id", "sourceId") || fallback || "layer");
}

function boxOf(layer) {
  const b = pick(layer, "box", "bounds") || {};
  return {
    x: finite(pick(b, "x", "left"), 0),
    y: finite(pick(b, "y", "top"), 0),
    w: Math.max(0.01, finite(pick(b, "w", "width"), 1)),
    h: Math.max(0.01, finite(pick(b, "h", "height"), 1)),
  };
}

function visibleBoxOf(layer) {
  const b = pick(layer, "visible_box", "visibleBox");
  if (!b) return boxOf(layer);
  return {
    x: finite(pick(b, "x", "left"), boxOf(layer).x),
    y: finite(pick(b, "y", "top"), boxOf(layer).y),
    w: Math.max(0.01, finite(pick(b, "w", "width"), boxOf(layer).w)),
    h: Math.max(0.01, finite(pick(b, "h", "height"), boxOf(layer).h)),
  };
}

function childrenOf(layer) {
  const children = pick(layer, "children", "layers");
  return Array.isArray(children) ? children : [];
}

function zIndexOf(layer) {
  return finite(
    pick(layer, "z_index", "zIndex", "z", "order"),
    finite(layer && layer.meta && layer.meta.z, 0)
  );
}

function sortLayers(layers) {
  const list = (layers || []).map(function (layer, index) {
    return { layer, index, explicit: pick(layer, "z_index", "zIndex", "z") !== undefined };
  });
  if (!list.some(function (entry) { return entry.explicit; })) return list.map(function (entry) { return entry.layer; });
  return list
    .sort(function (a, b) { return zIndexOf(a.layer) - zIndexOf(b.layer) || a.index - b.index; })
    .map(function (entry) { return entry.layer; });
}

function countLayers(layers) {
  let count = 0;
  (layers || []).forEach(function (layer) {
    count += 1 + countLayers(childrenOf(layer));
  });
  return count;
}

function canonicalType(layer) {
  const raw = normalizedToken(pick(layer, "type", "node_type", "nodeType", "kind"));
  const shapeKind = normalizedToken(pick(layer, "shape_kind", "shapeKind"));
  if (raw === "TEXT") return "text";
  if (raw === "IMAGE" || raw === "PHOTO" || raw === "RASTER") return "image";
  if (raw === "FRAME" || raw === "CONTAINER" || raw === "SECTION") return "frame";
  if (raw === "GROUP") return "group";
  if (raw === "VECTOR" || raw === "SVG" || raw === "ICON" || raw === "PATH") return "vector";
  if (raw === "SHAPE") return shapeKind === "PATH" ? "vector" : "shape";
  if (raw === "RECT" || raw === "RECTANGLE" || raw === "ELLIPSE" || raw === "CIRCLE") return "shape";
  if (childrenOf(layer).length) return "frame";
  return raw.toLowerCase() || "unknown";
}

function sourceCoordinateSpace(layer, context) {
  const explicit = normalizedToken(
    pick(layer, "coordinate_space", "coordinateSpace", "position_mode", "positionMode") ||
      (layer.meta && pick(layer.meta, "coordinate_space", "coordinateSpace"))
  );
  if (explicit === "LOCAL" || explicit === "RELATIVE") return "local";
  if (explicit === "ABSOLUTE" || explicit === "CANVAS") return "absolute";
  return context.defaultCoordinateSpace || "absolute";
}

function localBox(layer, context, useVisible) {
  const b = useVisible ? visibleBoxOf(layer) : boxOf(layer);
  const space = sourceCoordinateSpace(layer, context);
  if (space === "local") {
    return {
      x: b.x + finite(context.localOffset && context.localOffset.x, 0),
      y: b.y + finite(context.localOffset && context.localOffset.y, 0),
      w: b.w,
      h: b.h,
    };
  }
  return {
    x: b.x - finite(context.sourceOrigin && context.sourceOrigin.x, 0),
    y: b.y - finite(context.sourceOrigin && context.sourceOrigin.y, 0),
    w: b.w,
    h: b.h,
  };
}

function childSourceOrigin(layer, context) {
  const b = boxOf(layer);
  if (sourceCoordinateSpace(layer, context) === "local") {
    return {
      x: finite(context.sourceOrigin && context.sourceOrigin.x, 0) + b.x,
      y: finite(context.sourceOrigin && context.sourceOrigin.y, 0) + b.y,
    };
  }
  return { x: b.x, y: b.y };
}

function safeSet(object, key, value, context, label) {
  if (!object || value === undefined || value === null || !(key in object)) return false;
  try {
    object[key] = value;
    return true;
  } catch (error) {
    if (context) context.warn(label || ("Could not set " + key), String(error && error.message || error));
    return false;
  }
}

function parseColor(value, fallback, alphaOverride) {
  const fb = fallback || { r: 0, g: 0, b: 0, a: 1 };
  if (value && typeof value === "object" && !Array.isArray(value)) {
    let r = finite(value.r, fb.r);
    let g = finite(value.g, fb.g);
    let b = finite(value.b, fb.b);
    if (r > 1 || g > 1 || b > 1) { r /= 255; g /= 255; b /= 255; }
    return {
      r: clamp(r, 0, 1),
      g: clamp(g, 0, 1),
      b: clamp(b, 0, 1),
      a: opacityValue(alphaOverride !== undefined ? alphaOverride : pick(value, "a", "alpha", "opacity"), fb.a),
    };
  }
  let input = String(value || "").trim();
  const rgbMatch = input.match(/^rgba?\(([^)]+)\)$/i);
  if (rgbMatch) {
    const parts = rgbMatch[1].split(",").map(function (part) { return Number(part.trim()); });
    return {
      r: clamp(finite(parts[0], 0) / 255, 0, 1),
      g: clamp(finite(parts[1], 0) / 255, 0, 1),
      b: clamp(finite(parts[2], 0) / 255, 0, 1),
      a: opacityValue(alphaOverride !== undefined ? alphaOverride : parts[3], 1),
    };
  }
  input = input.replace(/^#/, "");
  if (input.length === 3 || input.length === 4) input = input.split("").map(function (c) { return c + c; }).join("");
  if (!/^[0-9a-f]{6}([0-9a-f]{2})?$/i.test(input)) return Object.assign({}, fb, alphaOverride !== undefined ? { a: opacityValue(alphaOverride, fb.a) } : {});
  return {
    r: parseInt(input.slice(0, 2), 16) / 255,
    g: parseInt(input.slice(2, 4), 16) / 255,
    b: parseInt(input.slice(4, 6), 16) / 255,
    a: opacityValue(alphaOverride !== undefined ? alphaOverride : (input.length === 8 ? parseInt(input.slice(6, 8), 16) / 255 : 1), 1),
  };
}

function solidPaint(value, opacity) {
  const color = parseColor(value || "#000000", undefined, opacity);
  return {
    type: "SOLID",
    color: { r: color.r, g: color.g, b: color.b },
    opacity: color.a,
    visible: true,
    blendMode: "NORMAL",
  };
}

function gradientTransform(fill, box) {
  const supplied = pick(fill, "gradientTransform", "gradient_transform", "transform");
  if (Array.isArray(supplied) && supplied.length === 2 &&
      supplied.every(function (row) { return Array.isArray(row) && row.length === 3 && row.every(function (n) { return Number.isFinite(Number(n)); }); })) {
    return supplied.map(function (row) { return row.map(Number); });
  }
  // Figma's gradientTransform maps the node's *normalized* (0-1 x 0-1) bounding box into
  // paint space. It is not a plain rotation matrix: a rotation matrix built from a bare
  // angle ignores the node's actual aspect ratio, so a 45-degree gradient on a wide,
  // short layer would render far steeper (skewed toward the shorter axis) than the same
  // angle rendered directly in pixel space. render_preview.py (the QA ground truth) always
  // computes gradients in true pixel space, so the matrix has to be corrected for the
  // layer's width/height to reproduce the same visible angle.
  const w = Math.max(0.01, finite(box && box.w, 1));
  const h = Math.max(0.01, finite(box && box.h, 1));
  const kind = normalizedToken(pick(fill, "kind", "type"));
  if (kind.indexOf("RADIAL") >= 0) {
    // render_preview.py's radial gradient is a literal circle in pixel space (distance
    // from the box center normalized by the half-diagonal, reaching the last stop at the
    // corners) regardless of angle. Figma's paint space normalizes each axis by the
    // node's own width/height, so an isotropic (identity) transform would render as an
    // ellipse on any non-square layer instead of a circle. Scale each axis by w/h against
    // the half-diagonal so the two definitions match.
    const halfDiagonal = Math.max(0.01, 0.5 * Math.hypot(w, h));
    const sx = (0.5 * w) / halfDiagonal;
    const sy = (0.5 * h) / halfDiagonal;
    return [
      [sx, 0, 0.5 - sx / 2],
      [0, sy, 0.5 - sy / 2],
    ];
  }
  // LINEAR (and, as a best-effort proxy, ANGULAR/DIAMOND): build the transform from the
  // requested angle so the gradient's progression axis matches the same pixel-space
  // direction render_preview.py uses (0 deg = left-to-right, increasing clockwise since
  // image y grows downward), aspect-corrected per layer.
  const angle = finite(pick(fill, "angle", "rotation"), 0) * Math.PI / 180;
  const dx = Math.cos(angle);
  const dy = Math.sin(angle);
  function axisRow(ux, uy) {
    const extent = Math.max(0.01, Math.abs(ux) * w + Math.abs(uy) * h);
    const a = (ux * w) / extent;
    const b = (uy * h) / extent;
    return [a, b, 0.5 - (a + b) / 2];
  }
  return [axisRow(dx, dy), axisRow(-dy, dx)];
}

function gradientStopPosition(value, fallback) {
  let position = finite(value, fallback);
  // CSS/SVG style extraction frequently reports "50" instead of "0.5".
  if (position > 1 && position <= 100) position /= 100;
  return clamp(position, 0, 1);
}

function paintFromSpec(spec, box) {
  if (spec === undefined || spec === null) return null;
  if (typeof spec === "string") return solidPaint(spec);
  const kind = normalizedToken(pick(spec, "kind", "type"));
  if (!kind || kind === "FLAT" || kind === "SOLID" || spec.color) {
    const paint = solidPaint(spec.color || "#000000", pick(spec, "opacity", "alpha"));
    paint.visible = pick(spec, "visible") !== false;
    const blend = normalizedToken(pick(spec, "blend_mode", "blendMode"));
    if (blend) paint.blendMode = blend;
    return paint;
  }
  if (kind.indexOf("GRADIENT") >= 0 || kind === "LINEAR" || kind === "RADIAL" || kind === "ANGULAR" || kind === "DIAMOND") {
    const map = {
      LINEAR: "GRADIENT_LINEAR",
      RADIAL: "GRADIENT_RADIAL",
      ANGULAR: "GRADIENT_ANGULAR",
      DIAMOND: "GRADIENT_DIAMOND",
    };
    const type = map[kind] || (kind.indexOf("RADIAL") >= 0 ? "GRADIENT_RADIAL" : kind.indexOf("ANGULAR") >= 0 ? "GRADIENT_ANGULAR" : kind.indexOf("DIAMOND") >= 0 ? "GRADIENT_DIAMOND" : "GRADIENT_LINEAR");
    const rawStops = pick(spec, "stops", "gradientStops", "gradient_stops") || [];
    const stops = rawStops.map(function (stop, index) {
      const color = parseColor(stop.color || "#000000", undefined, pick(stop, "opacity", "alpha"));
      return {
        position: gradientStopPosition(pick(stop, "offset", "position"), rawStops.length > 1 ? index / (rawStops.length - 1) : 0),
        color: color,
      };
    });
    if (!stops.length) stops.push({ position: 0, color: parseColor("#000000") }, { position: 1, color: parseColor("#ffffff") });
    // Figma requires two stops. Keep a one-stop extraction visually faithful instead
    // of letting the whole paint assignment fail.
    if (stops.length === 1) stops.push({ position: 1, color: Object.assign({}, stops[0].color) });
    stops.sort(function (a, b) { return a.position - b.position; });
    return {
      type,
      gradientTransform: gradientTransform(spec, box),
      gradientStops: stops,
      opacity: opacityValue(spec.opacity, 1),
      visible: spec.visible !== false,
      blendMode: normalizedToken(pick(spec, "blend_mode", "blendMode")) || "NORMAL",
    };
  }
  return null;
}

function fillSpecs(layer) {
  const many = pick(layer, "fills", "paints");
  if (Array.isArray(many)) return many;
  const one = pick(layer, "fill", "background");
  if (one !== undefined && one !== null) return [one];
  const style = layer.style || {};
  const styleMany = pick(style, "fills", "paints");
  if (Array.isArray(styleMany)) return styleMany;
  const styleOne = pick(style, "fill", "background", "color");
  return styleOne === undefined || styleOne === null ? [] : [styleOne];
}

function applyFills(node, layer, context, allowEmpty) {
  if (!("fills" in node)) return;
  const specs = fillSpecs(layer);
  const box = boxOf(layer);
  const paints = specs.map(function (spec) { return paintFromSpec(spec, box); }).filter(Boolean);
  if (specs.length && !paints.length && context) {
    context.fidelity("unsupported_paint", (layer.name || layerId(layer)) + " has a paint that could not be represented natively.");
  }
  if (paints.length || allowEmpty) safeSet(node, "fills", paints, context, "Fill could not be applied");
}

function strokeSpecs(layer) {
  const many = pick(layer, "strokes");
  if (Array.isArray(many)) return many;
  const one = pick(layer, "stroke");
  if (one !== undefined && one !== null) return [one];
  const style = layer.style || {};
  const styleMany = pick(style, "strokes");
  if (Array.isArray(styleMany)) return styleMany;
  const styleOne = pick(style, "stroke");
  return styleOne === undefined || styleOne === null ? [] : [styleOne];
}

function applyStrokes(node, layer, context) {
  if (!("strokes" in node)) return;
  const specs = strokeSpecs(layer);
  if (!specs.length) return;
  const box = boxOf(layer);
  const paints = specs.map(function (stroke) {
    if (typeof stroke === "string") return solidPaint(stroke);
    return paintFromSpec(stroke.paint || stroke.color || stroke, box);
  }).filter(Boolean);
  if (!paints.length && context) context.fidelity("unsupported_stroke", (layer.name || layerId(layer)) + " has a stroke that could not be represented natively.");
  safeSet(node, "strokes", paints, context, "Stroke paint could not be applied");
  const first = typeof specs[0] === "object" ? specs[0] : {};
  const weight = finite(pick(first, "width", "weight", "strokeWeight", "stroke_weight"), finite(pick(layer, "stroke_width", "strokeWidth"), 1));
  safeSet(node, "strokeWeight", Math.max(0, weight), context);
  const align = normalizedToken(pick(first, "align", "alignment", "strokeAlign", "stroke_align"));
  if (align) safeSet(node, "strokeAlign", align, context);
  const caps = normalizedToken(pick(first, "cap", "strokeCap", "stroke_cap"));
  if (caps) safeSet(node, "strokeCap", caps, context);
  const join = normalizedToken(pick(first, "join", "strokeJoin", "stroke_join"));
  if (join) safeSet(node, "strokeJoin", join, context);
  const dashes = pick(first, "dash", "dashes", "dashPattern", "dash_pattern");
  if (Array.isArray(dashes)) safeSet(node, "dashPattern", dashes.map(function (n) { return Math.max(0, finite(n, 0)); }), context);
}

function effectFromSpec(spec) {
  const type = normalizedToken(pick(spec, "type", "kind"));
  if (type === "DROP_SHADOW" || type === "SHADOW" || type === "INNER_SHADOW") {
    const offset = pick(spec, "offset") || {};
    return {
      type: type === "INNER_SHADOW" ? "INNER_SHADOW" : "DROP_SHADOW",
      color: parseColor(spec.color || "#00000040", undefined, pick(spec, "opacity", "alpha")),
      offset: {
        x: finite(pick(offset, "x"), finite(pick(spec, "x", "offsetX", "offset_x"), 0)),
        y: finite(pick(offset, "y"), finite(pick(spec, "y", "offsetY", "offset_y"), 4)),
      },
      radius: Math.max(0, finite(pick(spec, "radius", "blur"), 8)),
      spread: finite(pick(spec, "spread"), 0),
      visible: spec.visible !== false,
      blendMode: normalizedToken(pick(spec, "blend_mode", "blendMode")) || "NORMAL",
    };
  }
  if (type === "BLUR" || type === "LAYER_BLUR" || type === "BACKGROUND_BLUR") {
    return {
      type: type === "BACKGROUND_BLUR" ? "BACKGROUND_BLUR" : "LAYER_BLUR",
      radius: Math.max(0, finite(pick(spec, "radius", "blur"), 8)),
      visible: spec.visible !== false,
    };
  }
  return null;
}

function applyEffects(node, layer, context) {
  if (!("effects" in node)) return;
  const style = layer.style || {};
  const specs = Array.isArray(layer.effects) ? layer.effects :
    (Array.isArray(style.effects) ? style.effects : (layer.shadow ? [Object.assign({ type: "drop-shadow" }, layer.shadow)] : []));
  const effects = specs.map(effectFromSpec).filter(Boolean);
  if (specs.length && !effects.length && context) {
    context.fidelity("unsupported_effect", (layer.name || layerId(layer)) + " has an effect that could not be represented natively.");
  }
  if (effects.length) safeSet(node, "effects", effects, context, "Effect could not be applied");
}

function applyRadius(node, layer, context) {
  const style = layer.style || {};
  const meta = layer.meta || {};
  const radius = pick(layer, "radius", "corner_radius", "cornerRadius");
  const value = radius !== undefined ? radius : pick(style, "radius", "corner_radius", "cornerRadius") ?? pick(meta, "radius", "corner_radius", "cornerRadius");
  if (value === undefined || value === null) return;
  if (typeof value === "number") {
    safeSet(node, "cornerRadius", Math.max(0, value), context);
    return;
  }
  if (typeof value === "object") {
    safeSet(node, "topLeftRadius", Math.max(0, finite(pick(value, "topLeft", "top_left", "tl"), 0)), context);
    safeSet(node, "topRightRadius", Math.max(0, finite(pick(value, "topRight", "top_right", "tr"), 0)), context);
    safeSet(node, "bottomRightRadius", Math.max(0, finite(pick(value, "bottomRight", "bottom_right", "br"), 0)), context);
    safeSet(node, "bottomLeftRadius", Math.max(0, finite(pick(value, "bottomLeft", "bottom_left", "bl"), 0)), context);
  }
}

function applyConstraints(node, layer, context) {
  if (!("constraints" in node)) return;
  const source = layer.constraints;
  if (!source) return;
  let horizontal = normalizedToken(pick(source, "horizontal", "x")) || "LEFT";
  let vertical = normalizedToken(pick(source, "vertical", "y")) || "TOP";
  if (horizontal === "STRETCH") horizontal = "LEFT_RIGHT";
  if (vertical === "STRETCH") vertical = "TOP_BOTTOM";
  safeSet(node, "constraints", { horizontal, vertical }, context, "Constraints could not be applied");
}

function applyCommon(node, layer, context) {
  if (!node) return;
  node.name = String(layer.name || layerId(layer));
  try {
    node.setPluginData(LAYER_KEY, layerId(layer));
    node.setPluginData("adDecompilerNodeType", canonicalType(layer));
    node.setPluginData("adDecompilerZIndex", String(zIndexOf(layer)));
  } catch (error) {
    context.warn("Layer metadata could not be attached", String(error && error.message || error));
  }
  safeSet(node, "opacity", opacityValue(pick(layer, "opacity"), 1), context);
  const rotation = finite(pick(layer, "rotation"), 0);
  if (rotation) safeSet(node, "rotation", rotation, context);
  const blend = normalizedToken(pick(layer, "blend_mode", "blendMode"));
  if (blend) safeSet(node, "blendMode", blend, context, "Blend mode " + blend + " was not supported");
  if (layer.visible === false) safeSet(node, "visible", false, context);
  if (layer.locked === true) safeSet(node, "locked", true, context);
  applyConstraints(node, layer, context);
  applyEffects(node, layer, context);
}

function setGeometry(node, layer, context, useVisible) {
  const b = localBox(layer, context, useVisible);
  safeSet(node, "x", b.x, context);
  safeSet(node, "y", b.y, context);
  try {
    node.resize(Math.max(0.01, b.w), Math.max(0.01, b.h));
  } catch (error) {
    context.warn("Layer size could not be applied", layerId(layer) + ": " + String(error && error.message || error));
  }
  return b;
}

function textStyleOf(layer) {
  return Object.assign({}, layer.typography || {}, layer.style || {});
}

function weightFromStyleName(styleName) {
  const token = normalizedToken(styleName);
  if (token.indexOf("THIN") >= 0) return 100;
  if (token.indexOf("EXTRA_LIGHT") >= 0 || token.indexOf("ULTRA_LIGHT") >= 0) return 200;
  if (token.indexOf("LIGHT") >= 0) return 300;
  if (token.indexOf("MEDIUM") >= 0) return 500;
  if (token.indexOf("SEMI_BOLD") >= 0 || token.indexOf("DEMIBOLD") >= 0) return 600;
  if (token.indexOf("EXTRA_BOLD") >= 0 || token.indexOf("ULTRA_BOLD") >= 0) return 800;
  if (token.indexOf("BLACK") >= 0 || token.indexOf("HEAVY") >= 0) return 900;
  if (token.indexOf("BOLD") >= 0) return 700;
  return 400;
}

function requestedFont(style) {
  const family = String(pick(style, "fontFamily", "font_family", "family") || "Inter").trim();
  const exactStyle = pick(style, "fontStyle", "font_style", "styleName", "style_name", "style");
  const italic = style.italic === true || /italic|oblique/i.test(String(exactStyle || ""));
  const weight = clamp(finite(pick(style, "fontWeight", "font_weight", "weight"), weightFromStyleName(exactStyle)), 1, 1000);
  return { family, style: exactStyle ? String(exactStyle) : null, italic, weight };
}

function rankedFontRequests(style) {
  const rawCandidates = pick(style, "fontCandidates", "font_candidates", "candidates");
  const candidates = Array.isArray(rawCandidates) ? rawCandidates : [];
  const rawWeights = pick(style, "fontWeightCandidates", "font_weight_candidates");
  const weightCandidates = Array.isArray(rawWeights) ? rawWeights : [];
  const output = [];
  const hasPrimaryFamily = Boolean(pick(style, "fontFamily", "font_family", "family"));
  if (hasPrimaryFamily) {
    const primary = Object.assign(requestedFont(style), { score: finite(style.confidence, NaN), source: "primary" });
    if (weightCandidates.length && pick(style, "fontWeight", "font_weight", "weight") !== undefined) {
      primary.style = null;
    }
    output.push(primary);
    weightCandidates.forEach(function (entry) {
      const value = typeof entry === "object" ? pick(entry, "value", "weight", "fontWeight", "font_weight") : entry;
      const weight = finite(value, NaN);
      if (!Number.isFinite(weight)) return;
      output.push(Object.assign({}, primary, {
        weight,
        score: finite(entry && entry.score, NaN),
        source: "weight_candidate",
      }));
    });
  }
  candidates.forEach(function (candidate) {
    const source = typeof candidate === "string" ? { fontFamily: candidate } : candidate || {};
    output.push(Object.assign(requestedFont(source), {
      score: finite(source.score, NaN),
      source: source.source || "candidate",
    }));
  });
  if (!hasPrimaryFamily && weightCandidates.length) {
    const base = output.length ? output[0] : requestedFont(style);
    weightCandidates.forEach(function (entry) {
      const value = typeof entry === "object" ? pick(entry, "value", "weight", "fontWeight", "font_weight") : entry;
      const weight = finite(value, NaN);
      if (!Number.isFinite(weight)) return;
      output.push(Object.assign({}, base, {
        style: null,
        weight,
        score: finite(entry && entry.score, NaN),
        source: "weight_candidate",
      }));
    });
  }
  if (!output.length) output.push(Object.assign(requestedFont(style), { score: finite(style.confidence, NaN), source: "fallback" }));
  const seen = new Set();
  return output.filter(function (candidate) {
    const key = candidate.family.toLowerCase() + "\u0000" + String(candidate.style || "").toLowerCase() + "\u0000" + candidate.weight;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

class FontResolver {
  constructor(context) {
    this.context = context;
    this.fonts = [];
    this.loaded = new Map();
    this.ready = this.initialize();
  }

  async initialize() {
    try {
      this.fonts = await figma.listAvailableFontsAsync();
    } catch (error) {
      this.fonts = [];
    }
  }

  async resolve(style, label) {
    await this.ready;
    const requests = rankedFontRequests(style);
    const requested = requests[0];
    let chosen = null;
    let chosenRank = -1;
    for (let rank = 0; rank < requests.length && !chosen; rank += 1) {
      const request = requests[rank];
      const sameFamily = this.fonts.filter(function (entry) {
        return entry.fontName.family.toLowerCase() === request.family.toLowerCase();
      });
      if (!sameFamily.length) continue;
      if (request.style) {
        chosen = sameFamily.find(function (entry) {
          return entry.fontName.style.toLowerCase() === request.style.toLowerCase();
        }) || null;
      }
      if (!chosen) {
        chosen = sameFamily
          .map(function (entry) {
            const entryItalic = /italic|oblique/i.test(entry.fontName.style);
            const penalty = Math.abs(weightFromStyleName(entry.fontName.style) - request.weight) + (entryItalic === request.italic ? 0 : 250);
            return { entry, penalty };
          })
          .sort(function (a, b) { return a.penalty - b.penalty; })[0].entry;
      }
      if (chosen) chosenRank = rank;
    }
    if (!chosen) {
      chosen = this.fonts.find(function (entry) {
        return entry.fontName.family === "Inter" && entry.fontName.style === "Regular";
      }) || this.fonts[0] || { fontName: { family: "Inter", style: "Regular" } };
    }
    const result = chosen.fontName;
    const substituted = chosenRank !== 0 || result.family.toLowerCase() !== requested.family.toLowerCase() || (requested.style && result.style.toLowerCase() !== requested.style.toLowerCase());
    this.context.report.fonts.requested += 1;
    this.context.report.fonts.selections.push({
      label: label || "Text",
      requested: requested.family + (requested.style ? " " + requested.style : ""),
      selected: result.family + " " + result.style,
      rank: chosenRank >= 0 ? chosenRank + 1 : null,
      score: chosenRank >= 0 && Number.isFinite(requests[chosenRank].score) ? requests[chosenRank].score : null,
    });
    if (substituted) {
      this.context.report.fonts.substituted += 1;
      const rankDetail = chosenRank > 0
        ? "candidate #" + (chosenRank + 1) + (Number.isFinite(requests[chosenRank].score) ? " (score " + requests[chosenRank].score.toFixed(3) + ")" : "")
        : chosenRank === 0 ? "closest installed style" : "generic Figma fallback";
      this.context.warn(
        chosenRank > 0 ? "Ranked font candidate selected" : "Font substituted",
        (label || "Text") + ": " + requested.family + (requested.style ? " " + requested.style : "") + " → " + result.family + " " + result.style + " · " + rankDetail
      );
    }
    await this.load(result);
    return result;
  }

  async load(fontName) {
    const key = fontName.family + "\u0000" + fontName.style;
    if (!this.loaded.has(key)) this.loaded.set(key, figma.loadFontAsync(fontName));
    return this.loaded.get(key);
  }
}

function lineHeightValue(value) {
  if (value === undefined || value === null || value === "auto") return { unit: "AUTO" };
  if (typeof value === "object") {
    const unit = normalizedToken(value.unit || "PIXELS");
    if (unit === "AUTO") return { unit: "AUTO" };
    return { unit: unit === "PERCENT" || unit === "PERCENTAGE" ? "PERCENT" : "PIXELS", value: finite(value.value, 0) };
  }
  if (typeof value === "string" && value.trim().endsWith("%")) return { unit: "PERCENT", value: finite(parseFloat(value), 100) };
  return { unit: "PIXELS", value: Math.max(0, finite(value, 0)) };
}

function spacingValue(value) {
  if (value === undefined || value === null) return null;
  if (typeof value === "object") {
    return {
      unit: normalizedToken(value.unit) === "PERCENT" ? "PERCENT" : "PIXELS",
      value: finite(value.value, 0),
    };
  }
  if (typeof value === "string" && value.trim().endsWith("%")) return { unit: "PERCENT", value: finite(parseFloat(value), 0) };
  return { unit: "PIXELS", value: finite(value, 0) };
}

function textCaseValue(style) {
  const explicit = normalizedToken(pick(style, "textCase", "text_case", "case"));
  if (explicit) return explicit;
  if (style.uppercase === true) return "UPPER";
  return "ORIGINAL";
}

function alignmentValue(value, fallback) {
  const token = normalizedToken(value);
  if (token === "START") return "LEFT";
  if (token === "END") return "RIGHT";
  if (["LEFT", "CENTER", "RIGHT", "JUSTIFIED"].indexOf(token) >= 0) return token;
  return fallback;
}

function verticalAlignmentValue(value, fallback) {
  const token = normalizedToken(value);
  if (token === "MIDDLE") return "CENTER";
  if (["TOP", "CENTER", "BOTTOM"].indexOf(token) >= 0) return token;
  return fallback;
}

function textContent(layer) {
  if (layer.text !== undefined && layer.text !== null) return String(layer.text);
  const runs = pick(layer, "text_runs", "textRuns", "runs");
  if (Array.isArray(runs)) return runs.map(function (run) { return String(run.text || ""); }).join("");
  return "";
}

function normalizedRuns(layer, content) {
  const runs = pick(layer, "text_runs", "textRuns", "runs");
  if (!Array.isArray(runs)) return [];
  let cursor = 0;
  return runs.map(function (run) {
    const start = clamp(finite(pick(run, "start", "from"), cursor), 0, content.length);
    const inferredEnd = run.text !== undefined ? start + String(run.text).length : content.length;
    const end = clamp(finite(pick(run, "end", "to"), inferredEnd), start, content.length);
    cursor = end;
    return { start, end, style: Object.assign({}, run.typography || {}, run.style || {}, run) };
  }).filter(function (run) { return run.end > run.start; });
}

async function applyTextRuns(node, layer, content, context) {
  const runs = normalizedRuns(layer, content);
  for (let i = 0; i < runs.length; i += 1) {
    const run = runs[i];
    const style = run.style;
    try {
      const fontName = await context.fonts.resolve(style, layer.name || layerId(layer));
      node.setRangeFontName(run.start, run.end, fontName);
      const size = finite(pick(style, "fontSize", "font_size", "size"), NaN);
      if (Number.isFinite(size)) node.setRangeFontSize(run.start, run.end, Math.max(1, size));
      const color = pick(style, "color", "fill");
      if (color) node.setRangeFills(run.start, run.end, [solidPaint(color, style.opacity)]);
      const spacing = spacingValue(pick(style, "letterSpacing", "letter_spacing", "tracking"));
      if (spacing) node.setRangeLetterSpacing(run.start, run.end, spacing);
      const lh = pick(style, "lineHeight", "line_height", "leading");
      if (lh !== undefined) node.setRangeLineHeight(run.start, run.end, lineHeightValue(lh));
      const decoration = normalizedToken(pick(style, "textDecoration", "text_decoration", "decoration"));
      if (decoration) node.setRangeTextDecoration(run.start, run.end, decoration);
      const textCase = textCaseValue(style);
      if (textCase !== "ORIGINAL") node.setRangeTextCase(run.start, run.end, textCase);
    } catch (error) {
      context.warn("Text run style skipped", (layer.name || layerId(layer)) + ": " + String(error && error.message || error));
    }
  }
  return runs;
}

function pngDimensions(bytes) {
  if (!bytes || bytes.length < 24) return null;
  const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (data[0] !== 137 || data[1] !== 80 || data[2] !== 78 || data[3] !== 71) return null;
  try {
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    const width = view.getUint32(16, false);
    const height = view.getUint32(20, false);
    return width > 0 && height > 0 ? { width, height, source: "png" } : null;
  } catch (_) {
    return null;
  }
}

async function renderedTextDimensions(node) {
  try {
    const bytes = await node.exportAsync({ format: "PNG", constraint: { type: "SCALE", value: 1 } });
    const parsed = pngDimensions(bytes);
    if (parsed) return parsed;
  } catch (_) {}
  return { width: Math.max(0.01, node.width), height: Math.max(0.01, node.height), source: "node" };
}

function renderedFits(dimensions, target, multiline) {
  const widthFits = multiline || dimensions.width <= target.w + 0.75;
  return widthFits && dimensions.height <= target.h + 0.75;
}

async function binaryFitFontSize(node, target, multiline) {
  const initialSize = Math.max(1, finite(node.fontSize, 12));
  const initialLineHeight = node.lineHeight && node.lineHeight.unit === "PIXELS"
    ? finite(node.lineHeight.value, initialSize * 1.15)
    : null;
  let low = 1;
  let high = clamp(Math.max(initialSize * 3, target.h * 4, multiline ? target.h * 2 : target.w * 1.25), 2, 1000);
  let best = 1;
  for (let iteration = 0; iteration < 9; iteration += 1) {
    const midpoint = (low + high) / 2;
    node.fontSize = midpoint;
    if (initialLineHeight !== null) node.lineHeight = { unit: "PIXELS", value: Math.max(0.01, initialLineHeight * midpoint / initialSize) };
    if (multiline) node.resize(Math.max(1, target.w), Math.max(1, node.height));
    const dimensions = await renderedTextDimensions(node);
    if (renderedFits(dimensions, target, multiline)) {
      best = midpoint;
      low = midpoint;
    } else {
      high = midpoint;
    }
  }
  node.fontSize = clamp(best, 1, 1000);
  if (initialLineHeight !== null) node.lineHeight = { unit: "PIXELS", value: Math.max(0.01, initialLineHeight * node.fontSize / initialSize) };
  if (multiline) node.resize(Math.max(1, target.w), Math.max(1, node.height));
}

async function binaryFitMultilineLineHeight(node, target) {
  let low = Math.max(0.01, finite(node.fontSize, 12) * 0.65);
  let high = Math.max(low + 0.5, target.h * 1.25);
  let best = low;
  for (let iteration = 0; iteration < 7; iteration += 1) {
    const midpoint = (low + high) / 2;
    node.lineHeight = { unit: "PIXELS", value: midpoint };
    node.resize(Math.max(1, target.w), Math.max(1, node.height));
    const dimensions = await renderedTextDimensions(node);
    if (dimensions.height <= target.h + 0.75) {
      best = midpoint;
      low = midpoint;
    } else {
      high = midpoint;
    }
  }
  node.lineHeight = { unit: "PIXELS", value: best };
  node.resize(Math.max(1, target.w), Math.max(1, node.height));
}

async function fitTextToVisibleBox(node, layer, style, context, hasRuns) {
  const target = localBox(layer, context, true);
  const content = node.characters || "";
  const multiline = content.indexOf("\n") >= 0 || style.multiline === true || finite(pick(style, "lineCount", "line_count"), 1) > 1;
  const explicitFit = pick(style, "fit", "fitText", "fit_text", "fitMode", "fit_mode") !== undefined
    ? pick(style, "fit", "fitText", "fit_text", "fitMode", "fit_mode")
    : pick(layer, "fit_text", "fitText");
  const hasExplicitSize = Number.isFinite(Number(pick(style, "fontSize", "font_size", "size")));
  const hasVisibleBox = Boolean(pick(layer, "visible_box", "visibleBox"));
  if (multiline) {
    node.textAutoResize = "HEIGHT";
    node.resize(Math.max(1, target.w), Math.max(1, target.h));
  } else {
    node.textAutoResize = "WIDTH_AND_HEIGHT";
  }
  const overflows = node.width > target.w * 1.015 || node.height > target.h * 1.015;
  const shouldFit = explicitFit !== false && normalizedToken(explicitFit) !== "NONE" && (!hasExplicitSize || explicitFit === true || hasVisibleBox || overflows);
  const role = layerRole(layer);
  const buttonText = context.inAutoLayout && (isButtonLikeRole(role) || role === "LABEL");
  if (buttonText) {
    if (!multiline) node.textAutoResize = "WIDTH_AND_HEIGHT";
    if (shouldFit && !hasRuns && content.length) {
      if (hasVisibleBox) {
        await binaryFitFontSize(node, target, multiline);
        if (multiline && normalizedToken(pick(style, "lockLineHeight", "lock_line_height")) !== "TRUE") {
          await binaryFitMultilineLineHeight(node, target);
        }
      } else {
        const dimensions = await renderedTextDimensions(node);
        const widthRatio = multiline ? 1 : target.w / Math.max(0.01, dimensions.width);
        const heightRatio = target.h / Math.max(0.01, dimensions.height);
        const ratio = clamp(Math.min(widthRatio, heightRatio), 0.35, 2.5);
        node.fontSize = clamp(finite(node.fontSize, 12) * ratio * 0.995, 1, 1000);
        if (multiline) node.resize(Math.max(1, target.w), Math.max(1, node.height));
      }
    }
    node.x = 0;
    node.y = 0;
    return;
  }
  if (shouldFit && !hasRuns && content.length) {
    if (hasVisibleBox) {
      await binaryFitFontSize(node, target, multiline);
      if (multiline && normalizedToken(pick(style, "lockLineHeight", "lock_line_height")) !== "TRUE") {
        await binaryFitMultilineLineHeight(node, target);
      }
    } else {
      const dimensions = await renderedTextDimensions(node);
      const widthRatio = multiline ? 1 : target.w / Math.max(0.01, dimensions.width);
      const heightRatio = target.h / Math.max(0.01, dimensions.height);
      const ratio = clamp(Math.min(widthRatio, heightRatio), 0.35, 2.5);
      node.fontSize = clamp(finite(node.fontSize, 12) * ratio * 0.995, 1, 1000);
      if (multiline) node.resize(Math.max(1, target.w), Math.max(1, node.height));
    }
    const explicitSpacing = pick(style, "letterSpacing", "letter_spacing", "tracking") !== undefined;
    if (!multiline && !explicitSpacing && content.length > 1) {
      const dimensions = await renderedTextDimensions(node);
      const adjustment = (target.w - dimensions.width) / (content.length - 1);
      if (Math.abs(adjustment) <= finite(node.fontSize, 12) * 0.22) {
        node.letterSpacing = { unit: "PIXELS", value: adjustment };
      }
    }
  }
  const horizontal = alignmentValue(pick(style, "align", "textAlign", "text_align"), "LEFT");
  const vertical = verticalAlignmentValue(pick(style, "verticalAlign", "vertical_align"), "CENTER");
  const xFactor = horizontal === "CENTER" ? 0.5 : horizontal === "RIGHT" ? 1 : 0;
  const yFactor = vertical === "TOP" ? 0 : vertical === "BOTTOM" ? 1 : 0.5;
  node.x = target.x + (target.w - node.width) * xFactor;
  node.y = target.y + (target.h - node.height) * yFactor;
}

function styleSignature(style, fontName, node) {
  return JSON.stringify({
    font: fontName,
    size: finite(node.fontSize, 12),
    lineHeight: node.lineHeight,
    letterSpacing: node.letterSpacing,
    leadingTrim: node.leadingTrim,
    textCase: node.textCase,
    decoration: node.textDecoration,
  });
}

async function attachTextStyle(node, layer, style, fontName, context, hasRuns) {
  if (hasRuns) return;
  const explicit = pick(style, "styleName", "style_name") || pick(layer, "text_style_name", "textStyleName");
  const role = layer.meta && pick(layer.meta, "hierarchy", "role");
  if (!explicit && !role) return;
  const label = String(explicit || role || "Text").replace(/\//g, " ").trim();
  const key = context.docId + "|" + label + "|" + styleSignature(style, fontName, node);
  let textStyle = context.textStyles.get(key);
  if (!textStyle) {
    textStyle = context.localTextStyles.find(function (candidate) {
      try { return candidate.getPluginData(STYLE_KEY) === key; } catch (_) { return false; }
    });
  }
  if (!textStyle) textStyle = figma.createTextStyle();
  try {
    textStyle.name = "Ad Decompiler/" + context.docName + "/" + label;
    textStyle.fontName = fontName;
    textStyle.fontSize = finite(node.fontSize, 12);
    textStyle.lineHeight = node.lineHeight;
    textStyle.letterSpacing = node.letterSpacing;
    if ("leadingTrim" in textStyle && node.leadingTrim) textStyle.leadingTrim = node.leadingTrim;
    textStyle.textCase = node.textCase;
    textStyle.textDecoration = node.textDecoration;
    textStyle.setPluginData(STYLE_KEY, key);
    context.textStyles.set(key, textStyle);
    if (typeof node.setTextStyleIdAsync === "function") await node.setTextStyleIdAsync(textStyle.id);
    else node.textStyleId = textStyle.id;
  } catch (error) {
    context.warn("Text style link skipped", (layer.name || layerId(layer)) + ": " + String(error && error.message || error));
  }
}

async function createTextLayer(layer, parent, context) {
  const node = figma.createText();
  parent.appendChild(node);
  const style = textStyleOf(layer);
  const fontName = await context.fonts.resolve(style, layer.name || layerId(layer));
  node.fontName = fontName;
  const content = textContent(layer);
  node.characters = content;
  node.autoRename = false;
  node.fontSize = Math.max(1, finite(pick(style, "fontSize", "font_size", "size"), visibleBoxOf(layer).h * 0.78 || 12));
  const lineHeight = pick(style, "lineHeight", "line_height", "leading");
  if (lineHeight !== undefined) node.lineHeight = lineHeightValue(lineHeight);
  const spacing = spacingValue(pick(style, "letterSpacing", "letter_spacing", "tracking"));
  if (spacing) node.letterSpacing = spacing;
  node.textAlignHorizontal = alignmentValue(pick(style, "align", "textAlign", "text_align"), "LEFT");
  node.textAlignVertical = verticalAlignmentValue(pick(style, "verticalAlign", "vertical_align"), "TOP");
  safeSet(node, "leadingTrim", normalizedToken(pick(style, "leadingTrim", "leading_trim")) || "CAP_HEIGHT", context, "Cap-height trim was not available for this font");
  node.textCase = textCaseValue(style);
  const decoration = normalizedToken(pick(style, "textDecoration", "text_decoration", "decoration"));
  if (decoration) node.textDecoration = decoration;
  // Text can use the same native paint model as shapes: solid, gradient, or multiple
  // layered fills. This keeps outlined/gradient text editable instead of rasterizing it.
  applyFills(node, layer, context, false);
  const runs = await applyTextRuns(node, layer, content, context);
  await fitTextToVisibleBox(node, layer, style, context, runs.length > 0);
  await attachTextStyle(node, layer, style, fontName, context, runs.length > 0);
  applyStrokes(node, layer, context);
  applyCommon(node, layer, context);
  return node;
}

function shapeKindOf(layer) {
  const explicit = normalizedToken(pick(layer, "shape_kind", "shapeKind", "kind"));
  if (explicit) return explicit;
  const type = normalizedToken(layer.type);
  if (type === "ELLIPSE" || type === "CIRCLE") return "ELLIPSE";
  return "RECT";
}

async function createShapeLayer(layer, parent, context) {
  const kind = shapeKindOf(layer);
  let node;
  if (kind === "ELLIPSE" || kind === "CIRCLE" || kind === "OVAL") node = figma.createEllipse();
  else if (kind === "LINE" && typeof figma.createLine === "function") node = figma.createLine();
  else if (kind === "POLYGON" && typeof figma.createPolygon === "function") node = figma.createPolygon();
  else if (kind === "STAR" && typeof figma.createStar === "function") node = figma.createStar();
  else node = figma.createRectangle();
  parent.appendChild(node);
  setGeometry(node, layer, context, false);
  applyFills(node, layer, context, true);
  applyStrokes(node, layer, context);
  applyRadius(node, layer, context);
  applyCommon(node, layer, context);
  if (!fillSpecs(layer).length && !strokeSpecs(layer).length) context.warn("Shape has no paint", layer.name || layerId(layer));
  return node;
}

function escapeXml(value) {
  return String(value).replace(/&/g, "&amp;").replace(/\"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function vectorPathsOf(layer) {
  const fromLayer = pick(layer, "vector_paths", "vectorPaths", "paths");
  if (Array.isArray(fromLayer) && fromLayer.length) return fromLayer;
  const fromMeta = layer.meta && pick(layer.meta, "vector_paths", "vectorPaths");
  if (Array.isArray(fromMeta) && fromMeta.length) return fromMeta;
  const path = pick(layer, "path", "d");
  return path ? [{ d: path, fill: layer.fill, stroke: layer.stroke }] : [];
}

function svgForLayer(layer) {
  const raw = pick(layer, "svg", "svg_string", "svgString");
  if (typeof raw === "string" && raw.trim().indexOf("<svg") >= 0) return raw;
  const paths = vectorPathsOf(layer);
  if (!paths.length) return null;
  const box = boxOf(layer);
  const view = pick(layer, "view_box", "viewBox") || {};
  const vx = finite(pick(view, "x"), 0);
  const vy = finite(pick(view, "y"), 0);
  const vw = Math.max(0.01, finite(pick(view, "w", "width"), box.w));
  const vh = Math.max(0.01, finite(pick(view, "h", "height"), box.h));
  const layerFill = fillSpecs(layer)[0];
  const fallbackFill = typeof layerFill === "string" ? layerFill : (layerFill && layerFill.color) || "#000000";
  const body = paths.map(function (entry) {
    const d = typeof entry === "string" ? entry : pick(entry, "d", "path");
    if (!d) return "";
    const fillSpec = typeof entry === "object" ? pick(entry, "fill", "color") : null;
    const fill = typeof fillSpec === "string" ? fillSpec : (fillSpec && fillSpec.color) || fallbackFill;
    const strokeSpec = typeof entry === "object" ? entry.stroke : null;
    const stroke = typeof strokeSpec === "string" ? strokeSpec : (strokeSpec && strokeSpec.color) || "none";
    const width = typeof strokeSpec === "object" ? finite(pick(strokeSpec, "width", "weight"), 1) : 1;
    const rule = normalizedToken(typeof entry === "object" && pick(entry, "windingRule", "winding_rule", "fillRule", "fill_rule")) === "EVENODD" ? "evenodd" : "nonzero";
    const fillAlpha = typeof fillSpec === "object" ? opacityValue(pick(fillSpec, "opacity", "alpha"), 1) : 1;
    const strokeAlpha = typeof strokeSpec === "object" ? opacityValue(pick(strokeSpec, "opacity", "alpha"), 1) : 1;
    const cap = normalizedToken(typeof strokeSpec === "object" && pick(strokeSpec, "cap", "strokeCap", "stroke_cap"));
    const join = normalizedToken(typeof strokeSpec === "object" && pick(strokeSpec, "join", "strokeJoin", "stroke_join"));
    const dashes = typeof strokeSpec === "object" && pick(strokeSpec, "dash", "dashes", "dashPattern", "dash_pattern");
    const capSvg = { ROUND: "round", SQUARE: "square", BUTT: "butt" }[cap] || "";
    const joinSvg = { ROUND: "round", BEVEL: "bevel", MITER: "miter" }[join] || "";
    const dashSvg = Array.isArray(dashes) ? dashes.map(function (n) { return Math.max(0, finite(n, 0)); }).join(" ") : "";
    const pathOpacity = typeof entry === "object" ? opacityValue(pick(entry, "opacity", "alpha"), 1) : 1;
    return '<path d="' + escapeXml(d) + '" fill="' + escapeXml(fill || "none") + '" fill-opacity="' + fillAlpha + '" fill-rule="' + rule + '" stroke="' + escapeXml(stroke) + '" stroke-width="' + width + '" stroke-opacity="' + strokeAlpha + '"' +
      (capSvg ? ' stroke-linecap="' + capSvg + '"' : "") + (joinSvg ? ' stroke-linejoin="' + joinSvg + '"' : "") +
      (dashSvg ? ' stroke-dasharray="' + dashSvg + '"' : "") + (pathOpacity < 1 ? ' opacity="' + pathOpacity + '"' : "") + '/>';
  }).join("");
  return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="' + vx + " " + vy + " " + vw + " " + vh + '">' + body + "</svg>";
}

function maskLayerFor(mask, layer) {
  const source = mask || {};
  const explicit = pick(source, "box", "bounds");
  const hasBareBox = ["x", "y", "w", "h", "width", "height", "left", "top"].some(function (key) { return source[key] !== undefined; });
  const maskBox = explicit || (hasBareBox ? source : null);
  const result = Object.assign({}, layer, {
    name: (layer.name || layerId(layer)) + " — mask",
    box: maskBox || boxOf(layer),
    visible_box: undefined,
    visibleBox: undefined,
  });
  const position = pick(source, "coordinate_space", "coordinateSpace", "position_mode", "positionMode");
  if (position) result.coordinate_space = position;
  return result;
}

async function createVectorLayer(layer, parent, context) {
  const svg = svgForLayer(layer);
  if (!svg) throw new Error("Vector layer has no SVG or path geometry");
  let node;
  try {
    node = figma.createNodeFromSvg(svg);
  } catch (error) {
    throw new Error("SVG import failed: " + String(error && error.message || error));
  }
  parent.appendChild(node);
  setGeometry(node, layer, context, false);
  applyCommon(node, layer, context);
  return node;
}

function bytesForAsset(context, source) {
  if (!source) return null;
  const raw = context.assets[source] || context.assets[String(source).replace(/^\.\//, "")];
  if (!raw) return null;
  return raw instanceof Uint8Array ? raw : new Uint8Array(raw);
}

function imagePaint(image, layer) {
  const imageSpec = pick(layer, "image", "image_fill", "imageFill") || {};
  const requested = normalizedToken(pick(imageSpec, "scale_mode", "scaleMode") || pick(layer, "scale_mode", "scaleMode"));
  const allowed = ["FILL", "FIT", "CROP", "TILE"];
  const paint = {
    type: "IMAGE",
    scaleMode: allowed.indexOf(requested) >= 0 ? requested : "FILL",
    imageHash: image.hash,
    opacity: opacityValue(pick(imageSpec, "opacity"), 1),
    visible: true,
  };
  const transform = pick(imageSpec, "image_transform", "imageTransform", "transform");
  if (Array.isArray(transform) && transform.length === 2) {
    paint.scaleMode = "CROP";
    paint.imageTransform = transform;
  }
  // ImagePaint.rotation is only valid in increments of 90 degrees (0/90/180/270); Figma
  // throws on any other value, which would otherwise take down the whole image layer.
  // Snap to the closest supported increment instead of passing an arbitrary angle through.
  const rawRotation = finite(pick(imageSpec, "rotation"), 0);
  const rotation = ((Math.round(rawRotation / 90) * 90) % 360 + 360) % 360;
  if (rotation) paint.rotation = rotation;
  return paint;
}

function createMaskGeometry(mask, layer, parent, context) {
  const proxy = maskLayerFor(mask, layer);
  const kind = normalizedToken(pick(mask, "kind", "type"));
  if (kind === "PATH" || mask.path || mask.svg) {
    proxy.svg = mask.svg;
    proxy.path = mask.path;
    proxy.vector_paths = pick(mask, "vector_paths", "vectorPaths", "paths");
    proxy.fill = "#ffffff";
    const svg = svgForLayer(proxy);
    if (!svg) return null;
    const node = figma.createNodeFromSvg(svg);
    parent.appendChild(node);
    setGeometry(node, proxy, context, false);
    return node;
  }
  if (kind === "ELLIPSE" || kind === "CIRCLE") {
    const node = figma.createEllipse();
    parent.appendChild(node);
    setGeometry(node, proxy, context, false);
    node.fills = [solidPaint("#ffffff")];
    return node;
  }
  const node = figma.createRectangle();
  parent.appendChild(node);
  setGeometry(node, proxy, context, false);
  node.fills = [solidPaint("#ffffff")];
  if (kind === "RRECT" || kind === "ROUNDED_RECT") node.cornerRadius = Math.max(0, finite(pick(mask, "radius", "corner_radius", "cornerRadius"), 16));
  return node;
}

async function createImageLayer(layer, parent, context) {
  const source = pick(layer, "src", "source", "asset", "asset_path", "assetPath");
  const bytes = bytesForAsset(context, source);
  if (!bytes) {
    context.report.assets.missing += 1;
    throw new Error("Missing image asset: " + String(source || "(no source)"));
  }
  let image;
  try {
    image = figma.createImage(bytes);
  } catch (error) {
    throw new Error("Image could not be decoded: " + String(source) + " — " + String(error && error.message || error));
  }
  context.report.assets.loaded += 1;
  const mask = layer.mask || {};
  const maskKind = normalizedToken(pick(mask, "kind", "type"));
  let node;
  if (maskKind === "ELLIPSE" || maskKind === "CIRCLE") node = figma.createEllipse();
  else node = figma.createRectangle();
  parent.appendChild(node);
  // Ellipse/rounded-rect clips are representable as one native image-filled node.
  // Honour a detector-supplied mask box instead of stretching it back to the
  // original raster box.
  const directMaskGeometry = (maskKind === "ELLIPSE" || maskKind === "CIRCLE" || maskKind === "RRECT" || maskKind === "ROUNDED_RECT") &&
    (pick(mask, "box", "bounds") || ["x", "y", "w", "h", "width", "height", "left", "top"].some(function (key) { return mask[key] !== undefined; }));
  setGeometry(node, directMaskGeometry ? maskLayerFor(mask, layer) : layer, context, false);
  node.fills = [imagePaint(image, layer)];
  if (maskKind === "RRECT" || maskKind === "ROUNDED_RECT") node.cornerRadius = Math.max(0, finite(pick(mask, "radius", "corner_radius", "cornerRadius"), finite(pick(layer, "radius"), 16)));
  applyStrokes(node, layer, context);
  applyEffects(node, layer, context);

  const needsMaskGroup = maskKind === "PATH" || mask.path || mask.svg || (maskKind === "ALPHA" && pick(mask, "src", "source", "asset"));
  if (needsMaskGroup) {
    let maskNode = null;
    const maskSource = pick(mask, "src", "source", "asset", "asset_path", "assetPath");
    if (maskKind === "ALPHA" && maskSource) {
      const maskBytes = bytesForAsset(context, maskSource);
      if (maskBytes) {
        const maskImage = figma.createImage(maskBytes);
        maskNode = figma.createRectangle();
        parent.appendChild(maskNode);
        setGeometry(maskNode, maskLayerFor(mask, layer), context, false);
        maskNode.fills = [{ type: "IMAGE", scaleMode: "FILL", imageHash: maskImage.hash }];
        context.report.assets.loaded += 1;
      } else {
        context.report.assets.missing += 1;
        context.warn("Alpha mask missing", String(maskSource) + "; the image's own transparency was used.");
      }
    } else {
      maskNode = createMaskGeometry(mask, layer, parent, context);
    }
    if (maskNode) {
      // A Figma mask only affects following siblings. Mark it before grouping to
      // keep the import native in both the desktop app and plugin test harness.
      safeSet(maskNode, "isMask", true, context, "Mask could not be enabled");
      const group = figma.group([maskNode, node], parent);
      safeSet(maskNode, "isMask", true, context, "Mask could not be enabled");
      maskNode.name = (layer.name || layerId(layer)) + " — mask";
      node.name = (layer.name || layerId(layer)) + " — image";
      applyCommon(group, layer, context);
      return group;
    }
  }
  applyCommon(node, layer, context);
  return node;
}

function componentIntent(layer) {
  const component = layer.component;
  if (!component) return null;
  if (component === true) return { kind: "component", ref: layerId(layer) };
  if (typeof component === "string") return { kind: normalizedToken(component).toLowerCase(), ref: component };
  const rawKind = normalizedToken(pick(component, "kind", "type", "mode", "role"));
  const explicitRef = pick(component, "ref", "id", "component_id", "componentId", "key");
  // The v2 Python schema serializes an empty object for layers with no component
  // intent. It must not turn every semantic group into a frame.
  if (!rawKind && (explicitRef === undefined || explicitRef === null || explicitRef === "")) return null;
  return {
    kind: rawKind === "MASTER" ? "component" : rawKind.toLowerCase(),
    ref: String(explicitRef || layerId(layer)),
  };
}

function createContainerNode(layer, context) {
  const intent = componentIntent(layer);
  if (intent && intent.kind === "instance") {
    const source = context.components.get(intent.ref);
    if (source && typeof source.createInstance === "function") return source.createInstance();
    context.warn("Component instance expanded", (layer.name || layerId(layer)) + ": source " + intent.ref + " was not available yet.");
  }
  if (intent && intent.kind === "component" && typeof figma.createComponent === "function") return figma.createComponent();
  return figma.createFrame();
}

function paddingValues(layout) {
  const raw = pick(layout, "padding");
  if (typeof raw === "number") return { top: raw, right: raw, bottom: raw, left: raw };
  if (Array.isArray(raw)) {
    if (raw.length === 2) return { top: raw[0], right: raw[1], bottom: raw[0], left: raw[1] };
    if (raw.length === 4) return { top: raw[0], right: raw[1], bottom: raw[2], left: raw[3] };
  }
  const source = raw && typeof raw === "object" ? raw : layout;
  return {
    top: finite(pick(source, "top", "paddingTop", "padding_top"), 0),
    right: finite(pick(source, "right", "paddingRight", "padding_right"), 0),
    bottom: finite(pick(source, "bottom", "paddingBottom", "padding_bottom"), 0),
    left: finite(pick(source, "left", "paddingLeft", "padding_left"), 0),
  };
}

function axisAlignment(value, fallback) {
  const token = normalizedToken(value);
  const map = { START: "MIN", END: "MAX", LEFT: "MIN", RIGHT: "MAX", TOP: "MIN", BOTTOM: "MAX", BETWEEN: "SPACE_BETWEEN" };
  return map[token] || token || fallback;
}

function orderedContainerChildren(layer) {
  const children = childrenOf(layer).slice();
  const rawMode = normalizedToken(pick(layer.layout || {}, "mode", "direction", "layoutMode", "layout_mode"));
  const mode = rawMode === "ROW" ? "HORIZONTAL" : rawMode === "COLUMN" ? "VERTICAL" : rawMode;
  if (mode === "HORIZONTAL") return children.sort(function (a, b) { return boxOf(a).x - boxOf(b).x || boxOf(a).y - boxOf(b).y; });
  if (mode === "VERTICAL") return children.sort(function (a, b) { return boxOf(a).y - boxOf(b).y || boxOf(a).x - boxOf(b).x; });
  if (mode === "GRID") return children.sort(function (a, b) { return boxOf(a).y - boxOf(b).y || boxOf(a).x - boxOf(b).x; });
  return sortLayers(children);
}

function layerRole(layer) {
  return normalizedToken(layer.meta && pick(layer.meta, "role", "hierarchy")) || "";
}

function isButtonLikeRole(role) {
  return ["BUTTON", "CTA", "BADGE", "CHIP"].indexOf(role) >= 0;
}

function childHasSurface(child) {
  return fillSpecs(child).length > 0 || strokeSpecs(child).length > 0 ||
    pick(child, "radius", "corner_radius", "cornerRadius") !== undefined ||
    pick(child.style || {}, "radius", "corner_radius", "cornerRadius") !== undefined;
}

function boxInsideFraction(inner, outer) {
  const x1 = Math.max(inner.x, outer.x);
  const y1 = Math.max(inner.y, outer.y);
  const x2 = Math.min(inner.x + inner.w, outer.x + outer.w);
  const y2 = Math.min(inner.y + inner.h, outer.y + outer.h);
  if (x2 <= x1 || y2 <= y1) return 0;
  const overlap = (x2 - x1) * (y2 - y1);
  return overlap / Math.max(0.01, inner.w * inner.h);
}

function localGroupBox(layer) {
  const box = boxOf(layer);
  return { x: 0, y: 0, w: box.w, h: box.h };
}

function isShapeTextGroup(layer) {
  const children = childrenOf(layer);
  if (children.length < 2) return false;
  const shapes = children.filter(function (child) { return canonicalType(child) === "shape"; });
  const texts = children.filter(function (child) { return canonicalType(child) === "text"; });
  if (!shapes.length || !texts.length) return false;
  const groupBox = localGroupBox(layer);
  return shapes.some(function (shape) {
    return childHasSurface(shape) && boxInsideFraction(boxOf(shape), groupBox) >= 0.88;
  });
}

function hoistBackgroundShape(layer) {
  const children = childrenOf(layer);
  const groupBox = localGroupBox(layer);
  let best = null;
  let bestArea = 0;
  children.forEach(function (child) {
    if (canonicalType(child) !== "shape" || !childHasSurface(child)) return;
    const childBox = boxOf(child);
    if (boxInsideFraction(childBox, groupBox) < 0.88) return;
    const area = childBox.w * childBox.h;
    if (area < groupBox.w * groupBox.h * 0.72) return;
    if (area > bestArea) {
      bestArea = area;
      best = child;
    }
  });
  if (!best) return layer;
  const copy = Object.assign({}, layer);
  if (!fillSpecs(copy).length) {
    const fills = fillSpecs(best);
    if (fills.length === 1) copy.fill = fills[0];
    else if (fills.length) copy.fills = fills;
  }
  if (pick(copy, "radius", "corner_radius", "cornerRadius") === undefined) {
    const radius = pick(best, "radius", "corner_radius", "cornerRadius") ||
      pick(best.style || {}, "radius", "corner_radius", "cornerRadius");
    if (radius !== undefined) copy.radius = radius;
  }
  if (!strokeSpecs(copy).length && strokeSpecs(best).length) {
    const strokes = strokeSpecs(best);
    if (strokes.length === 1) copy.stroke = strokes[0];
    else copy.strokes = strokes;
  }
  copy.children = children.filter(function (child) { return layerId(child) !== layerId(best); });
  return copy;
}

function inferButtonLayout(layer) {
  const children = childrenOf(layer);
  if (!children.length) return null;
  const parentBox = localGroupBox(layer);
  const boxes = children.map(boxOf);
  const padding = {
    left: Math.max(0, Math.min.apply(null, boxes.map(function (box) { return box.x; }))),
    right: Math.max(0, parentBox.w - Math.max.apply(null, boxes.map(function (box) { return box.x + box.w; }))),
    top: Math.max(0, Math.min.apply(null, boxes.map(function (box) { return box.y; }))),
    bottom: Math.max(0, parentBox.h - Math.max.apply(null, boxes.map(function (box) { return box.y + box.h; }))),
  };
  const centersY = boxes.map(function (box) { return box.y + box.h / 2; });
  const centersX = boxes.map(function (box) { return box.x + box.w / 2; });
  const rowSpread = Math.max.apply(null, centersY) - Math.min.apply(null, centersY);
  const colSpread = Math.max.apply(null, centersX) - Math.min.apply(null, centersX);
  const mode = children.length === 1 || rowSpread <= colSpread ? "HORIZONTAL" : "VERTICAL";
  return {
    mode: mode,
    gap: 0,
    padding: padding,
    align: "CENTER",
    counterAlign: "CENTER",
  };
}

function prepareButtonFrame(layer) {
  let prepared = isShapeTextGroup(layer) ? hoistBackgroundShape(layer) : layer;
  const layout = prepared.layout || {};
  const rawMode = normalizedToken(pick(layout, "mode", "direction", "layoutMode", "layout_mode"));
  if (["HORIZONTAL", "VERTICAL", "GRID", "ROW", "COLUMN"].indexOf(rawMode) < 0) {
    const inferred = inferButtonLayout(prepared);
    if (inferred) prepared = Object.assign({}, prepared, { layout: Object.assign({}, layout, inferred) });
  }
  return Object.assign({}, prepared, { type: "frame" });
}

function groupHasSurface(layer) {
  return fillSpecs(layer).length || strokeSpecs(layer).length ||
    (Array.isArray(layer.effects) && layer.effects.length) ||
    pick(layer, "radius", "corner_radius", "cornerRadius") !== undefined ||
    pick(layer.style || {}, "radius", "corner_radius", "cornerRadius") !== undefined ||
    pick(layer.meta || {}, "radius", "corner_radius", "cornerRadius") !== undefined;
}

function shouldPromoteGroupToFrame(layer) {
  const layoutMode = normalizedToken(pick(layer.layout || {}, "mode", "direction", "layoutMode", "layout_mode"));
  if (["HORIZONTAL", "VERTICAL", "GRID", "ROW", "COLUMN"].indexOf(layoutMode) >= 0) return true;
  if (componentIntent(layer)) return true;
  if (groupHasSurface(layer)) return true;
  if (isButtonLikeRole(layerRole(layer))) return true;
  if (isShapeTextGroup(layer)) return true;
  return false;
}

function isButtonFrame(layer) {
  if (isButtonLikeRole(layerRole(layer))) return true;
  const layout = layer.layout || {};
  const mode = normalizedToken(pick(layout, "mode", "direction", "layoutMode", "layout_mode"));
  if (mode !== "HORIZONTAL" && mode !== "VERTICAL" && mode !== "ROW" && mode !== "COLUMN") return false;
  const align = axisAlignment(pick(layout, "align", "primary_align", "primaryAlign"), "");
  const counter = axisAlignment(pick(layout, "counter_align", "counterAlign", "counterAxisAlignItems"), align);
  return align === "CENTER" && counter === "CENTER";
}

function applyButtonTextLayout(node, layer, context) {
  const layout = layer.layout || {};
  const positioning = normalizedToken(pick(layout, "positioning", "position", "layoutPositioning", "layout_positioning"));
  if (positioning === "ABSOLUTE" || layout.absolute === true) return;
  safeSet(node, "layoutAlign", "CENTER", context);
  safeSet(node, "layoutGrow", 0, context);
  safeSet(node, "layoutSizingHorizontal", "HUG", context);
  safeSet(node, "layoutSizingVertical", "HUG", context);
}

function applyChildLayout(node, layer, context) {
  const layout = layer.layout || {};
  const positioning = normalizedToken(pick(layout, "positioning", "position", "layoutPositioning", "layout_positioning"));
  if (positioning === "ABSOLUTE" || layout.absolute === true) safeSet(node, "layoutPositioning", "ABSOLUTE", context);
  const align = normalizedToken(pick(layout, "align", "layoutAlign", "layout_align"));
  if (align) safeSet(node, "layoutAlign", align, context);
  const grow = finite(pick(layout, "grow", "layoutGrow", "layout_grow"), NaN);
  if (Number.isFinite(grow)) safeSet(node, "layoutGrow", grow, context);
  const horizontal = normalizedToken(pick(layout, "sizing_horizontal", "sizingHorizontal", "layoutSizingHorizontal", "layout_sizing_horizontal"));
  const vertical = normalizedToken(pick(layout, "sizing_vertical", "sizingVertical", "layoutSizingVertical", "layout_sizing_vertical"));
  if (horizontal) safeSet(node, "layoutSizingHorizontal", horizontal, context);
  if (vertical) safeSet(node, "layoutSizingVertical", vertical, context);
  ["minWidth", "maxWidth", "minHeight", "maxHeight"].forEach(function (key) {
    const snake = key.replace(/[A-Z]/g, function (match) { return "_" + match.toLowerCase(); });
    const value = pick(layout, key, snake);
    if (value !== undefined) safeSet(node, key, value === null ? null : Math.max(0, finite(value, 0)), context);
  });
}

function applyAutoLayout(node, layer, childResults, context) {
  const layout = layer.layout || {};
  const rawMode = normalizedToken(pick(layout, "mode", "direction", "layoutMode", "layout_mode"));
  const mode = rawMode === "ROW" ? "HORIZONTAL" : rawMode === "COLUMN" ? "VERTICAL" : rawMode;
  if (["HORIZONTAL", "VERTICAL", "GRID"].indexOf(mode) < 0) return;
  if (!("layoutMode" in node)) {
    context.warn("Auto Layout skipped", (layer.name || layerId(layer)) + " is not a frame-like node.");
    return;
  }
  const original = localBox(layer, context, false);
  try {
    node.layoutMode = mode;
    if ("primaryAxisSizingMode" in node) node.primaryAxisSizingMode = "FIXED";
    if ("counterAxisSizingMode" in node) node.counterAxisSizingMode = "FIXED";
    const padding = paddingValues(layout);
    node.paddingTop = padding.top;
    node.paddingRight = padding.right;
    node.paddingBottom = padding.bottom;
    node.paddingLeft = padding.left;
    node.itemSpacing = finite(pick(layout, "gap", "spacing", "itemSpacing", "item_spacing"), 0);
    const buttonFrame = isButtonFrame(layer);
    const primaryDefault = buttonFrame ? "CENTER" : "MIN";
    const counterDefault = buttonFrame ? "CENTER" : "MIN";
    const primary = axisAlignment(pick(layout, "primary_align", "primaryAlign", "justify", "primaryAxisAlignItems", "align"), primaryDefault);
    const counter = axisAlignment(pick(layout, "counter_align", "counterAlign", "counterAxisAlignItems", buttonFrame ? "align" : "align"), counterDefault);
    safeSet(node, "primaryAxisAlignItems", primary, context);
    safeSet(node, "counterAxisAlignItems", counter, context);
    const wrap = normalizedToken(pick(layout, "wrap", "layoutWrap", "layout_wrap"));
    if (wrap) safeSet(node, "layoutWrap", wrap === "TRUE" || wrap === "WRAP" ? "WRAP" : "NO_WRAP", context);
    if (layout.strokes_included === true || layout.strokesIncluded === true) safeSet(node, "strokesIncludedInLayout", true, context);
    childResults.forEach(function (entry) {
      applyChildLayout(entry.node, entry.layer, context);
      if (buttonFrame && canonicalType(entry.layer) === "text") applyButtonTextLayout(entry.node, entry.layer, context);
    });
    node.resize(Math.max(0.01, original.w), Math.max(0.01, original.h));
  } catch (error) {
    context.warn("Auto Layout partially applied", (layer.name || layerId(layer)) + ": " + String(error && error.message || error));
  }
}

async function createFrameLayer(layer, parent, context) {
  const node = createContainerNode(layer, context);
  parent.appendChild(node);
  setGeometry(node, layer, context, false);
  if (node.type === "INSTANCE") {
    applyCommon(node, layer, context);
    return node;
  }
  safeSet(node, "clipsContent", pick(layer, "clips_content", "clipsContent", "clip") === true, context);
  applyFills(node, layer, context, true);
  applyStrokes(node, layer, context);
  applyRadius(node, layer, context);
  applyCommon(node, layer, context);
  const intent = componentIntent(layer);
  if (intent && intent.kind === "component") context.components.set(intent.ref, node);
  const layoutMode = normalizedToken(pick(layer.layout || {}, "mode", "direction", "layoutMode", "layout_mode"));
  const hasAutoLayout = ["HORIZONTAL", "VERTICAL", "GRID", "ROW", "COLUMN"].indexOf(layoutMode) >= 0;
  const childContext = Object.assign({}, context, {
    sourceOrigin: childSourceOrigin(layer, context),
    localOffset: { x: 0, y: 0 },
    depth: context.depth + 1,
    inAutoLayout: hasAutoLayout,
  });
  const childResults = [];
  const children = orderedContainerChildren(layer);
  for (let i = 0; i < children.length; i += 1) {
    const result = await compileLayer(children[i], node, childContext);
    if (result) childResults.push({ node: result, layer: children[i] });
  }
  applyAutoLayout(node, layer, childResults, context);
  return node;
}

async function createGroupLayer(layer, parent, context) {
  if (shouldPromoteGroupToFrame(layer)) {
    context.warn("Group promoted to frame", (layer.name || layerId(layer)) + " carries layout or component intent.");
    const prepared = isShapeTextGroup(layer) || isButtonLikeRole(layerRole(layer))
      ? prepareButtonFrame(layer)
      : Object.assign({}, layer, { type: "frame" });
    return createFrameLayer(prepared, parent, context);
  }
  const children = sortLayers(childrenOf(layer));
  if (!children.length) {
    context.warn("Empty group promoted to frame", layer.name || layerId(layer));
    return createFrameLayer(Object.assign({}, layer, { type: "frame" }), parent, context);
  }
  const groupBox = localBox(layer, context, false);
  const childContext = Object.assign({}, context, {
    localOffset: { x: groupBox.x, y: groupBox.y },
    depth: context.depth + 1,
  });
  const nodes = [];
  for (let i = 0; i < children.length; i += 1) {
    const result = await compileLayer(children[i], parent, childContext);
    if (result) nodes.push(result);
  }
  if (!nodes.length) throw new Error("Group has no compilable children");
  const group = figma.group(nodes, parent);
  applyCommon(group, layer, context);
  return group;
}

async function compileLayer(layer, parent, context) {
  if (context.cancelled) {
    const error = new Error("Import cancelled");
    error.code = "CANCELLED";
    throw error;
  }
  const type = canonicalType(layer);
  context.progress += 1;
  post("progress", {
    phase: "compile",
    current: context.progress,
    total: context.total,
    message: "Building " + (layer.name || layerId(layer)) + "…",
  });
  try {
    let node = null;
    if (type === "text") node = await createTextLayer(layer, parent, context);
    else if (type === "image") node = await createImageLayer(layer, parent, context);
    else if (type === "shape") node = await createShapeLayer(layer, parent, context);
    else if (type === "vector") node = await createVectorLayer(layer, parent, context);
    else if (type === "frame") node = await createFrameLayer(layer, parent, context);
    else if (type === "group") node = await createGroupLayer(layer, parent, context);
    else throw new Error("Unsupported layer type: " + type);
    context.report.created += 1;
    context.report.byType[type] = (context.report.byType[type] || 0) + 1;
    return node;
  } catch (error) {
    context.report.skipped += 1;
    context.error("Layer failed", (layer.name || layerId(layer)) + ": " + String(error && error.message || error));
    return null;
  }
}

function findImportedRoots(docId) {
  return figma.currentPage.findAll(function (node) {
    try {
      return node.getPluginData(ROOT_KEY) === "true" && node.getPluginData(DOC_KEY) === docId;
    } catch (_) {
      return false;
    }
  });
}

function selectedImportedRoot(docId) {
  return (figma.currentPage.selection || []).find(function (node) {
    try {
      return node.getPluginData(ROOT_KEY) === "true" && node.getPluginData(DOC_KEY) === docId;
    } catch (_) {
      return false;
    }
  }) || null;
}

function importedAtOf(node) {
  try {
    const value = Number(node.getPluginData("adDecompilerImportedAt"));
    return Number.isFinite(value) ? value : -Infinity;
  } catch (_) {
    return -Infinity;
  }
}

// figma.currentPage.findAll() returns nodes in layer/traversal order, which a user
// can change simply by dragging a frame in the layers panel — it is not creation
// order. "Replace existing" must target the import that actually happened most
// recently, so rank candidates by the timestamp stamped at import time instead of
// by their position in that array. Ties fall back to array order.
function latestImportedRoot(existingRoots) {
  let latest = null;
  (existingRoots || []).forEach(function (node) {
    if (!latest || importedAtOf(node) >= importedAtOf(latest)) latest = node;
  });
  return latest;
}

function rootPlacement(doc, mode, existingRoots, replacement) {
  if (mode === "replace" && replacement) return { x: replacement.x, y: replacement.y };
  const canvas = doc.canvas || {};
  const width = Math.max(1, finite(pick(canvas, "w", "width"), 1080));
  const height = Math.max(1, finite(pick(canvas, "h", "height"), 1080));
  if (mode === "copy" && existingRoots.length) {
    const right = existingRoots.reduce(function (max, node) { return Math.max(max, node.x + node.width); }, -Infinity);
    const top = existingRoots.reduce(function (min, node) { return Math.min(min, node.y); }, Infinity);
    return { x: right + 96, y: top };
  }
  return { x: figma.viewport.center.x - width / 2, y: figma.viewport.center.y - height / 2 };
}

async function localTextStyles() {
  try {
    if (typeof figma.getLocalTextStylesAsync === "function") return await figma.getLocalTextStylesAsync();
    if (typeof figma.getLocalTextStyles === "function") return figma.getLocalTextStyles();
  } catch (_) {}
  return [];
}

function makeContext(doc, assets, settings) {
  const report = {
    ok: false,
    created: 0,
    skipped: 0,
    replaced: false,
    mode: settings.importMode,
    warnings: [],
    errors: [],
    events: [],
    fonts: { requested: 0, substituted: 0, selections: [] },
    assets: { loaded: 0, missing: 0 },
    fidelity: { unsupported_paint: 0, unsupported_stroke: 0, unsupported_effect: 0, notes: [] },
    byType: {},
    plugin_build: PLUGIN_BUILD,
  };
  const schemaVersion = finite(pick(doc, "schema_version", "schemaVersion"), 1);
  const declaredCoordinates = normalizedToken(pick(doc.meta || {}, "coordinate_space", "coordinateSpace"));
  const context = {
    doc,
    docId: String(doc.id || doc.run_id || doc.runId || doc.name || "ad"),
    docName: String(doc.name || doc.id || "Ad").replace(/\//g, " "),
    schemaVersion,
    defaultCoordinateSpace: declaredCoordinates === "ABSOLUTE" || declaredCoordinates === "CANVAS" ? "absolute" : (declaredCoordinates === "LOCAL" || schemaVersion >= 2 ? "local" : "absolute"),
    assets: assets || {},
    settings,
    total: countLayers(doc.layers || []),
    progress: 0,
    depth: 0,
    sourceOrigin: { x: 0, y: 0 },
    localOffset: { x: 0, y: 0 },
    cancelled: false,
    report,
    fonts: null,
    textStyles: new Map(),
    localTextStyles: [],
    components: new Map(),
    warn: function (title, detail) {
      const entry = { at: new Date().toISOString(), level: "warn", title: String(title), detail: String(detail || "") };
      report.warnings.push({ title: entry.title, detail: entry.detail });
      report.events.push(entry);
      post("log-event", entry);
    },
    error: function (title, detail) {
      const entry = { at: new Date().toISOString(), level: "error", title: String(title), detail: String(detail || "") };
      report.errors.push({ title: entry.title, detail: entry.detail });
      report.events.push(entry);
      post("log-event", entry);
    },
    fidelity: function (kind, detail) {
      const key = String(kind || "other");
      report.fidelity[key] = finite(report.fidelity[key], 0) + 1;
      report.fidelity.notes.push({ kind: key, detail: String(detail || "") });
      context.warn("Fidelity fallback", String(detail || key));
    },
  };
  context.fonts = new FontResolver(context);
  return context;
}

async function buildDocument(message) {
  if (activeJob) activeJob.cancelled = true;
  const doc = message.design || {};
  if (!doc.canvas || !Array.isArray(doc.layers)) {
    post("build-result", { report: { ok: false, created: 0, skipped: 0, warnings: [], errors: [{ title: "Invalid document", detail: "design.json needs canvas and layers." }] } });
    return;
  }
  const rawMode = normalizedToken(pick(message, "import_mode", "importMode"));
  const importMode = rawMode === "COPY" || rawMode === "CREATE_COPY" ? "copy" : "replace";
  const settings = { importMode };
  const assets = {};
  Object.keys(message.assets || {}).forEach(function (key) {
    const raw = message.assets[key];
    assets[key] = raw instanceof Uint8Array ? raw : new Uint8Array(raw);
  });
  const context = makeContext(doc, assets, settings);
  activeJob = context;
  context.localTextStyles = await localTextStyles();
  const existingRoots = findImportedRoots(context.docId);
  const replacement = selectedImportedRoot(context.docId) || latestImportedRoot(existingRoots);
  const position = rootPlacement(doc, importMode, existingRoots, replacement);
  const canvas = doc.canvas || {};
  const width = Math.max(1, finite(pick(canvas, "w", "width"), 1080));
  const height = Math.max(1, finite(pick(canvas, "h", "height"), 1080));
  let root = null;
  try {
    post("progress", { phase: "compile", current: 0, total: context.total, message: "Preparing Figma layers…" });
    root = figma.createFrame();
    root.name = String(doc.name || "Ad reconstruction");
    root.resize(width, height);
    root.x = position.x;
    root.y = position.y;
    root.clipsContent = true;
    root.fills = [];
    const canvasFill = pick(canvas, "fill", "background", "background_color", "backgroundColor");
    if (canvasFill) root.fills = [solidPaint(canvasFill)];
    root.setPluginData(ROOT_KEY, "true");
    root.setPluginData(DOC_KEY, context.docId);
    root.setPluginData("adDecompilerSchemaVersion", String(context.schemaVersion));
    root.setPluginData("adDecompilerImportedAt", String(Date.now()));
    const layers = sortLayers(doc.layers);
    for (let i = 0; i < layers.length; i += 1) await compileLayer(layers[i], root, context);
    if (context.cancelled) {
      const cancelled = new Error("Import cancelled");
      cancelled.code = "CANCELLED";
      throw cancelled;
    }
    if (context.report.errors.length) throw new Error(context.report.errors.length + " layer" + (context.report.errors.length === 1 ? "" : "s") + " could not be compiled");
    post("progress", { phase: "export", current: context.total, total: context.total, message: "Checking the finished frame…" });
    const png = await root.exportAsync({ format: "PNG" });
    // Point of no return: once the previous import is removed, a later failure
    // (selection/viewport/notify) must never fall through to the catch below and
    // remove the freshly built `root` too — that would delete the only remaining
    // frame instead of merely leaving one duplicate. Everything from here on is
    // best-effort and reported as a warning rather than allowed to unwind the swap.
    if (importMode === "replace" && replacement && replacement !== root) {
      replacement.remove();
      context.report.replaced = true;
    }
    context.report.ok = true;
    context.report.rootId = root.id;
    context.report.docId = context.docId;
    context.report.render = {
      width: Math.round(finite(root.width, width)),
      height: Math.round(finite(root.height, height)),
      png_bytes: png.length || png.byteLength || 0,
      emitted: true,
    };
    try {
      figma.currentPage.selection = [root];
      figma.viewport.scrollAndZoomIntoView([root]);
    } catch (viewError) {
      context.warn("Canvas selection could not be updated", String(viewError && viewError.message || viewError));
    }
    post("build-result", { report: context.report });
    post("exported", { bytes: Array.from(png), export_to: message.export_to || null });
    try {
      figma.notify("Import complete");
    } catch (_) {}
  } catch (error) {
    // Only discard the new root if we never committed the swap. Once the old
    // import has been removed (context.report.replaced), removing `root` here
    // too would leave the user with neither the old nor the new frame.
    if (root && root.parent && !context.report.replaced) root.remove();
    if (error && error.code === "CANCELLED") context.warn("Import cancelled", "No Figma layers were changed.");
    else if (!context.report.errors.length) context.error("Import failed", String(error && error.message || error));
    context.report.ok = false;
    context.report.cancelled = Boolean(error && error.code === "CANCELLED");
    post("build-result", { report: context.report });
    if (!(error && error.code === "CANCELLED")) figma.notify("Import failed", { error: true });
  } finally {
    if (activeJob === context) activeJob = null;
  }
}

figma.ui.onmessage = async function (message) {
  if (!message || !message.type) return;
  if (message.type === "ui-ready") {
    const saved = await figma.clientStorage.getAsync(SETTINGS_KEY).catch(function () { return null; });
    post("init", { settings: Object.assign({}, DEFAULT_SETTINGS, saved || {}), build: PLUGIN_BUILD });
    return;
  }
  if (message.type === "save-settings") {
    const settings = Object.assign({}, DEFAULT_SETTINGS, message.settings || {});
    await figma.clientStorage.setAsync(SETTINGS_KEY, settings).catch(function () {});
    return;
  }
  if (message.type === "cancel") {
    if (activeJob) activeJob.cancelled = true;
    return;
  }
  if (message.type === "build") await buildDocument(message);
};
