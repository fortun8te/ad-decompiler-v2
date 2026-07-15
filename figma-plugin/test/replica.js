"use strict";
// replica.js — read-only re-implementations of the *pure* helper functions in
// figma-plugin/code.js. The harness uses these only to PREDICT what the compiler
// intends for a given design.json (layer type, expected box, group promotion,
// hoist absorption, flow-child status). They never drive the compiler itself.
//
// If code.js changes these semantics, update this file to match (grep markers:
// each function names its code.js counterpart).

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

function normalizedToken(value) {
  return String(value || "").trim().replace(/[\s-]+/g, "_").toUpperCase();
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

function countLayers(layers) {
  let count = 0;
  (layers || []).forEach((layer) => { count += 1 + countLayers(childrenOf(layer)); });
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

function sourceCoordinateSpace(layer, defaultSpace) {
  const explicit = normalizedToken(
    pick(layer, "coordinate_space", "coordinateSpace", "position_mode", "positionMode") ||
      (layer.meta && pick(layer.meta, "coordinate_space", "coordinateSpace"))
  );
  if (explicit === "LOCAL" || explicit === "RELATIVE") return "local";
  if (explicit === "ABSOLUTE" || explicit === "CANVAS") return "absolute";
  return defaultSpace || "absolute";
}

function defaultCoordinateSpaceOf(doc) {
  const schemaVersion = finite(pick(doc, "schema_version", "schemaVersion"), 1);
  const declared = normalizedToken(pick(doc.meta || {}, "coordinate_space", "coordinateSpace"));
  if (declared === "ABSOLUTE" || declared === "CANVAS") return "absolute";
  return declared === "LOCAL" || schemaVersion >= 2 ? "local" : "absolute";
}

function layoutModeOf(layer) {
  const rawMode = normalizedToken(pick(layer.layout || {}, "mode", "direction", "layoutMode", "layout_mode"));
  if (rawMode === "ROW") return "HORIZONTAL";
  if (rawMode === "COLUMN") return "VERTICAL";
  return rawMode;
}

function hasAutoLayoutIntent(layer) {
  return ["HORIZONTAL", "VERTICAL", "GRID"].indexOf(layoutModeOf(layer)) >= 0;
}

function layerRole(layer) {
  return normalizedToken(layer.meta && pick(layer.meta, "role", "hierarchy")) || "";
}

function isButtonLikeRole(role) {
  return ["BUTTON", "CTA", "BADGE", "CHIP"].indexOf(role) >= 0;
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

function componentIntent(layer) {
  const component = layer.component;
  if (!component) return null;
  if (component === true) return { kind: "component", ref: layerId(layer) };
  if (typeof component === "string") return { kind: normalizedToken(component).toLowerCase(), ref: component };
  const rawKind = normalizedToken(pick(component, "kind", "type", "mode", "role"));
  const explicitRef = pick(component, "ref", "id", "component_id", "componentId", "key");
  if (!rawKind && (explicitRef === undefined || explicitRef === null || explicitRef === "")) return null;
  return {
    kind: rawKind === "MASTER" ? "component" : rawKind.toLowerCase(),
    ref: String(explicitRef || layerId(layer)),
  };
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
  const shapes = children.filter((child) => canonicalType(child) === "shape");
  const texts = children.filter((child) => canonicalType(child) === "text");
  if (!shapes.length || !texts.length) return false;
  const groupBox = localGroupBox(layer);
  return shapes.some((shape) => childHasSurface(shape) && boxInsideFraction(boxOf(shape), groupBox) >= 0.88);
}

// Predicts which child id createGroupLayer -> hoistBackgroundShape would absorb.
function hoistedBackgroundChildId(layer) {
  if (!isShapeTextGroup(layer)) return null;
  const children = childrenOf(layer);
  const groupBox = localGroupBox(layer);
  let best = null;
  let bestArea = 0;
  children.forEach((child) => {
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
  return best ? layerId(best) : null;
}

function groupHasSurface(layer) {
  return fillSpecs(layer).length || strokeSpecs(layer).length ||
    (Array.isArray(layer.effects) && layer.effects.length) ||
    pick(layer, "radius", "corner_radius", "cornerRadius") !== undefined ||
    pick(layer.style || {}, "radius", "corner_radius", "cornerRadius") !== undefined ||
    pick(layer.meta || {}, "radius", "corner_radius", "cornerRadius") !== undefined;
}

function shouldPromoteGroupToFrame(layer) {
  const children = childrenOf(layer);
  if (!children.length) return false;
  if (hasAutoLayoutIntent(layer)) return true;
  if (componentIntent(layer)) return true;
  if (groupHasSurface(layer)) return true;
  if (isButtonLikeRole(layerRole(layer))) return true;
  if (isShapeTextGroup(layer)) return true;
  return false;
}

function isFlowAutoLayoutChild(layer, parentHasAutoLayout) {
  if (!parentHasAutoLayout) return false;
  const childLayout = layer.layout || {};
  const positioning = normalizedToken(pick(childLayout, "positioning", "position", "layoutPositioning", "layout_positioning"));
  return positioning !== "ABSOLUTE" && childLayout.absolute !== true;
}

function textContent(layer) {
  if (layer.text !== undefined && layer.text !== null) return String(layer.text);
  const runs = pick(layer, "text_runs", "textRuns", "runs");
  if (Array.isArray(runs)) return runs.map((run) => String(run.text || "")).join("");
  return "";
}

function textRuns(layer) {
  const runs = pick(layer, "text_runs", "textRuns", "runs");
  return Array.isArray(runs) ? runs : [];
}

function maskKindOf(layer) {
  const mask = layer.mask || {};
  return normalizedToken(pick(mask, "kind", "type"));
}

// Collect the font families a design requests (primary + candidates + runs).
function requestedFontFamilies(doc) {
  const families = new Set();
  const visit = (layers) => (layers || []).forEach((layer) => {
    const styles = [Object.assign({}, layer.typography || {}, layer.style || {})];
    textRuns(layer).forEach((run) => styles.push(Object.assign({}, run.typography || {}, run.style || {}, run)));
    styles.forEach((style) => {
      const fam = pick(style, "fontFamily", "font_family", "family");
      if (fam) families.add(String(fam).trim());
      const candidates = pick(style, "fontCandidates", "font_candidates", "candidates");
      (Array.isArray(candidates) ? candidates : []).forEach((c) => {
        const f = typeof c === "string" ? c : c && (c.family || c.fontFamily || c.font_family);
        if (f) families.add(String(f).trim());
      });
    });
    visit(childrenOf(layer));
  });
  visit(doc.layers || []);
  return Array.from(families);
}

// Mirrors ui.html assetReferences(): image srcs are required, mask srcs optional.
function assetReferences(doc) {
  const refs = new Map();
  const visit = (layers) => (layers || []).forEach((layer) => {
    const source = layer.src || layer.source || layer.asset || layer.asset_path || layer.assetPath;
    if (canonicalType(layer) === "image" && source) refs.set(String(source), true);
    const mask = layer.mask || {};
    const maskSource = mask.src || mask.source || mask.asset || mask.asset_path || mask.assetPath;
    if (maskSource && !refs.has(String(maskSource))) refs.set(String(maskSource), false);
    visit(childrenOf(layer));
  });
  visit(doc.layers || []);
  return refs;
}

// ---------------------------------------------------------------------------
// Expectation rows: one per design layer, with predicted geometry and flags.
// ---------------------------------------------------------------------------
//
// Geometry model (matches code.js localBox/childSourceOrigin):
//   local space   -> node position relative to parent's node == layer box origin
//   absolute space -> layer boxes are canvas coordinates; relative = box - parentAbs
// Either way the *canvas-absolute* expected box is what we assert against, by
// accumulating the node's x/y up to (excluding) the imported root frame.

function buildExpectations(doc) {
  const defaultSpace = defaultCoordinateSpaceOf(doc);
  const rows = [];

  const walk = (layer, index, parentRow, ctx) => {
    const type = canonicalType(layer);
    const id = layerId(layer, "layer-" + rows.length);
    const box = boxOf(layer);
    const space = sourceCoordinateSpace(layer, defaultSpace);
    const abs = space === "local"
      ? { x: ctx.originX + box.x, y: ctx.originY + box.y, w: box.w, h: box.h }
      : { x: box.x, y: box.y, w: box.w, h: box.h };
    const vBox = visibleBoxOf(layer);
    const absVisible = space === "local"
      ? { x: ctx.originX + vBox.x, y: ctx.originY + vBox.y, w: vBox.w, h: vBox.h }
      : { x: vBox.x, y: vBox.y, w: vBox.w, h: vBox.h };

    const intent = componentIntent(layer);
    const row = {
      layer,
      id,
      path: (parentRow ? parentRow.path + "/" : "") + id,
      type,
      parentRow,
      expectedBox: abs,
      expectedVisibleBox: absVisible,
      rotation: finite(pick(layer, "rotation"), 0),
      flowChild: isFlowAutoLayoutChild(layer, ctx.parentHasAutoLayout),
      autoLayoutMode: hasAutoLayoutIntent(layer) ? layoutModeOf(layer) : null,
      layoutSpec: layer.layout || {},
      isInstance: Boolean(intent && intent.kind === "instance"),
      insideInstance: ctx.insideInstance,
      absorbed: ctx.absorbedId === id,
      maskKind: maskKindOf(layer),
      hasMask: Boolean(layer.mask && (maskKindOf(layer) || layer.mask.path || layer.mask.svg)),
      textContent: type === "text" ? textContent(layer) : null,
      hasRuns: type === "text" && textRuns(layer).length > 0,
      constraints: layer.constraints || null,
      visible: layer.visible !== false,
    };
    rows.push(row);

    // Children context. Promoted groups and frames behave the same for geometry.
    const promotedGroup = type === "group" && shouldPromoteGroupToFrame(layer);
    const absorbedId = promotedGroup ? hoistedBackgroundChildId(layer) : null;
    // prepareButtonFrame() infers an auto layout for shape+text / button-role groups
    // even when the design carries none, so their children become flow children too.
    const buttonPrepared = promotedGroup &&
      (isShapeTextGroup(layer) || isButtonLikeRole(layerRole(layer))) &&
      childrenOf(layer).length > 0;
    const childCtx = {
      originX: abs.x,
      originY: abs.y,
      parentHasAutoLayout: hasAutoLayoutIntent(layer) || buttonPrepared,
      insideInstance: ctx.insideInstance || row.isInstance,
      absorbedId,
    };
    childrenOf(layer).forEach((child, i) => walk(child, i, row, childCtx));
  };

  (doc.layers || []).forEach((layer, i) => walk(layer, i, null, {
    originX: 0,
    originY: 0,
    parentHasAutoLayout: false,
    insideInstance: false,
    absorbedId: null,
  }));

  return rows;
}

module.exports = {
  pick,
  finite,
  normalizedToken,
  layerId,
  boxOf,
  visibleBoxOf,
  childrenOf,
  countLayers,
  canonicalType,
  layoutModeOf,
  hasAutoLayoutIntent,
  layerRole,
  isButtonLikeRole,
  componentIntent,
  isShapeTextGroup,
  hoistedBackgroundChildId,
  shouldPromoteGroupToFrame,
  isFlowAutoLayoutChild,
  textContent,
  textRuns,
  requestedFontFamilies,
  assetReferences,
  buildExpectations,
  defaultCoordinateSpaceOf,
};
