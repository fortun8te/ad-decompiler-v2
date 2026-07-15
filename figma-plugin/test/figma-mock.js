"use strict";
// figma-mock.js — a faithful mock of the Figma plugin-sandbox API surface that
// figma-plugin/code.js actually uses (and nothing more).
//
// Faithfulness rules that matter for catching real compiler bugs:
//   * Property surface: code.js gates writes with `key in object` (safeSet), so each
//     mock node type exposes exactly the properties its real Figma counterpart has.
//   * Enum validation: assigning an invalid enum value throws, like the real sandbox
//     ("in set_x: ..."), so safeSet's catch/warn paths are exercised honestly.
//   * Font discipline: mutating text with an unloaded font throws; loadFontAsync only
//     resolves for a configurable "installed fonts" list and rejects otherwise.
//   * remove()/group() semantics: remove() kills the whole subtree; figma.group()
//     throws on removed nodes; grouping re-bases child coordinates and group resize
//     scales children (real GroupNode behavior).
//   * exportAsync returns a real PNG whose IHDR encodes round(width) x round(height),
//     because code.js parses PNG dimensions out of the bytes (pngDimensions()).
//   * Text measurement is a deterministic synthetic model (char width = 0.52 * fontSize,
//     auto line height = 1.2 * fontSize). It is monotonic in fontSize so code.js's
//     binary-search fitting converges. Fidelity is NOT the goal; structure is.
//
// Deliberately absent (code.js feature-detects these):
//   figma.createAutoLayout — not a real Figma API; leaving it out routes containers
//   through figma.createFrame, exactly like production Figma.

const crypto = require("crypto");
const zlib = require("zlib");

// --------------------------------------------------------------------------
// enums (Figma plugin API)
// --------------------------------------------------------------------------

const BLEND_MODES = [
  "PASS_THROUGH", "NORMAL", "DARKEN", "MULTIPLY", "LINEAR_BURN", "COLOR_BURN",
  "LIGHTEN", "SCREEN", "LINEAR_DODGE", "COLOR_DODGE", "OVERLAY", "SOFT_LIGHT",
  "HARD_LIGHT", "DIFFERENCE", "EXCLUSION", "HUE", "SATURATION", "COLOR", "LUMINOSITY",
];
const CONSTRAINT_TYPES = ["MIN", "CENTER", "MAX", "STRETCH", "SCALE"];
const STROKE_ALIGNS = ["CENTER", "INSIDE", "OUTSIDE"];
const STROKE_CAPS = ["NONE", "ROUND", "SQUARE", "ARROW_LINES", "ARROW_EQUILATERAL"];
const STROKE_JOINS = ["MITER", "BEVEL", "ROUND"];
const PAINT_TYPES = ["SOLID", "GRADIENT_LINEAR", "GRADIENT_RADIAL", "GRADIENT_ANGULAR", "GRADIENT_DIAMOND", "IMAGE", "VIDEO"];
const IMAGE_SCALE_MODES = ["FILL", "FIT", "CROP", "TILE"];
const EFFECT_TYPES = ["DROP_SHADOW", "INNER_SHADOW", "LAYER_BLUR", "BACKGROUND_BLUR"];
const LAYOUT_MODES = ["NONE", "HORIZONTAL", "VERTICAL", "GRID"];
const AXIS_SIZING = ["FIXED", "AUTO"];
const PRIMARY_ALIGN = ["MIN", "MAX", "CENTER", "SPACE_BETWEEN"];
const COUNTER_ALIGN = ["MIN", "MAX", "CENTER", "BASELINE"];
const LAYOUT_ALIGN = ["MIN", "CENTER", "MAX", "STRETCH", "INHERIT"];
const LAYOUT_SIZING = ["FIXED", "HUG", "FILL"];
const LAYOUT_POSITIONING = ["AUTO", "ABSOLUTE"];
const LAYOUT_WRAP = ["NO_WRAP", "WRAP"];
const TEXT_ALIGN_H = ["LEFT", "CENTER", "RIGHT", "JUSTIFIED"];
const TEXT_ALIGN_V = ["TOP", "CENTER", "BOTTOM"];
const TEXT_AUTO_RESIZE = ["NONE", "WIDTH_AND_HEIGHT", "HEIGHT", "TRUNCATE"];
const TEXT_CASE = ["ORIGINAL", "UPPER", "LOWER", "TITLE", "SMALL_CAPS", "SMALL_CAPS_FORCED"];
const TEXT_DECORATION = ["NONE", "UNDERLINE", "STRIKETHROUGH"];
const LEADING_TRIM = ["NONE", "CAP_HEIGHT"];

// synthetic text metrics
const CHAR_WIDTH_RATIO = 0.52;
const AUTO_LINE_HEIGHT = 1.2;

// --------------------------------------------------------------------------
// PNG helpers (real, minimal PNG encode + dimension sniffing)
// --------------------------------------------------------------------------

const CRC_TABLE = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

function crc32(buf) {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i += 1) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length, 0);
  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([len, body, crc]);
}

// Encode a real (valid, decodable) solid-color RGBA PNG of the given size.
function encodePng(width, height, rgba) {
  const w = Math.max(1, Math.round(Number(width) || 1));
  const h = Math.max(1, Math.round(Number(height) || 1));
  const px = rgba || [136, 136, 136, 255];
  const raw = Buffer.alloc(h * (1 + w * 4));
  for (let y = 0; y < h; y += 1) {
    const row = y * (1 + w * 4);
    raw[row] = 0; // filter: none
    for (let x = 0; x < w; x += 1) {
      const o = row + 1 + x * 4;
      raw[o] = px[0]; raw[o + 1] = px[1]; raw[o + 2] = px[2]; raw[o + 3] = px[3];
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8;  // bit depth
  ihdr[9] = 6;  // color type RGBA
  const png = Buffer.concat([
    Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
    pngChunk("IHDR", ihdr),
    pngChunk("IDAT", zlib.deflateSync(raw)),
    pngChunk("IEND", Buffer.alloc(0)),
  ]);
  return new Uint8Array(png.buffer, png.byteOffset, png.byteLength);
}

function pngSize(bytes) {
  if (!bytes || bytes.length < 24) return null;
  if (bytes[0] !== 137 || bytes[1] !== 80 || bytes[2] !== 78 || bytes[3] !== 71) return null;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return { width: view.getUint32(16, false), height: view.getUint32(20, false) };
}

function sniffImage(bytes) {
  if (bytes.length >= 8 && bytes[0] === 137 && bytes[1] === 80 && bytes[2] === 78 && bytes[3] === 71) return "png";
  if (bytes.length >= 3 && bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return "jpeg";
  if (bytes.length >= 6 && bytes[0] === 0x47 && bytes[1] === 0x49 && bytes[2] === 0x46) return "gif";
  if (bytes.length >= 12 && bytes[8] === 0x57 && bytes[9] === 0x45 && bytes[10] === 0x42 && bytes[11] === 0x50) return "webp";
  return null;
}

// --------------------------------------------------------------------------
// validation helpers
// --------------------------------------------------------------------------

function fail(where, message) {
  throw new Error("in " + where + ": " + message);
}

function assertEnum(where, value, allowed) {
  if (allowed.indexOf(value) < 0) fail(where, 'Expected one of [' + allowed.join(", ") + '], got "' + String(value) + '"');
  return value;
}

function assertNumber(where, value) {
  if (typeof value !== "number" || !Number.isFinite(value)) fail(where, "Expected a finite number, got " + String(value));
  return value;
}

function assertBool(where, value) {
  if (typeof value !== "boolean") fail(where, "Expected a boolean, got " + String(value));
  return value;
}

function assertUnitColor(where, c, withAlpha) {
  if (!c || typeof c !== "object") fail(where, "Expected a color object");
  const keys = withAlpha ? ["r", "g", "b", "a"] : ["r", "g", "b"];
  keys.forEach((k) => {
    const v = c[k];
    if (typeof v !== "number" || !Number.isFinite(v) || v < 0 || v > 1) {
      fail(where, "Color component " + k + " must be a number in [0, 1], got " + String(v));
    }
  });
}

function validatePaint(where, paint) {
  if (!paint || typeof paint !== "object") fail(where, "Expected a paint object");
  assertEnum(where, paint.type, PAINT_TYPES);
  if (paint.visible !== undefined) assertBool(where, paint.visible);
  if (paint.opacity !== undefined) {
    assertNumber(where, paint.opacity);
    if (paint.opacity < 0 || paint.opacity > 1) fail(where, "Paint opacity must be in [0, 1]");
  }
  if (paint.blendMode !== undefined) assertEnum(where, paint.blendMode, BLEND_MODES);
  if (paint.type === "SOLID") {
    assertUnitColor(where, paint.color, false);
  } else if (paint.type.indexOf("GRADIENT_") === 0) {
    if (!Array.isArray(paint.gradientStops) || paint.gradientStops.length < 2) {
      fail(where, "Gradient paint needs at least 2 gradientStops");
    }
    paint.gradientStops.forEach((stop) => {
      assertNumber(where, stop.position);
      if (stop.position < 0 || stop.position > 1) fail(where, "Gradient stop position must be in [0, 1]");
      assertUnitColor(where, stop.color, true);
    });
    const t = paint.gradientTransform;
    const okT = Array.isArray(t) && t.length === 2 && t.every((row) => Array.isArray(row) && row.length === 3 && row.every((n) => typeof n === "number" && Number.isFinite(n)));
    if (!okT) fail(where, "gradientTransform must be a 2x3 matrix of finite numbers");
  } else if (paint.type === "IMAGE") {
    if (typeof paint.imageHash !== "string" || !paint.imageHash) fail(where, "IMAGE paint needs imageHash");
    if (paint.scaleMode !== undefined) assertEnum(where, paint.scaleMode, IMAGE_SCALE_MODES);
    if (paint.rotation !== undefined && [0, 90, 180, 270].indexOf(paint.rotation) < 0) {
      fail(where, "ImagePaint.rotation must be 0, 90, 180 or 270");
    }
    if (paint.imageTransform !== undefined) {
      const m = paint.imageTransform;
      const ok = Array.isArray(m) && m.length === 2 && m.every((row) => Array.isArray(row) && row.length === 3 && row.every((n) => typeof n === "number" && Number.isFinite(n)));
      if (!ok) fail(where, "imageTransform must be a 2x3 matrix of finite numbers");
    }
  }
  return paint;
}

function validatePaintArray(where, value) {
  if (!Array.isArray(value)) fail(where, "Expected an array of paints");
  value.forEach((p) => validatePaint(where, p));
  return value.map((p) => Object.assign({}, p));
}

function validateEffect(where, effect) {
  if (!effect || typeof effect !== "object") fail(where, "Expected an effect object");
  assertEnum(where, effect.type, EFFECT_TYPES);
  assertNumber(where, effect.radius);
  if (effect.radius < 0) fail(where, "Effect radius must be >= 0");
  if (effect.visible !== undefined) assertBool(where, effect.visible);
  if (effect.type === "DROP_SHADOW" || effect.type === "INNER_SHADOW") {
    assertUnitColor(where, effect.color, true);
    if (!effect.offset || typeof effect.offset !== "object") fail(where, "Shadow effect needs an offset");
    assertNumber(where, effect.offset.x);
    assertNumber(where, effect.offset.y);
    if (effect.blendMode !== undefined) assertEnum(where, effect.blendMode, BLEND_MODES);
  }
  return effect;
}

function validateLineHeight(where, value) {
  if (!value || typeof value !== "object") fail(where, "Expected a LineHeight object");
  if (value.unit === "AUTO") return { unit: "AUTO" };
  assertEnum(where, value.unit, ["PIXELS", "PERCENT", "AUTO"]);
  assertNumber(where, value.value);
  if (value.value <= 0) fail(where, "LineHeight value must be > 0");
  return { unit: value.unit, value: value.value };
}

function validateLetterSpacing(where, value) {
  if (!value || typeof value !== "object") fail(where, "Expected a LetterSpacing object");
  assertEnum(where, value.unit, ["PIXELS", "PERCENT"]);
  assertNumber(where, value.value);
  return { unit: value.unit, value: value.value };
}

function validateFontName(where, value) {
  if (!value || typeof value !== "object" || typeof value.family !== "string" || typeof value.style !== "string") {
    fail(where, "Expected a FontName ({ family, style })");
  }
  return { family: value.family, style: value.style };
}

function fontKey(f) {
  return f.family + " " + f.style;
}

function clonePlain(value) {
  if (typeof structuredClone === "function") {
    try { return structuredClone(value); } catch (_) { /* fall through */ }
  }
  return JSON.parse(JSON.stringify(value));
}

// --------------------------------------------------------------------------
// the mock factory
// --------------------------------------------------------------------------

const DEFAULT_FONT_STYLES = ["Regular", "Bold", "Italic", "Bold Italic", "Light", "Medium", "SemiBold", "Black"];

function normalizeFonts(spec) {
  const out = [];
  (spec || []).forEach((entry) => {
    if (typeof entry === "string") {
      DEFAULT_FONT_STYLES.forEach((style) => out.push({ family: entry, style }));
    } else if (entry && entry.family) {
      out.push({ family: String(entry.family), style: String(entry.style || "Regular") });
    }
  });
  return out;
}

function createFigmaMock(options) {
  options = options || {};

  const state = {
    uiMessages: [],
    notifications: [],
    images: [],
    svgImports: [],
    fontLoads: [],
    fontLoadFailures: [],
    loadedFonts: new Set(),
    localTextStyles: [],
    clientStorage: new Map(),
    nodesById: new Map(),
    nodeCounter: 0,
    // Audit trail of every SUCCESSFUL layoutSizingHorizontal/Vertical assignment (after
    // the enum + FILL/HUG legality guards below passed), so a fixture can assert the
    // plugin actually drove per-dimension sizing rather than leaving nodes at the FIXED
    // default. Illegal sets throw (like real Figma) and never reach this log.
    sizingSets: [],
  };
  const installedFonts = normalizeFonts(options.installedFonts);
  const installedKeys = new Set(installedFonts.map(fontKey));

  function nextId(prefix) {
    state.nodeCounter += 1;
    return (prefix || "mock") + ":" + state.nodeCounter;
  }

  function assertAlive(node, where) {
    if (node.removed) fail(where, "The node with id " + node.id + ' ("' + node.name + '") has been removed');
  }

  function detach(node) {
    if (node.parent && Array.isArray(node.parent.children)) {
      const idx = node.parent.children.indexOf(node);
      if (idx >= 0) node.parent.children.splice(idx, 1);
    }
    node.parent = null;
  }

  function markRemoved(node) {
    node.removed = true;
    if (Array.isArray(node.children)) node.children.forEach(markRemoved);
  }

  // ---------------------------------------------------------------- BaseNode

  class BaseNode {
    constructor(type) {
      this.id = nextId("mock");
      this.type = type;
      this.name = type;
      this.parent = null;
      this.removed = false;
      this._pluginData = Object.create(null);
      state.nodesById.set(this.id, this);
    }
    setPluginData(key, value) {
      assertAlive(this, "setPluginData");
      this._pluginData[String(key)] = String(value);
    }
    getPluginData(key) {
      return this._pluginData[String(key)] || "";
    }
    remove() {
      assertAlive(this, "remove");
      detach(this);
      markRemoved(this);
    }
  }

  // ---------------------------------------------------------------- SceneNode

  class SceneNode extends BaseNode {
    constructor(type) {
      super(type);
      const container = type === "FRAME" || type === "GROUP" || type === "COMPONENT" || type === "INSTANCE";
      this._x = 0;
      this._y = 0;
      this._w = 100;
      this._h = 100;
      this._rotation = 0;
      this._opacity = 1;
      this._visible = true;
      this._locked = false;
      this._blendMode = container ? "PASS_THROUGH" : "NORMAL";
      this._layoutAlign = "INHERIT";
      this._layoutGrow = 0;
      this._layoutPositioning = "AUTO";
      this._layoutSizingH = "FIXED";
      this._layoutSizingV = "FIXED";
      this._minWidth = null;
      this._maxWidth = null;
      this._minHeight = null;
      this._maxHeight = null;
    }

    get x() { return this._x; }
    set x(v) { assertNumber("set_x", v); this._x = v; }
    get y() { return this._y; }
    set y(v) { assertNumber("set_y", v); this._y = v; }
    get width() { return this._w; }
    get height() { return this._h; }
    get rotation() { return this._rotation; }
    set rotation(v) { assertNumber("set_rotation", v); this._rotation = v; }
    get opacity() { return this._opacity; }
    set opacity(v) {
      assertNumber("set_opacity", v);
      if (v < 0 || v > 1) fail("set_opacity", "Opacity must be in [0, 1]");
      this._opacity = v;
    }
    get visible() { return this._visible; }
    set visible(v) { assertBool("set_visible", v); this._visible = v; }
    get locked() { return this._locked; }
    set locked(v) { assertBool("set_locked", v); this._locked = v; }
    get blendMode() { return this._blendMode; }
    set blendMode(v) { this._blendMode = assertEnum("set_blendMode", v, BLEND_MODES); }

    get layoutAlign() { return this._layoutAlign; }
    set layoutAlign(v) { this._layoutAlign = assertEnum("set_layoutAlign", v, LAYOUT_ALIGN); }
    get layoutGrow() { return this._layoutGrow; }
    set layoutGrow(v) { assertNumber("set_layoutGrow", v); this._layoutGrow = v; }
    get layoutPositioning() { return this._layoutPositioning; }
    set layoutPositioning(v) {
      assertEnum("set_layoutPositioning", v, LAYOUT_POSITIONING);
      if (v === "ABSOLUTE") {
        const p = this.parent;
        if (!p || !("layoutMode" in p) || p.layoutMode === "NONE") {
          fail("set_layoutPositioning", "ABSOLUTE positioning requires an auto-layout parent");
        }
      }
      this._layoutPositioning = v;
    }
    _inAutoLayoutParent() {
      const p = this.parent;
      return Boolean(p && "layoutMode" in p && p.layoutMode !== "NONE" && this._layoutPositioning !== "ABSOLUTE");
    }
    get layoutSizingHorizontal() { return this._layoutSizingH; }
    set layoutSizingHorizontal(v) {
      assertEnum("set_layoutSizingHorizontal", v, LAYOUT_SIZING);
      if (v === "FILL" && !this._inAutoLayoutParent()) fail("set_layoutSizingHorizontal", "FILL is only valid on auto-layout children");
      if (v === "HUG" && !(this.type === "TEXT" || ("layoutMode" in this && this.layoutMode !== "NONE"))) {
        fail("set_layoutSizingHorizontal", "HUG is only valid on auto-layout frames and text nodes");
      }
      this._layoutSizingH = v;
      state.sizingSets.push({ id: this.id, name: this.name, type: this.type, axis: "horizontal", value: v });
    }
    get layoutSizingVertical() { return this._layoutSizingV; }
    set layoutSizingVertical(v) {
      assertEnum("set_layoutSizingVertical", v, LAYOUT_SIZING);
      if (v === "FILL" && !this._inAutoLayoutParent()) fail("set_layoutSizingVertical", "FILL is only valid on auto-layout children");
      if (v === "HUG" && !(this.type === "TEXT" || ("layoutMode" in this && this.layoutMode !== "NONE"))) {
        fail("set_layoutSizingVertical", "HUG is only valid on auto-layout frames and text nodes");
      }
      this._layoutSizingV = v;
      state.sizingSets.push({ id: this.id, name: this.name, type: this.type, axis: "vertical", value: v });
    }
    _setMinMax(prop, v) {
      if (v !== null) {
        assertNumber("set_" + prop, v);
        if (v < 0) fail("set_" + prop, prop + " must be >= 0");
      }
      const selfAuto = "layoutMode" in this && this.layoutMode !== "NONE";
      if (!selfAuto && !this._inAutoLayoutParent()) {
        fail("set_" + prop, prop + " is only valid on auto-layout frames and their direct children");
      }
      this["_" + prop] = v;
    }
    get minWidth() { return this._minWidth; }
    set minWidth(v) { this._setMinMax("minWidth", v); }
    get maxWidth() { return this._maxWidth; }
    set maxWidth(v) { this._setMinMax("maxWidth", v); }
    get minHeight() { return this._minHeight; }
    set minHeight(v) { this._setMinMax("minHeight", v); }
    get maxHeight() { return this._maxHeight; }
    set maxHeight(v) { this._setMinMax("maxHeight", v); }

    resize(w, h) {
      assertAlive(this, "resize");
      assertNumber("resize", w);
      assertNumber("resize", h);
      if (this.type === "LINE") {
        if (h !== 0) fail("resize", "Line nodes cannot be resized vertically (height must be 0, got " + h + ")");
        if (w < 0.01) fail("resize", "Expected width >= 0.01");
        this._w = w;
        return;
      }
      if (w < 0.01 || h < 0.01) fail("resize", "Expected width/height >= 0.01");
      if (this.type === "GROUP") {
        const sx = w / Math.max(0.01, this._w);
        const sy = h / Math.max(0.01, this._h);
        (this.children || []).forEach(function scale(child) {
          child._x *= sx; child._y *= sy;
          child._w = Math.max(0.01, child._w * sx);
          child._h = Math.max(0.01, child._h * sy);
          if (Array.isArray(child.children)) child.children.forEach(scale);
        });
        this._w = w;
        this._h = h;
        return;
      }
      this._w = w;
      this._h = h;
      if (this.type === "TEXT") this._measure({ resized: true });
    }

    async exportAsync(settings) {
      assertAlive(this, "exportAsync");
      const fmt = (settings && settings.format) || "PNG";
      if (fmt !== "PNG") fail("exportAsync", "Mock only supports PNG export");
      return encodePng(Math.max(1, this.width), Math.max(1, this.height));
    }
  }

  // ------------------------------------------------------------ GeometryNode

  class GeometryNode extends SceneNode {
    constructor(type) {
      super(type);
      this._fills = [{ type: "SOLID", visible: true, opacity: 1, blendMode: "NORMAL", color: { r: 0.85, g: 0.85, b: 0.85 } }];
      this._strokes = [];
      this._strokeWeight = 1;
      this._strokeAlign = "INSIDE";
      this._strokeCap = "NONE";
      this._strokeJoin = "MITER";
      this._dashPattern = [];
      this._effects = [];
      this._constraints = { horizontal: "MIN", vertical: "MIN" };
      this._isMask = false;
    }
    get fills() { return this._fills; }
    set fills(v) { this._fills = validatePaintArray("set_fills", v); }
    get strokes() { return this._strokes; }
    set strokes(v) { this._strokes = validatePaintArray("set_strokes", v); }
    get strokeWeight() { return this._strokeWeight; }
    set strokeWeight(v) {
      assertNumber("set_strokeWeight", v);
      if (v < 0) fail("set_strokeWeight", "strokeWeight must be >= 0");
      this._strokeWeight = v;
    }
    get strokeAlign() { return this._strokeAlign; }
    set strokeAlign(v) { this._strokeAlign = assertEnum("set_strokeAlign", v, STROKE_ALIGNS); }
    get strokeCap() { return this._strokeCap; }
    set strokeCap(v) { this._strokeCap = assertEnum("set_strokeCap", v, STROKE_CAPS); }
    get strokeJoin() { return this._strokeJoin; }
    set strokeJoin(v) { this._strokeJoin = assertEnum("set_strokeJoin", v, STROKE_JOINS); }
    get dashPattern() { return this._dashPattern; }
    set dashPattern(v) {
      if (!Array.isArray(v)) fail("set_dashPattern", "Expected an array of numbers");
      v.forEach((n) => {
        assertNumber("set_dashPattern", n);
        if (n < 0) fail("set_dashPattern", "Dash values must be >= 0");
      });
      this._dashPattern = v.slice();
    }
    get effects() { return this._effects; }
    set effects(v) {
      if (!Array.isArray(v)) fail("set_effects", "Expected an array of effects");
      v.forEach((e) => validateEffect("set_effects", e));
      this._effects = v.map((e) => Object.assign({}, e));
    }
    get constraints() { return this._constraints; }
    set constraints(v) {
      if (!v || typeof v !== "object") fail("set_constraints", "Expected a constraints object");
      assertEnum("set_constraints", v.horizontal, CONSTRAINT_TYPES);
      assertEnum("set_constraints", v.vertical, CONSTRAINT_TYPES);
      this._constraints = { horizontal: v.horizontal, vertical: v.vertical };
    }
    get isMask() { return this._isMask; }
    set isMask(v) { assertBool("set_isMask", v); this._isMask = v; }
  }

  function addCornerRadius(node, allCorners) {
    node._cornerRadius = 0;
    Object.defineProperty(node, "cornerRadius", {
      enumerable: true,
      get() { return this._cornerRadius; },
      set(v) {
        assertNumber("set_cornerRadius", v);
        if (v < 0) fail("set_cornerRadius", "cornerRadius must be >= 0");
        this._cornerRadius = v;
        if (allCorners) {
          this._topLeftRadius = v; this._topRightRadius = v;
          this._bottomRightRadius = v; this._bottomLeftRadius = v;
        }
      },
    });
    if (allCorners) {
      ["topLeftRadius", "topRightRadius", "bottomRightRadius", "bottomLeftRadius"].forEach((prop) => {
        node["_" + prop] = 0;
        Object.defineProperty(node, prop, {
          enumerable: true,
          get() { return this["_" + prop]; },
          set(v) {
            assertNumber("set_" + prop, v);
            if (v < 0) fail("set_" + prop, prop + " must be >= 0");
            this["_" + prop] = v;
          },
        });
      });
    }
  }

  // ----------------------------------------------------------------- shapes

  class RectangleNode extends GeometryNode {
    constructor() { super("RECTANGLE"); this.name = "Rectangle"; addCornerRadius(this, true); }
  }
  class EllipseNode extends GeometryNode {
    constructor() { super("ELLIPSE"); this.name = "Ellipse"; }
  }
  class LineNode extends GeometryNode {
    constructor() { super("LINE"); this.name = "Line"; this._h = 0; this._fills = []; }
  }
  class PolygonNode extends GeometryNode {
    constructor() { super("POLYGON"); this.name = "Polygon"; this.pointCount = 3; addCornerRadius(this, false); }
  }
  class StarNode extends GeometryNode {
    constructor() { super("STAR"); this.name = "Star"; this.pointCount = 5; this.innerRadius = 0.382; addCornerRadius(this, false); }
  }
  class VectorNode extends GeometryNode {
    constructor() { super("VECTOR"); this.name = "Vector"; this.vectorData = null; }
  }

  // ------------------------------------------------------------------ frames

  class FrameNode extends GeometryNode {
    constructor(type) {
      super(type || "FRAME");
      this.name = "Frame";
      this.children = [];
      this._clipsContent = true;
      this._layoutMode = "NONE";
      this._primaryAxisSizingMode = "FIXED";
      this._counterAxisSizingMode = "FIXED";
      this._primaryAxisAlignItems = "MIN";
      this._counterAxisAlignItems = "MIN";
      this._paddingTop = 0;
      this._paddingRight = 0;
      this._paddingBottom = 0;
      this._paddingLeft = 0;
      this._itemSpacing = 0;
      this._layoutWrap = "NO_WRAP";
      this._strokesIncludedInLayout = false;
      this._fills = [{ type: "SOLID", visible: true, opacity: 1, blendMode: "NORMAL", color: { r: 1, g: 1, b: 1 } }];
      addCornerRadius(this, true);
    }
    appendChild(child) {
      assertAlive(this, "appendChild");
      if (!child || !(child instanceof BaseNode)) fail("appendChild", "Expected a node");
      if (child.removed) fail("appendChild", "Cannot move removed node " + child.id);
      let p = this;
      while (p) {
        if (p === child) fail("appendChild", "Cannot append a node inside itself");
        p = p.parent;
      }
      detach(child);
      child.parent = this;
      this.children.push(child);
    }
    get clipsContent() { return this._clipsContent; }
    set clipsContent(v) { assertBool("set_clipsContent", v); this._clipsContent = v; }
    get layoutMode() { return this._layoutMode; }
    set layoutMode(v) { this._layoutMode = assertEnum("set_layoutMode", v, LAYOUT_MODES); }
    _requireAutoLayout(where) {
      if (this._layoutMode === "NONE") fail(where, "Node must have auto-layout enabled (layoutMode !== NONE)");
    }
    get primaryAxisSizingMode() { return this._primaryAxisSizingMode; }
    set primaryAxisSizingMode(v) { this._requireAutoLayout("set_primaryAxisSizingMode"); this._primaryAxisSizingMode = assertEnum("set_primaryAxisSizingMode", v, AXIS_SIZING); }
    get counterAxisSizingMode() { return this._counterAxisSizingMode; }
    set counterAxisSizingMode(v) { this._requireAutoLayout("set_counterAxisSizingMode"); this._counterAxisSizingMode = assertEnum("set_counterAxisSizingMode", v, AXIS_SIZING); }
    get primaryAxisAlignItems() { return this._primaryAxisAlignItems; }
    set primaryAxisAlignItems(v) { this._requireAutoLayout("set_primaryAxisAlignItems"); this._primaryAxisAlignItems = assertEnum("set_primaryAxisAlignItems", v, PRIMARY_ALIGN); }
    get counterAxisAlignItems() { return this._counterAxisAlignItems; }
    set counterAxisAlignItems(v) { this._requireAutoLayout("set_counterAxisAlignItems"); this._counterAxisAlignItems = assertEnum("set_counterAxisAlignItems", v, COUNTER_ALIGN); }
    get itemSpacing() { return this._itemSpacing; }
    set itemSpacing(v) { this._requireAutoLayout("set_itemSpacing"); assertNumber("set_itemSpacing", v); this._itemSpacing = v; }
    get layoutWrap() { return this._layoutWrap; }
    set layoutWrap(v) { this._requireAutoLayout("set_layoutWrap"); this._layoutWrap = assertEnum("set_layoutWrap", v, LAYOUT_WRAP); }
    get strokesIncludedInLayout() { return this._strokesIncludedInLayout; }
    set strokesIncludedInLayout(v) { this._requireAutoLayout("set_strokesIncludedInLayout"); assertBool("set_strokesIncludedInLayout", v); this._strokesIncludedInLayout = v; }
  }
  ["paddingTop", "paddingRight", "paddingBottom", "paddingLeft"].forEach((prop) => {
    Object.defineProperty(FrameNode.prototype, prop, {
      enumerable: true,
      get() { return this["_" + prop]; },
      set(v) {
        this._requireAutoLayout("set_" + prop);
        assertNumber("set_" + prop, v);
        this["_" + prop] = v;
      },
    });
  });

  class ComponentNode extends FrameNode {
    constructor() {
      super("COMPONENT");
      this.name = "Component";
    }
    createInstance() {
      assertAlive(this, "createInstance");
      const inst = new InstanceNode();
      inst.mainComponentId = this.id;
      inst.name = this.name;
      inst._w = this._w;
      inst._h = this._h;
      inst._fills = this._fills.map((p) => Object.assign({}, p));
      currentPage.appendChild(inst);
      return inst;
    }
  }

  class InstanceNode extends FrameNode {
    constructor() {
      super("INSTANCE");
      this.name = "Instance";
      this.mainComponentId = null;
    }
  }

  class GroupNode extends SceneNode {
    constructor() {
      super("GROUP");
      this.name = "Group";
      this.children = [];
      this._effects = [];
      this._isMask = false;
    }
    appendChild(child) {
      assertAlive(this, "appendChild");
      if (!child || child.removed) fail("appendChild", "Cannot move removed node");
      detach(child);
      child.parent = this;
      this.children.push(child);
    }
    get effects() { return this._effects; }
    set effects(v) {
      if (!Array.isArray(v)) fail("set_effects", "Expected an array of effects");
      v.forEach((e) => validateEffect("set_effects", e));
      this._effects = v.map((e) => Object.assign({}, e));
    }
    get isMask() { return this._isMask; }
    set isMask(v) { assertBool("set_isMask", v); this._isMask = v; }
  }

  // ------------------------------------------------------------------- text

  class TextNode extends GeometryNode {
    constructor() {
      super("TEXT");
      this.name = "Text";
      this._w = 0.01;
      this._h = 14.4;
      this._characters = "";
      this._fontName = { family: "Inter", style: "Regular" };
      this._fontSize = 12;
      this._textAutoResize = "WIDTH_AND_HEIGHT";
      this._textAlignHorizontal = "LEFT";
      this._textAlignVertical = "TOP";
      this._letterSpacing = { unit: "PERCENT", value: 0 };
      this._lineHeight = { unit: "AUTO" };
      this._textCase = "ORIGINAL";
      this._textDecoration = "NONE";
      this._leadingTrim = "NONE";
      this._autoRename = true;
      this._textStyleId = "";
      this._styleRanges = [];
      this._fills = [{ type: "SOLID", visible: true, opacity: 1, blendMode: "NORMAL", color: { r: 0, g: 0, b: 0 } }];
    }

    _requireFontLoaded(where, font) {
      const f = font || this._fontName;
      if (!state.loadedFonts.has(fontKey(f))) {
        fail(where, 'Cannot write to node with unloaded font "' + f.family + " " + f.style + '". Please call figma.loadFontAsync first');
      }
    }

    // ---- synthetic layout model -------------------------------------------
    _rangeValue(prop, index, fallback) {
      let value = fallback;
      this._styleRanges.forEach((r) => {
        if (r.prop === prop && index >= r.start && index < r.end) value = r.value;
      });
      return value;
    }
    _charWidth(index) {
      const size = this._rangeValue("fontSize", index, this._fontSize);
      return size * CHAR_WIDTH_RATIO;
    }
    _lineHeightPx() {
      const lh = this._lineHeight;
      if (lh.unit === "PIXELS") return Math.max(0.01, lh.value);
      if (lh.unit === "PERCENT") return Math.max(0.01, this._fontSize * lh.value / 100);
      return this._fontSize * AUTO_LINE_HEIGHT;
    }
    _letterSpacingPx() {
      const ls = this._letterSpacing;
      if (ls.unit === "PIXELS") return ls.value;
      return this._fontSize * ls.value / 100;
    }
    _measure(opts) {
      const text = this._characters;
      const spacing = this._letterSpacingPx();
      const lineH = this._lineHeightPx();
      const logical = text.split("\n");
      let charIndex = 0;
      const lineWidths = logical.map((line) => {
        let w = 0;
        for (let i = 0; i < line.length; i += 1) w += this._charWidth(charIndex + i);
        if (line.length > 1) w += spacing * (line.length - 1);
        charIndex += line.length + 1;
        return w;
      });
      const contentWidth = Math.max(0.01, Math.max.apply(null, lineWidths.concat([0])));
      if (this._textAutoResize === "WIDTH_AND_HEIGHT") {
        this._w = contentWidth;
        this._h = Math.max(0.01, logical.length * lineH);
      } else if (this._textAutoResize === "HEIGHT") {
        // wrap each logical line into the fixed width
        const avail = Math.max(0.01, this._w);
        let total = 0;
        let ci = 0;
        logical.forEach((line) => {
          if (!line.length) { total += 1; ci += 1; return; }
          let lineCount = 1;
          let used = 0;
          for (let i = 0; i < line.length; i += 1) {
            const cw = this._charWidth(ci + i) + (used > 0 ? spacing : 0);
            if (used > 0 && used + cw > avail) {
              lineCount += 1;
              used = this._charWidth(ci + i);
            } else {
              used += cw;
            }
          }
          total += lineCount;
          ci += line.length + 1;
        });
        this._h = Math.max(0.01, total * lineH);
      }
      // NONE / TRUNCATE: keep both dimensions as-is
    }

    get characters() { return this._characters; }
    set characters(v) {
      if (typeof v !== "string") fail("set_characters", "Expected a string");
      this._requireFontLoaded("set_characters");
      this._characters = v;
      this._styleRanges = this._styleRanges.filter((r) => r.start < v.length);
      this._measure();
    }
    get fontName() { return this._fontName; }
    set fontName(v) {
      const font = validateFontName("set_fontName", v);
      this._requireFontLoaded("set_fontName", font);
      this._fontName = font;
      this._measure();
    }
    get fontSize() { return this._fontSize; }
    set fontSize(v) {
      assertNumber("set_fontSize", v);
      if (v < 1) fail("set_fontSize", "fontSize must be >= 1");
      this._requireFontLoaded("set_fontSize");
      this._fontSize = v;
      this._measure();
    }
    get textAutoResize() { return this._textAutoResize; }
    set textAutoResize(v) {
      this._textAutoResize = assertEnum("set_textAutoResize", v, TEXT_AUTO_RESIZE);
      this._measure();
    }
    get textAlignHorizontal() { return this._textAlignHorizontal; }
    set textAlignHorizontal(v) { this._textAlignHorizontal = assertEnum("set_textAlignHorizontal", v, TEXT_ALIGN_H); }
    get textAlignVertical() { return this._textAlignVertical; }
    set textAlignVertical(v) { this._textAlignVertical = assertEnum("set_textAlignVertical", v, TEXT_ALIGN_V); }
    get letterSpacing() { return this._letterSpacing; }
    set letterSpacing(v) {
      this._requireFontLoaded("set_letterSpacing");
      this._letterSpacing = validateLetterSpacing("set_letterSpacing", v);
      this._measure();
    }
    get lineHeight() { return this._lineHeight; }
    set lineHeight(v) {
      this._requireFontLoaded("set_lineHeight");
      this._lineHeight = validateLineHeight("set_lineHeight", v);
      this._measure();
    }
    get textCase() { return this._textCase; }
    set textCase(v) {
      this._requireFontLoaded("set_textCase");
      this._textCase = assertEnum("set_textCase", v, TEXT_CASE);
    }
    get textDecoration() { return this._textDecoration; }
    set textDecoration(v) {
      this._requireFontLoaded("set_textDecoration");
      this._textDecoration = assertEnum("set_textDecoration", v, TEXT_DECORATION);
    }
    get leadingTrim() { return this._leadingTrim; }
    set leadingTrim(v) {
      this._requireFontLoaded("set_leadingTrim");
      this._leadingTrim = assertEnum("set_leadingTrim", v, LEADING_TRIM);
    }
    get autoRename() { return this._autoRename; }
    set autoRename(v) { assertBool("set_autoRename", v); this._autoRename = v; }
    get textStyleId() { return this._textStyleId; }
    set textStyleId(v) { this._textStyleId = String(v); }
    async setTextStyleIdAsync(id) { this._textStyleId = String(id); }

    _assertRange(where, start, end) {
      if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end > this._characters.length || start >= end) {
        fail(where, "Range [" + start + ", " + end + ") is outside the bounds of the text (length " + this._characters.length + ")");
      }
    }
    // Real Figma throws when a range is mutated while the font ACTUALLY COVERING that
    // range is not loaded — that font is the last setRangeFontName applied over each
    // char, falling back to the node's base fontName. Checking only the base font (the
    // old behavior) let a "set range size before loading the range's own font" bug slip
    // through the mock while crashing in the real sandbox. Enforce the per-range font.
    _requireRangeFontsLoaded(where, start, end) {
      for (let i = start; i < end; i += 1) {
        const font = this._rangeValue("fontName", i, this._fontName);
        if (!state.loadedFonts.has(fontKey(font))) {
          fail(where, 'Cannot write to a range with unloaded font "' + font.family + " " + font.style +
            '". Please call figma.loadFontAsync first');
        }
      }
    }
    _pushRange(where, start, end, prop, value) {
      this._assertRange(where, start, end);
      this._requireRangeFontsLoaded(where, start, end);
      this._styleRanges.push({ start, end, prop, value });
      this._measure();
    }
    setRangeFontName(start, end, font) {
      const f = validateFontName("setRangeFontName", font);
      this._assertRange("setRangeFontName", start, end);
      if (!state.loadedFonts.has(fontKey(f))) {
        fail("setRangeFontName", 'Cannot write to node with unloaded font "' + f.family + " " + f.style + '"');
      }
      this._styleRanges.push({ start, end, prop: "fontName", value: f });
    }
    setRangeFontSize(start, end, size) {
      assertNumber("setRangeFontSize", size);
      if (size < 1) fail("setRangeFontSize", "fontSize must be >= 1");
      this._pushRange("setRangeFontSize", start, end, "fontSize", size);
    }
    setRangeFills(start, end, fills) {
      const value = validatePaintArray("setRangeFills", fills);
      this._pushRange("setRangeFills", start, end, "fills", value);
    }
    setRangeLetterSpacing(start, end, v) {
      this._pushRange("setRangeLetterSpacing", start, end, "letterSpacing", validateLetterSpacing("setRangeLetterSpacing", v));
    }
    setRangeLineHeight(start, end, v) {
      this._pushRange("setRangeLineHeight", start, end, "lineHeight", validateLineHeight("setRangeLineHeight", v));
    }
    setRangeTextDecoration(start, end, v) {
      assertEnum("setRangeTextDecoration", v, TEXT_DECORATION);
      this._pushRange("setRangeTextDecoration", start, end, "textDecoration", v);
    }
    setRangeTextCase(start, end, v) {
      assertEnum("setRangeTextCase", v, TEXT_CASE);
      this._pushRange("setRangeTextCase", start, end, "textCase", v);
    }
  }

  // ------------------------------------------------------------- text styles

  class TextStyle {
    constructor() {
      this.id = nextId("S");
      this.type = "TEXT";
      this._name = "Text style";
      this._fontName = { family: "Inter", style: "Regular" };
      this._fontSize = 12;
      this._lineHeight = { unit: "AUTO" };
      this._letterSpacing = { unit: "PERCENT", value: 0 };
      this._leadingTrim = "NONE";
      this._textCase = "ORIGINAL";
      this._textDecoration = "NONE";
      this._pluginData = Object.create(null);
      this.removed = false;
    }
    get name() { return this._name; }
    set name(v) {
      if (typeof v !== "string" || !v.length) fail("set_name", "Style name must be a non-empty string");
      this._name = v;
    }
    get fontName() { return this._fontName; }
    set fontName(v) {
      const font = validateFontName("set_fontName", v);
      if (!state.loadedFonts.has(fontKey(font))) {
        fail("set_fontName", 'Cannot set style font to unloaded font "' + font.family + " " + font.style + '"');
      }
      this._fontName = font;
    }
    get fontSize() { return this._fontSize; }
    set fontSize(v) {
      assertNumber("set_fontSize", v);
      if (v < 1) fail("set_fontSize", "fontSize must be >= 1");
      this._fontSize = v;
    }
    get lineHeight() { return this._lineHeight; }
    set lineHeight(v) { this._lineHeight = validateLineHeight("set_lineHeight", v); }
    get letterSpacing() { return this._letterSpacing; }
    set letterSpacing(v) { this._letterSpacing = validateLetterSpacing("set_letterSpacing", v); }
    get leadingTrim() { return this._leadingTrim; }
    set leadingTrim(v) { this._leadingTrim = assertEnum("set_leadingTrim", v, LEADING_TRIM); }
    get textCase() { return this._textCase; }
    set textCase(v) { this._textCase = assertEnum("set_textCase", v, TEXT_CASE); }
    get textDecoration() { return this._textDecoration; }
    set textDecoration(v) { this._textDecoration = assertEnum("set_textDecoration", v, TEXT_DECORATION); }
    setPluginData(key, value) { this._pluginData[String(key)] = String(value); }
    getPluginData(key) { return this._pluginData[String(key)] || ""; }
    remove() {
      this.removed = true;
      const idx = state.localTextStyles.indexOf(this);
      if (idx >= 0) state.localTextStyles.splice(idx, 1);
    }
  }

  // -------------------------------------------------------------------- page

  class PageNode extends BaseNode {
    constructor() {
      super("PAGE");
      this.name = "Page 1";
      this.children = [];
      this._selection = [];
    }
    appendChild(child) {
      if (!child || child.removed) fail("appendChild", "Cannot move removed node");
      detach(child);
      child.parent = this;
      this.children.push(child);
    }
    findAll(callback) {
      const out = [];
      const walk = (node) => {
        (node.children || []).forEach((child) => {
          let matched = true;
          if (typeof callback === "function") {
            try { matched = Boolean(callback(child)); } catch (_) { matched = false; }
          }
          if (matched) out.push(child);
          walk(child);
        });
      };
      walk(this);
      return out;
    }
    get selection() { return this._selection; }
    set selection(v) {
      if (!Array.isArray(v)) fail("set_selection", "Expected an array of nodes");
      v.forEach((n) => {
        if (!n || !(n instanceof BaseNode) || n.removed) fail("set_selection", "Selection contains a removed or invalid node");
      });
      this._selection = v.slice();
    }
  }

  const currentPage = new PageNode();

  function autoAppend(node) {
    currentPage.appendChild(node);
    return node;
  }

  // ------------------------------------------------------------- figma object

  const figma = {
    mixed: Symbol("figma.mixed"),

    showUI(html, opts) {
      state.showUI = { htmlLength: typeof html === "string" ? html.length : 0, options: opts || {} };
    },

    ui: {
      onmessage: null,
      postMessage(message) {
        state.uiMessages.push({ at: Date.now(), message: clonePlain(message) });
      },
      resize() {},
      close() {},
    },

    currentPage,

    viewport: {
      center: { x: 0, y: 0 },
      zoom: 1,
      scrollAndZoomIntoView(nodes) {
        state.viewportZoomedTo = (nodes || []).map((n) => n.id);
      },
    },

    notify(message, opts) {
      state.notifications.push({ message: String(message), error: Boolean(opts && opts.error) });
      return { cancel() {} };
    },

    clientStorage: {
      async getAsync(key) {
        return state.clientStorage.has(key) ? clonePlain(state.clientStorage.get(key)) : undefined;
      },
      async setAsync(key, value) {
        state.clientStorage.set(key, clonePlain(value));
      },
      async deleteAsync(key) {
        state.clientStorage.delete(key);
      },
    },

    async listAvailableFontsAsync() {
      return installedFonts.map((f) => ({ fontName: { family: f.family, style: f.style } }));
    },

    async loadFontAsync(fontName) {
      const font = validateFontName("loadFontAsync", fontName);
      if (!installedKeys.has(fontKey(font))) {
        state.fontLoadFailures.push(font);
        fail("loadFontAsync", 'Font "' + font.family + " " + font.style + '" is not installed');
      }
      state.loadedFonts.add(fontKey(font));
      state.fontLoads.push(font);
    },

    createFrame() { return autoAppend(new FrameNode()); },
    createComponent() { return autoAppend(new ComponentNode()); },
    createRectangle() { return autoAppend(new RectangleNode()); },
    createEllipse() { return autoAppend(new EllipseNode()); },
    createLine() { return autoAppend(new LineNode()); },
    createPolygon() { return autoAppend(new PolygonNode()); },
    createStar() { return autoAppend(new StarNode()); },
    createText() { return autoAppend(new TextNode()); },

    createTextStyle() {
      const style = new TextStyle();
      state.localTextStyles.push(style);
      return style;
    },
    async getLocalTextStylesAsync() { return state.localTextStyles.slice(); },
    getLocalTextStyles() { return state.localTextStyles.slice(); },

    createImage(bytes) {
      if (!(bytes instanceof Uint8Array)) fail("createImage", "Expected a Uint8Array");
      if (!bytes.length) fail("createImage", "Image data is empty");
      const kind = sniffImage(bytes);
      if (!kind) fail("createImage", "Unsupported image type (expected PNG, JPEG, GIF or WebP)");
      let size = null;
      if (kind === "png") {
        size = pngSize(bytes);
        if (size && (size.width > 4096 || size.height > 4096)) {
          fail("createImage", "Image is too large (max 4096x4096, got " + size.width + "x" + size.height + ")");
        }
      }
      const hash = crypto.createHash("sha1").update(bytes).digest("hex");
      state.images.push({ hash, kind, byteLength: bytes.length, size });
      return {
        hash,
        async getBytesAsync() { return bytes; },
        async getSizeAsync() { return size || { width: 0, height: 0 }; },
      };
    },

    createNodeFromSvg(svg) {
      if (typeof svg !== "string" || svg.indexOf("<svg") < 0) fail("createNodeFromSvg", "Invalid SVG: missing <svg> root");
      const selfClosed = /<svg\b[^>]*\/>/.test(svg);
      if (svg.indexOf("</svg>") < 0 && !selfClosed) fail("createNodeFromSvg", "Malformed SVG: unterminated <svg> element");
      const openPaths = (svg.match(/<path\b/g) || []).length;
      const closedPathTags = (svg.match(/<path\b[^>]*>/g) || []).length;
      if (openPaths !== closedPathTags) fail("createNodeFromSvg", "Malformed SVG: unterminated <path> element");

      const rootTag = (svg.match(/<svg\b[^>]*>/) || [""])[0];
      const attr = (name) => {
        const m = rootTag.match(new RegExp(name + '\\s*=\\s*"([^"]*)"'));
        return m ? m[1] : null;
      };
      let w = parseFloat(attr("width"));
      let h = parseFloat(attr("height"));
      const viewBox = (attr("viewBox") || "").trim().split(/[\s,]+/).map(Number);
      if (!Number.isFinite(w) && viewBox.length === 4 && Number.isFinite(viewBox[2])) w = viewBox[2];
      if (!Number.isFinite(h) && viewBox.length === 4 && Number.isFinite(viewBox[3])) h = viewBox[3];

      const frame = new FrameNode();
      frame.name = "Vector";
      frame._fills = [];
      frame._clipsContent = false;
      frame._w = Math.max(0.01, Number.isFinite(w) ? w : 100);
      frame._h = Math.max(0.01, Number.isFinite(h) ? h : 100);

      const pathTags = svg.match(/<path\b[^>]*>/g) || [];
      pathTags.forEach((tag) => {
        const node = new VectorNode();
        const dMatch = tag.match(/\bd\s*=\s*"([^"]*)"/);
        const fillMatch = tag.match(/\bfill\s*=\s*"([^"]*)"/);
        node.vectorData = { d: dMatch ? dMatch[1] : "", fill: fillMatch ? fillMatch[1] : null };
        node.name = "Path";
        node.parent = frame;
        frame.children.push(node);
      });
      const otherTags = (svg.match(/<(rect|circle|ellipse|polygon|polyline|line)\b/g) || []).length;
      state.svgImports.push({ bytes: svg.length, pathCount: pathTags.length, otherShapeCount: otherTags });
      return autoAppend(frame);
    },

    group(nodes, parent, index) {
      if (!Array.isArray(nodes) || !nodes.length) fail("group", "First argument must be an array of at least one node");
      nodes.forEach((n) => {
        if (!n || !(n instanceof BaseNode)) fail("group", "Array contains an invalid node");
        if (n.removed) fail("group", "Cannot group removed node " + n.id + ' ("' + n.name + '")');
      });
      if (!parent || parent.removed || !Array.isArray(parent.children)) fail("group", "Invalid parent node");
      const minX = Math.min.apply(null, nodes.map((n) => n.x));
      const minY = Math.min.apply(null, nodes.map((n) => n.y));
      const maxX = Math.max.apply(null, nodes.map((n) => n.x + n.width));
      const maxY = Math.max.apply(null, nodes.map((n) => n.y + n.height));
      const group = new GroupNode();
      group._x = minX;
      group._y = minY;
      group._w = Math.max(0.01, maxX - minX);
      group._h = Math.max(0.01, maxY - minY);
      nodes.forEach((n) => {
        detach(n);
        n._x -= minX;
        n._y -= minY;
        n.parent = group;
        group.children.push(n);
      });
      group.parent = parent;
      if (Number.isInteger(index) && index >= 0 && index <= parent.children.length) parent.children.splice(index, 0, group);
      else parent.children.push(group);
      return group;
    },
  };

  return { figma, state, currentPage };
}

module.exports = { createFigmaMock, encodePng, pngSize, fontKey };
