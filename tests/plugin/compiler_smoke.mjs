import assert from "node:assert/strict";
import fs from "node:fs";
import vm from "node:vm";

let nextId = 1;

class BaseNode {
  constructor(type) {
    this.type = type;
    this.id = String(nextId++);
    this.name = type;
    this.parent = null;
    this.children = [];
    this.x = 0;
    this.y = 0;
    this.width = 100;
    this.height = 100;
    this.opacity = 1;
    this.rotation = 0;
    this.visible = true;
    this.locked = false;
    this.fills = [];
    this.strokes = [];
    this.effects = [];
    this.constraints = { horizontal: "LEFT", vertical: "TOP" };
    this.strokeWeight = 1;
    this.strokeAlign = "CENTER";
    this.strokeCap = "NONE";
    this.strokeJoin = "MITER";
    this.dashPattern = [];
    this.isMask = false;
    this.cornerRadius = 0;
    this.topLeftRadius = 0;
    this.topRightRadius = 0;
    this.bottomLeftRadius = 0;
    this.bottomRightRadius = 0;
    this.blendMode = "PASS_THROUGH";
    this.clipsContent = false;
    this.layoutMode = "NONE";
    this.layoutWrap = "NO_WRAP";
    this.primaryAxisSizingMode = "FIXED";
    this.counterAxisSizingMode = "FIXED";
    this.primaryAxisAlignItems = "MIN";
    this.counterAxisAlignItems = "MIN";
    this.paddingTop = 0;
    this.paddingRight = 0;
    this.paddingBottom = 0;
    this.paddingLeft = 0;
    this.itemSpacing = 0;
    this.layoutPositioning = "AUTO";
    this.layoutAlign = "INHERIT";
    this.layoutGrow = 0;
    this.layoutSizingHorizontal = "FIXED";
    this.layoutSizingVertical = "FIXED";
    this._data = {};
  }

  appendChild(node) {
    if (node.parent) {
      const index = node.parent.children.indexOf(node);
      if (index >= 0) node.parent.children.splice(index, 1);
    }
    node.parent = this;
    this.children.push(node);
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
  }

  setPluginData(key, value) {
    this._data[key] = value;
  }

  getPluginData(key) {
    return this._data[key] || "";
  }

  remove() {
    if (!this.parent) return;
    const index = this.parent.children.indexOf(this);
    if (index >= 0) this.parent.children.splice(index, 1);
    this.parent = null;
  }

  async exportAsync() {
    return new Uint8Array([1, 2, 3]);
  }
}

class TextNode extends BaseNode {
  constructor() {
    super("TEXT");
    this._characters = "";
    this._fontSize = 12;
    this.fontName = { family: "Inter", style: "Regular" };
    this.textAutoResize = "WIDTH_AND_HEIGHT";
    this.lineHeight = { unit: "AUTO" };
    this.letterSpacing = { unit: "PIXELS", value: 0 };
    this.textCase = "ORIGINAL";
    this.textDecoration = "NONE";
    this.leadingTrim = "NONE";
    this.textAlignHorizontal = "LEFT";
    this.textAlignVertical = "TOP";
    this.autoRename = true;
    this.recalculate();
  }

  set characters(value) {
    this._characters = String(value);
    this.recalculate();
  }

  get characters() {
    return this._characters;
  }

  set fontSize(value) {
    this._fontSize = Number(value);
    this.recalculate();
  }

  get fontSize() {
    return this._fontSize;
  }

  recalculate() {
    const lines = (this._characters || "").split("\n");
    const longest = Math.max(1, ...lines.map((line) => line.length));
    if (this.textAutoResize !== "HEIGHT") this.width = Math.max(1, longest * this._fontSize * 0.58);
    this.height = Math.max(1, lines.length * this._fontSize * 1.2);
  }

  resize(width, height) {
    this.width = width;
    this.height = height;
    this.recalculate();
    if (this.textAutoResize === "HEIGHT") this.width = width;
  }

  setRangeFontName() {}
  setRangeFontSize() {}
  setRangeFills() {}
  setRangeLetterSpacing() {}
  setRangeLineHeight() {}
  setRangeTextDecoration() {}
  setRangeTextCase() {}

  async setTextStyleIdAsync(id) {
    this.textStyleId = id;
  }

  async exportAsync() {
    const bytes = new Uint8Array(24);
    bytes.set([137, 80, 78, 71], 0);
    const view = new DataView(bytes.buffer);
    view.setUint32(16, Math.max(1, Math.ceil(this.width)), false);
    view.setUint32(20, Math.max(1, Math.ceil(this.height)), false);
    return bytes;
  }
}

class ComponentNode extends BaseNode {
  constructor() {
    super("COMPONENT");
  }

  createInstance() {
    return new BaseNode("INSTANCE");
  }
}

class PageNode extends BaseNode {
  constructor() {
    super("PAGE");
    this.selection = [];
  }

  findAll(predicate) {
    const output = [];
    function walk(node) {
      node.children.forEach((child) => {
        if (predicate(child)) output.push(child);
        walk(child);
      });
    }
    walk(this);
    return output;
  }
}

const page = new PageNode();
const posted = [];
const textStyles = [];

function autoParent(node) {
  page.appendChild(node);
  return node;
}

const figma = {
  showUI() {},
  ui: {
    onmessage: null,
    postMessage(message) {
      posted.push(message);
    },
  },
  clientStorage: {
    async getAsync() { return null; },
    async setAsync() {},
  },
  currentPage: page,
  viewport: {
    center: { x: 500, y: 500 },
    scrollAndZoomIntoView() {},
  },
  notify() {},
  async listAvailableFontsAsync() {
    return [
      { fontName: { family: "Inter", style: "Regular" } },
      { fontName: { family: "Inter", style: "Bold" } },
    ];
  },
  async loadFontAsync() {},
  async getLocalTextStylesAsync() {
    return textStyles;
  },
  createTextStyle() {
    const style = {
      id: `style-${nextId++}`,
      name: "",
      fontName: { family: "Inter", style: "Regular" },
      fontSize: 12,
      lineHeight: { unit: "AUTO" },
      letterSpacing: { unit: "PIXELS", value: 0 },
      textCase: "ORIGINAL",
      textDecoration: "NONE",
      _data: {},
      setPluginData(key, value) { this._data[key] = value; },
      getPluginData(key) { return this._data[key] || ""; },
    };
    textStyles.push(style);
    return style;
  },
  createText() { return autoParent(new TextNode()); },
  createFrame() { return autoParent(new BaseNode("FRAME")); },
  createComponent() { return autoParent(new ComponentNode()); },
  createRectangle() { return autoParent(new BaseNode("RECTANGLE")); },
  createEllipse() { return autoParent(new BaseNode("ELLIPSE")); },
  createLine() { return autoParent(new BaseNode("LINE")); },
  createPolygon() { return autoParent(new BaseNode("POLYGON")); },
  createStar() { return autoParent(new BaseNode("STAR")); },
  createNodeFromSvg(svg) {
    const frame = new BaseNode("FRAME");
    frame.svg = svg;
    frame.appendChild(new BaseNode("VECTOR"));
    return autoParent(frame);
  },
  createImage() {
    return { hash: `image-${nextId++}` };
  },
  group(nodes, parent) {
    const minX = Math.min(...nodes.map((node) => node.x));
    const minY = Math.min(...nodes.map((node) => node.y));
    const group = new BaseNode("GROUP");
    group.x = minX;
    group.y = minY;
    group.width = Math.max(...nodes.map((node) => node.x + node.width)) - minX;
    group.height = Math.max(...nodes.map((node) => node.y + node.height)) - minY;
    parent.appendChild(group);
    nodes.forEach((node) => {
      node.x -= minX;
      node.y -= minY;
      group.appendChild(node);
    });
    return group;
  },
};

const sandbox = {
  figma,
  __html__: "",
  console,
  Uint8Array,
  Map,
  Set,
  Promise,
  Date,
  Math,
  Number,
  String,
  Object,
  Array,
  Boolean,
  JSON,
  Error,
  RegExp,
  parseInt,
  parseFloat,
  setTimeout,
  clearTimeout,
};

const code = fs.readFileSync("figma-plugin/code.js", "utf8");
vm.runInNewContext(code, sandbox, { filename: "figma-plugin/code.js" });

const sceneV2 = {
  schema_version: 2,
  id: "scene-v2-smoke",
  name: "Scene v2 smoke",
  canvas: { w: 600, h: 600 },
  layers: [
    {
      id: "background",
      type: "shape",
      name: "Background",
      box: { x: 0, y: 0, w: 600, h: 600 },
      fill: { kind: "flat", color: "#fafafa" },
      z_index: 0,
    },
    {
      id: "copy-stack",
      type: "frame",
      name: "Copy stack",
      box: { x: 40, y: 30, w: 300, h: 180 },
      layout: { mode: "vertical", gap: 12, padding: 16 },
      z_index: 1,
      children: [
        {
          id: "headline",
          type: "text",
          name: "Headline",
          box: { x: 16, y: 16, w: 268, h: 60 },
          visible_box: { x: 16, y: 16, w: 268, h: 60 },
          text: "A better\nheadline",
          style: { fontFamily: "Missing Sans", fontStyle: "Bold", fontWeight: 700, color: "#111111", lineCount: 2, fontCandidates: [{ family: "Missing Sans", style: "Bold", score: 0.9 }, { family: "Inter", style: "Bold", weight: 700, score: 0.82 }] },
          meta: { role: "Headline" },
        },
        {
          id: "arrow",
          type: "svg",
          name: "Arrow",
          box: { x: 16, y: 90, w: 32, h: 32 },
          svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M2 12h18M14 6l6 6-6 6" fill="none" stroke="black"/></svg>',
        },
      ],
    },
    {
      id: "gradient-copy",
      type: "text",
      name: "Gradient copy",
      box: { x: 40, y: 230, w: 220, h: 34 },
      text: "COLOUR TYPE",
      style: {
        fontFamily: "Inter", fontSize: 20, color: "#111111",
        fills: [{ kind: "linear", angle: 0, stops: [{ offset: 0, color: "#ff3366" }, { offset: 1, color: "#6633ff" }] }],
        strokes: [{ color: "#111111", width: 1 }],
      },
      z_index: 1.5,
    },
    {
      id: "photo",
      type: "image",
      name: "Photo",
      box: { x: 360, y: 80, w: 180, h: 220 },
      src: "assets/photo.png",
      mask: { kind: "ellipse" },
      z_index: 2,
    },
    {
      id: "badge",
      type: "group",
      name: "Badge",
      box: { x: 50, y: 420, w: 160, h: 60 },
      component: {},
      children: [
        { id: "pill", type: "shape", name: "Pill", box: { x: 0, y: 0, w: 160, h: 60 }, fill: "#111111", radius: 30 },
        { id: "label", type: "text", name: "Label", box: { x: 20, y: 14, w: 120, h: 28 }, text: "BUY NOW", style: { fontFamily: "Inter", fontSize: 18, color: "#ffffff", align: "center" } },
      ],
    },
    {
      id: "offer-master",
      type: "group",
      name: "Offer master",
      box: { x: 260, y: 410, w: 130, h: 54 },
      fill: { kind: "flat", color: "#ffdd55" },
      radius: 10,
      layout: { mode: "horizontal", padding: 12, align: "center", counterAlign: "center" },
      component: { key: "offer", role: "master" },
      z_index: 3,
      children: [
        { id: "offer-master-label", type: "text", name: "Offer", box: { x: 12, y: 12, w: 106, h: 30 }, text: "SAVE 20%", style: { fontFamily: "Inter", fontSize: 18, color: "#111111" } },
      ],
    },
    {
      id: "offer-instance",
      type: "group",
      name: "Offer instance",
      box: { x: 410, y: 410, w: 130, h: 54 },
      component: { key: "offer", role: "instance" },
      z_index: 4,
      children: [
        { id: "offer-instance-label", type: "text", name: "Offer", box: { x: 12, y: 12, w: 106, h: 30 }, text: "SAVE 20%", style: { fontFamily: "Inter", fontSize: 18, color: "#111111" } },
      ],
    },
    {
      id: "styled-gradient-card",
      type: "shape",
      name: "Styled gradient card",
      box: { x: 40, y: 510, w: 220, h: 64 },
      style: {
        fills: [{ kind: "linear-gradient", angle: 90, stops: [{ color: "#ff2200", offset: 0 }, { color: "#0044ff", offset: 100 }] }],
        strokes: [{ color: "#ffffff", width: 2, align: "inside", dash: [4, 2] }],
        effects: [{ type: "drop-shadow", color: "#00000066", x: 0, y: 3, blur: 8, spread: 1 }],
      },
      radius: { topLeft: 12, topRight: 8, bottomRight: 4, bottomLeft: 0 },
      z_index: 5,
    },
    {
      id: "masked-photo",
      type: "image",
      name: "Masked photo",
      box: { x: 300, y: 490, w: 200, h: 90 },
      src: "assets/photo.png",
      mask: { kind: "rounded-rect", box: { x: 324, y: 500, w: 140, h: 70 }, radius: 14 },
      z_index: 6,
    },
    {
      id: "generated-vector",
      type: "vector",
      name: "Generated vector",
      box: { x: 520, y: 490, w: 42, h: 42 },
      vector_paths: [{ d: "M2 2 L38 38", fill: "none", stroke: { color: "#141414", width: 3, cap: "round", join: "bevel", dash: [5, 2] } }],
      z_index: 7,
    },
    {
      id: "diagonal-gradient-banner",
      type: "shape",
      name: "Diagonal gradient banner",
      // Wide, non-square box: an aspect-naive rotation matrix would visibly skew this
      // 45-degree gradient toward the short axis instead of matching the true 45-degree
      // diagonal render_preview.py produces in pixel space.
      box: { x: 0, y: 600, w: 400, h: 100 },
      fill: { kind: "linear", angle: 45, stops: [{ offset: 0, color: "#000000" }, { offset: 1, color: "#ffffff" }] },
      z_index: 8,
    },
    {
      id: "radial-gradient-banner",
      type: "shape",
      name: "Radial gradient banner",
      box: { x: 400, y: 600, w: 400, h: 100 },
      fill: { kind: "radial", stops: [{ offset: 0, color: "#000000" }, { offset: 1, color: "#ffffff" }] },
      z_index: 9,
    },
    {
      // A path/SVG-shaped mask is the one masking flavor neither "photo" (bare ellipse,
      // no explicit mask box, so it takes the single-native-ellipse-node shortcut) nor
      // "masked-photo" (rounded-rect with an explicit box, also a single-node shortcut)
      // actually exercises: this forces createImageLayer's figma.group([maskNode, node], ...)
      // path, which is the fragile part of "mask is the correct sibling above the content".
      id: "path-masked-photo",
      type: "image",
      name: "Path masked photo",
      box: { x: 40, y: 700, w: 200, h: 120 },
      src: "assets/photo.png",
      mask: { kind: "path", svg: '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path d="M2 2 L22 2 L12 22 Z" fill="#fff"/></svg>' },
      z_index: 10,
    },
    {
      // No layout mode here, and the children's x-order is the reverse of their
      // z_index/array order — this is the one case that can distinguish "z-order
      // follows the design JSON's declared stack" from "z-order follows incidental
      // x/y geometry" (the two are indistinguishable in every other fixture above).
      id: "stack-frame",
      type: "frame",
      name: "Stack frame",
      box: { x: 460, y: 700, w: 120, h: 120 },
      z_index: 11,
      children: [
        { id: "stack-back", type: "shape", name: "Back", box: { x: 60, y: 60, w: 60, h: 60 }, fill: "#222222", z_index: 0 },
        { id: "stack-front", type: "shape", name: "Front", box: { x: 0, y: 0, w: 60, h: 60 }, fill: "#eeeeee", z_index: 1 },
      ],
    },
  ],
};

const assets = { "assets/photo.png": [1, 2, 3] };

async function build(design, importMode, suppliedAssets = assets) {
  const before = posted.length;
  await figma.ui.onmessage({
    type: "build",
    design,
    assets: suppliedAssets,
    import_mode: importMode,
  });
  return posted.slice(before).find((message) => message.type === "build-result").report;
}

function roots(docId) {
  return page.findAll((node) => node.getPluginData && node.getPluginData("adDecompilerRoot") === "true" && node.getPluginData("adDecompilerDocId") === docId);
}

function nodeForLayer(root, sourceId) {
  let match = null;
  function walk(node) {
    if (node.getPluginData && node.getPluginData("adDecompilerLayerId") === sourceId) match = node;
    node.children.forEach(walk);
  }
  walk(root);
  return match;
}

const first = await build(sceneV2, "replace");
assert.equal(first.ok, true);
assert.equal(first.render.width, 600);
assert.equal(first.render.height, 600);
assert.equal(first.render.png_bytes, 3);
assert.equal(first.render.emitted, true);
assert.equal(roots(sceneV2.id).length, 1);
assert.ok(first.created >= 8);
assert.ok(page.findAll((node) => node.type === "COMPONENT").length >= 1);
assert.ok(page.findAll((node) => node.type === "INSTANCE").length >= 1);
const firstRoot = roots(sceneV2.id)[0];
assert.equal(nodeForLayer(firstRoot, "copy-stack").x, 40, "v2 top-level boxes stay root-local");
assert.equal(nodeForLayer(firstRoot, "headline").x, 16, "v2 nested boxes stay parent-relative");
assert.equal(nodeForLayer(firstRoot, "headline").leadingTrim, "CAP_HEIGHT");
assert.deepEqual(nodeForLayer(firstRoot, "headline").fontName, { family: "Inter", style: "Bold" });
assert.equal(first.fonts.selections.find((selection) => selection.label === "Headline").rank, 2);
assert.equal(nodeForLayer(firstRoot, "gradient-copy").fills[0].type, "GRADIENT_LINEAR");
assert.equal(nodeForLayer(firstRoot, "gradient-copy").strokes[0].color.r > 0, true);
assert.equal(nodeForLayer(firstRoot, "badge").type, "GROUP", "empty schema component object does not promote normal groups to frames");
const styledCard = nodeForLayer(firstRoot, "styled-gradient-card");
assert.equal(styledCard.fills[0].type, "GRADIENT_LINEAR");
assert.equal(styledCard.fills[0].gradientStops[1].position, 1, "percentage gradient stop is normalized");
assert.equal(styledCard.strokes[0].type, "SOLID");
assert.equal(styledCard.strokeAlign, "INSIDE");
assert.deepEqual(styledCard.dashPattern, [4, 2]);
assert.equal(styledCard.effects[0].type, "DROP_SHADOW");
assert.equal(styledCard.topLeftRadius, 12);
const masked = nodeForLayer(firstRoot, "masked-photo");
assert.equal(masked.type, "RECTANGLE", "rounded image mask compiles to one native image-filled rectangle");
assert.equal(masked.width, 140, "mask geometry does not inherit full image width");
assert.equal(masked.height, 70, "mask geometry does not inherit full image height");
assert.equal(masked.cornerRadius, 14);
const generated = nodeForLayer(firstRoot, "generated-vector");
assert.match(generated.svg, /stroke-linecap="round"/);
assert.match(generated.svg, /stroke-linejoin="bevel"/);
assert.match(generated.svg, /stroke-dasharray="5 2"/);
assert.equal(first.fidelity.unsupported_paint, 0);

// Mask: the mask node must be the sibling directly below the content in z-order
// (index 0, i.e. "first child" in Figma's own isMask contract) so it masks the
// content sibling that follows it, not the other way around.
const pathMasked = nodeForLayer(firstRoot, "path-masked-photo");
assert.equal(pathMasked.type, "GROUP", "a path/SVG mask compiles to a mask+content group");
assert.equal(pathMasked.children.length, 2, "the mask group holds exactly the mask and the masked content");
assert.equal(pathMasked.children[0].isMask, true, "the mask must be the first (bottom) sibling so it masks the content above it");
assert.equal(pathMasked.children[1].isMask, false, "the content is the masked (subsequent) sibling, not itself a mask");
assert.equal(pathMasked.children[1].fills[0].type, "IMAGE", "the masked sibling still carries the photo's image fill");

// Z-order: children must be appended back-to-front in the design JSON's declared
// stack (z_index / array order), not incidentally sorted by x/y position. This frame
// has no auto-layout, and "stack-front" (z_index 1) sits geometrically to the left of
// "stack-back" (z_index 0) — a position-based sort would get this backwards.
const stackFrame = nodeForLayer(firstRoot, "stack-frame");
assert.deepEqual(
  stackFrame.children.map((child) => child.getPluginData("adDecompilerLayerId")),
  ["stack-back", "stack-front"],
  "z-order follows the design JSON's declared layer stack, not incidental x/y geometry"
);

function approxEqual(actual, expected, label) {
  assert.ok(Math.abs(actual - expected) < 1e-6, `${label}: expected ${expected}, got ${actual}`);
}

// A 45-degree gradient on a wide, non-square layer must keep the same pixel-space
// direction the QA renderer (render_preview.py) draws it in — not a naive rotation of
// the normalized 0-1 paint square, which would skew toward the shorter axis.
const diagonalBanner = nodeForLayer(firstRoot, "diagonal-gradient-banner");
const diagonalTransform = diagonalBanner.fills[0].gradientTransform;
approxEqual(diagonalTransform[0][0], 0.8, "diagonal gradient row0[0]");
approxEqual(diagonalTransform[0][1], 0.2, "diagonal gradient row0[1]");
approxEqual(diagonalTransform[0][2], 0, "diagonal gradient row0[2]");
approxEqual(diagonalTransform[1][0], -0.8, "diagonal gradient row1[0]");
approxEqual(diagonalTransform[1][1], 0.2, "diagonal gradient row1[1]");
approxEqual(diagonalTransform[1][2], 0.8, "diagonal gradient row1[2]");

// A radial gradient on a non-square layer must reach its last stop at the box corners
// (a true circle in pixel space, matching render_preview.py) rather than rendering as an
// ellipse from an uncorrected identity transform.
const radialBanner = nodeForLayer(firstRoot, "radial-gradient-banner");
assert.equal(radialBanner.fills[0].type, "GRADIENT_RADIAL");
const radialTransform = radialBanner.fills[0].gradientTransform;
approxEqual(radialTransform[0][0], 0.970142500145332, "radial gradient sx");
approxEqual(radialTransform[0][1], 0, "radial gradient row0[1]");
approxEqual(radialTransform[0][2], 0.014928749927334, "radial gradient row0[2]");
approxEqual(radialTransform[1][0], 0, "radial gradient row1[0]");
approxEqual(radialTransform[1][1], 0.242535625036333, "radial gradient sy");
approxEqual(radialTransform[1][2], 0.378732187481833, "radial gradient row1[2]");

const replacement = await build(sceneV2, "replace");
assert.equal(replacement.ok, true);
assert.equal(replacement.replaced, true);
assert.equal(roots(sceneV2.id).length, 1);

const copy = await build(sceneV2, "copy");
assert.equal(copy.ok, true);
assert.equal(copy.mode, "copy");
assert.equal(roots(sceneV2.id).length, 2);

// Regression: figma.currentPage.findAll() returns nodes in layer/traversal order,
// which a user changes just by dragging a frame in the layers panel — it is not
// creation order. "Replace existing" must target the import that most recently
// happened (by the adDecompilerImportedAt timestamp), not whichever root happens
// to sort last in that array, or it can silently replace the wrong frame.
{
  const [originalRoot, copyRoot] = roots(sceneV2.id);
  assert.ok(originalRoot && copyRoot && originalRoot !== copyRoot);
  originalRoot.setPluginData("adDecompilerImportedAt", "1000");
  copyRoot.setPluginData("adDecompilerImportedAt", "2000");
  // Simulate dragging the older import above the newer one in the layers panel.
  const idx0 = page.children.indexOf(originalRoot);
  const idx1 = page.children.indexOf(copyRoot);
  page.children[idx0] = copyRoot;
  page.children[idx1] = originalRoot;
  assert.deepEqual(roots(sceneV2.id), [copyRoot, originalRoot], "copy import now sorts first in layer order");
  // Clear the current selection so the plugin must fall back to ranking existingRoots
  // itself instead of trusting an explicit user selection — this isolates the
  // array-order bug in the fallback path.
  page.selection = [];

  const reordered = await build(sceneV2, "replace");
  assert.equal(reordered.ok, true);
  assert.equal(reordered.replaced, true);
  assert.equal(roots(sceneV2.id).length, 2, "replace still does not stack duplicates after reordering");
  assert.equal(copyRoot.parent, null, "Replace must remove the most recently imported root, not whichever sorts last in layer order");
  assert.equal(originalRoot.parent, page, "the older import must be left untouched when a newer one exists");
}

// Regression: once the previous import has been removed (report.replaced = true),
// a later, unrelated failure must not also delete the freshly built replacement —
// that would leave the user with neither the old nor the new frame.
{
  const before = roots(sceneV2.id);
  assert.equal(before.length, 2, "two roots exist before the crash-safety replace test");
  const originalPostMessage = figma.ui.postMessage;
  let throwOnce = true;
  figma.ui.postMessage = function (message) {
    if (throwOnce && message.type === "build-result") {
      throwOnce = false;
      throw new Error("simulated failure after the swap");
    }
    return originalPostMessage.call(figma.ui, message);
  };
  let crashed;
  try {
    crashed = await build(sceneV2, "replace");
  } finally {
    figma.ui.postMessage = originalPostMessage;
  }
  assert.equal(crashed.ok, false, "the reported outcome reflects the late failure");
  assert.equal(crashed.replaced, true, "the swap itself completed before the simulated failure");
  assert.equal(roots(sceneV2.id).length, 2, "a late failure after the swap must not also delete the freshly built replacement");
}

const failed = await build(sceneV2, "replace", {});
assert.equal(failed.ok, false);
assert.ok(failed.errors.some((error) => error.detail.includes("Missing image asset")));
assert.equal(roots(sceneV2.id).length, 2, "failed replace keeps existing imports intact");

const legacy = {
  id: "legacy-smoke",
  name: "Legacy smoke",
  canvas: { w: 320, h: 240 },
  layers: [
    { id: "shape", type: "shape", name: "Card", box: { x: 20.25, y: 18.5, w: 280.5, h: 204.25 }, shape_kind: "rect", fill: { kind: "flat", color: "#ffffff" } },
    { id: "text", type: "text", name: "Title", box: { x: 40.5, y: 40.25, w: 240, h: 48 }, text: "Legacy stays editable", style: { fontSize: 24, lineHeight: 29, letterSpacing: -0.3 } },
  ],
};

const legacyResult = await build(legacy, "replace", {});
assert.equal(legacyResult.ok, true);
assert.equal(roots(legacy.id).length, 1);
assert.equal(nodeForLayer(roots(legacy.id)[0], "shape").x, 20.25, "legacy float positions are preserved");

const html = fs.readFileSync("figma-plugin/ui.html", "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/);
assert.ok(script, "plugin UI contains a script");
new Function(script[1]);
const manifest = JSON.parse(fs.readFileSync("figma-plugin/manifest.json", "utf8"));
assert.deepEqual(manifest.networkAccess.allowedDomains, ["none"]);
// Figma's manifest validator rejects IP-literal devAllowedDomains entries (e.g.
// "http://127.0.0.1:8790" errors with "must be a valid URL"), so only localhost is
// allow-listed; the UI script must normalize any 127.0.0.1 address a user types instead.
assert.deepEqual(manifest.networkAccess.devAllowedDomains, ["http://localhost:8790"]);
assert.ok(fs.existsSync("figma-plugin/icon.svg"), "plugin icon asset exists");

{
  const cleanBaseMatch = script[1].match(/function cleanBase\(\)[\s\S]*?\n {4}\}/);
  assert.ok(cleanBaseMatch, "ui.html defines cleanBase()");
  const sandbox = { value: "http://127.0.0.1:8790/" };
  const fn = new Function("$", "state", `${cleanBaseMatch[0]}\nreturn cleanBase();`);
  const normalized = fn(() => sandbox, { settings: {} });
  assert.equal(normalized, "http://localhost:8790", "cleanBase() rewrites 127.0.0.1 to the allow-listed localhost host");
}

console.log("plugin smoke passed", {
  v2Created: first.created,
  replaceRoots: roots(sceneV2.id).length,
  legacyCreated: legacyResult.created,
});
