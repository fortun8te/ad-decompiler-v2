// code.js — ad-decompiler Figma import plugin (main thread).
// Receives a design.json (schema.DesignDoc) + resolved asset data-URLs from the UI, builds
// real editable nodes on a new frame, then exports a PNG of that frame back through the UI.
//
// Mapping: text -> TextNode | shape(rect/ellipse) -> Rectangle/Ellipse + fill |
//   shape(path) / icon -> VectorNode from d-string | image -> Rectangle w/ image fill,
//   mask via ellipse/rrect/alpha. Names come straight from layer.name (semantic).

figma.showUI(__html__, { width: 340, height: 260 });

const clamp = (n) => Math.max(0, Math.round(n || 0));

function hexToRGB(hex) {
  const h = String(hex || "#000").replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map(c => c + c).join("") : h.slice(0, 6), 16);
  return { r: ((n >> 16) & 255) / 255, g: ((n >> 8) & 255) / 255, b: (n & 255) / 255 };
}

async function makeText(L) {
  const t = figma.createText();
  const st = L.style || {};
  const family = st.fontFamily || "Inter";
  const style = (st.fontWeight >= 700) ? "Bold" : (st.fontWeight >= 600 ? "Semi Bold" : "Regular");
  try { await figma.loadFontAsync({ family, style }); t.fontName = { family, style }; }
  catch (e) { await figma.loadFontAsync({ family: "Inter", style: "Regular" }); t.fontName = { family: "Inter", style: "Regular" }; }
  t.characters = L.text || "";
  if (st.fontSize) t.fontSize = st.fontSize;
  t.x = clamp(L.box.x); t.y = clamp(L.box.y);
  t.resize(clamp(L.box.w) || 10, clamp(L.box.h) || t.height);
  t.textAutoResize = "HEIGHT";
  if (st.align) t.textAlignHorizontal = st.align.toUpperCase();
  if (st.color) t.fills = [{ type: "SOLID", color: hexToRGB(st.color) }];
  if (st.uppercase) t.textCase = "UPPER";
  if (st.letterSpacing) t.letterSpacing = { unit: "PIXELS", value: st.letterSpacing };
  return t;
}

function applyFill(node, fill) {
  if (!fill) return;
  if (fill.kind === "flat" || fill.color) node.fills = [{ type: "SOLID", color: hexToRGB(fill.color) }];
  else if (fill.kind === "linear" || fill.kind === "radial") {
    const stops = (fill.stops || []).map((s, i, a) => ({ position: s.offset != null ? s.offset : i / Math.max(1, a.length - 1), color: { ...hexToRGB(s.color), a: 1 } }));
    node.fills = [{ type: fill.kind === "radial" ? "GRADIENT_RADIAL" : "GRADIENT_LINEAR",
      gradientTransform: [[1, 0, 0], [0, 1, 0]], gradientStops: stops.length ? stops : [{ position: 0, color: { r:0,g:0,b:0,a:1 } }] }];
  }
}

async function makeShape(L) {
  if (L.shape_kind === "path" && L.path) {
    const v = figma.createVector();
    v.x = clamp(L.box.x); v.y = clamp(L.box.y); v.resize(clamp(L.box.w) || 1, clamp(L.box.h) || 1);
    try { v.vectorPaths = [{ windingRule: "NONZERO", data: L.path }]; } catch (e) {}
    applyFill(v, L.fill || { kind: "flat", color: "#000000" });
    v.strokeWeight = 0;
    return v;
  }
  const node = L.shape_kind === "ellipse" ? figma.createEllipse() : figma.createRectangle();
  node.x = clamp(L.box.x); node.y = clamp(L.box.y);
  node.resize(clamp(L.box.w) || 1, clamp(L.box.h) || 1);
  applyFill(node, L.fill || { kind: "flat", color: "#cccccc" });
  if (L.style && L.style.radius) node.cornerRadius = L.style.radius;
  return node;
}

async function makeImage(L, assets) {
  const node = figma.createRectangle();
  node.x = clamp(L.box.x); node.y = clamp(L.box.y);
  node.resize(clamp(L.box.w) || 1, clamp(L.box.h) || 1);
  const bytes = assets[L.src];
  if (bytes) {
    const img = figma.createImage(bytes);
    node.fills = [{ type: "IMAGE", scaleMode: "FILL", imageHash: img.hash }];
  } else {
    node.fills = [{ type: "SOLID", color: { r: 0.9, g: 0.9, b: 0.9 } }];
  }
  if (L.mask && L.mask.kind === "ellipse") node.cornerRadius = Math.min(L.box.w, L.box.h) / 2;
  else if (L.mask && L.mask.kind === "rrect") node.cornerRadius = L.mask.radius || 16;
  return node;
}

figma.ui.onmessage = async (msg) => {
  if (msg.type === "build") {
    const doc = msg.design;
    const assets = {};
    for (const k in (msg.assets || {})) assets[k] = new Uint8Array(msg.assets[k]);
    const frame = figma.createFrame();
    frame.name = doc.name || "ad";
    frame.resize(doc.canvas.w, doc.canvas.h);
    frame.x = 0; frame.y = 0; frame.clipsContent = true;
    for (const L of doc.layers) {
      let node = null;
      try {
        if (L.type === "text") node = await makeText(L);
        else if (L.type === "shape") node = await makeShape(L);
        else if (L.type === "image") node = await makeImage(L, assets);
      } catch (e) { console.log("layer failed", L.id, e); }
      if (node) { node.name = L.name || L.id; frame.appendChild(node); }
    }
    figma.currentPage.selection = [frame];
    figma.viewport.scrollAndZoomIntoView([frame]);
    const png = await frame.exportAsync({ format: "PNG" });
    figma.ui.postMessage({ type: "exported", bytes: Array.from(png), export_to: msg.export_to });
    figma.notify(`Imported ${doc.layers.length} layers`);
  }
};
