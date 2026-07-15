"use strict";
// test_paint_effects.js — focused render assertions for the shipped richer-reconstruction
// features that flow into figma-plugin/code.js:
//
//   Feature 1  hand-drawn annotation -> editable VECTOR stroke (createNodeFromSvg)
//   Feature 3  multi-stop linear + radial gradient fills (GRADIENT_LINEAR / GRADIENT_RADIAL)
//
// (Feature 2, glassmorphism / BACKGROUND_BLUR, was dropped from scope.)
//
// run_e2e.js validates that every layer BUILDS (the mock throws on a malformed paint/effect,
// which safeSet swallows into a warning), but it does not assert the specific native paint /
// effect landed on the node. This test does, so a regression in applyFills / applyEffects /
// createVectorLayer is caught. Standalone: does not change the run_e2e 16/16 fixture count.
//
// Usage: node figma-plugin/test/test_paint_effects.js

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const { createFigmaMock } = require("./figma-mock.js");

const CODE_JS = path.join(path.dirname(__dirname), "code.js");
const ROOT_KEY = "adDecompilerRoot";
const LAYER_KEY = "adDecompilerLayerId";

const failures = [];
function check(cond, message) {
  if (!cond) failures.push(message);
}

const design = {
  id: "paint-effects",
  name: "paint-effects",
  canvas: { w: 400, h: 400 },
  layers: [
    {
      id: "bg",
      type: "image",
      name: "Background — clean plate",
      box: { x: 0, y: 0, w: 400, h: 400 },
      z_index: -1000000,
      // no src: harness leaves it out; the mock records it as a degradation, not a crash.
    },
    {
      // Feature 3a — multi-stop (3-stop) linear gradient
      id: "grad-multistop",
      type: "shape",
      shape_kind: "rect",
      name: "grad-multistop",
      box: { x: 20, y: 20, w: 160, h: 100 },
      z_index: 10,
      fill: {
        kind: "linear",
        angle: 0,
        stops: [
          { position: 0, color: "#ff0000" },
          { position: 0.5, color: "#ffffff" },
          { position: 1, color: "#0000ff" },
        ],
      },
    },
    {
      // Feature 3b — radial gradient
      id: "grad-radial",
      type: "shape",
      shape_kind: "ellipse",
      name: "grad-radial",
      box: { x: 220, y: 20, w: 160, h: 100 },
      z_index: 10,
      fill: {
        kind: "radial",
        stops: [
          { position: 0, color: "#ffffff" },
          { position: 1, color: "#101030" },
        ],
      },
    },
    {
      // Feature 1 — hand-drawn annotation as an editable vector stroke
      id: "annotation-arrow",
      type: "shape",
      shape_kind: "path",
      name: "annotation-arrow",
      box: { x: 40, y: 320, w: 200, h: 40 },
      z_index: 30,
      svg:
        '<svg xmlns="http://www.w3.org/2000/svg" width="200" height="40" viewBox="0 0 200 40">' +
        '<path d="M0 20 L200 20" fill="none" stroke="#e01b1b" stroke-width="5" stroke-linecap="round"/></svg>',
      src: null,
    },
  ],
};

function walk(node, cb) {
  cb(node);
  (node.children || []).forEach((c) => walk(c, cb));
}

(async function main() {
  if (!fs.existsSync(CODE_JS)) {
    console.error("code.js not found at " + CODE_JS);
    process.exit(2);
  }
  const source = fs.readFileSync(CODE_JS, "utf8");
  const factory = vm.compileFunction(source, ["figma", "__html__"], { filename: CODE_JS });

  const mock = createFigmaMock({ installedFonts: ["Inter"] });
  const { figma, state } = mock;

  const unhandled = [];
  const onUnhandled = (r) => unhandled.push(String((r && r.stack) || r));
  process.on("unhandledRejection", onUnhandled);

  factory(figma, "<mock-ui/>");
  if (typeof figma.ui.onmessage !== "function") {
    console.error("code.js did not register figma.ui.onmessage");
    process.exit(1);
  }

  let buildError = null;
  try {
    await figma.ui.onmessage({ type: "ui-ready" });
    await figma.ui.onmessage({
      type: "build",
      design,
      assets: {},
      export_to: null,
      manifest_doc_id: "paint-effects",
      roundtrip_token: "tok",
      import_mode: "replace",
    });
  } catch (error) {
    buildError = error;
  }
  await new Promise((r) => setImmediate(r));
  process.removeListener("unhandledRejection", onUnhandled);

  check(!buildError, "import threw out of onmessage: " + (buildError && (buildError.stack || buildError)));
  check(!unhandled.length, "unhandled rejection(s): " + unhandled.join("; "));

  const roots = mock.currentPage.findAll((n) => {
    try { return n.getPluginData(ROOT_KEY) === "true"; } catch (_) { return false; }
  });
  check(roots.length === 1, "expected exactly 1 import root, got " + roots.length);
  const root = roots[roots.length - 1];

  const byId = new Map();
  if (root) walk(root, (n) => {
    const tag = n.getPluginData ? n.getPluginData(LAYER_KEY) : "";
    if (tag) (byId.get(tag) || byId.set(tag, []).get(tag)).push(n);
  });
  const one = (id) => (byId.get(id) || [])[0];

  // Feature 3a — multi-stop linear gradient
  const multistop = one("grad-multistop");
  check(!!multistop, "grad-multistop node was not created");
  if (multistop) {
    const paint = (multistop.fills || [])[0];
    check(paint && paint.type === "GRADIENT_LINEAR", "grad-multistop fill is not GRADIENT_LINEAR: " + (paint && paint.type));
    check(paint && Array.isArray(paint.gradientStops) && paint.gradientStops.length === 3,
      "grad-multistop should have 3 gradientStops, got " + (paint && paint.gradientStops && paint.gradientStops.length));
    // stops arrive sorted by position; the middle white stop must survive.
    if (paint && paint.gradientStops && paint.gradientStops.length === 3) {
      const mid = paint.gradientStops[1];
      check(Math.abs(mid.position - 0.5) < 0.01, "middle stop position drifted: " + mid.position);
      check(mid.color.r > 0.9 && mid.color.g > 0.9 && mid.color.b > 0.9, "middle stop should be white");
    }
  }

  // Feature 3b — radial gradient
  const radial = one("grad-radial");
  check(!!radial, "grad-radial node was not created");
  if (radial) {
    const paint = (radial.fills || [])[0];
    check(paint && paint.type === "GRADIENT_RADIAL", "grad-radial fill is not GRADIENT_RADIAL: " + (paint && paint.type));
    check(paint && Array.isArray(paint.gradientStops) && paint.gradientStops.length >= 2,
      "grad-radial needs >= 2 gradientStops");
  }

  // Feature 1 — annotation as an editable vector stroke via createNodeFromSvg
  const arrow = one("annotation-arrow");
  check(!!arrow, "annotation-arrow node was not created");
  if (arrow) {
    check(arrow.type === "FRAME", "annotation-arrow should import as an SVG FRAME, got " + arrow.type);
    const paths = (arrow.children || []).filter((c) => c.type === "VECTOR");
    check(paths.length >= 1, "annotation-arrow should contain >= 1 imported VECTOR path");
    const svgImport = state.svgImports.find((s) => s.pathCount >= 1);
    check(!!svgImport, "createNodeFromSvg was never used for a stroked path");
  }

  if (failures.length) {
    console.log("\n FAIL — paint/effects render assertions");
    failures.forEach((f) => console.log("   ✗ " + f));
    console.log("\n " + failures.length + " assertion(s) failed\n");
    process.exit(1);
  }
  console.log("\n PASS — annotation vector stroke + multi-stop and radial gradients all rendered\n");
  process.exit(0);
})();
