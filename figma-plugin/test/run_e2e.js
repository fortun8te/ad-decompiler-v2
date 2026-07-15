"use strict";
// run_e2e.js — automated mock-Figma end-to-end harness for figma-plugin/code.js.
//
// Loads the real compiler (code.js) with a faithful `figma` mock, feeds it the same
// {type:"build"} message ui.html sends, lets the whole import run, then asserts the
// resulting node tree against the design.json contract:
//   * import finishes, no unhandled rejections, exactly one build-result
//   * every design layer -> exactly one tagged node OR a recorded degradation
//   * hierarchy matches design nesting
//   * geometry within 0.5px (flow/auto-layout and text handled by containment rules)
//   * text nodes carry characters + a resolved & loaded font
//   * auto-layout intent -> layoutMode/spacing/padding applied
//   * report / progress / warning events are well-formed
//
// Usage:
//   node figma-plugin/test/run_e2e.js --all
//   node figma-plugin/test/run_e2e.js --fixture kitchen-sink --verbose
//   node figma-plugin/test/run_e2e.js --all --json out.json
//
// Exit code 0 = all fixtures pass; 1 = at least one failure (CI gate).

const fs = require("fs");
const path = require("path");
const vm = require("vm");

const { createFigmaMock, encodePng, fontKey } = require("./figma-mock.js");
const R = require("./replica.js");

const TEST_DIR = __dirname;
const PLUGIN_DIR = path.dirname(TEST_DIR);
const PROJECT_DIR = path.dirname(PLUGIN_DIR);
const CODE_JS = path.join(PLUGIN_DIR, "code.js");
const FIXTURES_DIR = path.join(TEST_DIR, "fixtures");
const GOLDEN_DIR = path.join(PROJECT_DIR, "runs", "golden-optimized-check");

const GEO_TOLERANCE = 0.5;      // px, hard assert for non-text geometry
const TEXT_SLACK_PX = 3;        // text containment slack (synthetic font metrics)
const TEXT_SLACK_FRAC = 0.06;   // ...or 6% of the target box, whichever is larger

const LAYER_KEY = "adDecompilerLayerId";
const ROOT_KEY = "adDecompilerRoot";

// ----------------------------------------------------------------------------
// CLI
// ----------------------------------------------------------------------------

const argv = process.argv.slice(2);
function hasFlag(name) { return argv.indexOf("--" + name) >= 0; }
function flagValue(name) {
  const i = argv.indexOf("--" + name);
  return i >= 0 && argv[i + 1] && argv[i + 1].indexOf("--") !== 0 ? argv[i + 1] : null;
}
const VERBOSE = hasFlag("verbose");
const ONLY = flagValue("fixture");
const JSON_OUT = flagValue("json");
const RUNS_DIR = flagValue("runs-dir") || GOLDEN_DIR;

// ----------------------------------------------------------------------------
// fixture discovery
// ----------------------------------------------------------------------------

function loadDesignFromDir(dir) {
  const js = path.join(dir, "design.js");
  if (fs.existsSync(js)) {
    delete require.cache[require.resolve(js)];
    return require(js)();
  }
  return JSON.parse(fs.readFileSync(path.join(dir, "design.json"), "utf8"));
}

function discoverFixtures() {
  const fixtures = [];

  if (fs.existsSync(RUNS_DIR)) {
    fs.readdirSync(RUNS_DIR).sort().forEach((entry) => {
      const dir = path.join(RUNS_DIR, entry);
      if (fs.existsSync(path.join(dir, "design.json"))) {
        fixtures.push({
          name: "golden/" + entry,
          dir,
          assetsBase: dir,
          harness: { installedFonts: "auto", expect: { ok: true } },
          golden: true,
        });
      }
    });
  }

  if (fs.existsSync(FIXTURES_DIR)) {
    fs.readdirSync(FIXTURES_DIR).sort().forEach((entry) => {
      const dir = path.join(FIXTURES_DIR, entry);
      if (!fs.statSync(dir).isDirectory()) return;
      if (!fs.existsSync(path.join(dir, "design.json")) && !fs.existsSync(path.join(dir, "design.js"))) return;
      let harness = { installedFonts: "auto", expect: { ok: true } };
      const harnessPath = path.join(dir, "harness.json");
      if (fs.existsSync(harnessPath)) {
        harness = Object.assign(harness, JSON.parse(fs.readFileSync(harnessPath, "utf8")));
      }
      fixtures.push({ name: "edge/" + entry, dir, assetsBase: dir, harness, golden: false });
    });
  }

  return fixtures;
}

// ----------------------------------------------------------------------------
// asset staging (mirrors ui.html gatherAssets: number[] over the wire)
// ----------------------------------------------------------------------------

function gatherAssets(fixture, design) {
  const refs = R.assetReferences(design);
  const generated = (fixture.harness && fixture.harness.generateAssets) || {};
  const assets = {};
  const notes = [];
  refs.forEach((required, source) => {
    if (Object.prototype.hasOwnProperty.call(generated, source)) {
      const spec = generated[source];
      assets[source] = Array.from(encodePng(spec.w || 8, spec.h || 8, spec.rgba));
      return;
    }
    const rel = source.replace(/\\/g, path.sep).replace(/\//g, path.sep);
    const file = path.join(fixture.assetsBase, rel);
    if (fs.existsSync(file)) {
      assets[source] = Array.from(new Uint8Array(fs.readFileSync(file)));
    } else {
      notes.push((required ? "required" : "optional") + " asset not staged: " + source);
    }
  });
  return { assets, notes };
}

// ----------------------------------------------------------------------------
// compiler loading
// ----------------------------------------------------------------------------

let compiledFactory = null;
function loadCompiler() {
  if (!compiledFactory) {
    const source = fs.readFileSync(CODE_JS, "utf8");
    compiledFactory = vm.compileFunction(source, ["figma", "__html__"], { filename: CODE_JS });
  }
  return compiledFactory;
}

// ----------------------------------------------------------------------------
// node-tree helpers
// ----------------------------------------------------------------------------

function findImportRoots(page) {
  return page.findAll((n) => {
    try { return n.getPluginData(ROOT_KEY) === "true"; } catch (_) { return false; }
  });
}

function walkNodes(node, cb, depth) {
  cb(node, depth || 0);
  (node.children || []).forEach((c) => walkNodes(c, cb, (depth || 0) + 1));
}

function absBox(node, root) {
  let x = 0;
  let y = 0;
  let cur = node;
  while (cur && cur !== root) {
    x += cur.x;
    y += cur.y;
    cur = cur.parent;
  }
  if (cur !== root) return null; // not under root
  return { x, y, w: node.width, h: node.height };
}

function isDescendantOf(node, ancestor) {
  let cur = node.parent;
  while (cur) {
    if (cur === ancestor) return true;
    cur = cur.parent;
  }
  return false;
}

function indexTaggedNodes(root) {
  const byId = new Map();
  walkNodes(root, (node) => {
    if (node === root) return;
    const tag = node.getPluginData ? node.getPluginData(LAYER_KEY) : "";
    if (!tag) return;
    if (!byId.has(tag)) byId.set(tag, []);
    byId.get(tag).push(node);
  });
  return byId;
}

function eventsOfType(state, type) {
  return state.uiMessages.map((m) => m.message).filter((m) => m && m.type === type);
}

function reportMentionsLayer(report, row) {
  const name = String(row.layer.name || "");
  const hay = [];
  (report.warnings || []).forEach((w) => hay.push(String(w.title) + " " + String(w.detail)));
  (report.errors || []).forEach((e) => hay.push(String(e.title) + " " + String(e.detail)));
  return hay.some((line) => line.indexOf(row.id) >= 0 || (name && line.indexOf(name) >= 0));
}

// ----------------------------------------------------------------------------
// per-fixture run + assertions
// ----------------------------------------------------------------------------

async function runFixture(fixture) {
  const result = {
    name: fixture.name,
    pass: true,
    failures: [],
    degradations: [],
    info: [],
    stats: {},
  };
  const fail = (msg) => { result.pass = false; result.failures.push(msg); };
  const degrade = (msg) => { result.degradations.push(msg); };

  let design;
  try {
    design = loadDesignFromDir(fixture.dir);
  } catch (error) {
    fail("fixture design could not be loaded: " + error.message);
    return result;
  }

  const expect = Object.assign({ ok: true }, fixture.harness.expect || {});
  const failedAllowed = new Set(expect.failedLayers || []);
  const skipGeometry = new Set(expect.skipGeometry || []);
  const skipTextContainment = new Set(expect.skipTextContainment || []);

  // fonts: "auto" installs every family the design requests (plus Inter)
  let installedFonts = fixture.harness.installedFonts;
  if (installedFonts === "auto" || installedFonts === undefined) {
    installedFonts = R.requestedFontFamilies(design).concat(["Inter"]);
  }

  const mock = createFigmaMock({ installedFonts });
  const { figma, state } = mock;

  const unhandled = [];
  const onUnhandled = (reason) => unhandled.push(String((reason && reason.stack) || reason));
  process.on("unhandledRejection", onUnhandled);

  try {
    loadCompiler()(figma, "<mock-ui/>");
  } catch (error) {
    process.removeListener("unhandledRejection", onUnhandled);
    fail("code.js crashed at load time: " + (error && error.stack || error));
    return result;
  }

  if (typeof figma.ui.onmessage !== "function") {
    process.removeListener("unhandledRejection", onUnhandled);
    fail("code.js did not register figma.ui.onmessage");
    return result;
  }

  const { assets, notes } = gatherAssets(fixture, design);
  notes.forEach((n) => result.info.push(n));

  let buildError = null;
  try {
    // handshake first, the way ui.html does
    await figma.ui.onmessage({ type: "ui-ready" });
    await figma.ui.onmessage({
      type: "build",
      design,
      assets,
      export_to: null,
      manifest_doc_id: "e2e-" + fixture.name,
      roundtrip_token: "e2e-token",
      import_mode: "replace",
    });
    if (expect.rebuild) {
      // second import in the same file: exercises the replace path end-to-end
      await figma.ui.onmessage({
        type: "build",
        design,
        assets,
        export_to: null,
        manifest_doc_id: "e2e-" + fixture.name,
        roundtrip_token: "e2e-token-2",
        import_mode: "replace",
      });
    }
  } catch (error) {
    buildError = error;
  }
  await new Promise((resolve) => setImmediate(resolve));
  process.removeListener("unhandledRejection", onUnhandled);

  if (buildError) fail("import threw out of figma.ui.onmessage (must never happen): " + (buildError.stack || buildError));
  if (unhandled.length) fail("unhandled promise rejection(s) during import:\n  " + unhandled.join("\n  "));

  // -------------------------------------------------- protocol-level checks
  const inits = eventsOfType(state, "init");
  if (inits.length !== 1) fail("expected exactly 1 'init' reply to ui-ready, got " + inits.length);

  const buildResults = eventsOfType(state, "build-result");
  const expectedBuilds = expect.rebuild ? 2 : 1;
  if (buildResults.length !== expectedBuilds) {
    fail("expected " + expectedBuilds + " 'build-result' message(s), got " + buildResults.length);
    return result;
  }
  const report = buildResults[buildResults.length - 1].report || {};
  result.stats.report = {
    ok: report.ok, created: report.created, skipped: report.skipped,
    warnings: (report.warnings || []).length, errors: (report.errors || []).length,
    fonts: report.fonts, assets: report.assets,
  };

  // report shape
  ["ok", "created", "skipped", "warnings", "errors"].forEach((key) => {
    if (report[key] === undefined) fail("build-result report is missing '" + key + "'");
  });
  (report.warnings || []).concat(report.errors || []).forEach((entry, i) => {
    if (!entry || typeof entry.title !== "string" || typeof entry.detail !== "string") {
      fail("report warning/error #" + i + " is malformed (need string title+detail): " + JSON.stringify(entry));
    }
  });
  (report.events || []).forEach((entry, i) => {
    if (!entry || typeof entry.at !== "string" || isNaN(Date.parse(entry.at)) ||
        ["warn", "error"].indexOf(entry.level) < 0 || typeof entry.title !== "string") {
      fail("report event #" + i + " is malformed: " + JSON.stringify(entry));
    }
  });

  // progress stream
  const progress = eventsOfType(state, "progress");
  const expectedTotal = R.countLayers(design.layers || []);
  progress.forEach((p) => {
    if (typeof p.current !== "number" || typeof p.total !== "number" || !p.phase) {
      fail("malformed progress event: " + JSON.stringify(p));
    } else if (p.current > p.total) {
      fail("progress current > total: " + JSON.stringify(p));
    }
  });
  if (report.ok && !progress.length) fail("no progress events were emitted");
  if (progress.length && progress[0].total !== expectedTotal && expect.rebuild !== true) {
    degrade("progress total " + progress[0].total + " != countLayers " + expectedTotal + " (informational)");
  }

  if (Boolean(report.ok) !== Boolean(expect.ok)) {
    fail("report.ok=" + report.ok + " but fixture expects ok=" + expect.ok +
      (report.errors && report.errors.length ? " · first error: " + JSON.stringify(report.errors[0]) : ""));
  }
  if (expect.ok && report.ok) {
    const exported = eventsOfType(state, "exported");
    if (exported.length !== expectedBuilds) fail("expected " + expectedBuilds + " 'exported' message(s), got " + exported.length);
    else if (!Array.isArray(exported[0].bytes) || !exported[0].bytes.length) fail("'exported' bytes missing/empty");
    if (!report.render || report.render.emitted !== true) fail("report.render.emitted is not true");
  }

  if (expect.errorsContain) {
    expect.errorsContain.forEach((needle) => {
      const found = (report.errors || []).some((e) => (e.title + " " + e.detail).indexOf(needle) >= 0);
      if (!found) fail("expected an error containing '" + needle + "'");
    });
  }
  if (expect.warningsContain) {
    expect.warningsContain.forEach((needle) => {
      const found = (report.warnings || []).some((w) => (w.title + " " + w.detail).indexOf(needle) >= 0);
      if (!found) fail("expected a warning containing '" + needle + "'");
    });
  }
  if (expect.assetsMissing !== undefined) {
    const missing = report.assets ? report.assets.missing : undefined;
    if (missing !== expect.assetsMissing) fail("report.assets.missing=" + missing + ", expected " + expect.assetsMissing);
  }

  // -------------------------------------------------------- tree-level checks
  const roots = findImportRoots(mock.currentPage);
  if (expect.ok) {
    if (roots.length !== 1) {
      fail("expected exactly 1 imported root frame on the page, found " + roots.length +
        (expect.rebuild ? " (replace mode must remove the previous import)" : ""));
    }
    if (expect.rebuild && !report.replaced) fail("second import did not set report.replaced=true");
  }
  const root = roots[roots.length - 1];
  if (!root) {
    if (expect.ok) fail("no imported root frame found");
    finishScore(result, report, [], design);
    return result;
  }

  // canvas
  const canvas = design.canvas || {};
  const canvasW = Math.max(1, Number(canvas.w || canvas.width || 1080));
  const canvasH = Math.max(1, Number(canvas.h || canvas.height || 1080));
  if (Math.abs(root.width - canvasW) > GEO_TOLERANCE || Math.abs(root.height - canvasH) > GEO_TOLERANCE) {
    fail("root frame is " + root.width + "x" + root.height + ", canvas says " + canvasW + "x" + canvasH);
  }

  const rows = R.buildExpectations(design);
  const tagged = indexTaggedNodes(root);

  const stats = {
    layers: rows.length,
    built: 0,
    absorbed: 0,
    failedRecorded: 0,
    silentDrops: 0,
    geometryChecked: 0,
    geometryFailed: 0,
    geometryMaxDelta: 0,
    textChecked: 0,
    textFailed: 0,
    autoLayoutChecked: 0,
    autoLayoutFailed: 0,
    sizingChecked: 0,
    sizingFailed: 0,
    hierarchyFailed: 0,
  };

  const nodeOfRow = new Map();

  rows.forEach((row) => {
    if (row.insideInstance) { result.info.push(row.path + ": inside component instance, represented by master (exempt)"); return; }
    const candidates = tagged.get(row.id) || [];

    if (row.absorbed) {
      stats.absorbed += 1;
      if (candidates.length) degrade(row.path + ": predicted hoist-absorbed but a node exists (hoist prediction drift — informational)");
      else result.info.push(row.path + ": absorbed into parent button frame (hoisted background), by design");
      return;
    }

    if (!candidates.length) {
      if (reportMentionsLayer(report, row)) {
        stats.failedRecorded += 1;
        if (failedAllowed.size && !failedAllowed.has(row.id) && !failedAllowed.has("*")) {
          fail(row.path + ": layer failed with a recorded error but fixture does not allow it to fail");
        } else {
          degrade(row.path + ": no node, degradation recorded in report (acceptable)");
        }
      } else {
        stats.silentDrops += 1;
        fail(row.path + ": SILENT DROP — no node created and no warning/error mentions this layer");
      }
      return;
    }
    if (candidates.length > 1) {
      fail(row.path + ": expected exactly one node, found " + candidates.length + " nodes tagged with this layer id");
    }
    const node = candidates[0];
    nodeOfRow.set(row, node);
    stats.built += 1;
    if (failedAllowed.has(row.id)) {
      degrade(row.path + ": fixture allowed failure but layer built fine");
    }

    // hierarchy
    if (row.parentRow && !row.parentRow.insideInstance && !row.parentRow.absorbed) {
      const parentNode = nodeOfRow.get(row.parentRow);
      if (parentNode && !isDescendantOf(node, parentNode)) {
        stats.hierarchyFailed += 1;
        fail(row.path + ": HIERARCHY — node is not inside its design parent's node (" + row.parentRow.path + ")");
      }
    }

    // geometry
    if (!skipGeometry.has(row.id)) {
      const box = absBox(node, root);
      if (!box) {
        fail(row.path + ": node is not attached under the imported root frame");
      } else if (row.type === "text") {
        checkText(row, node, box, report, stats, fail, degrade, skipTextContainment, mock);
      } else if (row.rotation) {
        // design.json rotation is degrees CLOCKWISE-positive; Figma's node.rotation is
        // counter-clockwise-positive, so the compiler correctly assigns the negation.
        stats.geometryChecked += 1;
        if (Math.abs((node.rotation || 0) + row.rotation) > 0.5) {
          stats.geometryFailed += 1;
          fail(row.path + ": rotation=" + node.rotation + ", design says " + row.rotation + " (expected node.rotation = -design rotation)");
        }
      } else {
        stats.geometryChecked += 1;
        const e = row.expectedBox;
        const checkY = !row.flowChild;
        const deltas = {
          x: checkY ? Math.abs(box.x - e.x) : 0,
          y: checkY ? Math.abs(box.y - e.y) : 0,
          w: Math.abs(box.w - e.w),
          h: Math.abs(box.h - e.h),
        };
        // auto-layout HUG frames legitimately re-derive their own size
        const hugging = ("layoutMode" in node && node.layoutMode !== "NONE" &&
          (node.primaryAxisSizingMode === "AUTO" || node.counterAxisSizingMode === "AUTO"));
        if (hugging) { deltas.w = 0; deltas.h = 0; }
        const max = Math.max(deltas.x, deltas.y, deltas.w, deltas.h);
        stats.geometryMaxDelta = Math.max(stats.geometryMaxDelta, max);
        if (max > GEO_TOLERANCE) {
          stats.geometryFailed += 1;
          fail(row.path + ": GEOMETRY off by " + max.toFixed(2) + "px — node " +
            fmtBox(box) + " vs design " + fmtBox(e) + (row.flowChild ? " (flow child: x/y exempt)" : ""));
        }
      }
    }

    // masked images: the inner "<name> — image" shape must land on the layer box
    if (row.type === "image" && row.hasMask && node.children && !skipGeometry.has(row.id)) {
      const imageChild = findDeep(node, (n) => /— image$/.test(n.name || ""));
      if (imageChild) {
        const b = absBox(imageChild, root);
        if (b) {
          const e = row.maskBoxOverride || row.expectedBox;
          const max = Math.max(Math.abs(b.x - e.x), Math.abs(b.y - e.y));
          if (max > GEO_TOLERANCE && !row.layer.mask.box && !row.layer.mask.bounds) {
            stats.geometryFailed += 1;
            fail(row.path + ": MASKED IMAGE content off by " + max.toFixed(2) + "px — image shape at " + fmtBox(b) + " vs layer box " + fmtBox(e));
          }
        }
      }
    }

    // auto-layout
    if (row.autoLayoutMode) {
      stats.autoLayoutChecked += 1;
      if (!("layoutMode" in node)) {
        stats.autoLayoutFailed += 1;
        fail(row.path + ": AUTO-LAYOUT — design wants " + row.autoLayoutMode + " but node type " + node.type + " has no layoutMode");
      } else if (node.layoutMode !== row.autoLayoutMode) {
        stats.autoLayoutFailed += 1;
        fail(row.path + ": AUTO-LAYOUT — layoutMode=" + node.layoutMode + ", design wants " + row.autoLayoutMode);
      } else {
        const gap = Number(R.pick(row.layoutSpec, "gap", "spacing", "itemSpacing", "item_spacing"));
        if (Number.isFinite(gap) && Math.abs(node.itemSpacing - gap) > GEO_TOLERANCE) {
          stats.autoLayoutFailed += 1;
          fail(row.path + ": AUTO-LAYOUT — itemSpacing=" + node.itemSpacing + ", design wants " + gap);
        }
      }
    }

    // per-dimension sizing (Codia DimensionSpec parity): design.json layer.sizing.w/h
    // must land on node.layoutSizingHorizontal / layoutSizingVertical. Hard-assert every
    // axis whose value would legally apply; note the rest (an illegal combo the plugin
    // correctly skipped) instead of failing.
    if (row.layer && row.layer.sizing && typeof row.layer.sizing === "object") {
      [
        ["horizontal", "layoutSizingHorizontal", sizingExpect(R.pick(row.layer.sizing, "w", "width", "horizontal"))],
        ["vertical", "layoutSizingVertical", sizingExpect(R.pick(row.layer.sizing, "h", "height", "vertical"))],
      ].forEach((entry) => {
        const axis = entry[0];
        const prop = entry[1];
        const want = entry[2];
        if (!want || !(prop in node)) return;
        stats.sizingChecked = (stats.sizingChecked || 0) + 1;
        if (!sizingWouldApply(node, want)) {
          result.info.push(row.path + ": SIZING — " + axis + " " + want + " does not legally apply to this node (plugin skipped, by design)");
          return;
        }
        if (node[prop] !== want) {
          stats.sizingFailed = (stats.sizingFailed || 0) + 1;
          fail(row.path + ": SIZING — " + prop + "=" + node[prop] + ", design sizing." + (axis === "horizontal" ? "w" : "h") + " wants " + want);
        }
      });
    }

    // constraints (informational: code.js applies them best-effort via safeSet)
    if (row.constraints && "constraints" in node) {
      const want = R.normalizedToken(row.constraints.horizontal || row.constraints.x || "");
      const got = node.constraints && node.constraints.horizontal;
      if (want && got === "MIN" && want !== "MIN" && want !== "LEFT") {
        result.info.push(row.path + ": constraints not applied (design wants " + want + ", node kept default MIN) — see warnings");
      }
    }

    // rotation recorded for rotated non-text layers handled above; text below.
  });

  if (expect.mustBuild) {
    expect.mustBuild.forEach((id) => {
      const row = rows.find((r) => r.id === id);
      if (!row || !nodeOfRow.get(row)) fail("mustBuild layer '" + id + "' did not produce a node");
    });
  }

  // created should equal the number of tagged builds (each compiled layer counts once)
  if (report.ok && typeof report.created === "number" && !expect.rebuild) {
    if (report.created !== stats.built) {
      degrade("report.created=" + report.created + " but harness matched " + stats.built + " design layers to nodes (informational; extra helper nodes are unnamed)");
    }
  }
  if (expect.minCreated !== undefined && (report.created || 0) < expect.minCreated) {
    fail("report.created=" + report.created + " < expected minimum " + expect.minCreated);
  }

  finishScore(result, report, rows, design, stats);
  return result;
}

function findDeep(node, predicate) {
  const stack = (node.children || []).slice();
  while (stack.length) {
    const n = stack.shift();
    if (predicate(n)) return n;
    (n.children || []).forEach((c) => stack.push(c));
  }
  return null;
}

// Map a design.json sizing token to the Figma unified enum the node should carry.
function sizingExpect(value) {
  const token = R.normalizedToken(value);
  if (token === "FILL") return "FILL";
  if (token === "HUG" || token === "FIT" || token === "FIT_CONTENT" || token === "CONTENT" || token === "AUTO") return "HUG";
  if (token === "FIXED" || token === "EXACT" || token === "ABSOLUTE") return "FIXED";
  return "";
}

// Would this sizing token legally apply to this node (mirrors code.js applyAxisSizing and
// the Figma docs)? FILL only on an auto-layout child; HUG only on an auto-layout frame or
// text; FIXED always. Lets the harness hard-assert legal sizing and merely note the rest.
function sizingWouldApply(node, token) {
  if (token === "FIXED") return true;
  if (token === "FILL") {
    const p = node.parent;
    return Boolean(p && "layoutMode" in p && p.layoutMode && p.layoutMode !== "NONE");
  }
  if (token === "HUG") {
    return node.type === "TEXT" || ("layoutMode" in node && node.layoutMode && node.layoutMode !== "NONE");
  }
  return false;
}

function checkText(row, node, box, report, stats, fail, degrade, skipContainment, mock) {
  stats.textChecked += 1;
  let bad = false;

  if (node.type !== "TEXT") {
    fail(row.path + ": TEXT — expected a TEXT node, got " + node.type);
    stats.textFailed += 1;
    return;
  }
  if (node.characters !== row.textContent) {
    bad = true;
    fail(row.path + ": TEXT — characters mismatch: " + JSON.stringify(node.characters) + " vs design " + JSON.stringify(row.textContent));
  }
  const f = node.fontName;
  if (!f || !f.family || !mock.state.loadedFonts.has(fontKey(f))) {
    bad = true;
    fail(row.path + ": TEXT — fontName " + JSON.stringify(f) + " was never loaded via loadFontAsync");
  }
  if (row.hasRuns && (!node._styleRanges || !node._styleRanges.length)) {
    degrade(row.path + ": TEXT — design has text_runs but no range styles were recorded (informational)");
  }
  if (row.rotation) {
    // CW-positive design rotation maps to CCW-positive Figma rotation (negated).
    if (Math.abs((node.rotation || 0) + row.rotation) > 0.5) {
      bad = true;
      fail(row.path + ": TEXT — rotation=" + node.rotation + ", design says " + row.rotation + " (expected node.rotation = -design rotation)");
    }
  } else if (!row.flowChild && !skipContainment.has(row.id) && row.textContent && row.textContent.trim()) {
    const t = row.expectedVisibleBox;
    const slackX = Math.max(TEXT_SLACK_PX, t.w * TEXT_SLACK_FRAC);
    const slackY = Math.max(TEXT_SLACK_PX, t.h * TEXT_SLACK_FRAC);
    const inside = box.x >= t.x - slackX && box.y >= t.y - slackY &&
      box.x + box.w <= t.x + t.w + slackX && box.y + box.h <= t.y + t.h + slackY;
    if (!inside) {
      const overflowRecorded = (report.errors || []).concat(report.warnings || []).some((e) =>
        /overflow/i.test(e.title + " " + e.detail) && (e.detail || "").indexOf(row.layer.name || row.id) >= 0);
      if (overflowRecorded) {
        degrade(row.path + ": TEXT overflows its box but the compiler recorded it (acceptable)");
      } else {
        bad = true;
        fail(row.path + ": TEXT PLACEMENT — node box " + fmtBox(box) + " escapes target " + fmtBox(t) +
          " (slack " + slackX.toFixed(1) + "/" + slackY.toFixed(1) + "px) with no overflow warning");
      }
    }
  }
  if (bad) stats.textFailed += 1;
}

function fmtBox(b) {
  return "[" + b.x.toFixed(1) + "," + b.y.toFixed(1) + " " + b.w.toFixed(1) + "x" + b.h.toFixed(1) + "]";
}

function finishScore(result, report, rows, design, stats) {
  result.stats = Object.assign(result.stats, stats || {});
}

// ----------------------------------------------------------------------------
// scorecard
// ----------------------------------------------------------------------------

function printScorecard(results) {
  const line = "-".repeat(78);
  console.log("\n" + line);
  console.log(" FIGMA PLUGIN COMPILER — MOCK E2E SCORECARD");
  console.log(" code.js: " + CODE_JS);
  console.log(line);
  results.forEach((r) => {
    const s = r.stats || {};
    const rep = s.report || {};
    console.log(
      (r.pass ? " PASS " : " FAIL ") + "| " + r.name.padEnd(42) +
      "| layers " + String(s.layers === undefined ? "-" : s.layers).padStart(3) +
      " built " + String(s.built === undefined ? "-" : s.built).padStart(3) +
      " geoΔmax " + (s.geometryMaxDelta === undefined ? "  - " : s.geometryMaxDelta.toFixed(2).padStart(5)) +
      " warn " + String(rep.warnings === undefined ? "-" : rep.warnings).padStart(3) +
      " err " + String(rep.errors === undefined ? "-" : rep.errors).padStart(2)
    );
    if (VERBOSE || !r.pass) {
      r.failures.forEach((f) => console.log("        ✗ " + f));
    }
    if (VERBOSE) {
      r.degradations.forEach((d) => console.log("        ~ " + d));
      r.info.forEach((i) => console.log("        · " + i));
    }
  });
  console.log(line);
  const failed = results.filter((r) => !r.pass);
  console.log(" " + (results.length - failed.length) + "/" + results.length + " fixtures passed" +
    (failed.length ? "  — FAILED: " + failed.map((f) => f.name).join(", ") : ""));
  console.log(line + "\n");
}

// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------

(async function main() {
  if (!fs.existsSync(CODE_JS)) {
    console.error("code.js not found at " + CODE_JS);
    process.exit(2);
  }
  let fixtures = discoverFixtures();
  if (ONLY) fixtures = fixtures.filter((f) => f.name.indexOf(ONLY) >= 0);
  if (!fixtures.length) {
    console.error("No fixtures matched" + (ONLY ? " '" + ONLY + "'" : "") + ".");
    process.exit(2);
  }

  const results = [];
  for (const fixture of fixtures) {
    process.stdout.write("running " + fixture.name + " ...\n");
    let res;
    try {
      res = await runFixture(fixture);
    } catch (error) {
      res = { name: fixture.name, pass: false, failures: ["harness crashed: " + (error.stack || error)], degradations: [], info: [], stats: {} };
    }
    results.push(res);
  }

  printScorecard(results);

  if (JSON_OUT) {
    fs.writeFileSync(JSON_OUT, JSON.stringify(results, null, 2));
    console.log("JSON results written to " + JSON_OUT);
  }

  process.exit(results.every((r) => r.pass) ? 0 : 1);
})();
