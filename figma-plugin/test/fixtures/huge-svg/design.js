"use strict";
// Generated design: one vector layer with 2500 paths, one malformed vector.
module.exports = function () {
  const paths = [];
  for (let i = 0; i < 2500; i += 1) {
    const x = (i % 50) * 4;
    const y = Math.floor(i / 50) * 4;
    paths.push('<path d="M' + x + " " + y + "h3v3h-3z" + '" fill="#3366ff"/>');
  }
  const svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">' + paths.join("") + "</svg>";
  return {
    id: "huge-svg",
    name: "huge-svg",
    canvas: { w: 1000, h: 800 },
    schema_version: 2,
    meta: { coordinate_space: "local" },
    layers: [
      { id: "bg", type: "shape", shape_kind: "rect", name: "bg", box: { x: 0, y: 0, w: 1000, h: 800 }, fill: { kind: "flat", color: "#ffffff" } },
      { id: "monster", type: "vector", name: "monster", box: { x: 100, y: 100, w: 400, h: 400 }, svg },
      { id: "bad", type: "vector", name: "bad", box: { x: 600, y: 100, w: 100, h: 100 }, svg: '<svg viewBox="0 0 10 10"><path d="M0 0L10' },
    ],
  };
};
