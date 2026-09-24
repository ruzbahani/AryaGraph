/*
 * =============================================================================
 *
 *      _                     ____                 _
 *     / \   _ __ _   _  __ _ / ___|_ __ __ _ _ __ | |__
 *    / _ \ | '__| | | |/ _` | |  _| '__/ _` | '_ \| '_ \
 *   / ___ \| |  | |_| | (_| | |_| | | | (_| | |_) | | | |
 *  /_/   \_\_|   \__, |\__,_|\____|_|  \__,_| .__/|_| |_|
 *               |___/                     |_|
 *
 *          Graph & DAG visualization, analysis and simulation.
 *
 * -----------------------------------------------------------------------------
 *  Copyright (c) 2026 Ali Mohammadi Ruzbahani
 *  SPDX-License-Identifier: MIT
 *
 *  https://ruzbahani.com/aryagraph
 * =============================================================================
 */

/* AryaGraph interactive runtime (no dependencies). Initialises every .ag-app on the page. */
(function () {
  "use strict";

  // ------------------------------------------------------------------ geometry
  function lerp(a, b, t) { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]; }
  function dist(a, b) { return Math.hypot(a[0] - b[0], a[1] - b[1]); }
  function pointAt(s, t) {
    var u = 1 - t, a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
    return [a * s[0][0] + b * s[1][0] + c * s[2][0] + d * s[3][0], a * s[0][1] + b * s[1][1] + c * s[2][1] + d * s[3][1]];
  }
  function tangentAt(s, t) {
    var u = 1 - t;
    var dx = 3 * u * u * (s[1][0] - s[0][0]) + 6 * u * t * (s[2][0] - s[1][0]) + 3 * t * t * (s[3][0] - s[2][0]);
    var dy = 3 * u * u * (s[1][1] - s[0][1]) + 6 * u * t * (s[2][1] - s[1][1]) + 3 * t * t * (s[3][1] - s[2][1]);
    var n = Math.hypot(dx, dy);
    if (n < 1e-9) { dx = s[3][0] - s[0][0]; dy = s[3][1] - s[0][1]; n = Math.hypot(dx, dy) || 1; }
    return [dx / n, dy / n];
  }
  function split(s, t) {
    var a = lerp(s[0], s[1], t), b = lerp(s[1], s[2], t), c = lerp(s[2], s[3], t);
    var d = lerp(a, b, t), e = lerp(b, c, t), f = lerp(d, e, t);
    return [[s[0], a, d, f], [f, e, c, s[3]]];
  }
  function line(a, b) { return [a, lerp(a, b, 1 / 3), lerp(a, b, 2 / 3), b]; }
  function quad(a, c, b) { return [a, lerp(a, c, 2 / 3), lerp(b, c, 2 / 3), b]; }
  function fmt(v) { return (Math.round(v * 100) / 100).toString(); }
  function toD(segs) {
    if (!segs.length) return "";
    var out = "M" + fmt(segs[0][0][0]) + "," + fmt(segs[0][0][1]);
    segs.forEach(function (s) {
      out += "C" + fmt(s[1][0]) + "," + fmt(s[1][1]) + " " + fmt(s[2][0]) + "," + fmt(s[2][1]) + " " + fmt(s[3][0]) + "," + fmt(s[3][1]);
    });
    return out;
  }

  function makeShapes(spec) {
    function contains(n, x, y) {
      var sp = spec[n.shape] || { t: "e" }, hw = n.w / 2, hh = n.h / 2, dx = x - n.x, dy = y - n.y;
      if (sp.t === "e") return (dx / Math.max(hw, 1e-9)) * (dx / Math.max(hw, 1e-9)) + (dy / Math.max(hh, 1e-9)) * (dy / Math.max(hh, 1e-9)) <= 1;
      if (sp.t === "r") {
        var r = sp.r === null ? Math.min(hw, hh) : Math.min(sp.r, hw, hh);
        var ax = Math.abs(dx), ay = Math.abs(dy);
        if (ax > hw || ay > hh) return false;
        if (ax <= hw - r || ay <= hh - r) return true;
        return (ax - (hw - r)) * (ax - (hw - r)) + (ay - (hh - r)) * (ay - (hh - r)) <= r * r;
      }
      var v = sp.v, sign = 0;
      for (var i = 0; i < v.length; i++) {
        var x1 = n.x + v[i][0] * hw, y1 = n.y + v[i][1] * hh;
        var j = (i + 1) % v.length, x2 = n.x + v[j][0] * hw, y2 = n.y + v[j][1] * hh;
        var cr = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1);
        if (Math.abs(cr) < 1e-12) continue;
        var s = cr > 0 ? 1 : -1;
        if (!sign) sign = s; else if (s !== sign) return false;
      }
      return true;
    }
    function boundary(n, tx, ty) {
      var dx = tx - n.x, dy = ty - n.y, L = Math.hypot(dx, dy);
      if (L < 1e-9) return [n.x, n.y];
      var ux = dx / L, uy = dy / L, lo = 0, hi = Math.hypot(n.w, n.h);
      for (var i = 0; i < 32; i++) {
        var mid = (lo + hi) / 2;
        if (contains(n, n.x + ux * mid, n.y + uy * mid)) lo = mid; else hi = mid;
      }
      return [n.x + ux * lo, n.y + uy * lo];
    }
    return { contains: contains, boundary: boundary };
  }

  function exitParam(seg, inside) {
    var lo = 0, hi = null;
    for (var k = 1; k <= 32; k++) { var t = k / 32; if (!inside(pointAt(seg, t))) { hi = t; break; } lo = t; }
    if (hi === null) return 1;
    for (var i = 0; i < 30; i++) { var m = (lo + hi) / 2; if (inside(pointAt(seg, m))) lo = m; else hi = m; }
    return hi;
  }
  function trimEnd(segs, d) {
    segs = segs.slice();
    var end = segs[segs.length - 1][3];
    while (segs.length) {
      var s = segs[segs.length - 1], prev = 1;
      for (var k = 47; k >= 0; k--) {
        var t = k / 48;
        if (dist(pointAt(s, t), end) >= d) {
          var lo = t, hi = prev;
          for (var i = 0; i < 28; i++) { var m = (lo + hi) / 2; if (dist(pointAt(s, m), end) >= d) lo = m; else hi = m; }
          segs[segs.length - 1] = split(s, lo)[0];
          return segs;
        }
        prev = t;
      }
      if (segs.length === 1) break;
      segs.pop();
    }
    var tan = tangentAt(segs[segs.length - 1], 1), p = [end[0] - tan[0] * d, end[1] - tan[1] * d];
    return [line(p, p)];
  }
  function arrowHead(tip, dir, len, wid) {
    var bx = tip[0] - dir[0] * len, by = tip[1] - dir[1] * len, px = -dir[1] * wid / 2, py = dir[0] * wid / 2;
    var back = [tip[0] - dir[0] * len * 0.78, tip[1] - dir[1] * len * 0.78];
    return {
      d: "M" + fmt(tip[0]) + "," + fmt(tip[1]) + "L" + fmt(bx + px) + "," + fmt(by + py) + "L" + fmt(back[0]) + "," + fmt(back[1]) +
        "L" + fmt(bx - px) + "," + fmt(by - py) + "Z",
      back: len * 0.78 + 0.3
    };
  }
  function flowSegs(pts, axis) {
    var segs = [];
    for (var i = 0; i + 1 < pts.length; i++) {
      var a = pts[i], b = pts[i + 1];
      if (axis === "y") { var k = (b[1] - a[1]) * 0.5; segs.push([a, [a[0], a[1] + k], [b[0], b[1] - k], b]); }
      else { var q = (b[0] - a[0]) * 0.5; segs.push([a, [a[0] + q, a[1]], [b[0] - q, b[1]], b]); }
    }
    return segs;
  }
  function orthoSegs(pts, axis) {
    var c = [pts[0]];
    for (var i = 0; i + 1 < pts.length; i++) {
      var a = pts[i], b = pts[i + 1];
      if (axis === "y") { var m = (a[1] + b[1]) / 2; c.push([a[0], m], [b[0], m], b); }
      else { var mx = (a[0] + b[0]) / 2; c.push([mx, a[1]], [mx, b[1]], b); }
    }
    var segs = [];
    for (var j = 0; j + 1 < c.length; j++) if (dist(c[j], c[j + 1]) > 1e-6) segs.push(line(c[j], c[j + 1]));
    return segs.length ? segs : [line(c[0], c[0])];
  }

  // ------------------------------------------------------------------ helpers
  function el(tag, attrs, text) {
    var e = document.createElement(tag);
    if (attrs) for (var k in attrs) { if (k === "class") e.className = attrs[k]; else e.setAttribute(k, attrs[k]); }
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }
  function fmtVal(v) {
    if (v === null || v === undefined) return "n/a";
    if (typeof v === "number") {
      if (!isFinite(v)) return String(v);
      if (Number.isInteger(v)) return v.toLocaleString();
      var a = Math.abs(v);
      if (a >= 1000) return Math.round(v).toLocaleString();
      if (a >= 0.001) return Number(v.toPrecision(4)).toString();
      return v.toExponential(2);
    }
    if (typeof v === "boolean") return v ? "true" : "false";
    if (Array.isArray(v)) return v.map(fmtVal).join(", ");
    if (typeof v === "object") return JSON.stringify(v);
    return String(v);
  }
  function b64bytes(s) {
    var bin = atob(s), out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function download(name, blob) {
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = name;
    document.body.appendChild(a); a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  }

  // ------------------------------------------------------------------ app
  function init(app) {
    var data = JSON.parse(app.querySelector("script.ag-data").textContent);
    var stage = app.querySelector(".ag-stage");
    var svg = stage.querySelector("svg");
    var viewport = svg.querySelector(".ag-viewport");
    var tip = stage.querySelector(".ag-tip");
    var panel = stage.querySelector(".ag-panel");
    var shapes = makeShapes(data.shapes);
    var off = data.offset, W = data.size[0], H = data.size[1];

    var nodes = new Map(), edges = [], adj = new Map(), inc = new Map();
    var nodeEls = {}, lblEls = {}, edgeEls = {}, elblEls = {};
    svg.querySelectorAll(".ag-n").forEach(function (e) { nodeEls[e.getAttribute("data-k")] = e; });
    svg.querySelectorAll(".ag-l").forEach(function (e) { lblEls[e.getAttribute("data-k")] = e; });
    svg.querySelectorAll(".ag-e").forEach(function (e) { edgeEls[e.getAttribute("data-k")] = e; });
    svg.querySelectorAll(".ag-el").forEach(function (e) { elblEls[e.getAttribute("data-k")] = e; });
    data.nodes.forEach(function (n, i) {
      n.i = i; n.x0 = n.x; n.y0 = n.y; n.el = nodeEls[n.k]; n.lbl = lblEls[n.k];
      nodes.set(n.k, n); adj.set(n.k, new Set()); inc.set(n.k, []);
    });
    data.edges.forEach(function (e, i) {
      e.i = i; e.el = edgeEls[e.k]; e.lab = elblEls[e.k];
      if (e.el) { e.pathEl = e.el.querySelector(".ag-edge"); e.arrowEl = e.el.querySelector(".ag-arrow"); }
      edges.push(e);
      adj.get(e.u).add(e.v); adj.get(e.v).add(e.u);
      inc.get(e.u).push(e); if (e.v !== e.u) inc.get(e.v).push(e);
    });

    // ---------------------------------------------------------------- view
    var view = { x: 0, y: 0, k: 1 };
    function apply() { viewport.setAttribute("transform", "translate(" + fmt(view.x) + "," + fmt(view.y) + ") scale(" + view.k.toFixed(4) + ")"); }
    function toSvg(ev) {
      var p = svg.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY;
      return p.matrixTransform(svg.getScreenCTM().inverse());
    }
    function toContent(p) { return [(p.x - view.x) / view.k - off[0], (p.y - view.y) / view.k - off[1]]; }
    function zoomAt(px, py, factor) {
      var k = Math.min(Math.max(view.k * factor, 0.1), 40);
      factor = k / view.k;
      view.x = px - (px - view.x) * factor; view.y = py - (py - view.y) * factor; view.k = k; apply();
    }
    function fit() { view = { x: 0, y: 0, k: 1 }; apply(); }
    function centerOn(n, k) {
      view.k = Math.max(view.k, k || 1.8);
      view.x = W / 2 - view.k * (n.x + off[0]); view.y = H / 2 - view.k * (n.y + off[1]); apply();
    }

    // ---------------------------------------------------------------- emphasis
    var selected = null, groupOn = null;
    function clearOn() { svg.querySelectorAll(".ag-on").forEach(function (e) { e.classList.remove("ag-on"); }); }
    function focusNodes(keys) {
      clearOn();
      if (!keys) { svg.classList.remove("ag-focus"); return; }
      svg.classList.add("ag-focus");
      keys.forEach(function (k) { var n = nodes.get(k); if (n && n.el) n.el.classList.add("ag-on"); if (n && n.lbl) n.lbl.classList.add("ag-on"); });
      edges.forEach(function (e) {
        if (keys.has(e.u) && keys.has(e.v)) { if (e.el) e.el.classList.add("ag-on"); if (e.lab) e.lab.classList.add("ag-on"); }
      });
    }
    function neighborhood(k) {
      var s = new Set([k]); adj.get(k).forEach(function (x) { s.add(x); });
      return s;
    }
    function focusNode(k) {
      focusNodes(neighborhood(k));
      inc.get(k).forEach(function (e) { if (e.el) e.el.classList.add("ag-on"); if (e.lab) e.lab.classList.add("ag-on"); });
    }
    function restore() {
      if (selected) focusNode(selected);
      else if (groupOn !== null) focusGroup(groupOn);
      else focusNodes(null);
    }

    // ---------------------------------------------------------------- tooltip
    function stateOf(n) {
      if (!anim) return null;
      var code = anim.frames[frame * anim.n + n.i];
      if (anim.values) return anim.values[frame * anim.n + n.i];
      return anim.states[code];
    }
    function tipRows(n) {
      var rows = [];
      var s = stateOf(n);
      if (s !== null) rows.push([anim.values ? "value" : "state", s]);
      var t = n.tip || {};
      ["degree", "in", "out"].forEach(function (k) { if (k in t) rows.push([k, t[k]]); });
      var at = t.attrs || {};
      Object.keys(at).forEach(function (k) { rows.push([k, at[k]]); });
      return rows;
    }
    function showTip(ev, title, rows) {
      tip.textContent = "";
      tip.appendChild(el("b", null, title));
      if (rows.length) {
        var dl = el("dl");
        rows.slice(0, 14).forEach(function (r) { dl.appendChild(el("dt", null, r[0])); dl.appendChild(el("dd", null, fmtVal(r[1]))); });
        tip.appendChild(dl);
      }
      tip.hidden = false;
      moveTip(ev);
    }
    function moveTip(ev) {
      if (tip.hidden) return;
      var r = stage.getBoundingClientRect(), x = ev.clientX - r.left + 14, y = ev.clientY - r.top + 14;
      var w = tip.offsetWidth, h = tip.offsetHeight;
      if (x + w > r.width - 6) x = ev.clientX - r.left - w - 14;
      if (y + h > r.height - 6) y = ev.clientY - r.top - h - 14;
      tip.style.left = Math.max(4, x) + "px"; tip.style.top = Math.max(4, y) + "px";
    }
    function hideTip() { tip.hidden = true; }

    // ---------------------------------------------------------------- details panel
    function openPanel(n) {
      if (!panel) return;
      panel.textContent = "";
      var close = el("button", { class: "ag-close", "aria-label": "Close details" }, "×");
      close.addEventListener("click", function () { select(null); });
      panel.appendChild(close);
      panel.appendChild(el("h3", null, n.label));
      panel.appendChild(el("div", { class: "ag-sub" }, "id " + n.id));
      var rows = tipRows(n);
      if (rows.length) {
        panel.appendChild(el("h4", null, "Attributes"));
        var dl = el("dl");
        rows.forEach(function (r) { dl.appendChild(el("dt", null, r[0])); dl.appendChild(el("dd", null, fmtVal(r[1]))); });
        panel.appendChild(dl);
      }
      var nb = Array.from(adj.get(n.k)).filter(function (k) { return k !== n.k; });
      panel.appendChild(el("h4", null, "Neighbors (" + nb.length + ")"));
      var ul = el("ul");
      nb.slice(0, 60).forEach(function (k) {
        var m = nodes.get(k), li = el("li"), b = el("button", { type: "button" }, m.label);
        b.addEventListener("click", function () { select(k); centerOn(m); });
        li.appendChild(b); ul.appendChild(li);
      });
      panel.appendChild(ul);
      panel.hidden = false;
    }
    function select(k) {
      if (selected && nodes.get(selected).el) nodes.get(selected).el.classList.remove("ag-sel");
      selected = k; groupOn = null;
      if (k) { var n = nodes.get(k); if (n.el) n.el.classList.add("ag-sel"); openPanel(n); }
      else if (panel) panel.hidden = true;
      restore();
    }

    // ---------------------------------------------------------------- dragging
    function updateEdge(e) {
      if (!e.pathEl) return;
      var U = nodes.get(e.u), V = nodes.get(e.v), segs, route = e.route || [];
      if (e.u === e.v) {
        e.el.setAttribute("transform", "translate(" + fmt(U.x - U.x0) + "," + fmt(U.y - U.y0) + ")");
        return;
      }
      var inside = function (n) { return function (p) { return shapes.contains(n, p[0], p[1]); }; };
      if (e.style === "flow" || e.style === "orthogonal") {
        var ax = data.axis === "x" ? 0 : 1;
        var first = route.length ? route[0] : [V.x, V.y], last = route.length ? route[route.length - 1] : [U.x, U.y];
        var su = first[ax] >= (ax ? U.y : U.x) ? 1 : -1, sv = last[ax] >= (ax ? V.y : V.x) ? 1 : -1;
        var pu = ax ? shapes.boundary(U, U.x, U.y + su * U.h) : shapes.boundary(U, U.x + su * U.w, U.y);
        var pv = ax ? shapes.boundary(V, V.x, V.y + sv * V.h) : shapes.boundary(V, V.x + sv * V.w, V.y);
        var pts = [pu].concat(route, [pv]);
        segs = e.style === "flow" ? flowSegs(pts, data.axis) : orthoSegs(pts, data.axis);
      } else if (e.style === "curved") {
        var a = [U.x, U.y], b = [V.x, V.y], dx = b[0] - a[0], dy = b[1] - a[1], c = e.curv;
        var ctrl = [(a[0] + b[0]) / 2 + dy * c * 2, (a[1] + b[1]) / 2 - dx * c * 2];
        var s = quad(a, ctrl, b);
        s = split(s, exitParam(s, inside(U)))[1];
        var rev = [s[3], s[2], s[1], s[0]];
        var t = exitParam(rev, inside(V));
        segs = [split(s, 1 - t)[0]];
      } else {
        var aa = shapes.boundary(U, route.length ? route[0][0] : V.x, route.length ? route[0][1] : V.y);
        var bb = shapes.boundary(V, route.length ? route[route.length - 1][0] : U.x, route.length ? route[route.length - 1][1] : U.y);
        var all = [aa].concat(route, [bb]);
        segs = [];
        for (var i = 0; i + 1 < all.length; i++) segs.push(line(all[i], all[i + 1]));
      }
      if (e.al > 0 && e.arrowEl) {
        segs = trimEnd(segs, 1.2);
        var last2 = segs[segs.length - 1], tipP = last2[3], dir = tangentAt(last2, 1);
        var ah = arrowHead(tipP, dir, e.al, e.al * 0.78);
        segs = trimEnd(segs, ah.back);
        e.arrowEl.setAttribute("d", ah.d);
      }
      e.pathEl.setAttribute("d", toD(segs));
      if (e.lab) {
        var len = e.pathEl.getTotalLength(), mp = e.pathEl.getPointAtLength(len / 2);
        if (e.m0 === undefined) e.m0 = e.mid || [mp.x, mp.y];
        e.lab.setAttribute("transform", "translate(" + fmt(mp.x - e.m0[0]) + "," + fmt(mp.y - e.m0[1]) + ")");
      }
    }
    function moveNode(n, x, y) {
      n.x = x; n.y = y;
      var tr = "translate(" + fmt(x) + "," + fmt(y) + ")";
      if (n.el) n.el.setAttribute("transform", tr);
      if (n.lbl) n.lbl.setAttribute("transform", tr);
      inc.get(n.k).forEach(updateEdge);
    }

    // ---------------------------------------------------------------- pointer input
    var pointers = new Map(), drag = null, pinch = null;
    svg.addEventListener("pointerdown", function (ev) {
      if (ev.button !== 0 && ev.pointerType === "mouse") return;
      svg.setPointerCapture(ev.pointerId);
      pointers.set(ev.pointerId, [ev.clientX, ev.clientY]);
      if (pointers.size === 2) {
        var ps = Array.from(pointers.values());
        pinch = { d: dist(ps[0], ps[1]), k: view.k };
        drag = null; return;
      }
      var g = ev.target.closest && ev.target.closest(".ag-n");
      var p = toSvg(ev);
      drag = { node: g ? nodes.get(g.getAttribute("data-k")) : null, sx: ev.clientX, sy: ev.clientY, px: p.x, py: p.y, vx: view.x, vy: view.y, moved: false };
      if (drag.node) { var c = toContent(p); drag.dx = drag.node.x - c[0]; drag.dy = drag.node.y - c[1]; }
    });
    svg.addEventListener("pointermove", function (ev) {
      if (pointers.has(ev.pointerId)) pointers.set(ev.pointerId, [ev.clientX, ev.clientY]);
      if (pinch && pointers.size === 2) {
        var ps = Array.from(pointers.values()), d = dist(ps[0], ps[1]);
        var mid = { clientX: (ps[0][0] + ps[1][0]) / 2, clientY: (ps[0][1] + ps[1][1]) / 2 }, p = toSvg(mid);
        zoomAt(p.x, p.y, (pinch.k * d / pinch.d) / view.k); return;
      }
      if (!drag) { moveTip(ev); return; }
      if (!drag.moved && Math.hypot(ev.clientX - drag.sx, ev.clientY - drag.sy) < 3) return;
      drag.moved = true; hideTip();
      var p2 = toSvg(ev);
      if (drag.node) {
        stage.classList.add("ag-dragging");
        var c = toContent(p2); moveNode(drag.node, c[0] + drag.dx, c[1] + drag.dy);
      } else {
        stage.classList.add("ag-panning");
        view.x = drag.vx + (p2.x - drag.px); view.y = drag.vy + (p2.y - drag.py); apply();
      }
    });
    function endPointer(ev) {
      pointers.delete(ev.pointerId);
      if (pointers.size < 2) pinch = null;
      if (!drag) return;
      var d = drag; drag = null;
      stage.classList.remove("ag-dragging", "ag-panning");
      if (d.moved) return;
      if (d.node) select(selected === d.node.k ? null : d.node.k);
      else if (!ev.target.closest(".ag-legend")) { groupOn = null; select(null); }
    }
    svg.addEventListener("pointerup", endPointer);
    svg.addEventListener("pointercancel", endPointer);
    svg.addEventListener("wheel", function (ev) {
      ev.preventDefault();
      var p = toSvg(ev);
      zoomAt(p.x, p.y, Math.exp(-ev.deltaY * (ev.deltaMode === 1 ? 0.05 : 0.0018)));
    }, { passive: false });
    svg.addEventListener("dblclick", function (ev) { if (!ev.target.closest(".ag-n")) fit(); });
    svg.addEventListener("pointerover", function (ev) {
      if (drag && drag.moved) return;
      var g = ev.target.closest(".ag-n");
      if (g) {
        var n = nodes.get(g.getAttribute("data-k"));
        if (!selected) focusNode(n.k);
        showTip(ev, n.label, tipRows(n));
        return;
      }
      var e = ev.target.closest(".ag-e");
      if (e) {
        var ed = edges[+e.getAttribute("data-k").slice(1)];
        var rows = [];
        var at = (ed.tip && ed.tip.attrs) || {};
        Object.keys(at).forEach(function (k) { rows.push([k, at[k]]); });
        showTip(ev, nodes.get(ed.u).label + (data.directed ? " → " : " - ") + nodes.get(ed.v).label, rows);
      }
    });
    svg.addEventListener("pointerout", function (ev) {
      var g = ev.target.closest(".ag-n") || ev.target.closest(".ag-e");
      if (!g) return;
      var to = ev.relatedTarget;
      if (to && g.contains(to)) return;
      hideTip();
      if (!selected && (!drag || !drag.moved)) restore();
    });

    // keyboard
    svg.setAttribute("tabindex", "0");
    svg.addEventListener("keydown", function (ev) {
      var step = 40;
      if (ev.key === "+" || ev.key === "=") zoomAt(W / 2, H / 2, 1.25);
      else if (ev.key === "-" || ev.key === "_") zoomAt(W / 2, H / 2, 0.8);
      else if (ev.key === "0") fit();
      else if (ev.key === "ArrowLeft") { view.x += step; apply(); }
      else if (ev.key === "ArrowRight") { view.x -= step; apply(); }
      else if (ev.key === "ArrowUp") { view.y += step; apply(); }
      else if (ev.key === "ArrowDown") { view.y -= step; apply(); }
      else if (ev.key === "Escape") { select(null); }
      else if (ev.key === " " && anim) { togglePlay(); }
      else return;
      ev.preventDefault();
    });

    // ---------------------------------------------------------------- legend
    function focusGroup(v) {
      var keys = new Set();
      data.nodes.forEach(function (n) { if (n.g !== undefined && String(n.g) === v) keys.add(n.k); });
      focusNodes(keys);
    }
    svg.querySelectorAll('.ag-legend-block[data-channel="node_color"] .ag-legend-item[data-value]').forEach(function (it) {
      var v = it.getAttribute("data-value");
      it.addEventListener("mouseenter", function () { if (!selected && groupOn === null) focusGroup(v); });
      it.addEventListener("mouseleave", function () { if (!selected && groupOn === null) focusNodes(null); });
      it.addEventListener("click", function (ev) {
        ev.stopPropagation();
        if (selected) select(null);
        groupOn = groupOn === v ? null : v;
        restore();
      });
    });

    // ---------------------------------------------------------------- toolbar
    function btn(name) { return app.querySelector('[data-act="' + name + '"]'); }
    function on(name, fn) { var b = btn(name); if (b) b.addEventListener("click", fn); }
    on("fit", fit);
    on("zoom-in", function () { zoomAt(W / 2, H / 2, 1.3); });
    on("zoom-out", function () { zoomAt(W / 2, H / 2, 1 / 1.3); });
    var labelMode = 0;
    on("labels", function () {
      labelMode = (labelMode + 1) % 3;
      svg.classList.toggle("ag-labels-all", labelMode === 1);
      svg.classList.toggle("ag-labels-none", labelMode === 2);
      btn("labels").setAttribute("aria-pressed", labelMode ? "true" : "false");
      btn("labels").querySelector(".ag-txt").textContent = ["Labels: auto", "Labels: all", "Labels: off"][labelMode];
    });
    function serialize() {
      var clone = svg.cloneNode(true);
      clone.classList.remove("ag-focus");
      clone.removeAttribute("tabindex");
      clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
      return new XMLSerializer().serializeToString(clone);
    }
    on("svg", function () { download((data.name || "graph") + ".svg", new Blob([serialize()], { type: "image/svg+xml" })); });
    on("png", function () {
      var img = new Image(), url = URL.createObjectURL(new Blob([serialize()], { type: "image/svg+xml" }));
      img.onload = function () {
        var s = 2, c = document.createElement("canvas");
        c.width = W * s; c.height = H * s;
        var ctx = c.getContext("2d"); ctx.drawImage(img, 0, 0, W * s, H * s);
        URL.revokeObjectURL(url);
        c.toBlob(function (b) { download((data.name || "graph") + ".png", b); });
      };
      img.src = url;
    });

    // ---------------------------------------------------------------- search
    var input = app.querySelector(".ag-search input"), results = app.querySelector(".ag-results"), active = -1, matches = [];
    function renderResults() {
      results.textContent = "";
      if (!matches.length) { results.hidden = true; return; }
      matches.forEach(function (n, i) {
        var li = el("li", { role: "option", "aria-selected": i === active ? "true" : "false" });
        li.appendChild(el("span", null, n.label));
        li.appendChild(el("small", null, (adj.get(n.k).size) + " links"));
        li.addEventListener("mousedown", function (ev) { ev.preventDefault(); pick(n); });
        results.appendChild(li);
      });
      results.hidden = false;
    }
    function pick(n) { select(n.k); centerOn(n); matches = []; renderResults(); input.blur(); }
    if (input) {
      input.addEventListener("input", function () {
        var q = input.value.trim().toLowerCase();
        active = -1;
        if (!q) { matches = []; renderResults(); return; }
        var hits = [];
        data.nodes.forEach(function (n) {
          var l = n.label.toLowerCase(), id = String(n.id).toLowerCase();
          var at = l.indexOf(q), ai = id.indexOf(q);
          if (at >= 0 || ai >= 0) hits.push({ n: n, s: (at === 0 || ai === 0 ? 0 : 1), d: -adj.get(n.k).size });
        });
        hits.sort(function (a, b) { return a.s - b.s || a.d - b.d; });
        matches = hits.slice(0, 8).map(function (h) { return h.n; });
        renderResults();
      });
      input.addEventListener("keydown", function (ev) {
        if (ev.key === "ArrowDown") { active = Math.min(active + 1, matches.length - 1); renderResults(); ev.preventDefault(); }
        else if (ev.key === "ArrowUp") { active = Math.max(active - 1, 0); renderResults(); ev.preventDefault(); }
        else if (ev.key === "Enter") { if (matches.length) pick(matches[Math.max(active, 0)]); }
        else if (ev.key === "Escape") { input.value = ""; matches = []; renderResults(); }
      });
      input.addEventListener("blur", function () { setTimeout(function () { results.hidden = true; }, 120); });
    }

    // ---------------------------------------------------------------- table view
    var tableBox = app.querySelector(".ag-table"), tableBuilt = false;
    on("table", function () {
      if (!tableBox) return;
      tableBox.hidden = !tableBox.hidden;
      btn("table").setAttribute("aria-pressed", tableBox.hidden ? "false" : "true");
      if (!tableBuilt) { buildTable(); tableBuilt = true; }
    });
    function buildTable() {
      var keys = [], seen = {};
      data.nodes.forEach(function (n) {
        Object.keys((n.tip && n.tip.attrs) || {}).forEach(function (k) { if (!seen[k] && keys.length < 10) { seen[k] = 1; keys.push(k); } });
      });
      var base = data.directed ? ["in", "out"] : ["degree"];
      var cols = ["label"].concat(base, keys);
      var rows = data.nodes.map(function (n) {
        var t = n.tip || {}, at = t.attrs || {};
        return { n: n, v: cols.map(function (c) { return c === "label" ? n.label : (c in t ? t[c] : at[c]); }) };
      });
      var table = el("table"), thead = el("thead"), tr = el("tr"), tbody = el("tbody");
      var sortCol = -1, asc = false;
      cols.forEach(function (c, i) {
        var th = el("th", { scope: "col" }, c);
        th.addEventListener("click", function () {
          asc = sortCol === i ? !asc : (i === 0);
          sortCol = i;
          tr.querySelectorAll("th").forEach(function (h) { h.removeAttribute("aria-sort"); });
          th.setAttribute("aria-sort", asc ? "ascending" : "descending");
          rows.sort(function (a, b) {
            var x = a.v[i], y = b.v[i];
            if (x === undefined || x === null) return 1;
            if (y === undefined || y === null) return -1;
            var r = (typeof x === "number" && typeof y === "number") ? x - y : String(x).localeCompare(String(y), undefined, { numeric: true });
            return asc ? r : -r;
          });
          fill();
        });
        tr.appendChild(th);
      });
      thead.appendChild(tr); table.appendChild(thead); table.appendChild(tbody); tableBox.appendChild(table);
      function fill() {
        tbody.textContent = "";
        rows.slice(0, 2000).forEach(function (r) {
          var row = el("tr");
          r.v.forEach(function (v) { row.appendChild(el("td", typeof v === "number" ? { class: "num" } : null, fmtVal(v))); });
          row.addEventListener("click", function () { select(r.n.k); centerOn(r.n); });
          tbody.appendChild(row);
        });
      }
      fill();
    }

    // ---------------------------------------------------------------- animation
    var anim = null, frame = 0, playing = false, timer = null;
    var togglePlay = function () {};
    if (data.anim) {
      var A = data.anim;
      anim = {
        n: data.nodes.length, T: A.times.length, times: A.times, states: A.states || [], palette: A.palette,
        frames: b64bytes(A.frames), values: A.values ? new Float32Array(b64bytes(A.values).buffer) : null,
        fire: A.fire || null, chart: A.chart || null, fmt: A.time_format || "t = {t}"
      };
      var slider = app.querySelector(".ag-player input[type=range]"), tlabel = app.querySelector(".ag-time");
      var readout = app.querySelector(".ag-readout"), cursor = app.querySelector(".ag-chart .ag-cursor");
      var shapeEls = data.nodes.map(function (n) { return n.el ? n.el.querySelector(".ag-shape") : null; });
      var textEls = data.nodes.map(function (n) { return n.el ? n.el.querySelector("text") : null; });
      var current = new Int16Array(anim.n).fill(-1);
      var fired = [];
      var setFrame = function (i) {
        frame = Math.max(0, Math.min(anim.T - 1, i));
        var base = frame * anim.n;
        for (var j = 0; j < anim.n; j++) {
          var c = anim.frames[base + j];
          if (c === current[j] || !shapeEls[j]) continue;
          current[j] = c;
          var p = anim.palette[c];
          shapeEls[j].setAttribute("fill", p[0]);
          if (p[1]) shapeEls[j].setAttribute("stroke", p[1]);
          if (p[2] && textEls[j]) textEls[j].setAttribute("fill", p[2]);
        }
        fired.forEach(function (e) { e.classList.remove("ag-fire"); });
        fired = [];
        if (anim.fire && anim.fire[frame]) anim.fire[frame].forEach(function (ei) { var e = edges[ei]; if (e && e.el) { e.el.classList.add("ag-fire"); fired.push(e.el); } });
        if (slider) slider.value = frame;
        var t = anim.times[frame];
        if (tlabel) tlabel.textContent = anim.fmt.replace("{t}", fmtVal(Math.round(t * 1000) / 1000)).replace("{i}", frame);
        if (cursor && anim.chart) {
          var ch = anim.chart, x = ch.x0 + (t - ch.t0) / ((ch.t1 - ch.t0) || 1) * (ch.x1 - ch.x0);
          cursor.setAttribute("x1", fmt(x)); cursor.setAttribute("x2", fmt(x));
        }
        if (readout && !anim.values) {
          var counts = new Array(anim.states.length).fill(0);
          for (var q = 0; q < anim.n; q++) counts[anim.frames[base + q]]++;
          readout.textContent = "";
          anim.states.forEach(function (s, si) {
            var span = el("span", { class: "ag-ro" });
            var key = el("i"); key.style.background = anim.palette[si][3] || anim.palette[si][0];
            span.appendChild(key); span.appendChild(el("b", null, counts[si].toLocaleString())); span.appendChild(el("small", null, s));
            readout.appendChild(span);
          });
        }
      };
      var speedSel = app.querySelector(".ag-player select");
      var fps = function () { return Math.max(3, Math.min(30, anim.T / 10)) * (speedSel ? +speedSel.value : 1); };
      var tick = function () {
        if (!playing) return;
        if (frame >= anim.T - 1) { stop(); return; }
        setFrame(frame + 1);
        timer = setTimeout(tick, 1000 / fps());
      };
      var playBtn = btn("play");
      var stop = function () { playing = false; clearTimeout(timer); if (playBtn) { playBtn.setAttribute("aria-pressed", "false"); playBtn.querySelector(".ag-txt").textContent = "Play"; } };
      togglePlay = function () {
        if (playing) { stop(); return; }
        if (frame >= anim.T - 1) setFrame(0);
        playing = true;
        if (playBtn) { playBtn.setAttribute("aria-pressed", "true"); playBtn.querySelector(".ag-txt").textContent = "Pause"; }
        tick();
      };
      on("play", function () { togglePlay(); });
      on("step-back", function () { stop(); setFrame(frame - 1); });
      on("step-fwd", function () { stop(); setFrame(frame + 1); });
      if (slider) { slider.max = anim.T - 1; slider.addEventListener("input", function () { stop(); setFrame(+slider.value); }); }
      var chartBox = app.querySelector(".ag-chart svg");
      if (chartBox && anim.chart) {
        chartBox.addEventListener("pointermove", function (ev) {
          if (ev.buttons !== 1) return;
          var p = chartBox.createSVGPoint(); p.x = ev.clientX; p.y = ev.clientY;
          var q = p.matrixTransform(chartBox.getScreenCTM().inverse()), ch = anim.chart;
          var t = ch.t0 + (q.x - ch.x0) / (ch.x1 - ch.x0) * (ch.t1 - ch.t0), best = 0;
          for (var i = 0; i < anim.T; i++) if (Math.abs(anim.times[i] - t) < Math.abs(anim.times[best] - t)) best = i;
          stop(); setFrame(best);
        });
        chartBox.addEventListener("pointerdown", function (ev) { chartBox.setPointerCapture(ev.pointerId); chartBox.dispatchEvent(new PointerEvent("pointermove", { clientX: ev.clientX, clientY: ev.clientY, buttons: 1 })); });
      }
      setFrame(data.anim.start || 0);
      if (data.anim.autoplay) togglePlay();
    }

    apply();
    app.agReady = true;
  }

  function boot() {
    document.querySelectorAll(".ag-app").forEach(function (app) {
      if (app.agReady) return;
      try { init(app); } catch (err) { app.setAttribute("data-error", String(err && err.message || err)); if (window.console) console.error(err); }
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
