# =============================================================================
#
#      _                     ____                 _
#     / \   _ __ _   _  __ _ / ___|_ __ __ _ _ __ | |__
#    / _ \ | '__| | | |/ _` | |  _| '__/ _` | '_ \| '_ \
#   / ___ \| |  | |_| | (_| | |_| | | | (_| | |_) | | | |
#  /_/   \_\_|   \__, |\__,_|\____|_|  \__,_| .__/|_| |_|
#               |___/                     |_|
#
#          Graph & DAG visualization, analysis and simulation.
#
# -----------------------------------------------------------------------------
#  Copyright (c) 2026 Ali Mohammadi Ruzbahani
#  SPDX-License-Identifier: MIT
#
#  https://ruzbahani.com/aryagraph
# =============================================================================

"""Mermaid flowcharts (``.mmd``): a writer and a reader for the flowchart subset.

:func:`to_mermaid` emits ``flowchart`` syntax that GitHub, GitLab and the
Mermaid live editor render. Node names that are not safe Mermaid ids
(``[A-Za-z][A-Za-z0-9_]*`` or digits, not a keyword) are replaced by ``n0``,
``n1``, … and shown through their quoted label, so any name (Persian text,
spaces, quotes) displays correctly. In labels ``"`` and ``#`` are written as
the entity codes ``#quot;`` and ``#35;``, and line breaks as ``<br>``.

:func:`from_mermaid` reads ``graph``/``flowchart`` diagrams: node shapes
(``A[text]``, ``A(text)``, ``A{text}``, ``A((text))`` and the other bracket
shapes) set the ``label`` and ``shape`` node attributes; links ``-->``,
``---``, ``-.->``, ``==>``, ``--o``, ``--x``, ``<-->`` (any length), with
``|label|`` or ``-- label -->`` text; chains ``A --> B --> C`` and ``&``
groups; several statements per line separated by ``;``. Subgraphs are
flattened (members get the ``cluster`` attribute, the subgraph id);
``style``, ``classDef``, ``class``, ``click``, ``linkStyle``, ``direction``
and ``%%`` comment lines are skipped; ``~~~`` (invisible) links add no edge.

A diagram with at least one arrowhead becomes a :class:`DiGraph` (its open
links become arcs with ``arrow="none"``); otherwise a :class:`Graph`. Dotted
and thick links set the edge attribute ``style``; ``o``/``x`` heads set
``arrow`` to ``"circle"``/``"cross"``. Mermaid has no types, so node names
are always strings.
"""

from __future__ import annotations

import html
import os
import re
from typing import Any

from ..core.graph import DiGraph, Graph

PathLike = str | os.PathLike

_DIRECTIONS = ("TD", "TB", "BT", "LR", "RL")
_RESERVED = frozenset(
    {"end", "graph", "flowchart", "subgraph", "click", "call", "class", "classdef", "style", "linkstyle",
     "direction", "default", "href", "interpolate", "o", "x"}
)  # fmt: skip
_SAFE_ID = re.compile(r"(?:[A-Za-z][A-Za-z0-9_]*|[0-9]+)\Z")

# shape name -> (opener, closer)
_SHAPES: dict[str, tuple[str, str]] = {
    "double_circle": ("(((", ")))"),
    "circle": ("((", "))"),
    "stadium": ("([", "])"),
    "subroutine": ("[[", "]]"),
    "cylinder": ("[(", ")]"),
    "hexagon": ("{{", "}}"),
    "parallelogram": ("[/", "/]"),
    "trapezoid": ("[/", "\\]"),
    "parallelogram_alt": ("[\\", "\\]"),
    "trapezoid_alt": ("[\\", "/]"),
    "round": ("(", ")"),
    "rect": ("[", "]"),
    "diamond": ("{", "}"),
    "asymmetric": (">", "]"),
}
_SHAPE_ALIASES = {"rhombus": "diamond", "box": "rect", "rectangle": "rect", "rounded": "round"}
_OPENERS = sorted({o for o, _ in _SHAPES.values()}, key=len, reverse=True)  # longest first

_NODE_ID = re.compile(r"\w+")
_CLASS = re.compile(r"[\w-]+")
_LINK = re.compile(
    r"(?P<start><)?(?:(?P<dotted>-\.+-)|(?P<thick>={2,})|(?P<solid>-{2,})|(?P<invisible>~{3,}))(?P<head>[>ox])?"
)
_TEXT_LINK_OPEN = re.compile(r"(?P<start><)?(?:(?P<dotted>-\.)|(?P<thick>==)|(?P<solid>--))(?=[ \t])")
_TEXT_LINK_CLOSE = {
    "dotted": re.compile(r"\.+-(?P<head>[>ox])?"),
    "thick": re.compile(r"={2,}(?P<head>[>ox])?"),
    "solid": re.compile(r"-{2,}(?P<head>[>ox])?"),
}
_KINDS = ("dotted", "thick", "solid", "invisible")
_ARROW_OF_HEAD = {"o": "circle", "x": "cross"}
_HEAD_OF_ARROW = {"none": "", "normal": ">", "circle": "o", "cross": "x"}
_LINK_TEXT = {  # (style, head) -> link token
    ("solid", ""): "---", ("solid", ">"): "-->", ("solid", "o"): "--o", ("solid", "x"): "--x",
    ("dotted", ""): "-.-", ("dotted", ">"): "-.->", ("dotted", "o"): "-.-o", ("dotted", "x"): "-.-x",
    ("thick", ""): "===", ("thick", ">"): "==>", ("thick", "o"): "==o", ("thick", "x"): "==x",
}  # fmt: skip
_HEADER = re.compile(r"(?:graph|flowchart)(?:[ \t]+(TD|TB|BT|LR|RL))?[ \t]*(?:;|\Z)", re.IGNORECASE)
_SKIP = re.compile(r"(?:style|classDef|class|click|linkStyle|direction|accTitle|accDescr)\b")
_SUBGRAPH_NAME = re.compile(r'"([^"]*)"|(\w+)')


# ---------------------------------------------------------------------- #
# writer
# ---------------------------------------------------------------------- #
def _escape(text: str) -> str:
    text = text.replace("#", "#35;").replace('"', "#quot;")
    return text.replace("\r\n", "\n").replace("\n", "<br>")


def to_mermaid(
    g: Graph,
    *,
    direction: str = "TD",
    label: str | None = None,
    edge_label: str | None = "label",
) -> str:
    """Mermaid ``flowchart`` text for *g*.

    Parameters
    ----------
    direction:
        ``"TD"``/``"TB"`` (top-down), ``"BT"``, ``"LR"`` or ``"RL"``.
    label:
        Node attribute holding the display text. By default a node shows its
        ``label`` attribute when it has one, else its name.
    edge_label:
        Edge attribute shown on the link (``None`` for no link text).

    A node's ``shape`` attribute (``rect``, ``round``, ``stadium``,
    ``circle``, ``diamond``, ``hexagon``, …) picks its brackets; an edge's
    ``style`` (``"dotted"``/``"thick"``) and ``arrow`` (``"none"``,
    ``"normal"``, ``"circle"``, ``"cross"``) pick the link.

    >>> print(to_mermaid(DiGraph([("start", "end")])), end="")
    flowchart TD
        start
        n1["end"]
        start --> n1
    """
    direction = direction.upper()
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}, got {direction!r}")
    ids: dict[Any, str] = {}
    used: set[str] = set()
    for n in g._node:
        text = str(n)
        if not isinstance(n, bool) and _SAFE_ID.match(text) and text.lower() not in _RESERVED and text not in used:
            ids[n] = text
            used.add(text)
    for i, n in enumerate(g._node):
        if n not in ids:
            candidate = f"n{i}"
            while candidate in used:
                candidate += "_"
            ids[n] = candidate
            used.add(candidate)
    lines = [f"flowchart {direction}"]
    for n, attrs in g._node.items():
        shown = attrs.get("label" if label is None else label)
        text = str(n) if shown is None else str(shown)
        shape = attrs.get("shape")
        shape = _SHAPE_ALIASES.get(shape, shape) if isinstance(shape, str) else None
        if shape is None and text == ids[n]:
            lines.append(f"    {ids[n]}")
        else:
            opener, closer = _SHAPES.get(shape, _SHAPES["rect"])  # type: ignore[arg-type]
            lines.append(f'    {ids[n]}{opener}"{_escape(text)}"{closer}')
    for u, v, attrs in g._iter_edges():
        style = attrs.get("style") if attrs.get("style") in ("dotted", "thick") else "solid"
        head = _HEAD_OF_ARROW.get(attrs.get("arrow"), ">" if g.directed else "")
        link = _LINK_TEXT[style, head]
        text = attrs.get(edge_label) if edge_label is not None else None
        if text is not None:
            link += f'|"{_escape(str(text))}"|'
        lines.append(f"    {ids[u]} {link} {ids[v]}")
    return "\n".join(lines) + "\n"


def write_mermaid(g: Graph, path: PathLike, **kwargs: Any) -> None:
    """Write :func:`to_mermaid` output to *path* (UTF-8)."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(to_mermaid(g, **kwargs))


# ---------------------------------------------------------------------- #
# reader
# ---------------------------------------------------------------------- #
def _unescape(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    return re.sub(
        r"#(\d+|[A-Za-z]+);",
        lambda m: html.unescape(f"&#{m.group(1)};" if m.group(1).isdigit() else f"&{m.group(1)};"),
        text,
    )


class _Cursor:
    """Position in one source line; errors report the line and 1-based column."""

    def __init__(self, text: str, line: int, pos: int = 0) -> None:
        self.text = text
        self.line = line
        self.pos = pos

    def error(self, message: str) -> ValueError:
        return ValueError(f"Mermaid syntax error at line {self.line}, column {self.pos + 1}: {message}")

    def skip_ws(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] in " \t":
            self.pos += 1

    def at_end(self) -> bool:
        self.skip_ws()
        return self.pos >= len(self.text)

    def peek(self, s: str) -> bool:
        return self.text.startswith(s, self.pos)

    def match(self, pattern: re.Pattern) -> re.Match | None:
        m = pattern.match(self.text, self.pos)
        if m:
            self.pos = m.end()
        return m


def from_mermaid(text: str, *, node_names: str = "id") -> Graph:
    """Parse a Mermaid flowchart (the subset described in the module docstring).

    Parameters
    ----------
    node_names:
        ``"id"`` names nodes by their Mermaid id (``A`` in ``A[Start]``) and
        keeps the text as the ``label`` attribute; ``"label"`` names them by
        their text when they have one, which recovers the node names of a
        :func:`to_mermaid` export.

    The flow direction is stored as ``g.attrs["direction"]``. Unsupported
    syntax raises ``ValueError`` with the line and column.

    >>> g = from_mermaid("flowchart LR\\n  A[Start] --> B{Ok?} -->|yes| C((Done))")
    >>> g.nodes["B"]
    {'label': 'Ok?', 'shape': 'diamond'}
    """
    if node_names not in ("id", "label"):
        raise ValueError(f"node_names must be 'id' or 'label', got {node_names!r}")
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[tuple[str, str, dict[str, Any]]] = []
    direction: str | None = None
    clusters: list[str] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        s = raw.strip()
        if not s or s.startswith("%%"):
            continue
        indent = len(raw) - len(raw.lstrip())
        if direction is None:
            m = _HEADER.match(s)
            if m is None:
                raise ValueError(
                    f"Mermaid syntax error at line {line_no}, column {indent + 1}: "
                    f"expected a 'flowchart' or 'graph' header, found {s!r}"
                )
            direction = (m.group(1) or "TD").upper()
            cursor = _Cursor(raw, line_no, indent + m.end())
        else:
            word = s.split(None, 1)[0]
            if word == "subgraph":
                named = _SUBGRAPH_NAME.search(s, len(word))
                clusters.append(next(x for x in named.groups() if x is not None) if named else f"subgraph{len(clusters)}")
                continue
            if s.rstrip("; \t") == "end":
                if not clusters:
                    raise ValueError(f"Mermaid syntax error at line {line_no}: 'end' without a matching 'subgraph'")
                clusters.pop()
                continue
            if _SKIP.match(s):
                continue
            cursor = _Cursor(raw, line_no, indent)
        cluster = clusters[-1] if clusters else None
        while not cursor.at_end():
            _statement(cursor, nodes, edges, cluster)
            cursor.skip_ws()
            if cursor.peek(";"):
                cursor.pos += 1
            elif not cursor.at_end():
                raise cursor.error(f"unexpected {cursor.text[cursor.pos:]!r}")
    if direction is None:
        raise ValueError("empty Mermaid text: expected a 'flowchart' or 'graph' header")
    heads = [attrs.pop("_head") for _, _, attrs in edges]
    directed = any(heads)
    g: Graph = DiGraph() if directed else Graph()
    g.attrs["direction"] = direction
    names: dict[str, Any] = {}
    for mid, attrs in nodes.items():
        if node_names == "label" and "label" in attrs:
            attrs = dict(attrs)
            name = attrs.pop("label")
        else:
            name = mid
        if name in g._node:
            raise ValueError(f"two Mermaid nodes would both be named {name!r}; use node_names='id'")
        names[mid] = name
        g.add_node(name, **attrs)
    for (u, v, attrs), head in zip(edges, heads):
        if directed and not head:
            attrs["arrow"] = "none"
        if not directed:
            attrs.pop("arrow", None)
        g.add_edge(names[u], names[v], **attrs)
    return g


def _statement(cur: _Cursor, nodes: dict, edges: list, cluster: str | None) -> None:
    """``group (link group)*``: one node statement or edge chain."""
    left = _group(cur, nodes, cluster)
    while True:
        cur.skip_ws()
        link = _link(cur)
        if link is None:
            return
        attrs, both_ways, invisible = link
        right = _group(cur, nodes, cluster)
        if not invisible:
            for u in left:
                for v in right:
                    edges.append((u, v, dict(attrs)))
                    if both_ways:
                        edges.append((v, u, dict(attrs)))
        left = right


def _group(cur: _Cursor, nodes: dict, cluster: str | None) -> list[str]:
    """``node ('&' node)*``."""
    group = [_node(cur, nodes, cluster)]
    while True:
        cur.skip_ws()
        if not cur.peek("&"):
            return group
        cur.pos += 1
        group.append(_node(cur, nodes, cluster))


def _node(cur: _Cursor, nodes: dict, cluster: str | None) -> str:
    """``id [shape] [:::class]``; records label, shape, class and cluster."""
    cur.skip_ws()
    m = cur.match(_NODE_ID)
    if m is None:
        raise cur.error("expected a node id")
    mid = m.group()
    attrs = nodes.setdefault(mid, {})
    if cluster is not None:
        attrs.setdefault("cluster", cluster)
    for opener in _OPENERS:
        if cur.peek(opener):
            cur.pos += len(opener)
            text, closer = _shape_text(cur, opener)
            attrs["label"] = _unescape(text)
            attrs["shape"] = next(name for name, oc in _SHAPES.items() if oc == (opener, closer))
            break
    if cur.peek(":::"):
        cur.pos += 3
        cls = cur.match(_CLASS)
        if cls is None:
            raise cur.error("expected a class name after ':::'")
        attrs["class"] = cls.group()
    return mid


def _shape_text(cur: _Cursor, opener: str) -> tuple[str, str]:
    """Text inside a node shape (quoted or bare) and the closer that ended it."""
    closers = [c for o, c in _SHAPES.values() if o == opener]
    if cur.peek('"'):
        end = cur.text.find('"', cur.pos + 1)
        if end < 0:
            raise cur.error("unterminated quoted label")
        text = cur.text[cur.pos + 1 : end]
        cur.pos = end + 1
        for closer in closers:
            if cur.peek(closer):
                cur.pos += len(closer)
                return text, closer
        raise cur.error(f"expected {' or '.join(map(repr, closers))} after the label")
    found = [(k, c) for c in closers if (k := cur.text.find(c, cur.pos)) >= 0]
    if not found:
        raise cur.error(f"unterminated node shape {opener!r}")
    k, closer = min(found)
    text = cur.text[cur.pos : k].strip()
    cur.pos = k + len(closer)
    return text, closer


def _kind(m: re.Match) -> str:
    groups = m.groupdict()
    return next(k for k in _KINDS if groups.get(k))


def _link(cur: _Cursor) -> tuple[dict[str, Any], bool, bool] | None:
    """A link at the cursor as ``(edge_attrs, both_ways, invisible)``, or ``None``."""
    start = cur.pos
    m = cur.match(_TEXT_LINK_OPEN)
    if m is not None:  # "-- text -->" form: the text ends at a closing link preceded by a blank
        kind = _kind(m)
        closer = _TEXT_LINK_CLOSE[kind]
        for k in range(cur.pos + 1, len(cur.text)):
            if cur.text[k - 1] in " \t" and (cm := closer.match(cur.text, k)):
                label = cur.text[cur.pos : k].strip()
                cur.pos = cm.end()
                return _link_attrs(kind, cm.group("head"), label, m.group("start") is not None)
        cur.pos = start
    m = cur.match(_LINK)
    if m is None:
        return None
    kind = _kind(m)
    head = m.group("head")
    if kind == "solid" and len(m.group("solid")) == 2 and head is None:
        cur.pos = start
        raise cur.error("'--' must be followed by '-', '>', 'o', 'x' or by link text and a closing link")
    label = None
    cur.skip_ws()
    if cur.peek("|"):
        end = cur.text.find("|", cur.pos + 1)
        if end < 0:
            raise cur.error("unterminated |link text|")
        label = cur.text[cur.pos + 1 : end].strip()
        if len(label) >= 2 and label[0] == label[-1] == '"':
            label = label[1:-1]
        cur.pos = end + 1
    return _link_attrs(kind, head, label, m.group("start") is not None)


def _link_attrs(kind: str, head: str | None, label: str | None, start: bool) -> tuple[dict[str, Any], bool, bool]:
    attrs: dict[str, Any] = {"_head": head is not None}
    if kind == "invisible":
        return attrs, False, True
    if kind in ("dotted", "thick"):
        attrs["style"] = kind
    if head in _ARROW_OF_HEAD:
        attrs["arrow"] = _ARROW_OF_HEAD[head]
    if label is not None:
        attrs["label"] = _unescape(label)
    return attrs, start and head == ">", False


def read_mermaid(path: PathLike, *, node_names: str = "id") -> Graph:
    """Read a Mermaid flowchart file; see :func:`from_mermaid`."""
    with open(path, encoding="utf-8-sig") as fh:
        return from_mermaid(fh.read(), node_names=node_names)


__all__ = ["to_mermaid", "from_mermaid", "write_mermaid", "read_mermaid"]
