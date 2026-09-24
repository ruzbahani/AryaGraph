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

"""Graphviz DOT: a writer and a complete parser for the DOT language.

The parser implements the grammar of https://graphviz.org/doc/info/lang.html:
``strict``, ``graph``/``digraph``, node and edge statements, edge chains
``a -> b -> c``, subgraph endpoints ``{a b} -> c``, attribute lists,
``graph``/``node``/``edge`` default statements (inherited by nested scopes,
applied to nodes and edges created afterwards), ``k=v`` statements, nested
subgraphs, ports (``a:p:n``), comments (``//``, ``/* */`` and ``#`` lines),
quoted strings with ``+`` concatenation, HTML strings and numerals.

Mapping onto AryaGraph:

* Subgraphs are flattened. A node inside a *cluster* (a subgraph named
  ``cluster…``) gets the node attribute ``cluster`` = the innermost cluster's
  name, and each cluster's own attributes are kept in
  ``g.attrs["clusters"][name]``. Graph attributes of other subgraphs (such
  as ``rank=same``) are dropped.
* Unquoted numerals become ``int`` (canonical integers) or ``float`` (with a
  decimal point); every other ID, including quoted ``"42"``, is a string.
  HTML strings keep their angle brackets (``"<b>x</b>"`` → ``'<<b>x</b>>'``)
  so the writer can emit them as HTML again.
* Ports become the edge attributes ``tailport`` / ``headport``; parallel
  edges merge (later attributes win).

Quoted strings follow Graphviz's lexer exactly: ``\\"`` is a quote, a
backslash before a newline continues the line, every other backslash is kept
verbatim (Graphviz interprets ``\\n``, ``\\l`` … when it renders labels).
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np

from ..core.dag import DAG
from ..core.exceptions import AryaGraphError
from ..core.graph import DiGraph, Graph
from .jsonio import _jsonable

PathLike = str | os.PathLike

_KEYWORDS = frozenset({"strict", "graph", "digraph", "subgraph", "node", "edge"})
_ID_START = "A-Za-z_\u0080-\U0010ffff"
_ID_RE = re.compile(f"[{_ID_START}][{_ID_START}0-9]*")
_ID_FULL = re.compile(f"[{_ID_START}][{_ID_START}0-9]*\\Z")
_NUMERAL_RE = re.compile(r"-?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)")
_CANON_INT = re.compile(r"(?:0|-?[1-9][0-9]*)\Z")
_ID_CHAR = re.compile(f"[{_ID_START}0-9.]")
_ODD_BACKSLASHES = re.compile(r'(?<!\\)(?:\\\\)*\\(?=["\n]|\Z)')


class DotSyntaxError(AryaGraphError, ValueError):
    """Malformed DOT text; ``line`` and ``column`` (1-based) locate the problem."""

    def __init__(self, message: str, line: int, column: int) -> None:
        self.line = line
        self.column = column
        super().__init__(f"DOT syntax error at line {line}, column {column}: {message}")


# ---------------------------------------------------------------------- #
# lexer
# ---------------------------------------------------------------------- #
class _Tok(NamedTuple):
    kind: str  # "id", "num", "str", "html", "punct", "eof"
    value: str
    line: int
    col: int


def _tokenize(text: str) -> list[_Tok]:
    toks: list[_Tok] = []
    n = len(text)
    i = 0
    line = 1
    line_start = 0

    def newlines(a: int, b: int) -> None:
        nonlocal line, line_start
        k = text.count("\n", a, b)
        if k:
            line += k
            line_start = text.rindex("\n", a, b) + 1

    while i < n:
        c = text[i]
        if c == "\n":
            i += 1
            line += 1
            line_start = i
            continue
        if c in " \t\r\f\v﻿":
            i += 1
            continue
        col = i - line_start + 1
        if c == "#" and not text[line_start:i].strip():
            end = text.find("\n", i)
            i = n if end < 0 else end
            continue
        if text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            if end < 0:
                raise DotSyntaxError("unterminated /* comment", line, col)
            newlines(i, end + 2)
            i = end + 2
            continue
        if c in "{}[];,=:+":
            toks.append(_Tok("punct", c, line, col))
            i += 1
            continue
        if c == "-" and text[i + 1 : i + 2] in ("-", ">"):
            toks.append(_Tok("punct", text[i : i + 2], line, col))
            i += 2
            continue
        if c == '"':
            start_line = line
            j = i + 1
            buf: list[str] = []
            while True:
                if j >= n:
                    raise DotSyntaxError("unterminated quoted string", start_line, col)
                ch = text[j]
                if ch == '"':
                    j += 1
                    break
                if ch == "\\" and j + 1 < n:
                    nxt = text[j + 1]
                    if nxt == '"':
                        buf.append('"')
                        j += 2
                        continue
                    if nxt == "\\":
                        buf.append("\\\\")
                        j += 2
                        continue
                    if nxt == "\n":  # line continuation
                        j += 2
                        line += 1
                        line_start = j
                        continue
                if ch == "\n":
                    line += 1
                    line_start = j + 1
                buf.append(ch)
                j += 1
            toks.append(_Tok("str", "".join(buf), start_line, col))
            i = j
            continue
        if c == "<":
            depth = 0
            j = i
            while j < n:
                if text[j] == "<":
                    depth += 1
                elif text[j] == ">":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if j >= n:
                raise DotSyntaxError("unterminated HTML string (unbalanced '<' … '>')", line, col)
            toks.append(_Tok("html", text[i : j + 1], line, col))
            newlines(i, j + 1)
            i = j + 1
            continue
        m = _NUMERAL_RE.match(text, i)
        if m:
            j = m.end()
            if j < n and _ID_CHAR.match(text[j]):
                raise DotSyntaxError(f"badly delimited number {text[i:j + 1]!r}", line, col)
            toks.append(_Tok("num", m.group(), line, col))
            i = j
            continue
        m = _ID_RE.match(text, i)
        if m:
            toks.append(_Tok("id", m.group(), line, col))
            i = m.end()
            continue
        raise DotSyntaxError(f"unexpected character {c!r}", line, col)
    toks.append(_Tok("eof", "", line, i - line_start + 1))
    return toks


def _numeral(text: str) -> Any:
    """Canonical integers → int, numerals with a decimal point → float, others ('007', '-0') stay text."""
    if _CANON_INT.match(text):
        return int(text)
    if "." in text:
        return float(text)
    return text


# ---------------------------------------------------------------------- #
# parser
# ---------------------------------------------------------------------- #
@dataclass
class _Scope:
    node_defaults: dict = field(default_factory=dict)
    edge_defaults: dict = field(default_factory=dict)
    graph_attrs: dict = field(default_factory=dict)


class _Parser:
    def __init__(self, text: str) -> None:
        self.toks = _tokenize(text)
        self.i = 0
        self.g: Graph = Graph()
        self.directed = False
        self.clusters: dict[str, dict] = {}
        self.cluster_of: dict[Any, str] = {}

    # -- token helpers ---------------------------------------------------
    def peek(self, k: int = 0) -> _Tok:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def next(self) -> _Tok:
        tok = self.peek()
        self.i += 1
        return tok

    @staticmethod
    def _is(tok: _Tok, punct: str) -> bool:
        return tok.kind == "punct" and tok.value == punct

    @staticmethod
    def _is_kw(tok: _Tok, *words: str) -> bool:
        return tok.kind == "id" and tok.value.lower() in words

    @staticmethod
    def _is_atom(tok: _Tok) -> bool:
        return tok.kind in ("str", "num", "html") or (tok.kind == "id" and tok.value.lower() not in _KEYWORDS)

    def error(self, message: str, tok: _Tok | None = None) -> DotSyntaxError:
        tok = tok or self.peek()
        return DotSyntaxError(message, tok.line, tok.col)

    @staticmethod
    def describe(tok: _Tok) -> str:
        if tok.kind == "eof":
            return "end of input"
        if tok.kind == "str":
            return f"string {tok.value!r}"
        return repr(tok.value)

    def expect(self, punct: str) -> _Tok:
        tok = self.next()
        if not self._is(tok, punct):
            raise self.error(f"expected {punct!r} but found {self.describe(tok)}", tok)
        return tok

    def atom(self, what: str = "an identifier") -> Any:
        """One ID: identifier, numeral, HTML string or ``+``-joined quoted strings."""
        tok = self.next()
        if tok.kind == "str":
            parts = [tok.value]
            while self._is(self.peek(), "+"):
                self.next()
                more = self.next()
                if more.kind != "str":
                    raise self.error(f"'+' must join two quoted strings, found {self.describe(more)}", more)
                parts.append(more.value)
            return "".join(parts)
        if tok.kind == "num":
            return _numeral(tok.value)
        if tok.kind == "html":
            return tok.value
        if tok.kind == "id":
            if tok.value.lower() in _KEYWORDS:
                raise self.error(f"{tok.value!r} is a keyword; quote it to use it as {what}", tok)
            return tok.value
        raise self.error(f"expected {what} but found {self.describe(tok)}", tok)

    # -- grammar -----------------------------------------------------------
    def parse(self) -> Graph:
        tok = self.next()
        if self._is_kw(tok, "strict"):
            tok = self.next()
        if not self._is_kw(tok, "graph", "digraph"):
            raise self.error(f"expected 'graph' or 'digraph' but found {self.describe(tok)}", tok)
        self.directed = tok.value.lower() == "digraph"
        self.g = DiGraph() if self.directed else Graph()
        name = None
        if not self._is(self.peek(), "{"):
            name = self.atom("the graph name")
        self.expect("{")
        root = _Scope(graph_attrs=self.g.attrs)
        self.stmt_list(root, {})
        self.expect("}")
        tail = self.peek()
        if tail.kind != "eof":
            if self._is_kw(tail, "graph", "digraph", "strict"):
                raise self.error("the text holds several graphs; only one is supported", tail)
            raise self.error(f"unexpected {self.describe(tail)} after the end of the graph", tail)
        if name is not None:
            self.g.attrs["name"] = str(name)
        for n, cname in self.cluster_of.items():
            self.g._node[n]["cluster"] = cname
        if self.clusters:
            self.g.attrs["clusters"] = self.clusters
        return self.g

    def stmt_list(self, scope: _Scope, members: dict) -> None:
        while True:
            tok = self.peek()
            if self._is(tok, "}") or tok.kind == "eof":
                return
            if self._is(tok, ";"):
                self.next()
                continue
            self.stmt(scope, members)

    def stmt(self, scope: _Scope, members: dict) -> None:
        tok = self.peek()
        if self._is_kw(tok, "graph", "node", "edge"):
            self.next()
            if not self._is(self.peek(), "["):
                raise self.error(f"expected '[' after {tok.value!r}")
            attrs = self.attr_lists()
            kind = tok.value.lower()
            target = scope.graph_attrs if kind == "graph" else scope.node_defaults if kind == "node" else scope.edge_defaults
            target.update(attrs)
            return
        if self._is_atom(tok) and self._is(self.peek(1), "="):
            key = str(self.atom("an attribute name"))
            self.next()
            scope.graph_attrs[key] = self.atom("an attribute value")
            return
        operands = [self.operand(scope, members, "a statement")]
        while self.peek().kind == "punct" and self.peek().value in ("->", "--"):
            op = self.next()
            if (op.value == "->") != self.directed:
                want = "->" if self.directed else "--"
                raise self.error(f"{op.value!r} in {'a digraph' if self.directed else 'an undirected graph'}; use {want!r}", op)
            operands.append(self.operand(scope, members, f"a node or subgraph after {op.value!r}"))
        if len(operands) > 1:
            attrs = self.attr_lists()
            for (left, _), (right, _) in zip(operands, operands[1:]):
                for u, uport in left:
                    for v, vport in right:
                        edge = {**scope.edge_defaults, **attrs}
                        if uport is not None:
                            edge["tailport"] = uport
                        if vport is not None:
                            edge["headport"] = vport
                        self.g.add_edge(u, v, **edge)
            return
        endpoints, is_subgraph = operands[0]
        if self._is(self.peek(), "["):
            if is_subgraph:
                raise self.error("a subgraph cannot take an attribute list")
            self.g._node[endpoints[0][0]].update(self.attr_lists())

    def operand(self, scope: _Scope, members: dict, expected: str) -> tuple[list[tuple[Any, str | None]], bool]:
        tok = self.peek()
        if self._is_kw(tok, "subgraph") or self._is(tok, "{"):
            return [(n, None) for n in self.subgraph(scope, members)], True
        if not self._is_atom(tok):
            if tok.kind == "id":
                raise self.error(f"{tok.value!r} is a keyword; quote it to use it as a node name")
            raise self.error(f"expected {expected} but found {self.describe(tok)}")
        node = self.atom("a node name")
        port = None
        if self._is(self.peek(), ":"):
            self.next()
            port = str(self.atom("a port"))
            if self._is(self.peek(), ":"):
                self.next()
                port += ":" + str(self.atom("a compass point"))
        if node not in self.g._node:
            self.g.add_node(node, **scope.node_defaults)
        members[node] = None
        return [(node, port)], False

    def subgraph(self, scope: _Scope, members: dict) -> list:
        name = None
        if self._is_kw(self.peek(), "subgraph"):
            self.next()
            if not self._is(self.peek(), "{"):
                name = self.atom("a subgraph name")
        self.expect("{")
        child = _Scope(dict(scope.node_defaults), dict(scope.edge_defaults), {})
        inner: dict = {}
        self.stmt_list(child, inner)
        self.expect("}")
        is_cluster = name is not None and (
            str(name).lower().startswith("cluster") or str(child.graph_attrs.get("cluster", "")).lower() == "true"
        )
        if is_cluster:
            cname = str(name)
            self.clusters.setdefault(cname, {}).update(child.graph_attrs)
            for n in inner:  # inner clusters finished first, so they keep their claim
                self.cluster_of.setdefault(n, cname)
        members.update(inner)
        return list(inner)

    def attr_lists(self) -> dict:
        attrs: dict = {}
        while self._is(self.peek(), "["):
            self.next()
            while not self._is(self.peek(), "]"):
                if self.peek().kind == "eof":
                    raise self.error("unterminated attribute list: expected ']'")
                key = str(self.atom("an attribute name"))
                self.expect("=")
                attrs[key] = self.atom("an attribute value")
                if self.peek().kind == "punct" and self.peek().value in (",", ";"):
                    self.next()
            self.expect("]")
        return attrs


def from_dot(text: str, *, dag: bool = False) -> Graph:
    """Parse DOT text into a :class:`Graph` (``graph``) or :class:`DiGraph` (``digraph``).

    With *dag*, a ``digraph`` is returned as a :class:`DAG` (raising
    :class:`CycleError` on a cycle). Syntax errors raise
    :class:`DotSyntaxError` (a ``ValueError``) with the line and column.

    >>> g = from_dot('digraph { a -> {b c} [color=red]; b -> c }')
    >>> sorted(g.edges)
    [('a', 'b'), ('a', 'c'), ('b', 'c')]
    """
    g = _Parser(text).parse()
    if dag:
        if not g.directed:
            raise ValueError("dag=True needs a 'digraph'")
        return DAG(g)
    return g


def read_dot(path: PathLike, *, dag: bool = False, encoding: str = "utf-8") -> Graph:
    """Read a DOT file; see :func:`from_dot`."""
    with open(path, encoding="utf-8-sig" if encoding.lower().replace("-", "") == "utf8" else encoding) as fh:
        return from_dot(fh.read(), dag=dag)


# ---------------------------------------------------------------------- #
# writer
# ---------------------------------------------------------------------- #
def _html_like(s: str) -> bool:
    if len(s) < 2 or s[0] != "<" or s[-1] != ">":
        return False
    depth = 0
    for i, c in enumerate(s):
        if c == "<":
            depth += 1
        elif c == ">":
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return False
        if depth < 0:
            return False
    return depth == 0


def _quote(s: str) -> str:
    if _ODD_BACKSLASHES.search(s):
        raise ValueError(
            f"{s!r} cannot be written in DOT: Graphviz reads an odd run of backslashes before a quote, "
            "a line break or the end of a string as an escape"
        )
    return '"' + s.replace('"', '\\"') + '"'


def _dot_id(s: str) -> str:
    """*s* as a DOT ID: bare when it is a plain identifier, HTML when it looks like ``<…>``, else quoted."""
    if _ID_FULL.match(s) and s.lower() not in _KEYWORDS:
        return s
    if _html_like(s):
        return s
    return _quote(s)


def _dot_number(x: float) -> str:
    if math.isfinite(x):
        return np.format_float_positional(x, trim="0")
    return _quote(repr(x))


def _dot_value(value: Any) -> str | None:
    """DOT text for an attribute value or node id (``None`` → omit)."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _dot_number(value)
    if isinstance(value, str):
        return _dot_id(value)
    if isinstance(value, (tuple, list, np.ndarray)) and all(
        isinstance(x, (int, float, np.number)) and not isinstance(x, bool) for x in value
    ):
        return _quote(",".join(_dot_number(float(x)) if isinstance(x, float) else str(x) for x in value))
    return _quote(json.dumps(_jsonable(value), ensure_ascii=False))


def _node_text(n: Any) -> str:
    if isinstance(n, tuple):
        return _quote(str(n))
    text = _dot_value(n)
    assert text is not None  # nodes are never None
    return text


def _attr_text(attrs: Mapping[str, Any]) -> str:
    parts = []
    for k, v in attrs.items():
        text = _dot_value(v)
        if text is not None:
            parts.append(f"{_dot_id(str(k))}={text}")
    return ", ".join(parts)


def _select(attrs: Mapping[str, Any], which: bool | Iterable[str], skip: tuple = ()) -> dict:
    if which is True:
        return {k: v for k, v in attrs.items() if k not in skip}
    if which is False:
        return {}
    wanted = set(which)
    return {k: v for k, v in attrs.items() if k in wanted and k not in skip}


def to_dot(
    g: Graph,
    *,
    name: str | None = None,
    node_attrs: bool | Iterable[str] = True,
    edge_attrs: bool | Iterable[str] = True,
    rankdir: str | None = None,
) -> str:
    """DOT text for *g*.

    Parameters
    ----------
    name:
        Graph ID (defaults to ``g.name``).
    node_attrs, edge_attrs:
        Write all attributes, none, or the named ones. Numbers are written as
        DOT numerals, booleans as ``true``/``false``, sequences of numbers as
        ``"x,y"`` (Graphviz's point syntax) and other containers as JSON.
    rankdir:
        Graphviz layout direction (``"TB"``, ``"LR"``, …), written as a graph attribute.

    Notes
    -----

    Nodes with a ``cluster`` attribute are grouped into ``subgraph cluster_…``
    blocks (with the attributes in ``g.attrs["clusters"]``), which is exactly
    what :func:`from_dot` produces, so clusters round-trip. Scalar graph
    attributes are written in a ``graph [...]`` statement.

    Examples
    --------

    >>> print(to_dot(DiGraph([("a", "b b")])), end="")
    digraph {
      a;
      "b b";
      a -> "b b";
    }
    """
    op = "->" if g.directed else "--"
    gname = name if name is not None else (g.name or None)
    header = "digraph" if g.directed else "graph"
    lines = [f"{header} {_dot_id(str(gname))} {{" if gname else f"{header} {{"]
    gattrs = {
        k: v
        for k, v in g.attrs.items()
        if k not in ("name", "clusters") and isinstance(_jsonable(v), (str, int, float, bool))
    }
    if rankdir is not None:
        gattrs["rankdir"] = rankdir
    if gattrs:
        lines.append(f"  graph [{_attr_text(gattrs)}];")

    use_clusters = node_attrs is True or (not isinstance(node_attrs, bool) and "cluster" in set(node_attrs))
    groups: dict[Any, list[str]] = {}
    for n, attrs in g._node.items():
        shown = _attr_text(_select(attrs, node_attrs, skip=("cluster",) if use_clusters else ()))
        stmt = _node_text(n) + (f" [{shown}]" if shown else "") + ";"
        cluster = attrs.get("cluster") if use_clusters else None
        if cluster is None:
            lines.append("  " + stmt)
        else:
            groups.setdefault(cluster, []).append(stmt)
    cluster_attrs = g.attrs.get("clusters") if isinstance(g.attrs.get("clusters"), Mapping) else {}
    for cluster, stmts in groups.items():
        cname = str(cluster) if str(cluster).lower().startswith("cluster") else f"cluster_{cluster}"
        lines.append(f"  subgraph {_dot_id(cname)} {{")
        extra = cluster_attrs.get(cname) or cluster_attrs.get(cluster) or {}
        if isinstance(extra, Mapping) and extra:
            lines.append(f"    graph [{_attr_text(extra)}];")
        lines.extend("    " + s for s in stmts)
        lines.append("  }")
    for u, v, attrs in g._iter_edges():
        shown = _attr_text(_select(attrs, edge_attrs))
        lines.append(f"  {_node_text(u)} {op} {_node_text(v)}" + (f" [{shown}]" if shown else "") + ";")
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_dot(g: Graph, path: PathLike, **kwargs: Any) -> None:
    """Write :func:`to_dot` output to *path* (UTF-8); keyword arguments are passed through."""
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(to_dot(g, **kwargs))


__all__ = ["to_dot", "write_dot", "from_dot", "read_dot", "DotSyntaxError"]
