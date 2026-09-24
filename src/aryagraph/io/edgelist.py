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

"""Edge lists (CSV/TSV/whitespace) and adjacency lists.

Both formats share one tokenizer with a simple, predictable typing rule:

* a **quoted** field (``"..."``, with ``""`` for a literal quote) is always a
  string (this is how ``"42"`` stays the string ``'42'``);
* an **unquoted** field is parsed as an ``int`` when it is a canonical
  integer (``"007"`` stays text) or a ``float`` when it has a decimal point
  or an exponent; in attribute columns ``true``/``false`` become booleans
  and ``nan``/``inf`` floats; anything else is a string;
* an empty unquoted attribute field means "attribute absent".

The writers quote exactly the strings that would otherwise be misread
(numeric-looking, empty, containing the delimiter, quotes, line breaks,
leading/trailing blanks, or starting with the comment character), so files
round-trip. Node ids should be strings or numbers; other values (tuples,
lists, …) are written as JSON text and read back as strings.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Callable, Mapping
from typing import Any, Sequence

import numpy as np

from ..core.dag import DAG
from ..core.graph import DiGraph, Graph
from .jsonio import _jsonable

PathLike = str | os.PathLike
Field = tuple[str, bool]  # (text, was_quoted)
Types = Callable[[str], Any] | Mapping[Any, Callable[[str], Any]] | None

_INT = re.compile(r"-?(?:0|[1-9][0-9]*)\Z")
_FLOAT = re.compile(r"-?(?:(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|[0-9]+[eE][+-]?[0-9]+)\Z")
_NUMERIC_LIKE = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
_SPECIAL_FLOATS = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf, "+inf": math.inf}
_BOOLS = {"true": True, "false": False, "True": True, "False": False}
_HEADER_WORDS = {
    "source", "target", "from", "to", "src", "dst", "u", "v", "node", "node1", "node2",
    "head", "tail", "weight", "id", "source_id", "target_id",
}  # fmt: skip
_SNIFF_CANDIDATES = (",", "\t", ";", "|")


# ---------------------------------------------------------------------- #
# typing
# ---------------------------------------------------------------------- #
def _auto(text: str, node: bool) -> Any:
    """Auto-typed value of an unquoted field (``node`` columns never become bools/NaN)."""
    if _INT.match(text):
        return int(text)
    if _FLOAT.match(text):
        return float(text)
    if not node:
        if text in _BOOLS:
            return _BOOLS[text]
        special = _SPECIAL_FLOATS.get(text.lower())
        if special is not None:
            return special
    return text


def _format(value: Any, delimiter: str | None, node: bool) -> str:
    """Text of one field, quoted when a reader would otherwise mistype or mis-split it."""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ""
    if isinstance(value, bool):
        return ("true" if value else "false") if not node else _quote(str(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if not isinstance(value, str):
        value = json.dumps(_jsonable(value), ensure_ascii=False)
    return _quote(value) if _needs_quote(value, delimiter, node) else value


def _needs_quote(s: str, delimiter: str | None, node: bool) -> bool:
    if not s or s != s.strip() or s.startswith("#") or '"' in s or "\n" in s or "\r" in s:
        return True
    if delimiter is None:
        if " " in s or "\t" in s:
            return True
    elif delimiter in s:
        return True
    # Also quote numerals AryaGraph would keep as text ("007", "+5"): other tools read them as numbers.
    return _NUMERIC_LIKE.match(s) is not None or not isinstance(_auto(s, node), str)


def _quote(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


# ---------------------------------------------------------------------- #
# tokenizer
# ---------------------------------------------------------------------- #
def _records(
    text: str,
    delimiter: str | None,
    comments: str | None,
    limit: int | None = None,
) -> list[tuple[int, list[Field]]]:
    """Split *text* into ``(line_number, fields)`` records, skipping blank and comment lines.

    ``delimiter=None`` splits on runs of spaces/tabs; otherwise on that single
    character, stripping blanks around unquoted fields. Quoted fields may
    contain the delimiter, doubled quotes and line breaks. Stops after
    *limit* records when given.
    """
    out: list[tuple[int, list[Field]]] = []
    n = len(text)
    pos = 0
    line = 1
    line_start = 0
    blanks = " \t" if delimiter != "\t" else " "

    def error(msg: str, at: int) -> ValueError:
        return ValueError(f"line {line}, column {at - line_start + 1}: {msg}")

    def quoted(start: int) -> tuple[str, int]:
        nonlocal line, line_start
        parts = []
        i = start + 1
        while True:
            j = text.find('"', i)
            if j < 0:
                raise ValueError(f"line {line}, column {start - line_start + 1}: unterminated quoted field")
            chunk = text[i:j]
            k = chunk.rfind("\n")
            if k >= 0:
                line += chunk.count("\n")
                line_start = i + k + 1
            parts.append(chunk)
            if j + 1 < n and text[j + 1] == '"':
                parts.append('"')
                i = j + 2
                continue
            return "".join(parts), j + 1

    while pos < n:
        rec_line = line
        fields: list[Field] = []
        while True:
            while pos < n and text[pos] in blanks:
                pos += 1
            if delimiter is None and (pos >= n or text[pos] == "\n"):
                break
            if comments and text.startswith(comments, pos):
                end = text.find("\n", pos)
                pos = n if end < 0 else end
                if delimiter is None:
                    continue
                break
            if pos < n and text[pos] == '"':
                value, pos = quoted(pos)
                fields.append((value, True))
                if delimiter is None:
                    if pos < n and text[pos] not in " \t\n":
                        raise error("unexpected character after a quoted field", pos)
                else:
                    while pos < n and text[pos] in blanks:
                        pos += 1
            else:
                start = pos
                if delimiter is None:
                    while pos < n and text[pos] not in " \t\n":
                        pos += 1
                else:
                    while pos < n and text[pos] != delimiter and text[pos] != "\n":
                        pos += 1
                fields.append((text[start:pos].strip(blanks), False))
            if pos >= n or text[pos] == "\n":
                break
            if delimiter is None:
                continue  # at a blank, skipped at the top of the loop
            if text[pos] == delimiter:
                pos += 1
            else:
                raise error(f"expected {delimiter!r} after a quoted field", pos)
        if pos < n and text[pos] == "\n":
            pos += 1
            line += 1
            line_start = pos
        if fields and fields != [("", False)]:
            out.append((rec_line, fields))
            if limit is not None and len(out) >= limit:
                break
    return out


def _sniff(text: str, comments: str | None) -> str | None:
    """Delimiter of *text* judged on its first 20 records (``None`` = runs of whitespace).

    The first candidate that splits every sampled record into the same number
    (≥ 2) of fields wins; failing that, the first that splits every record at
    all. A candidate that occurs on every sampled line but cannot be parsed
    signals a malformed file, and its error is raised.
    """
    head = text[:65536].splitlines()[:200]
    lines = [s for s in (raw.strip() for raw in head) if s and not (comments and s.startswith(comments))]
    splits: dict[str, list[int]] = {}
    for cand in _SNIFF_CANDIDATES:
        try:
            recs = _records(text, cand, comments, limit=20)
        except ValueError:
            if lines and all(cand in s for s in lines[:20]):
                raise
            continue
        if recs:
            splits[cand] = [len(fields) for _, fields in recs]
    for cand, counts in splits.items():
        if len(set(counts)) == 1 and counts[0] >= 2:
            return cand
    for cand, counts in splits.items():
        if min(counts) >= 2:
            return cand
    return None


def _read_text(path: PathLike, encoding: str) -> str:
    with open(path, encoding="utf-8-sig" if encoding.lower().replace("-", "") == "utf8" else encoding) as fh:
        return fh.read()


def _converter(types: Types, column: int, name: str | None) -> Callable[[str], Any] | None:
    if types is None:
        return None
    if isinstance(types, Mapping):
        if name is not None and name in types:
            return types[name]
        return types.get(column)
    if callable(types):
        return types
    raise TypeError(f"types must be None, a callable or a mapping, got {types!r}")


def _value(field: Field, conv: Callable[[str], Any] | None, node: bool, line_no: int = 0) -> Any:
    text, was_quoted = field
    if conv is not None:
        try:
            return conv(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"line {line_no}: cannot convert {text!r}: {exc}") from None
    return text if was_quoted else _auto(text, node)


def _new_graph(directed: bool, dag: bool) -> Graph:
    return DAG() if dag else DiGraph() if directed else Graph()


# ---------------------------------------------------------------------- #
# edge lists
# ---------------------------------------------------------------------- #
def read_edgelist(
    path: PathLike,
    *,
    delimiter: str | None = None,
    directed: bool = False,
    dag: bool = False,
    header: bool | str = "auto",
    source: int | str = 0,
    target: int | str = 1,
    types: Types = None,
    comments: str | None = "#",
    encoding: str = "utf-8",
) -> Graph:
    """Read an edge list: one edge per line, extra columns become edge attributes.

    Parameters
    ----------
    delimiter:
        Field separator. ``None`` sniffs it: the first of ``,``, tab, ``;``,
        ``|`` that splits each of the first 20 records into the same number
        (≥ 2) of fields (or, failing that, splits every one of them), else
        runs of whitespace. ``" "`` also means runs of whitespace.
    directed, dag:
        Build a :class:`DiGraph` / :class:`DAG` (``dag`` implies directed).
    header:
        ``True``, ``False`` or ``"auto"``: the first line is a header when all
        its fields are text and either one is a usual column name (``source``,
        ``target``, ``from``, ``to``, ``weight``, …) or some column is numeric
        in the data lines but not in the first line. Naming *source* or
        *target* by name implies a header. Without a header, the first extra
        column is called ``weight`` and later ones ``col<index>``.
    source, target:
        Column index or header name of the endpoints.
    types:
        ``None`` for the automatic typing described in the module docstring,
        a callable applied to the raw text of every field (``types=str``
        keeps everything as text), or a mapping from column name/index to a
        callable (other columns stay automatic).

    A line whose target field is empty or missing declares an isolated node
    (this is how :func:`write_edgelist` stores isolates). Repeated edges merge
    their attributes, later lines winning.

    Examples
    --------
    >>> g = read_edgelist("flights.csv", source="origin", target="dest", directed=True)  # doctest: +SKIP
    """
    text = _read_text(path, encoding)
    if delimiter == " ":
        delimiter = None
    elif delimiter is None:
        delimiter = _sniff(text, comments)
    records = _records(text, delimiter, comments)
    g = _new_graph(directed, dag)
    if not records:
        return g

    named = isinstance(source, str) or isinstance(target, str)
    if header == "auto":
        header = named or _looks_like_header(records)
    elif not isinstance(header, bool):
        raise ValueError(f"header must be True, False or 'auto', got {header!r}")
    if named and not header:
        raise ValueError("source/target given by name, but the file is read without a header")

    names: list[str] | None = None
    if header:
        line_no, first = records[0]
        names = [t for t, _ in first]
        records = records[1:]
        if len(set(names)) != len(names):
            raise ValueError(f"line {line_no}: duplicate column names in the header {names}")
    s_col = _column(source, names, "source")
    t_col = _column(target, names, "target")
    if s_col == t_col:
        raise ValueError("source and target must be different columns")

    width = len(names) if names is not None else None
    attr_cols: dict[int, str] = {}

    def attr_name(k: int) -> str:
        if k not in attr_cols:
            if names is not None:
                attr_cols[k] = names[k]
            else:
                extra = [c for c in range(k + 1) if c not in (s_col, t_col)]
                attr_cols[k] = "weight" if extra[0] == k else f"col{k}"
        return attr_cols[k]

    s_conv = _converter(types, s_col, names[s_col] if names else None)
    t_conv = _converter(types, t_col, names[t_col] if names else None)
    batch = []
    for line_no, fields in records:
        if width is not None and len(fields) > width:
            raise ValueError(f"line {line_no}: {len(fields)} fields, but the header has {width}")
        if s_col >= len(fields) or fields[s_col] == ("", False):
            raise ValueError(f"line {line_no}: missing source field (column {s_col})")
        u = _value(fields[s_col], s_conv, True, line_no)
        if t_col >= len(fields) or fields[t_col] == ("", False):
            g.add_node(u)
            continue
        v = _value(fields[t_col], t_conv, True, line_no)
        attrs = {}
        for k, f in enumerate(fields):
            if k in (s_col, t_col) or f == ("", False):
                continue
            name = attr_name(k)
            attrs[name] = _value(f, _converter(types, k, name), False, line_no)
        if dag:
            g.add_node(u)
            g.add_node(v)
            batch.append((u, v, attrs))
        else:
            g.add_edge(u, v, **attrs)
    if batch:
        g.add_edges(batch)
    return g


def _column(spec: int | str, names: list[str] | None, what: str) -> int:
    if isinstance(spec, str):
        if names is None or spec not in names:
            raise ValueError(f"{what} column {spec!r} is not in the header {names}")
        return names.index(spec)
    if not isinstance(spec, int) or spec < 0:
        raise ValueError(f"{what} must be a column name or a non-negative index, got {spec!r}")
    if names is not None and spec >= len(names):
        raise ValueError(f"{what} column {spec} is out of range for the header {names}")
    return spec


def _looks_like_header(records: list[tuple[int, list[Field]]]) -> bool:
    first = records[0][1]
    if not all(q or isinstance(_auto(t, node=False), str) for t, q in first):
        return False
    if any(t.strip().lower() in _HEADER_WORDS for t, _ in first):
        return True
    rest = [fields for _, fields in records[1:21]]
    if not rest:
        return False
    for k in range(len(first)):
        column = [f[k] for f in rest if k < len(f) and f[k] != ("", False)]
        if column and all(not q and not isinstance(_auto(t, node=False), str) for t, q in column):
            return True
    return False


def write_edgelist(
    g: Graph,
    path: PathLike,
    *,
    delimiter: str = ",",
    attrs: bool | Sequence[str] = True,
    header: bool = True,
    isolates: bool = True,
    encoding: str = "utf-8",
) -> None:
    """Write *g* as a delimited edge list (CSV by default).

    Parameters
    ----------
    delimiter:
        ``","`` (CSV), ``"\\t"`` (TSV), ``" "`` (whitespace-separated), …
    attrs:
        ``True`` writes one column per edge attribute (the union of all
        keys, in first-seen order; missing values stay empty), ``False``
        none, or a list of attribute names.
    header:
        Write the ``source,target,<attrs>`` header line.
    isolates:
        Also write nodes without edges, as lines with an empty target, so
        :func:`read_edgelist` restores them. (Other tools may not understand
        such lines; pass ``False`` for a pure edge list.)
    """
    if len(delimiter) != 1 or delimiter in '"\n\r':
        raise ValueError(f"delimiter must be a single character other than a quote or newline, got {delimiter!r}")
    mode = None if delimiter == " " else delimiter
    if attrs is True:
        keys = list(dict.fromkeys(k for _, _, d in g._iter_edges() for k in d))
    elif attrs is False:
        keys = []
    else:
        keys = list(attrs)
    for reserved in ("source", "target"):
        if header and reserved in keys:
            raise ValueError(f"an edge attribute is named {reserved!r}, which clashes with the header")
    lines = []
    if header:
        lines.append(delimiter.join(_format(c, mode, node=False) for c in ["source", "target", *keys]))
    for u, v, d in g._iter_edges():
        row = [_format(u, mode, node=True), _format(v, mode, node=True)]
        row += [_format(d.get(k), mode, node=False) for k in keys]
        if mode is None:
            # Runs of blanks collapse on reading, so an empty field can only be trailing.
            while len(row) > 2 and row[-1] == "":
                row.pop()
            if "" in row:
                missing = keys[row.index("") - 2]
                raise ValueError(
                    f"edge ({u!r}, {v!r}) has no {missing!r} but has later attributes; a whitespace-separated "
                    "file cannot hold empty fields, so use delimiter=',' or '\\t'"
                )
        lines.append(delimiter.join(row))
    if isolates:
        # Padded to the full width so the file keeps a constant field count.
        pad = "" if mode is None else delimiter * (1 + len(keys))
        for n in g._node:
            if not g._succ[n] and not g._pred[n]:
                lines.append(_format(n, mode, node=True) + pad)
    with open(path, "w", encoding=encoding, newline="\n") as fh:
        fh.write("\n".join(lines))
        if lines:
            fh.write("\n")


# ---------------------------------------------------------------------- #
# adjacency lists
# ---------------------------------------------------------------------- #
def read_adjacency_list(
    path: PathLike,
    *,
    delimiter: str | None = None,
    directed: bool = False,
    dag: bool = False,
    types: Callable[[str], Any] | None = None,
    comments: str | None = "#",
    encoding: str = "utf-8",
) -> Graph:
    """Read an adjacency list: each line is a node followed by its neighbors (successors if directed).

    A line with a single node declares it (possibly isolated). Fields follow
    the typing rule of the module docstring; *types* is an optional callable
    applied to every field's text instead.
    """
    if types is not None and not callable(types):
        raise TypeError("types must be a callable for adjacency lists")
    text = _read_text(path, encoding)
    records = _records(text, None if delimiter == " " else delimiter, comments)
    g = _new_graph(directed, dag)
    batch = []
    for line_no, fields in records:
        if fields[0] == ("", False):
            raise ValueError(f"line {line_no}: empty node field")
        u = _value(fields[0], types, True, line_no)
        g.add_node(u)
        for f in fields[1:]:
            if f == ("", False):
                continue
            v = _value(f, types, True, line_no)
            if dag:
                g.add_node(v)
                batch.append((u, v))
            else:
                g.add_edge(u, v)
    if batch:
        g.add_edges(batch)
    return g


def write_adjacency_list(
    g: Graph,
    path: PathLike,
    *,
    delimiter: str = " ",
    encoding: str = "utf-8",
) -> None:
    """Write *g* as an adjacency list (networkx's ``adjlist`` layout, with quoting).

    Every node gets a line listing its successors, so isolated nodes
    survive; an undirected edge is listed once, on the line of whichever
    endpoint comes first in graph order. Attributes are not stored.
    """
    if len(delimiter) != 1 or delimiter in '"\n\r':
        raise ValueError(f"delimiter must be a single character other than a quote or newline, got {delimiter!r}")
    mode = None if delimiter == " " else delimiter
    seen: set = set()
    lines = []
    for u, nbrs in g._succ.items():
        row = [_format(u, mode, node=True)]
        row += [_format(v, mode, node=True) for v in nbrs if g.directed or v not in seen]
        seen.add(u)
        lines.append(delimiter.join(row))
    with open(path, "w", encoding=encoding, newline="\n") as fh:
        fh.write("\n".join(lines))
        if lines:
            fh.write("\n")


__all__ = ["read_edgelist", "write_edgelist", "read_adjacency_list", "write_adjacency_list"]
