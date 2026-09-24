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

"""DAG algorithms and cycle finding, checked against networkx and brute force."""

from __future__ import annotations

import itertools
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pytest

from aryagraph import DAG, CycleError, DiGraph, Graph, GraphTypeError, NodeNotFound
from aryagraph.algorithms import dag as D

SEEDS = range(6)


def pair(nxg: nx.Graph, cls: type | None = None) -> tuple[nx.Graph, Graph]:
    nodes = list(nxg.nodes(data=True))
    edges = list(nxg.edges(data=True))
    h = nxg.__class__()
    h.add_nodes_from(nodes)
    h.add_edges_from(edges)
    g = (cls or (DiGraph if nxg.is_directed() else Graph))()
    g.add_nodes((n, dict(d)) for n, d in nodes)
    g.add_edges((u, v, dict(d)) for u, v, d in edges)
    return h, g


def rand_dag(seed: int, n: int = 40, p: float = 0.1, weights: str | None = None, cls: type = DAG) -> tuple[nx.DiGraph, Graph]:
    """A random DAG whose node order is *not* topological (labels are shuffled)."""
    rng = random.Random(seed)
    base = nx.gnp_random_graph(n, p, seed=seed, directed=True)
    perm = list(range(n))
    rng.shuffle(perm)
    nxg = nx.DiGraph()
    nxg.add_nodes_from(perm)  # graph order differs from topological order
    for u, v in base.edges:
        if u < v:
            attrs = {}
            if weights == "int":
                attrs["weight"] = rng.randint(1, 9)
            elif weights == "signed":
                attrs["weight"] = rng.randint(-5, 9)
            nxg.add_edge(perm[u], perm[v], **attrs)
    return pair(nxg, cls)


def rand_digraph(seed: int, n: int = 30, p: float = 0.08) -> tuple[nx.DiGraph, DiGraph]:
    return pair(nx.gnp_random_graph(n, p, seed=seed, directed=True))


def is_topological(g: Graph, order: list) -> bool:
    pos = {n: i for i, n in enumerate(order)}
    return len(order) == len(g) and set(order) == set(g) and all(pos[u] < pos[v] for u, v in g.edges)


def assert_cycle(g: Graph, cycle: list, *, simple: bool = True) -> None:
    assert len(cycle) >= 2 and cycle[0] == cycle[-1]
    for u, v in zip(cycle, cycle[1:]):
        assert g.has_edge(u, v)
    if simple:
        assert len(set(cycle[:-1])) == len(cycle) - 1


# ---------------------------------------------------------------------- #
# orders
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cls", [DAG, DiGraph])
def test_topological_sort(seed, cls):
    h, g = rand_dag(seed, cls=cls)
    order = D.topological_sort(g)
    assert is_topological(g, order)
    if cls is DAG:
        assert order == g.topological_order()
    for key in (lambda n: n, lambda n: -n, str):
        got = D.topological_sort(g, key=key)
        assert got == list(nx.lexicographical_topological_sort(h, key=key))


def test_topological_sort_errors():
    cyc = DiGraph([(0, 1), (1, 2), (2, 0), (2, 3)])
    for key in (None, str):
        with pytest.raises(CycleError) as info:
            D.topological_sort(cyc, key=key)
        assert_cycle(cyc, info.value.cycle)
    with pytest.raises(GraphTypeError):
        D.topological_sort(Graph([(0, 1)]))
    assert D.topological_sort(DAG()) == []
    with pytest.raises(CycleError):
        D.topological_sort(DiGraph([(0, 0)]))


def test_reported_cycle_does_not_depend_on_the_hash_seed():
    # Kahn's leftovers are traced from the first one in graph order, so string
    # nodes give the same cycle (and message) in every process.
    code = (
        "import aryagraph as ag\n"
        "try:\n"
        "    ag.DAG([('x', 'y'), ('y', 'z'), ('z', 'x'), ('z', 'w')])\n"
        "except ag.CycleError as err:\n"
        "    print(err.cycle)\n"
        "try:\n"
        "    ag.alg.topological_sort(ag.DiGraph([('b', 'a'), ('a', 'b')]))\n"
        "except ag.CycleError as err:\n"
        "    print(err.cycle)\n"
    )
    outputs = set()
    for hash_seed in ("0", "1", "2", "3"):
        src = str(Path(__file__).resolve().parents[1] / "src")
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")])))
        done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        outputs.add(done.stdout)
    assert outputs == {"['x', 'y', 'z', 'x']\n['b', 'a', 'b']\n"}


@pytest.mark.parametrize("seed", SEEDS)
def test_all_topological_sorts_match_networkx(seed):
    h, g = rand_dag(seed, n=7, p=0.25)
    got = list(D.all_topological_sorts(g))
    assert len(got) == len({tuple(o) for o in got})
    assert {tuple(o) for o in got} == {tuple(o) for o in nx.all_topological_sorts(h)}
    assert all(is_topological(g, o) for o in got)


def test_all_topological_sorts_small_and_errors():
    g = DAG([("a", "b"), ("a", "c")])
    assert list(D.all_topological_sorts(g)) == [["a", "b", "c"], ["a", "c", "b"]]
    assert list(D.all_topological_sorts(DAG())) == [[]]
    antichain = DiGraph()
    antichain.add_nodes([1, 2, 3])
    assert len(list(D.all_topological_sorts(antichain))) == 6
    with pytest.raises(CycleError):
        D.all_topological_sorts(DiGraph([(0, 1), (1, 0)]))  # eager


@pytest.mark.parametrize("seed", SEEDS)
def test_is_dag_matches_networkx(seed):
    for p in (0.03, 0.08):
        h, g = rand_digraph(seed, p=p)
        assert D.is_dag(g) == nx.is_directed_acyclic_graph(h)
    assert D.is_dag(rand_dag(seed)[1])
    assert not D.is_dag(Graph([(0, 1)]))
    assert not D.is_dag(DiGraph([(0, 0)]))


# ---------------------------------------------------------------------- #
# cycles
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("p", [0.03, 0.06, 0.12])
def test_find_cycle_directed(seed, p):
    h, g = rand_digraph(seed, p=p)
    cycle = D.find_cycle(g)
    if nx.is_directed_acyclic_graph(h):
        assert cycle is None
    else:
        assert_cycle(g, cycle)
    for s in (0, 13):
        reach = nx.descendants(h, s) | {s}
        cyclic_from_s = not nx.is_directed_acyclic_graph(h.subgraph(reach))
        found = D.find_cycle(g, s)
        assert (found is not None) == cyclic_from_s
        if found:
            assert_cycle(g, found)
            assert set(found) <= reach


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("p", [0.03, 0.06])
def test_find_cycle_undirected(seed, p):
    h, g = pair(nx.gnp_random_graph(40, p, seed=seed))
    cycle = D.find_cycle(g)
    has_cycle = h.number_of_edges() > h.number_of_nodes() - nx.number_connected_components(h)
    assert (cycle is not None) == has_cycle
    if cycle:
        assert_cycle(g, cycle)
        assert len(cycle) >= 4  # at least three distinct nodes


def test_find_cycle_small_cases():
    assert D.find_cycle(Graph([(0, 1), (1, 2)])) is None  # an edge is not a cycle
    assert D.find_cycle(Graph([(0, 1), (1, 1)])) == [1, 1]
    assert D.find_cycle(DiGraph([(0, 1), (1, 0)])) == [0, 1, 0]
    assert D.find_cycle(DiGraph([(0, 0)])) == [0, 0]
    assert D.find_cycle(DiGraph([("a", "b"), ("b", "c"), ("c", "a")])) == ["a", "b", "c", "a"]
    assert D.find_cycle(DAG([(0, 1), (1, 2)])) is None
    assert D.find_cycle(DiGraph()) is None
    g = DiGraph([(0, 1), (2, 3), (3, 2)])
    assert D.find_cycle(g, 0) is None
    assert D.find_cycle(g, [0, 2]) == [2, 3, 2]
    with pytest.raises(NodeNotFound):
        D.find_cycle(g, 99)


def _canon_directed(cycle: list) -> tuple:
    body = cycle[:-1] if cycle[0] == cycle[-1] and len(cycle) > 1 else cycle
    i = body.index(min(body))
    return tuple(body[i:] + body[:i])


def _canon_undirected(cycle: list) -> tuple:
    a = _canon_directed(cycle)
    b = _canon_directed(list(reversed(a)))
    return min(a, b)


@pytest.mark.parametrize("seed", SEEDS)
def test_simple_cycles_directed_match_networkx(seed):
    h, g = rand_digraph(seed, n=14, p=0.2)
    h.add_edge(3, 3)
    g.add_edge(3, 3)
    got = list(D.simple_cycles(g))
    ref = list(nx.simple_cycles(h))
    assert all(c[0] == c[-1] for c in got)
    for c in got:
        assert_cycle(g, c)
    assert len(got) == len(ref)
    assert {_canon_directed(c) for c in got} == {_canon_directed(c) for c in ref}
    index = g.node_index()
    assert all(index[c[0]] == min(index[x] for x in c) for c in got)  # starts at earliest node


@pytest.mark.parametrize("seed", SEEDS)
def test_simple_cycles_undirected_match_networkx(seed):
    h, g = pair(nx.gnp_random_graph(12, 0.3, seed=seed))
    got = list(D.simple_cycles(g))
    ref = list(nx.simple_cycles(h))
    assert len(got) == len(ref)
    assert {_canon_undirected(c) for c in got} == {_canon_undirected(c) for c in ref}
    for c in got:
        assert_cycle(g, c)
        assert len(c) >= 4


def test_simple_cycles_small():
    g = DiGraph([(0, 1), (1, 2), (2, 0), (2, 2), (1, 0)])
    assert sorted(D.simple_cycles(g)) == [[0, 1, 0], [0, 1, 2, 0], [2, 2]]
    assert list(D.simple_cycles(DAG([(0, 1)]))) == []
    tri = Graph([("a", "b"), ("b", "c"), ("c", "a")])
    assert list(D.simple_cycles(tri)) == [["a", "b", "c", "a"]]


# ---------------------------------------------------------------------- #
# longest paths, levels, CPM
# ---------------------------------------------------------------------- #
def path_weight(g: Graph, path: list, weight: str | None) -> float:
    total = 0
    for u, v in zip(path, path[1:]):
        assert g.has_edge(u, v)
        total += 1 if weight is None else g.edges[u, v].get(weight, 1)
    return total


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("weights", [None, "int", "signed"])
def test_dag_longest_path_matches_networkx(seed, weights):
    h, g = rand_dag(seed, weights=weights, cls=DiGraph if seed % 2 else DAG)
    w = "weight" if weights else None
    ref = nx.dag_longest_path_length(h, weight="weight" if weights else "__none__", default_weight=1)
    path = D.dag_longest_path(g, weight=w)
    assert path_weight(g, path, w) == ref
    assert D.dag_longest_path_length(g, weight=w) == ref
    if isinstance(g, DAG):
        assert path_weight(g, g.longest_path(), None) == nx.dag_longest_path_length(h, weight="__none__")


def test_dag_longest_path_small():
    g = DAG([("a", "b", {"cost": 2}), ("b", "c", {"cost": 3}), ("a", "c", {"cost": 10}), ("c", "d")])
    assert D.dag_longest_path(g) == ["a", "b", "c", "d"]
    assert D.dag_longest_path_length(g) == 3
    assert D.dag_longest_path(g, weight="cost") == ["a", "c", "d"]
    assert D.dag_longest_path_length(g, weight="cost") == 11  # missing attribute counts as default 1
    assert D.dag_longest_path_length(g, weight="cost", default=0) == 10
    assert D.dag_longest_path(g, weight=lambda u, v, d: -1) == ["a"]
    assert D.dag_longest_path(DAG()) == [] and D.dag_longest_path_length(DAG()) == 0
    single = DAG()
    single.add_node("x")
    assert D.dag_longest_path(single) == ["x"]
    with pytest.raises(CycleError):
        D.dag_longest_path(DiGraph([(0, 1), (1, 0)]))
    with pytest.raises(GraphTypeError):
        D.dag_longest_path(Graph([(0, 1)]))


@pytest.mark.parametrize("seed", SEEDS)
def test_dag_levels(seed):
    _, g = rand_dag(seed)
    levels = D.dag_levels(g)
    assert dict(levels) == dict(g.levels())
    assert list(levels) == list(g)
    assert all(levels[v] > levels[u] for u, v in g.edges)
    assert levels.name == "level"


def _makespan_by_networkx(h: nx.DiGraph, dur: dict) -> float:
    """Longest node-weighted path via an edge-weighted copy with a super sink."""
    t = nx.DiGraph()
    t.add_nodes_from(h)
    t.add_weighted_edges_from((u, v, dur[u]) for u, v in h.edges)
    t.add_weighted_edges_from((u, "__sink__", dur[u]) for u in h)
    return nx.dag_longest_path_length(t)


@pytest.mark.parametrize("seed", SEEDS)
def test_critical_path_random(seed):
    h, g = rand_dag(seed, n=50, p=0.08)
    rng = random.Random(seed)
    dur = {n: rng.choice([0, 0.5, 1, 2, 3.25, 7]) for n in g}
    for n, d in dur.items():
        if n % 7:  # leave some durations to the default
            g.nodes[n]["duration"] = d
        else:
            dur[n] = 1.0
    cp = D.critical_path(g)
    assert cp.length == pytest.approx(_makespan_by_networkx(h, dur))
    assert cp.makespan == cp.length
    # the path is a real source-to-sink chain of critical activities adding up to the makespan
    assert not g._pred[cp.path[0]] and not g._succ[cp.path[-1]]
    for u, v in zip(cp.path, cp.path[1:]):
        assert g.has_edge(u, v)
    assert sum(dur[n] for n in cp.path) == pytest.approx(cp.length)
    assert set(cp.path) <= cp.critical
    for n in g:
        assert cp.earliest_finish[n] == pytest.approx(cp.earliest_start[n] + dur[n])
        assert cp.latest_finish[n] == pytest.approx(cp.latest_start[n] + dur[n])
        assert cp.slack[n] >= -1e-9
        assert (cp.slack[n] == 0) == (n in cp.critical)
        assert cp.latest_finish[n] <= cp.length + 1e-9
    for u, v in g.edges:
        assert cp.earliest_start[v] >= cp.earliest_finish[u] - 1e-9
        assert cp.latest_finish[u] <= cp.latest_start[v] + 1e-9
    # slack is exactly how long an activity may slip: longest path through n
    for n in list(g)[:10]:
        through = max((cp.earliest_finish[n] + _tail(g, dur, n)), cp.earliest_finish[n])
        assert cp.slack[n] == pytest.approx(cp.length - through, abs=1e-9)


def _tail(g: Graph, dur: dict, n) -> float:
    """Longest duration sum strictly after n (brute-force DP)."""
    memo: dict = {}
    order = D.topological_sort(g)[::-1]
    for v in order:
        memo[v] = max((dur[w] + memo[w] for w in g._succ[v]), default=0.0)
    return memo[n]


def test_critical_path_textbook_example():
    dag = DAG([("design", "build"), ("design", "docs"), ("build", "ship"), ("docs", "ship")])
    for n, d in {"design": 2, "build": 5, "docs": 1, "ship": 1}.items():
        dag.nodes[n]["duration"] = d
    cp = D.critical_path(dag)
    assert cp.path == ["design", "build", "ship"]
    assert cp.length == 8
    assert cp.critical == {"design", "build", "ship"}
    assert cp.slack["docs"] == 4
    assert dict(cp.earliest_start) == {"design": 0, "build": 2, "docs": 2, "ship": 7}
    assert dict(cp.latest_start) == {"design": 0, "build": 2, "docs": 6, "ship": 7}
    assert "length=8" in repr(cp) and "'design' → 'build' → 'ship'" in repr(cp)
    assert dag.critical_path().path == cp.path  # the DAG method delegates here
    by_callable = D.critical_path(dag, duration=lambda n: len(n))
    assert by_callable.length == len("design") + len("build") + len("ship")


def test_critical_path_edge_cases():
    empty = D.critical_path(DAG())
    assert empty.path == [] and empty.length == 0 and empty.critical == set()
    lone = DAG()
    lone.add_node("x")
    assert D.critical_path(lone, default=3).length == 3
    bad = DAG([("a", "b")])
    bad.nodes["a"]["duration"] = -1
    with pytest.raises(ValueError):
        D.critical_path(bad)
    bad.nodes["a"]["duration"] = "soon"
    with pytest.raises(ValueError):
        D.critical_path(bad)
    with pytest.raises(CycleError):
        D.critical_path(DiGraph([(0, 1), (1, 0)]))
    with pytest.raises(GraphTypeError):
        D.critical_path(Graph([(0, 1)]))
    zero = DAG([("a", "b"), ("b", "c")])
    for n in zero:
        zero.nodes[n]["duration"] = 0
    cp = D.critical_path(zero)
    assert cp.length == 0 and cp.critical == {"a", "b", "c"} and cp.path == ["a", "b", "c"]


# ---------------------------------------------------------------------- #
# closure and reduction
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("reflexive", [False, True, None])
def test_transitive_closure_matches_networkx(seed, reflexive):
    h, g = rand_digraph(seed, p=0.06)
    h.add_edge(2, 2)
    g.add_edge(2, 2)
    tc = D.transitive_closure(g, reflexive=reflexive)
    assert type(tc) is DiGraph
    assert set(tc.edges) == set(nx.transitive_closure(h, reflexive=reflexive).edges)
    hd, gd = rand_dag(seed)
    tcd = D.transitive_closure(gd, reflexive=reflexive)
    assert set(tcd.edges) == set(nx.transitive_closure(hd, reflexive=reflexive).edges)
    assert isinstance(tcd, DAG) == (reflexive is not True)


def test_transitive_closure_keeps_attributes():
    g = DAG([("a", "b", {"w": 5}), ("b", "c")], name="pipeline")
    g.nodes["a"]["color"] = "red"
    tc = D.transitive_closure(g)
    assert isinstance(tc, DAG) and tc.name == "pipeline"
    assert tc.edges["a", "b"] == {"w": 5} and tc.edges["a", "c"] == {}
    assert tc.nodes["a"] == {"color": "red"}
    assert set(g.transitive_closure().edges) == set(tc.edges)
    assert g.num_edges == 2  # the input is untouched
    with pytest.raises(GraphTypeError):
        D.transitive_closure(Graph([(0, 1)]))
    with pytest.raises(ValueError):
        D.transitive_closure(g, reflexive="yes")


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("cls", [DAG, DiGraph])
def test_transitive_reduction_matches_networkx(seed, cls):
    h, g = rand_dag(seed, p=0.15, weights="int", cls=cls)
    tr = D.transitive_reduction(g)
    assert isinstance(tr, DAG)
    assert list(tr) == list(g)
    assert set(tr.edges) == set(nx.transitive_reduction(h).edges)
    for u, v in tr.edges:
        assert tr.edges[u, v] == g.edges[u, v]
    # same reachability
    assert set(D.transitive_closure(tr).edges) == set(nx.transitive_closure(h).edges)


def test_transitive_reduction_small_and_errors():
    g = DAG([(1, 2), (2, 3), (1, 3, {"w": 1}), (3, 4)])
    g.nodes[1]["label"] = "start"
    tr = g.transitive_reduction()
    assert list(tr.edges) == [(1, 2), (2, 3), (3, 4)]
    assert tr.nodes[1] == {"label": "start"}
    with pytest.raises(CycleError):
        D.transitive_reduction(DiGraph([(0, 1), (1, 0)]))
    with pytest.raises(GraphTypeError):
        D.transitive_reduction(Graph([(0, 1)]))


# ---------------------------------------------------------------------- #
# ancestry and width
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", SEEDS)
def test_lowest_common_ancestors(seed):
    h, g = rand_dag(seed, n=30, p=0.12)
    rng = random.Random(seed)
    nodes = list(g)
    for _ in range(25):
        a, b = rng.choice(nodes), rng.choice(nodes)
        got = D.lowest_common_ancestors(g, a, b)
        common = (nx.ancestors(h, a) | {a}) & (nx.ancestors(h, b) | {b})
        expected = {x for x in common if not (nx.descendants(h, x) & common)}
        assert got == expected
        one = nx.lowest_common_ancestor(h, a, b)
        assert (one is None and not got) or one in got


def test_lowest_common_ancestors_small_and_errors():
    g = DAG([("r", "a"), ("r", "b"), ("a", "x"), ("b", "x"), ("a", "y"), ("b", "y")])
    assert D.lowest_common_ancestors(g, "x", "y") == {"a", "b"}
    assert D.lowest_common_ancestors(g, "a", "x") == {"a"}
    assert D.lowest_common_ancestors(g, "a", "a") == {"a"}
    disjoint = DAG([(0, 1), (2, 3)])
    assert D.lowest_common_ancestors(disjoint, 1, 3) == set()
    with pytest.raises(NodeNotFound):
        D.lowest_common_ancestors(g, "x", "nope")
    with pytest.raises(CycleError):
        D.lowest_common_ancestors(DiGraph([(0, 1), (1, 0)]), 0, 1)


def _brute_width(h: nx.DiGraph) -> int:
    tc = nx.transitive_closure(h)
    nodes = list(h)
    for k in range(len(nodes), 0, -1):
        for subset in itertools.combinations(nodes, k):
            if not any(tc.has_edge(u, v) for u in subset for v in subset if u != v):
                return k
    return 0


def _matching_width(h: nx.DiGraph) -> int:
    tc = nx.transitive_closure(h)
    bip = nx.Graph()
    bip.add_nodes_from((("L", n) for n in h), bipartite=0)
    bip.add_nodes_from((("R", n) for n in h), bipartite=1)
    bip.add_edges_from((("L", u), ("R", v)) for u, v in tc.edges)
    m = nx.bipartite.hopcroft_karp_matching(bip, top_nodes=[("L", n) for n in h])
    return len(h) - len(m) // 2


def _check_antichain_and_chains(g: Graph, h: nx.DiGraph, width: int) -> None:
    tc = nx.transitive_closure(h)
    anti = D.maximum_antichain(g)
    assert len(anti) == width
    assert not any(tc.has_edge(u, v) for u in anti for v in anti if u != v)
    chains = D.minimum_chain_partition(g)
    assert len(chains) == width
    assert sorted(n for c in chains for n in c) == sorted(g)
    for c in chains:
        assert all(tc.has_edge(u, v) for u, v in zip(c, c[1:]))


@pytest.mark.parametrize("seed", SEEDS)
def test_dag_width_brute_force(seed):
    h, g = rand_dag(seed, n=11, p=0.2)
    width = D.dag_width(g)
    assert width == _brute_width(h)
    _check_antichain_and_chains(g, h, width)


@pytest.mark.parametrize("seed", SEEDS)
def test_dag_width_matches_matching_oracle(seed):
    h, g = rand_dag(seed, n=60, p=0.06)
    width = D.dag_width(g)
    assert width == _matching_width(h)
    _check_antichain_and_chains(g, h, width)


def test_dag_width_small():
    assert D.dag_width(DAG()) == 0
    chain = DAG([(0, 1), (1, 2), (2, 3)])
    assert D.dag_width(chain) == 1
    assert D.minimum_chain_partition(chain) == [[0, 1, 2, 3]]
    diamond = DAG([("s", "a"), ("s", "b"), ("s", "c"), ("a", "t"), ("b", "t"), ("c", "t")])
    assert D.dag_width(diamond) == 3
    assert D.maximum_antichain(diamond) == {"a", "b", "c"}
    with pytest.raises(CycleError):
        D.dag_width(DiGraph([(0, 1), (1, 0)]))


def test_dag_functions_are_iterative():
    n = 30_000
    chain = DAG((i, i + 1) for i in range(n - 1))
    assert D.dag_longest_path_length(chain) == n - 1
    assert D.topological_sort(chain, key=lambda x: -x) == list(range(n))
    assert D.find_cycle(DiGraph(chain)) is None
    ring = DiGraph((i, (i + 1) % n) for i in range(n))
    assert len(D.find_cycle(ring)) == n + 1
    assert list(D.simple_cycles(ring)) == [[*range(n), 0]]
    assert D.critical_path(chain).length == n


def test_math_helpers_are_consistent():
    # the bit iterator underpins closure/reduction/width
    for x in (0, 1, 2, 5, 1 << 70 | 3):
        assert list(D._iter_bits(x)) == [i for i in range(x.bit_length()) if x >> i & 1]
    assert math.isclose(D.critical_path(DAG([("a", "b")]), default=0.1).length, 0.2)
