from networkx import MultiDiGraph
from networkx.classes.reportviews import NodeView
from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from typing import Set, Iterable


class FiniteAutomaton:
    def accepts(self, word: Iterable[Symbol]) -> bool:
        pass

    def is_empty(self) -> bool:
        pass


def intersect_automata(
    automaton1: FiniteAutomaton, automaton2: FiniteAutomaton
) -> FiniteAutomaton:
    pass


def paths_ends(
    graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int], regex: str
) -> list[tuple[NodeView, NodeView]]:
    pass


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    return Regex(regex).to_epsilon_nfa().minimize()


def graph_to_nfa(
    graph: MultiDiGraph,
    start_nodes: Set[int],
    final_nodes: Set[int],
) -> NondeterministicFiniteAutomaton:
    start_nodes = start_nodes or set(graph.nodes())
    final_nodes = final_nodes or set(graph.nodes())

    nfa = NondeterministicFiniteAutomaton()

    for start_node in start_nodes:
        nfa.add_start_state(State(start_node))

    for final_node in final_nodes:
        nfa.add_final_state(State(final_node))

    for from_node, to_node, label in graph.edges(data="label"):
        nfa.add_transition(State(from_node), Symbol(label), State(to_node))

    return nfa
