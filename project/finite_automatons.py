from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from networkx import MultiDiGraph
from typing import Set


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
