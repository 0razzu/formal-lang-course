from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
    Epsilon,
    EpsilonNFA,
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
    if start_nodes is None or len(start_nodes) == 0:
        start_nodes = graph.nodes()

    if final_nodes is None or len(final_nodes) == 0:
        final_nodes = graph.nodes()

    start_nodes = set(start_nodes)
    final_nodes = set(final_nodes)

    for node in graph.nodes(data=True):
        node[1].update({"is_start": node[0] in start_nodes})
        node[1].update({"is_final": node[0] in final_nodes})

    nfa = EpsilonNFA()

    for start_node in start_nodes:
        nfa.add_start_state(State(start_node))

    for final_node in final_nodes:
        nfa.add_final_state(State(final_node))

    print(start_nodes)
    print(final_nodes)
    for from_node, to_node, label in graph.edges(data="label"):
        print(from_node, to_node, label)
        nfa.add_transition(
            State(from_node),
            Epsilon() if label == "ɛ" else Symbol(label),
            State(to_node),
        )

    return nfa
