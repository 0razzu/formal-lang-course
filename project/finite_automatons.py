from networkx import MultiDiGraph, transitive_closure
from networkx.classes.reportviews import NodeView
from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from scipy.sparse import dok_matrix, kron
from typing import Dict, Iterable, Set


class FiniteAutomaton:
    def __init__(
        self,
        automaton,
        start_states: Set[State] = {},
        final_states: Set[State] = {},
        state_to_idx: Dict[State, int] = {},
    ):
        if isinstance(automaton, DeterministicFiniteAutomaton) or isinstance(
            automaton, NondeterministicFiniteAutomaton
        ):
            matrix = nfa_to_matrix(automaton)
            self.matrix = matrix.matrix
            self.start_states = matrix.start_states
            self.final_states = matrix.final_states
            self.state_to_idx = matrix.state_to_idx

        else:
            self.matrix = automaton
            self.start_states = start_states
            self.final_states = final_states
            self.state_to_idx = state_to_idx

    def accepts(self, word: Iterable[Symbol]) -> bool:
        return matrix_to_nfa(self).accepts(word)

    def is_empty(self) -> bool:
        return len(self.matrix) == 0 or len(list(self.matrix.values())[0]) == 0


def nfa_to_matrix(automaton: NondeterministicFiniteAutomaton) -> FiniteAutomaton:
    matrix = {}
    state_to_idx = {state: idx for idx, state in enumerate(automaton.states)}
    state_transitions = automaton.to_dict()
    states_quan = len(automaton.states)

    for label in automaton.symbols:
        matrix[label] = dok_matrix((states_quan, states_quan), dtype=bool)

        for from_state, transitions in state_transitions.items():
            if label in transitions:
                for to_state in {transitions[label]}:
                    matrix[label][
                        state_to_idx[from_state], state_to_idx[to_state]
                    ] = True

    return FiniteAutomaton(
        matrix,
        automaton.start_states,
        automaton.final_states,
        state_to_idx,
    )


def matrix_to_nfa(automaton: FiniteAutomaton) -> NondeterministicFiniteAutomaton:
    nfa = NondeterministicFiniteAutomaton()

    for label in automaton.matrix.keys():
        matrix_size = automaton.matrix[label].shape[0]

        for from_idx in range(matrix_size):
            from_state = State(from_idx)
            for to_idx in range(matrix_size):
                to_state = State(to_idx)
                if automaton.matrix[label][from_idx, to_idx]:
                    nfa.add_transition(from_state, label, to_state)

    for state in automaton.start_states:
        nfa.add_start_state(state)
    for state in automaton.final_states:
        nfa.add_final_state(state)

    return nfa


def intersect_automata(
    automaton1: FiniteAutomaton,
    automaton2: FiniteAutomaton,
) -> FiniteAutomaton:
    labels = automaton1.matrix.keys() & automaton2.matrix.keys()
    matrix = {}
    start_states = set()
    final_states = set()
    state_to_idx = {}

    for label in labels:
        matrix[label] = kron(automaton1.matrix[label], automaton2.matrix[label], "csr")

    for u, i in automaton1.state_to_idx.items():
        for v, j in automaton2.state_to_idx.items():
            k = len(automaton2.state_to_idx) * i + j
            state_to_idx[State(k)] = k

            if u in automaton1.start_states and v in automaton2.start_states:
                start_states.add(State(k))

            if u in automaton1.final_states and v in automaton2.final_states:
                final_states.add(State(k))

    return FiniteAutomaton(matrix, start_states, final_states, state_to_idx)


def paths_ends(
    graph: MultiDiGraph, start_nodes: set[int], final_nodes: set[int], regex: str
) -> list[tuple[NodeView, NodeView]]:
    intersection = intersect_automata(
        FiniteAutomaton(automaton=graph_to_nfa(graph)),
        FiniteAutomaton(automaton=regex_to_dfa(regex)),
    )

    return zip(intersection.start_states, intersection.final_states)


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
