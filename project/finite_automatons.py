from networkx import MultiDiGraph, DiGraph
from networkx.classes.reportviews import NodeView
from pyformlang.cfg import Epsilon
from pyformlang.regular_expression import Regex
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from pyformlang.rsa import RecursiveAutomaton
from scipy.sparse import dok_matrix, kron
from typing import Iterable


class FiniteAutomaton:
    def __init__(
        self,
        automaton,
        start_states: set[State] = {},
        final_states: set[State] = {},
        state_to_idx: dict[State, int] = {},
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
        if len(self.matrix) == 0:
            return True

        m = sum(self.matrix.values())
        for _ in range(m.shape[0]):
            m += m @ m

        if m.shape[0] != 0 or m.shape[1] != 0:
            return True

        for u in self.start_states:
            for v in self.final_states:
                if m[u, v] != 0:
                    return False

        return True

    def idx_to_state(self) -> dict[int, State]:
        return {i: s for s, i in self.state_to_idx.items()}

    def states_len(self) -> int:
        return len(self.state_to_idx)


def to_set(obj):
    if isinstance(obj, set):
        return obj
    return {obj}


def nfa_to_matrix(automaton: NondeterministicFiniteAutomaton) -> FiniteAutomaton:
    matrix = {}
    state_to_idx = {state: idx for idx, state in enumerate(automaton.states)}
    state_transitions = automaton.to_dict()
    states_quan = len(automaton.states)

    for label in automaton.symbols:
        matrix[label] = dok_matrix((states_quan, states_quan), dtype=bool)

        for from_state, transitions in state_transitions.items():
            if label in transitions:
                for to_state in to_set(transitions[label]):
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
    regex_automaton = FiniteAutomaton(regex_to_dfa(regex))
    intersection = intersect_automata(
        FiniteAutomaton(graph_to_nfa(graph, start_nodes, final_nodes)),
        regex_automaton,
    )
    size = len(regex_automaton.state_to_idx)

    res = []
    for i, j in zip(*transitive_closure(intersection).nonzero()):
        if (
            State(i) in intersection.start_states
            and State(j) in intersection.final_states
        ):
            res.append((i // size, j // size))

    return res


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    return Regex(regex).to_epsilon_nfa().minimize()


def graph_to_nfa(
    graph: DiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
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


def rsm_to_matrix(rsm: RecursiveAutomaton) -> FiniteAutomaton:
    states = set()
    start_states = set()
    final_states = set()
    nullable_symbols = set()

    for var, product in rsm.boxes.items():
        for state in product.dfa.states:
            s = State((var, state.value))

            states.add(s)

            if state in product.dfa.start_states:
                start_states.add(s)

            if state in product.dfa.final_states:
                final_states.add(s)

    states_len = len(states)
    state_to_idx = {
        s: i for i, s in enumerate(sorted(states, key=lambda x: x.value[1]))
    }

    matrix = {}
    for var, product in rsm.boxes.items():
        for fr, transition in product.dfa.to_dict().items():
            for symbol, tos in transition.items():
                label = symbol.value

                if symbol not in matrix:
                    matrix[label] = dok_matrix((states_len, states_len), dtype=bool)

                for to in to_set(tos):
                    matrix[label][
                        state_to_idx[State((var, fr.value))],
                        state_to_idx[State((var, to.value))],
                    ] = True

                if isinstance(tos, Epsilon):
                    nullable_symbols.add(label)

    res = FiniteAutomaton(matrix, start_states, final_states, state_to_idx)
    res.nullable_symbols = nullable_symbols

    return res


def transitive_closure(fa: FiniteAutomaton) -> dok_matrix:
    if fa.is_empty():
        return dok_matrix((0, 0), dtype=bool)

    f = None
    for m in fa.matrix.values():
        f = m if f is None else f + m

    p = 0
    while f.count_nonzero() != p:
        p = f.count_nonzero()
        f += f @ f

    return f


def reachability_with_constraints(
    fa: FiniteAutomaton, constraints_fa: FiniteAutomaton
) -> dict[int, set[int]]:
    intersected = intersect_automata(fa, constraints_fa)

    res = dict()
    for s in fa.start_states:
        res[fa.state_to_idx[s]] = set()

    fa_len = len(constraints_fa.state_to_idx)
    for i, j in zip(*transitive_closure(intersected).nonzero()):
        if (
            State(i) in intersected.start_states
            and State(j) in intersected.final_states
        ):
            res[i // fa_len].add(j // fa_len)

    return res
