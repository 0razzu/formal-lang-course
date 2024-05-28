import re
from copy import deepcopy

import networkx as nx
from pyformlang.cfg import Variable, Terminal, CFG, Epsilon
from pyformlang.finite_automaton import Symbol, State
from pyformlang.regular_expression import Regex
from pyformlang.rsa import RecursiveAutomaton, Box
from scipy.sparse import dok_matrix, eye

from project.finite_automatons import (
    graph_to_nfa,
    nfa_to_matrix,
    rsm_to_matrix,
    transitive_closure,
    intersect_automata,
)


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    cfg = cfg.eliminate_unit_productions().remove_useless_symbols()
    productions = cfg._decompose_productions(
        cfg._get_productions_with_only_single_terminals()
    )

    return CFG(productions=set(productions), start_symbol=Variable("S"))


def cfpq_with_hellings(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    if start_nodes is None:
        start_nodes = graph.nodes
    if final_nodes is None:
        final_nodes = graph.nodes

    cfg = cfg_to_weak_normal_form(cfg)

    to_term_rules = {}
    to_eps_rules = set()
    to_2_nonterms_rules = {}
    for production in cfg.productions:
        production_body_len = len(production.body)

        if production_body_len == 1:
            if isinstance(production.body[0], Terminal):
                to_term_rules.setdefault(production.head, set()).add(production.body[0])
            elif isinstance(production.body[0], Epsilon):
                to_eps_rules.add(production.head)
        elif production_body_len == 2:
            to_2_nonterms_rules.setdefault(production.head, set()).add(
                (production.body[0], production.body[1])
            )

    res = {(n, v, v) for n in to_eps_rules for v in graph.nodes}

    res |= {
        (n, fr, to)
        for (fr, to, label) in graph.edges.data("label")
        for n in to_term_rules
        if label in to_term_rules[n]
    }

    res_copy = res.copy()
    while len(res_copy) > 0:
        n_i, fr_i, to_i = res_copy.pop()

        inc_steps = set()

        for n_j, fr_j, to_j in res:
            ij, ji = fr_i == to_j, to_i == fr_j

            if ij or ji:
                rule = (n_j, n_i) if ij else (n_i, n_j)

                for n_k in to_2_nonterms_rules:
                    edge = (n_k, fr_j, to_i) if ij else (n_k, fr_i, to_j)

                    if rule in to_2_nonterms_rules[n_k] and edge not in res:
                        res_copy.add(edge)
                        inc_steps.add(edge)

        res |= inc_steps

    return {
        (fr, to)
        for (n, fr, to) in res
        if fr in start_nodes and to in final_nodes and Variable(n) == cfg.start_symbol
    }


def cfpq_with_matrix(
    cfg: CFG,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    if start_nodes is None:
        start_nodes = graph.nodes
    if final_nodes is None:
        final_nodes = graph.nodes

    cfg = cfg_to_weak_normal_form(cfg)

    n = graph.number_of_nodes()

    m_0 = {}
    to_term_rules = {}
    to_eps_rules = set()
    to_2_nonterms_rules = {}
    for production in cfg.productions:
        m_0[production.head.to_text()] = dok_matrix((n, n), dtype=bool)
        production_body_len = len(production.body)

        if production_body_len == 1:
            if isinstance(production.body[0], Terminal):
                to_term_rules.setdefault(production.head.to_text(), set()).add(
                    production.body[0].to_text()
                )
            elif isinstance(production.body[0], Epsilon):
                to_eps_rules.add(production.head.to_text())
        elif production_body_len == 2:
            to_2_nonterms_rules.setdefault(production.head.to_text(), set()).add(
                (production.body[0].to_text(), production.body[1].to_text())
            )

    for fr, to, label in graph.edges(data="label"):
        if label in to_term_rules:
            for term in to_term_rules[label]:
                m_0[term][fr, to] = True

    for non_term in to_eps_rules:
        m_0[non_term].setdiag(True)

    res = {
        production.head.to_text(): dok_matrix((n, n), dtype=bool)
        for production in cfg.productions
    }

    changed = True
    while changed:
        changed = False
        for n_i, non_terms in to_2_nonterms_rules.items():
            for n_j, n_k in non_terms:
                before = res[n_i].nnz
                res[n_i] += res[n_j] @ res[n_k]
                changed |= before != res[n_i].nnz

        for n, m in m_0.items():
            m_0[n] += m

    rows, cols = m_0[cfg.start_symbol.to_text()].nonzero()
    return {
        (row, col)
        for row, col in zip(rows, cols)
        if row in start_nodes and col in final_nodes
    }


def cfpq_with_tensor(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    final_nodes: set[int] = None,
    start_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    rsm_matrix = rsm_to_matrix(rsm)
    graph_matrix = nfa_to_matrix(graph_to_nfa(graph, start_nodes, final_nodes))
    n = graph_matrix.states_len()

    for var in rsm_matrix.nullable_symbols:
        if var not in graph_matrix.matrix:
            graph_matrix.matrix[var] = dok_matrix((n, n), dtype=bool)
        graph_matrix.matrix[var] += eye(n, dtype=bool)

    rsm_matrix_i2s = rsm_matrix.idx_to_state()
    graph_matrix_i2s = graph_matrix.idx_to_state()

    last_nonzero = 0
    while True:
        closure = [
            *zip(
                *transitive_closure(
                    intersect_automata(rsm_matrix, graph_matrix)
                ).nonzero()
            )
        ]

        cur_nonzero = len(closure)
        if cur_nonzero == last_nonzero:
            break
        last_nonzero = cur_nonzero

        for i, j in closure:
            fr = rsm_matrix_i2s[i // n]
            to = rsm_matrix_i2s[j // n]

            if fr in rsm_matrix.start_states and to in rsm_matrix.final_states:
                var = fr.value[0]

                if var not in graph_matrix.matrix:
                    graph_matrix.matrix[var] = dok_matrix((n, n), dtype=bool)

                graph_matrix.matrix[var][i % n, j % n] = True

    return {
        (i, j)
        for m in graph_matrix.matrix.values()
        for i, j in zip(*m.nonzero())
        if graph_matrix_i2s[i] in rsm_matrix.start_states
        and graph_matrix_i2s[j] in rsm_matrix.final_states
    }


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    res_productions = {}

    for production in cfg.productions:
        if len(production.body) == 0:
            regex = Regex(
                " ".join(
                    "$" if isinstance(var, Epsilon) else var.value
                    for var in production.body
                )
            )
        else:
            regex = Regex("$")

        if production.head not in res_productions:
            res_productions[production.head] = regex
        else:
            res_productions[production.head] = res_productions[production.head] or regex

    res_productions = {
        Symbol(var): Box(regex.to_epsilon_nfa().to_deterministic(), Symbol(var))
        for var, regex in res_productions.items()
    }

    return RecursiveAutomaton(
        res_productions.keys(), Symbol("S"), set(res_productions.values())
    )


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    res_productions = {}

    for production_str in ebnf.splitlines():
        production_str = production_str.strip()

        if "->" not in production_str:
            continue

        head, body = re.split(r"\s*->\s*", production_str)
        if body == "":
            body = Epsilon().to_text()

        if head in res_productions:
            res_productions[head] += f" | {body}"
        else:
            res_productions[head] = body

    res_productions = {
        Symbol(var): Box(Regex(regex).to_epsilon_nfa().to_deterministic(), Symbol(var))
        for var, regex in res_productions.items()
    }

    return RecursiveAutomaton(
        res_productions.keys(), Symbol("S"), set(res_productions.values())
    )


def cfpq_with_gll(
    rsm: RecursiveAutomaton,
    graph: nx.DiGraph,
    start_nodes: set[int] = None,
    final_nodes: set[int] = None,
) -> set[tuple[int, int]]:
    if start_nodes is None:
        start_nodes = graph.nodes
    if final_nodes is None:
        final_nodes = graph.nodes

    ini_label = rsm.initial_label.value if rsm.initial_label.value is not None else "S"

    dfa_start_state = rsm.boxes[ini_label].dfa.to_deterministic().start_state.value
    dfa = rsm.boxes[ini_label].dfa.to_dict()
    dfa.setdefault(State(dfa_start_state), {})
    stack = {(n, ini_label): set() for n in start_nodes}
    visited = {(n, (dfa_start_state, ini_label), (n, ini_label)) for n in start_nodes}
    queue = deepcopy(visited)

    def visit(node, rsm_state, stack_state):
        s = (node, rsm_state, stack_state)

        if s not in visited:
            visited.add(s)
            queue.add(s)

    res = set()
    while len(queue) > 0:
        node, (q_rsm_state, _), (stack_node, stack_label) = queue.pop()
        q_stack_state = (stack_node, stack_label)

        if (
            stack_node in start_nodes
            and stack_label == dfa_start_state
            and node in final_nodes
        ):
            res.add((stack_node, node))

        for s_rsm_state, stack_state in stack.setdefault(q_stack_state, set()):
            visit(node, s_rsm_state, stack_state)

        for symbol in dfa.keys():
            if symbol in rsm.labels:
                start_sym_state = rsm.boxes[symbol].dfa.start_state.value
                s_rsm_state = (start_sym_state, symbol.value)
                stack_state = (node, symbol.value)

                visit(node, s_rsm_state, stack_state)

    return res
