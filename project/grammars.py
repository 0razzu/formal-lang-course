import networkx as nx
from pyformlang.cfg import Variable, Terminal, CFG, Epsilon
from scipy.sparse import dok_matrix


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
