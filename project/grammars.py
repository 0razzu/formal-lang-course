import networkx as nx
import pyformlang
from pyformlang.cfg import Variable, Terminal, CFG, Epsilon


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
        for (fr, to, tag) in graph.edges.data("label")
        for n in to_term_rules
        if tag in to_term_rules[n]
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
