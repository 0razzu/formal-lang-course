import cfpq_data
from networkx import MultiDiGraph
from networkx.drawing import nx_pydot as pydot
from collections import namedtuple
from typing import Tuple


GraphProps = namedtuple("GraphProps", ["node_quan", "edge_quan", "labels"])


def download_by_name(name: str) -> MultiDiGraph:
    return cfpq_data.graph_from_csv(cfpq_data.download(name))


def get_labels(graph: MultiDiGraph) -> set:
    return set([label for _, _, label in graph.edges(data="label")])


def get_props(graph: MultiDiGraph) -> GraphProps:
    return GraphProps(
        node_quan=graph.number_of_nodes(),
        edge_quan=graph.number_of_edges(),
        labels=get_labels(graph),
    )


def get_props_by_name(name: str) -> GraphProps:
    return get_props(download_by_name(name))


def save_labeled_two_cycle_graph(
    node_quan_1: int, node_quan_2: int, labels: Tuple[str, str], path: str
) -> None:
    pydot.write_dot(
        G=cfpq_data.labeled_two_cycles_graph(
            n=node_quan_1,
            m=node_quan_2,
            labels=labels,
        ),
        path=path,
    )
