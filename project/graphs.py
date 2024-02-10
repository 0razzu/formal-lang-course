import cfpq_data
from networkx import MultiDiGraph
from collections import namedtuple


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
