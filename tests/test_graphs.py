import project.graphs as graphs
import pytest
from textwrap import dedent


def test_get_props_by_name():
    assert graphs.GraphProps(
        node_quan=1687, edge_quan=1453, labels={"a", "d"}
    ) == graphs.get_props_by_name("ls")


def test_save_labeled_two_cycle_graph(tmp_path):
    graph_path = tmp_path / "c2,5.dot"

    graphs.save_labeled_two_cycle_graph(
        node_quan_1=1, node_quan_2=4, labels=("a", "b"), path=graph_path
    )

    with open(graph_path, "r") as graph_file:
        expected = dedent(
            """\
            digraph  {
            1;
            0;
            2;
            3;
            4;
            5;
            1 -> 0  [key=0, label=a];
            0 -> 1  [key=0, label=a];
            0 -> 2  [key=0, label=b];
            2 -> 3  [key=0, label=b];
            3 -> 4  [key=0, label=b];
            4 -> 5  [key=0, label=b];
            5 -> 0  [key=0, label=b];
            }
            """
        )

        assert expected == "".join(graph_file.readlines())
