import project.graphs as graphs
import pytest
from textwrap import dedent


def test_get_props_by_name__ls():
    assert graphs.GraphProps(1687, 1453, {"a", "d"}) == graphs.get_props_by_name("ls")


def test_get_props_by_name__univ():
    assert graphs.GraphProps(
        node_quan=179,
        edge_quan=293,
        labels={
            "type",
            "label",
            "subClassOf",
            "domain",
            "range",
            "first",
            "rest",
            "someValuesFrom",
            "onProperty",
            "intersectionOf",
            "subPropertyOf",
            "inverseOf",
            "versionInfo",
            "comment",
        },
    ) == graphs.get_props_by_name("univ")


def test_save_labeled_two_cycle_graph__c_2_5(tmp_path):
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


def test_save_labeled_two_cycle_graph__c_4_4(tmp_path):
    graph_path = tmp_path / "c4,4.dot"

    graphs.save_labeled_two_cycle_graph(3, 3, ("0", "123"), graph_path)

    with open(graph_path, "r") as graph_file:
        expected = dedent(
            """\
            digraph  {
            1;
            2;
            3;
            0;
            4;
            5;
            6;
            1 -> 2  [key=0, label=0];
            2 -> 3  [key=0, label=0];
            3 -> 0  [key=0, label=0];
            0 -> 1  [key=0, label=0];
            0 -> 4  [key=0, label=123];
            4 -> 5  [key=0, label=123];
            5 -> 6  [key=0, label=123];
            6 -> 0  [key=0, label=123];
            }
            """
        )

        assert expected == "".join(graph_file.readlines())
