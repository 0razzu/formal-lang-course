import pytest
import project.graphs as graphs



def test_get_props_by_name():
    assert graphs.GraphProps(
        node_quan=1687,
        edge_quan=1453,
        labels={'a', 'd'}
    ) == graphs.get_props_by_name('ls')
