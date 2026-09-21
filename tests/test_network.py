"""Tes graf jaringan sungai 15 stasiun (``dhompo.data.network``)."""

from __future__ import annotations

import pytest

from dhompo.data.loader import ALL_STATIONS
from dhompo.data.network import (
    NETWORK_PATH,
    Reach,
    RiverNetwork,
    StationNode,
    load_network,
)


@pytest.fixture(scope="module")
def network() -> RiverNetwork:
    return load_network()


def test_network_file_exists():
    assert NETWORK_PATH.exists()


def test_network_has_all_loader_stations(network):
    assert set(network.station_names) == set(ALL_STATIONS)
    assert len(network.station_names) == 15


def test_target_and_aux_present(network):
    assert network.target == "Dhompo"
    assert network.downstream_aux == "Jalan Nasional"
    assert "Bd. Sentono" in network.station_names


def test_every_station_flows_to_target_except_aux(network):
    for name in network.station_names:
        if name == network.target:
            assert network.path_to_target(name) == [network.target]
        elif name == network.downstream_aux:
            assert network.path_to_target(name) == []
        else:
            path = network.path_to_target(name)
            assert path[0] == name
            assert path[-1] == network.target


def test_topological_order_respects_direction(network):
    order = network.order_upstream_to_downstream()
    assert len(order) == len(network.station_names)
    pos = {name: i for i, name in enumerate(order)}
    for upstream, downstream in network.edges:
        assert pos[upstream] < pos[downstream]
    assert pos["Dhompo"] < pos["Jalan Nasional"]


def test_dhompo_has_two_upstream_branches(network):
    upstream = set(network.upstream_of("Dhompo"))
    assert upstream == {"Bd. Grinting", "Klosod"}


def test_sentono_joins_west_branch(network):
    assert "Bd. Sentono" in network.upstream_of("Purwodadi")


def test_travel_hours_lookup(network):
    assert network.travel_hours("Klosod", "Dhompo") == pytest.approx(1.0)
    with pytest.raises(KeyError):
        network.travel_hours("Dhompo", "Klosod")


def _node(name: str) -> StationNode:
    return StationNode(
        name=name,
        elevation_m=1.0,
        travel_hours_to_target=1.0,
        branch="",
        role="upstream",
    )


def test_validate_rejects_unknown_reach():
    net = RiverNetwork(
        target="B",
        downstream_aux="C",
        nodes=(_node("A"), _node("B"), _node("C")),
        reaches=(Reach("A", "Z", 1.0), Reach("A", "B", 1.0), Reach("B", "C", 1.0)),
        defaults={},
    )
    with pytest.raises(ValueError, match="Reach tidak valid"):
        net.validate()


def test_validate_rejects_cycle():
    net = RiverNetwork(
        target="B",
        downstream_aux="C",
        nodes=(_node("A"), _node("B"), _node("C")),
        reaches=(Reach("A", "B", 1.0), Reach("B", "A", 1.0), Reach("B", "C", 1.0)),
        defaults={},
    )
    with pytest.raises(ValueError, match="siklus"):
        net.validate()


def test_validate_rejects_disconnected_station():
    net = RiverNetwork(
        target="B",
        downstream_aux="C",
        nodes=(_node("A"), _node("B"), _node("C"), _node("Z")),
        reaches=(Reach("A", "B", 1.0), Reach("B", "C", 1.0)),
        defaults={},
    )
    with pytest.raises(ValueError, match="tidak mengalir ke target"):
        net.validate()
