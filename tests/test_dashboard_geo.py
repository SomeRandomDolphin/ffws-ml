"""Tes geometri DAS dashboard: koordinat, topologi, dan waktu tempuh."""

from __future__ import annotations

import pandas as pd
import pytest

from dhompo.data.loader import TARGET_STATION

from dashboard.geo import (
    AUX_STATION,
    EAST_BRANCH,
    LOCAL_STATIONS,
    TRAVEL_ANCHORS,
    WEST_BRANCH,
    all_edges,
    load_station_geo,
    schematic_positions,
    travel_hours,
)


def test_station_geo_csv_has_all_stations():
    geo = load_station_geo()
    expected = set(
        WEST_BRANCH + EAST_BRANCH + LOCAL_STATIONS + [AUX_STATION]
    )
    assert expected <= set(geo.keys())
    assert len(geo) >= 14


def test_station_geo_coordinates_valid_range():
    geo = load_station_geo()
    for station, g in geo.items():
        assert -9.0 <= g.latitude <= -5.0, station
        assert 110.0 <= g.longitude <= 116.0, station


def test_travel_hours_anchors_exact():
    assert travel_hours("Purwodadi") == 3.5
    assert travel_hours("AWLR Kademungan") == 2.0
    assert travel_hours("Klosod") == 1.0
    assert travel_hours("Dhompo") == 0.0


def test_travel_hours_estimate_rounding_and_monotonic():
    # stasiun hulu harus punya waktu tempuh lebih lama ke Dhompo
    assert travel_hours("Bd. Suwoto") > travel_hours("Krajan Timur")
    assert travel_hours("Krajan Timur") > travel_hours("Purwodadi")
    for st in WEST_BRANCH + EAST_BRANCH:
        assert travel_hours(st) >= 0.5
        assert travel_hours(st) * 2 == round(travel_hours(st) * 2)  # kelipatan 0,5


def test_edges_form_connected_tree_to_dhompo():
    edges = all_edges()
    nodes = set()
    for a, b in edges:
        nodes.add(a)
        nodes.add(b)
    assert "Dhompo" in nodes
    assert len(nodes) >= 13
    # semua cabang mengalir menuju Dhompo: Dhompo harus ujung dari tiap cabang
    assert (WEST_BRANCH[-1], "Dhompo") in edges
    assert (EAST_BRANCH[-1], "Dhompo") in edges
    assert ("Dhompo", "Jalan Nasional") in edges


def test_schematic_positions_cover_all_nodes():
    pos = schematic_positions()
    edges = all_edges()
    for a, b in edges:
        assert a in pos and b in pos
    assert "Dhompo" in pos