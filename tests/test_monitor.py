"""Acceptance checks for the standalone simulated dashboard."""

import numpy as np

from dashboard.demo_fixture import COLORS, fixture, snapshot
from dashboard.monitor import app, hydro_figure, map_figure


def test_repeated_zoom_preserves_pan_and_resets():
    from dashboard.monitor import update_camera
    from dash import no_update

    first = update_camera(None, "wm-zoom-in")
    second = update_camera(first, "wm-zoom-in")
    assert second["map.zoom"] == 12.65
    center = {"lat": -7.7, "lon": 112.8}
    panned = update_camera(second, "wm-map", {"map.center": center})
    assert panned["map.zoom"] == 12.65
    assert panned["map.center"] == center
    assert update_camera(panned, "wm-map", {"map.center": center}) is no_update
    assert update_camera(panned, "wm-reset")["map.zoom"] == 10.65
    assert update_camera(panned, "wm-viewport", viewport=390)["map.zoom"] == 9.4


def test_fixture_is_complete_and_scenario_has_delayed_peak():
    data = fixture()
    assert len(data["stations"]) == 15
    assert len(data["times"]) == 59
    assert set(r["status"] for r in snapshot()) == set(COLORS)
    assert (
        data["stations"]["Purwodadi"]["values"].argmax()
        < data["stations"]["Dhompo"]["values"].argmax()
    )
    target = data["stations"]["Dhompo"]
    assert target["values"].argmax() == 54  # now + 3h
    assert next(r for r in snapshot() if r["name"] == "Dhompo")["status"] == "Waspada"
    assert next(r for r in snapshot(3) if r["name"] == "Dhompo")["status"] == "Bahaya"


def test_frames_and_figures_share_values_and_thresholds():
    for lead in range(6):
        rows = snapshot(lead)
        for row in rows:
            assert np.isfinite(row["value"])
            assert row["value"] == row["values"][48 + 2 * lead]
            assert np.isclose(
                row["delta"],
                row["values"][48 + 2 * lead] - row["values"][42 + 2 * lead],
            )
            fig = hydro_figure(row["name"], lead)
            assert fig.data[-1].y[0] == row["value"]
            assert fig.data[1].y[-1] == fig.data[2].y[0]
        markers = map_figure(rows, "Dhompo", ["markers", "labels"])
        assert sum(bool(trace.customdata) for trace in markers.data) == 15
        assert len(map_figure(rows, "Dhompo", []).data) == 0


def test_dashboard_loads_without_model_provider():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from dashboard.app import app; assert 'dashboard.data' not in sys.modules; assert 'dhompo.serving.file_predictor' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    client = app.server.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/_dash-layout").status_code == 200
    dependencies = client.get("/_dash-dependencies").get_json()
    outputs = [item["output"] for item in dependencies]
    assert len(outputs) == len(set(outputs))


def test_map_layers_include_topology_and_keep_optional_assets_safe():
    rows = snapshot(0)
    figure = map_figure(
        rows, "Dhompo", ["topology", "basin", "rivers", "radar", "aquifer"]
    )
    # Two topology branches are available from repository coordinates even when
    # optional official GeoJSON assets have not been added yet.
    assert len(figure.data) == 2
    assert figure.layout.map.style == "carto-positron"
    topo = map_figure(rows, "Dhompo", ["topology"], basemap="open-street-map")
    assert len(topo.data) == 2
    assert topo.layout.map.style == "open-street-map"
