"""Map-first frontend prototype with a self-contained simulation provider."""

from pathlib import Path
import json

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update

from dashboard.demo_fixture import COLORS, NOW, fixture, snapshot
from dashboard.geo import EAST_BRANCH, WEST_BRANCH, load_station_geo

GRAPH_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": True}
BASEMAP_STYLES = {
    "carto-positron": "carto-positron",
    # Plotly/MapLibre no longer renders the old Stamen style reliably.
    "stamen-terrain": "open-street-map",
    "open-street-map": "open-street-map",
    "carto-darkmatter": "carto-darkmatter",
}
GEO_DIR = Path(__file__).resolve().parents[1] / "data" / "geospatial" / "dhompo"


def _geojson_features(filename):
    path = GEO_DIR / filename
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get("features", [])
    except (OSError, ValueError, TypeError):
        return []


def _vector_layer(filename, color, fill=False, width=2):
    traces = []
    for feature in _geojson_features(filename):
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        kind = geometry.get("type")
        if kind == "LineString":
            groups = [coordinates]
        elif kind == "Polygon":
            groups = [coordinates[0]] if coordinates else []
        elif kind == "MultiLineString":
            groups = list(coordinates)
        elif kind == "MultiPolygon":
            groups = [part[0] for part in coordinates if part]
        else:
            continue
        for group in groups:
            if not group:
                continue
            traces.append(
                go.Scattermap(
                    lon=[point[0] for point in group],
                    lat=[point[1] for point in group],
                    mode="lines",
                    line=dict(color=color, width=width),
                    fill="toself" if fill else None,
                    fillcolor={
                        "#89a86c": "rgba(137,168,108,.18)",
                        "#e58a42": "rgba(229,138,66,.18)",
                    }.get(color, color)
                    if fill
                    else None,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    return traces


def _topology_layers():
    geo = load_station_geo()
    traces = []
    for branch, color in ((WEST_BRANCH, "#167cb5"), (EAST_BRANCH, "#55a8c7")):
        chain = branch + ["Dhompo"]
        traces.append(
            go.Scattermap(
                lon=[geo[name].longitude for name in chain if name in geo],
                lat=[geo[name].latitude for name in chain if name in geo],
                mode="lines",
                line=dict(color=color, width=3),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


def _relief_layers():
    """Soft hypsometric bands for the demo until a real DEM is available."""
    bands = [
        (-7.78, 112.73, 0.11, 0.075, "rgba(72,145,91,.17)", "Dataran rendah · 0–200 m"),
        (-7.72, 112.81, 0.095, 0.065, "rgba(188,181,73,.20)", "Lereng · 200–500 m"),
        (
            -7.67,
            112.78,
            0.075,
            0.055,
            "rgba(194,135,64,.22)",
            "Dataran tinggi · 500–900 m",
        ),
        (-7.64, 112.70, 0.058, 0.042, "rgba(135,91,61,.20)", "Dataran tinggi · 900+ m"),
    ]
    traces = []
    for lat, lon, rx, ry, color, label in bands:
        for size, opacity in ((220, 0.06), (145, 0.08), (85, 0.11)):
            traces.append(
                go.Scattermap(
                    lat=[lat],
                    lon=[lon],
                    mode="markers",
                    marker=dict(size=size, color=color, opacity=opacity),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    return traces


def pill(text, color):
    return html.Span(
        text, className="wm-pill", style={"color": color, "background": color + "15"}
    )


def station_popup(row):
    if not row:
        return None
    return html.Div(
        [
            html.Button("×", id="wm-popup-close", className="wm-popup-close"),
            html.H3(f"{row['name']} · DAS Welang"),
            html.Div(
                [
                    html.Span("Muka air, meter", className="wm-popup-label"),
                    html.Strong(f"{row['value']:.2f} m", className="wm-popup-value"),
                    html.Span(
                        f"{'↑' if row['delta'] >= 0 else '↓'} {abs(row['delta']):.2f} m dalam 3 jam",
                        className="wm-popup-trend",
                    ),
                ],
                className="wm-popup-reading",
            ),
            html.P(
                [pill(row["status"], row["color"]), " Data simulasi untuk prototipe"],
                className="wm-popup-status",
            ),
            html.Div(
                [
                    html.Button("⌖ Monitoring Location", className="wm-popup-action primary"),
                    html.Button("▥ Grafik statistik harian", className="wm-popup-action"),
                    html.Button("♟ Buat WaterAlert", className="wm-popup-action"),
                ],
                className="wm-popup-actions",
            ),
        ],
        className="wm-station-popup",
    )


def map_figure(rows, selected, layers, reset=0, basemap="carto-positron"):
    fig = go.Figure()
    if "topology" in layers:
        fig.add_traces(_topology_layers())
    if "basin" in layers:
        fig.add_traces(_vector_layer("basin_boundary.geojson", "#0d6b92", width=3))
    if "rivers" in layers:
        fig.add_traces(_vector_layer("rivers.geojson", "#2b8cc4", width=2))
    if "aquifer" in layers:
        fig.add_traces(_vector_layer("aquifers.geojson", "#89a86c", fill=True, width=1))
    if "radar" in layers:
        fig.add_traces(
            _vector_layer("radar_static.geojson", "#e58a42", fill=True, width=1)
        )
    if "relief" in layers:
        fig.add_traces(_relief_layers())
    if "markers" in layers:
        for row in rows:
            g = row["geo"]
            active = row["name"] == selected
            if active:
                fig.add_trace(
                    go.Scattermap(
                        lat=[g.latitude],
                        lon=[g.longitude],
                        mode="markers",
                        marker={
                            "size": 32,
                            "color": "#102D4E",
                            "opacity": 0.18,
                            "symbol": "circle",
                        },
                        hoverinfo="skip",
                    )
                )
            fig.add_trace(
                go.Scattermap(
                    lat=[g.latitude],
                    lon=[g.longitude],
                    mode="markers",
                    marker={
                        "size": 21 if active else 16,
                        "color": "white",
                        "symbol": "circle",
                    },
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scattermap(
                    lat=[g.latitude],
                    lon=[g.longitude],
                    mode="markers+text" if "labels" in layers else "markers",
                    marker={
                        "size": 14 if active else 10,
                        "color": row["color"],
                        "symbol": "circle",
                    },
                    text=[row["name"]],
                    textposition="top right",
                    textfont={"size": 11, "color": "#102D4E"},
                    customdata=[row["name"]],
                    hovertemplate=(
                        f"<b>{row['name'].upper()}</b><br>"
                        "<span style='color:#667085'>Muka air, meter</span><br>"
                        f"<b style='font-size:18px'>{row['value']:.2f} m</b><br>"
                        "<span style='color:#667085'>Waktu simulasi: sekarang</span><br>"
                        f"<span style='color:#667085'>Status: </span><b>{row['status']}</b><br>"
                        f"<span style=\"color:{'#b42318' if row['delta'] >= 0 else '#027a48'}\">"
                        f"{'Naik' if row['delta'] >= 0 else 'Turun'} {abs(row['delta']):.2f} m dalam 3 jam</span>"
                        "<extra></extra>"
                    ),
                )
            )
    fig.update_layout(
        template="none",
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="#c8d0d8",
            font=dict(family="IBM Plex Sans", color="#1e2a32", size=12),
            align="left",
            namelength=0,
        ),
        map=dict(
            style=BASEMAP_STYLES.get(basemap, "carto-positron"),
            center=dict(lat=-7.738, lon=112.794),
            zoom=10.65,
        ),
        legend=dict(
            orientation="v",
            x=0.01,
            y=0.01,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,.9)",
            bordercolor="#dbe4eb",
            borderwidth=1,
            font=dict(size=10),
        ),
        uirevision=f"welang-{reset}",
        paper_bgcolor="#e9eff0",
        font=dict(family="IBM Plex Sans"),
    )
    return fig


def hydro_figure(selected, lead):
    data = fixture()
    station = data["stations"][selected]
    times, values = data["times"], station["values"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=times,
            y=data["rain"],
            yaxis="y2",
            name="Hujan simulasi",
            marker_color="rgba(22,124,181,.13)",
            hovertemplate="%{y:.1f} mm / 30 menit<extra>Hujan</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times[:49],
            y=values[:49],
            name="Riwayat simulasi",
            mode="lines",
            line=dict(color="#167CB5", width=2.5),
            hovertemplate="%{y:.2f} m<extra>Riwayat</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=times[48:],
            y=values[48:],
            name="Proyeksi simulasi",
            mode="lines",
            line=dict(color="#167CB5", width=2.5, dash="dash"),
            hovertemplate="%{y:.2f} m<extra>Proyeksi</extra>",
        )
    )
    for name, key in [("Waspada", "alert"), ("Bahaya", "danger")]:
        fig.add_hline(
            y=station[key],
            line_dash="dot",
            line_color=COLORS[name],
            line_width=1,
            annotation_text=f"{name} · {station[key]:.2f}",
            annotation_position="top left",
            annotation_font_size=10,
        )
    fig.add_vline(
        x=NOW.timestamp() * 1000, line_color="#a4b2bf", line_dash="dash", line_width=1
    )
    active = NOW + pd.Timedelta(hours=lead)
    fig.add_trace(
        go.Scatter(
            x=[active],
            y=[values[48 + lead * 2]],
            mode="markers",
            marker=dict(size=9, color="#102D4E", line=dict(color="white", width=2)),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        template="none",
        margin=dict(l=48, r=40, t=15, b=28),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="IBM Plex Sans", size=10, color="#66798a"),
        showlegend=False,
        hovermode="x unified",
        xaxis=dict(tickformat="%H:%M", gridcolor="#f2f5f7", zeroline=False),
        yaxis=dict(
            title="m",
            gridcolor="#edf1f4",
            zeroline=False,
            range=[min(values) - 0.18, max(max(values), station["danger"]) + 0.30],
        ),
        yaxis2=dict(
            overlaying="y",
            side="right",
            title="mm",
            range=[0, 60],
            showgrid=False,
            zeroline=False,
            tickvals=[0, 20, 40],
        ),
        uirevision=selected,
        bargap=0.2,
    )
    return fig


NAV_TABS = ("overview", "layers", "legend", "tools")

_NAV_BAR = (
    ("overview", "ⓘ", "Overview", None),
    ("layers", "▤", "Layers", 3),
    ("legend", "≡", "Legend", 3),
    ("tools", "⚙", "Tools", None),
)


def _nav_row(icon, label, desc):
    return html.Div(
        className="wm-nav-row",
        children=[
            html.Span(icon, className="wm-nav-row-ico"),
            html.Span(label, className="wm-nav-row-label"),
            html.Span(desc, className="wm-nav-row-desc"),
        ],
    )


def _nav_page_overview():
    return html.Div(
        className="wm-nav-welcome",
        children=[
            html.H3("Welcome to the"),
            html.H2("Welang Water Dashboard"),
            html.Span("PENJELASAN NAVIGASI ATAS", className="wm-nav-heading"),
            _nav_row("▤", "Layers", "Tambah dan hapus lapisan peta"),
            _nav_row("≡", "Legend", "Penjelasan lapisan yang tampil"),
            _nav_row("⚙", "Tools", "Peralatan tambahan peta"),
            html.P(
                "Dashboard ini menampilkan kondisi muka air hasil simulasi di 15 stasiun "
                "pemantauan DAS Welang, Pasuruan, dalam konteks hujan dan ambang waspada.",
                className="wm-nav-note",
            ),
            html.Div(
                className="wm-nav-actions",
                children=[
                    html.Button("▤ Pernyataan Data", className="wm-nav-action"),
                    html.Button("? FAQ", className="wm-nav-action"),
                ],
            ),
        ],
    )


def _nav_page_layers():
    return html.Div(
        className="wm-nav-page",
        children=[
            html.H3("Layers"),
            html.P("Pilih lapisan yang ditampilkan pada peta."),
            html.Label("BASE MAP", className="wm-nav-field-label"),
            html.Div(
                [
                    html.Button(
                        [
                            html.Img(src="/assets/basemap-light.svg", alt="Preview basemap Terang"),
                            html.Span("Terang"),
                        ],
                        id={"type": "wm-basemap-card", "value": "carto-positron"},
                        n_clicks=0,
                        className="wm-basemap-card",
                    ),
                    html.Button(
                        [
                            html.Img(src="/assets/basemap-topo.svg", alt="Preview basemap Topografi"),
                            html.Span("Topografi"),
                        ],
                        id={"type": "wm-basemap-card", "value": "stamen-terrain"},
                        n_clicks=0,
                        className="wm-basemap-card",
                    ),
                    html.Button(
                        [
                            html.Img(src="/assets/basemap-street.svg", alt="Preview basemap Street map"),
                            html.Span("Street map"),
                        ],
                        id={"type": "wm-basemap-card", "value": "open-street-map"},
                        n_clicks=0,
                        className="wm-basemap-card",
                    ),
                    html.Button(
                        [
                            html.Img(src="/assets/basemap-dark.svg", alt="Preview basemap Gelap"),
                            html.Span("Gelap"),
                        ],
                        id={"type": "wm-basemap-card", "value": "carto-darkmatter"},
                        n_clicks=0,
                        className="wm-basemap-card",
                    ),
                ],
                className="wm-basemap-grid",
            ),
            dcc.RadioItems(
                id="wm-basemap",
                options=[
                    {"label": " Terang", "value": "carto-positron"},
                    {"label": "Topografi", "value": "stamen-terrain"},
                    {"label": "Street map", "value": "open-street-map"},
                    {"label": "Gelap", "value": "carto-darkmatter"},
                ],
                value="carto-positron",
                className="wm-basemap-options wm-hidden-control",
                persistence=True,
                persistence_type="session",
            ),
            html.Label("OVERLAY MAP", className="wm-nav-field-label"),
            dcc.Checklist(
                id="wm-layer-checklist",
                options=[
                    {"label": " Relief warna (Kategori ketinggian)", "value": "relief"},
                    {
                        "label": " Stasiun pemantauan (titik muka air)",
                        "value": "markers",
                    },
                    {"label": " Nama stasiun", "value": "labels"},
                    {
                        "label": " Topologi DAS Welang (jaringan stasiun)",
                        "value": "topology",
                    },
                    {"label": " Batas DAS Welang (GeoJSON lokal)", "value": "basin"},
                    {"label": " Rivers (GeoJSON lokal)", "value": "rivers"},
                    {"label": " Radar: Static (GeoJSON lokal)", "value": "radar"},
                    {"label": " Aquifer (GeoJSON lokal)", "value": "aquifer"},
                ],
                value=["relief", "markers", "labels", "topology"],
                persistence=True,
                persistence_type="session",
            ),
            html.Div(
                "Rivers, Radar, dan Aquifer menunggu aset lokal resmi.",
                className="wm-nav-data-note",
            ),
        ],
    )


def _nav_page_legend():
    return html.Div(
        className="wm-nav-page",
        children=[
            html.H3("Legend"),
            html.P("Layer peta dan topologi", className="wm-nav-note"),
            html.P("Status stasiun relatif terhadap ambang:", className="wm-nav-note"),
            html.H4("Topologi dan peta", className="wm-nav-subtitle"),
            html.Div(
                [
                    html.Div(
                        [
                            html.I(style={"background": "#167cb5"}),
                            " Topologi DAS Welang · jaringan stasiun",
                        ]
                    ),
                    html.Div(
                        [
                            html.I(style={"background": "#2b8cc4"}),
                            " Rivers · sungai dari GeoJSON lokal",
                        ]
                    ),
                    html.Div(
                        [
                            html.I(style={"background": "#e58a42"}),
                            " Radar · intensitas hujan dBZ",
                        ]
                    ),
                    html.Div(
                        [
                            html.I(style={"background": "#89a86c"}),
                            " Aquifer · klasifikasi geologi",
                        ]
                    ),
                ],
                className="wm-nav-legend-more",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.I(style={"background": color}),
                            html.Span(name),
                        ],
                        className="wm-nav-legend-row",
                    )
                    for name, color in COLORS.items()
                ],
                className="wm-nav-legend-list",
            ),
            html.Div(
                className="wm-nav-legend-more",
                children=[
                    html.Div(
                        [
                            html.I(style={"background": "#48915b"}),
                            " Dataran rendah · 0–200 m",
                        ]
                    ),
                    html.Div(
                        [
                            html.I(style={"background": "#bcb549"}),
                            " Lereng · 200–500 m",
                        ]
                    ),
                    html.Div(
                        [
                            html.I(style={"background": "#c28740"}),
                            " Dataran tinggi · 500+ m",
                        ]
                    ),
                    html.Div(
                        [
                            html.I(style={"background": "#8a5a33"}),
                            " Dataran tinggi · 900+ m",
                        ]
                    ),
                ],
            ),
            html.P(
                "━ Riwayat · ┄ Proyeksi · ▥ Hujan (mm/30 menit)",
                className="wm-nav-note",
            ),
        ],
    )


def _nav_page_tools():
    return html.Div(
        children=[
            html.H3("Tools"),
            _nav_row("⌖", "Cakupan DAS", "Kembali ke cakupan penuh DAS Welang"),
            _nav_row("＋", "Zoom", "Perbesar / perkecil peta"),
            _nav_row("▶", "Putar", "Animasi horizon proyeksi sampai +5 jam"),
            _nav_row("▥", "Lipat grafik", "Sembunyikan panel riwayat dan proyeksi"),
        ],
    )


def layout():
    return html.Div(
        id="wm-app",
        className="wm-app",
        children=[
            dcc.Store(id="wm-selected", data="Dhompo", storage_type="session"),
            dcc.Store(id="wm-camera", storage_type="session"),
            dcc.Store(id="wm-popup-station"),
            dcc.Store(id="wm-viewport", data=1440),
            dcc.Store(id="wm-nav-tab", data="overview"),
            dcc.Interval(id="wm-tick", interval=1500, disabled=True),
            html.Header(
                className="wm-header",
                children=[
                    html.Div(
                        className="wm-brand",
                        children=[
                            html.Span("≋", className="wm-logo"),
                            html.Div([html.H1("Welang"), html.Span("WATER MONITOR")]),
                        ],
                    ),
                    html.Div(
                        "DAS Welang / Pasuruan, Jawa Timur", className="wm-location"
                    ),
                    html.Div(
                        [html.Span(className="wm-clock-dot"), html.Span(id="wm-time")],
                        className="wm-clock",
                    ),
                    html.Span("DEMO · DATA SIMULASI", className="wm-demo"),
                ],
            ),
            html.Div(
                className="wm-nav-wrap",
                children=[
                    html.Nav(
                        className="wm-nav",
                        children=[
                            html.Button(
                                [
                                    html.Span("ⓘ", className="wm-nav-ico"),
                                    "Overview",
                                ],
                                id="wm-nav-overview",
                                n_clicks=0,
                                className="wm-nav-btn",
                            ),
                            html.Button(
                                [
                                    html.Span("▤", className="wm-nav-ico"),
                                    "Layers",
                                    html.Span("7", className="wm-nav-badge"),
                                ],
                                id="wm-nav-layers",
                                n_clicks=0,
                                className="wm-nav-btn",
                            ),
                            html.Button(
                                [
                                    html.Span("≡", className="wm-nav-ico"),
                                    "Legend",
                                    html.Span("7", className="wm-nav-badge"),
                                ],
                                id="wm-nav-legend",
                                n_clicks=0,
                                className="wm-nav-btn",
                            ),
                            html.Button(
                                [
                                    html.Span("⚙", className="wm-nav-ico"),
                                    "Tools",
                                ],
                                id="wm-nav-tools",
                                n_clicks=0,
                                className="wm-nav-btn",
                            ),
                        ],
                    ),
                    html.Div(
                        id="wm-nav-panel",
                        className="wm-nav-panel",
                        children=[
                            html.Button(
                                "✕ Close",
                                id="wm-nav-close",
                                n_clicks=0,
                                className="wm-nav-close",
                            ),
                            html.Div(
                                "Overview", id="wm-nav-title", className="wm-nav-title"
                            ),
                            html.Div(
                                id="wm-nav-body",
                                className="wm-nav-body",
                                children=[
                                    html.Div(
                                        id="wm-nav-page-overview",
                                        className="wm-nav-page-item",
                                        style={"display": "block"},
                                        children=[_nav_page_overview()],
                                    ),
                                    html.Div(
                                        id="wm-nav-page-layers",
                                        className="wm-nav-page-item",
                                        style={"display": "none"},
                                        children=[_nav_page_layers()],
                                    ),
                                    html.Div(
                                        id="wm-nav-page-legend",
                                        className="wm-nav-page-item",
                                        style={"display": "none"},
                                        children=[_nav_page_legend()],
                                    ),
                                    html.Div(
                                        id="wm-nav-page-tools",
                                        className="wm-nav-page-item",
                                        style={"display": "none"},
                                        children=[_nav_page_tools()],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Main(
                className="wm-workspace",
                children=[
                    html.Section(
                        className="wm-map-area",
                        children=[
                            dcc.Graph(
                                id="wm-map",
                                config=GRAPH_CONFIG,
                                className="wm-map",
                                responsive=True,
                            ),
                            html.Div(id="wm-map-popup"),
                            html.Div(
                                className="wm-map-top",
                                children=[
                                    html.Div(
                                        [
                                            html.Span(
                                                [
                                                    html.I(
                                                        style={"background": "#48915b"}
                                                    ),
                                                    " rendah",
                                                ]
                                            ),
                                            html.Span(
                                                [
                                                    html.I(
                                                        style={"background": "#bcb549"}
                                                    ),
                                                    " lereng",
                                                ]
                                            ),
                                            html.Span(
                                                [
                                                    html.I(
                                                        style={"background": "#c28740"}
                                                    ),
                                                    " tinggi",
                                                ]
                                            ),
                                        ],
                                        className="wm-relief-legend",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                "JARINGAN PEMANTAUAN",
                                                className="wm-eyebrow",
                                            ),
                                            html.H2("Dari hulu, menuju Dhompo."),
                                            html.P(
                                                "15 stasiun · Skenario hujan di hulu"
                                            ),
                                        ],
                                        className="wm-map-title",
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "+",
                                                id="wm-zoom-in",
                                                n_clicks=0,
                                                className="wm-map-button",
                                                title="Perbesar peta",
                                            ),
                                            html.Button(
                                                "−",
                                                id="wm-zoom-out",
                                                n_clicks=0,
                                                className="wm-map-button",
                                                title="Perkecil peta",
                                            ),
                                            html.Button(
                                                "⌖ Cakupan DAS",
                                                id="wm-reset",
                                                n_clicks=0,
                                                className="wm-map-button",
                                            ),
                                        ],
                                        className="wm-map-controls",
                                    ),
                                ],
                            ),
                            html.Div(id="wm-counts", className="wm-counts"),
                            html.Div(id="wm-dhompo", className="wm-dhompo"),
                            html.Div(
                                className="wm-map-bottom",
                                children=[
                                    html.Div(
                                        [
                                            html.Span(
                                                [
                                                    html.I(style={"background": color}),
                                                    name,
                                                ]
                                            )
                                            for name, color in COLORS.items()
                                        ],
                                        className="wm-legend",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Aside(
                        className="wm-rail",
                        children=[
                            html.Div(
                                [
                                    html.Div(
                                        "STASIUN PEMANTAUAN", className="wm-eyebrow"
                                    ),
                                    html.Span("15", className="wm-total"),
                                ],
                                className="wm-rail-heading",
                            ),
                            html.Label(
                                "Cari stasiun",
                                htmlFor="wm-search",
                                className="wm-sr-only",
                            ),
                            dcc.Input(
                                id="wm-search",
                                placeholder="Cari nama stasiun…",
                                type="search",
                                debounce=False,
                                persistence=True,
                                persistence_type="session",
                                className="wm-search",
                            ),
                            html.Div(
                                dcc.Dropdown(
                                    id="wm-filter",
                                    options=[{"label": "Semua status", "value": "all"}]
                                    + [{"label": s, "value": s} for s in COLORS],
                                    value="all",
                                    clearable=False,
                                    searchable=False,
                                    persistence=True,
                                    persistence_type="session",
                                ),
                                className="wm-filter-wrap",
                            ),
                            dcc.RadioItems(
                                id="wm-rail-tab",
                                options=["Daftar", "Detail"],
                                value="Daftar",
                                className="wm-rail-tabs",
                                persistence=True,
                                persistence_type="session",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            "STASIUN / STATUS",
                                            html.Span("MUKA AIR / Δ 3J"),
                                        ],
                                        className="wm-table-head",
                                    ),
                                    html.Div(id="wm-stations"),
                                ],
                                id="wm-list-area",
                                className="wm-list-area",
                            ),
                            html.Div(id="wm-detail", className="wm-detail"),
                        ],
                    ),
                ],
            ),
            html.Section(
                id="wm-chart-panel",
                className="wm-chart-panel",
                children=[
                    html.Div(
                        className="wm-chart-heading",
                        children=[
                            html.Div(
                                [
                                    html.H2(id="wm-chart-title"),
                                    html.Span(
                                        "24 jam riwayat / 5 jam proyeksi",
                                        className="wm-chart-subtitle",
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Span("━ Riwayat", className="wm-key"),
                                    html.Span("┄ Proyeksi", className="wm-key"),
                                    html.Span("▥ Hujan", className="wm-key"),
                                    dcc.Checklist(
                                        id="wm-collapse",
                                        options=[
                                            {
                                                "label": " Lipat grafik",
                                                "value": "collapsed",
                                            }
                                        ],
                                        value=[],
                                        persistence=True,
                                        persistence_type="session",
                                    ),
                                ],
                                className="wm-chart-actions",
                            ),
                        ],
                    ),
                    dcc.Graph(
                        id="wm-hydro",
                        config={"displayModeBar": False, "responsive": True},
                        className="wm-hydro",
                        responsive=True,
                    ),
                    html.Div(
                        className="wm-timeline",
                        children=[
                            html.Button("▶ Putar", id="wm-play", n_clicks=0),
                            html.Div(
                                [
                                    html.Span("WAKTU SIMULASI", className="wm-eyebrow"),
                                    html.Strong(id="wm-lead-label"),
                                ],
                                className="wm-timeline-label",
                            ),
                            dcc.Slider(
                                id="wm-lead",
                                min=0,
                                max=5,
                                step=1,
                                value=0,
                                marks={
                                    0: "Sekarang",
                                    **{i: f"+{i} jam" for i in range(1, 6)},
                                },
                                persistence=True,
                                persistence_type="session",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


app = dash.Dash(
    __name__,
    title="Welang · Water Monitor",
    assets_folder=str(Path(__file__).parent / "assets"),
    suppress_callback_exceptions=True,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap"
    ],
)
app.layout = layout


@app.callback(
    Output("wm-basemap", "value"),
    Input({"type": "wm-basemap-card", "value": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_basemap(cards):
    if isinstance(ctx.triggered_id, dict) and any(cards or []):
        return ctx.triggered_id["value"]
    return no_update


@app.callback(
    Output({"type": "wm-basemap-card", "value": ALL}, "className"),
    Input("wm-basemap", "value"),
)
def style_basemap_cards(value):
    values = ["carto-positron", "stamen-terrain", "open-street-map", "carto-darkmatter"]
    return [
        "wm-basemap-card is-selected" if current == value else "wm-basemap-card"
        for current in values
    ]

app.clientside_callback(
    """function(reset) {
        if (!window.welangResize) {
            window.welangResize = true;
            let pending;
            window.addEventListener('resize', function() {
                clearTimeout(pending);
                pending = setTimeout(function() {
                    dash_clientside.set_props('wm-viewport', {data: window.innerWidth});
                }, 150);
            });
        }
        return window.innerWidth;
    }""",
    Output("wm-viewport", "data"),
    Input("wm-reset", "n_clicks"),
)


@app.callback(
    Output("wm-selected", "data"),
    Output("wm-popup-station", "data"),
    Input("wm-map", "clickData"),
    Input({"type": "wm-station", "name": ALL}, "n_clicks"),
    Input("wm-popup-close", "n_clicks"),
    prevent_initial_call=True,
)
def select_station(click, clicks, close):
    if ctx.triggered_id == "wm-popup-close":
        return no_update, ""
    if isinstance(ctx.triggered_id, dict):
        name = ctx.triggered_id["name"] if any(clicks) else None
        return name or no_update, name or no_update
    points = (click or {}).get("points", [])
    name = points[0].get("customdata") if points else None
    return (
        (name, name)
        if name in fixture()["stations"]
        else (no_update, no_update)
    )


def update_camera(camera, action, relayout=None, viewport=1440):
    mobile = (viewport or 1440) <= 700
    defaults = {
        "map.center": {"lat": -7.69 if mobile else -7.738, "lon": 112.794},
        "map.zoom": 9.4 if mobile else 10.65,
    }
    current = {**defaults, **(camera or {})}
    revision = current.get("revision", 0)
    if action in ("wm-reset", "wm-viewport"):
        return {**defaults, "revision": revision + 1}
    if action in ("wm-zoom-in", "wm-zoom-out"):
        change = 1 if action == "wm-zoom-in" else -1
        return {
            **current,
            "map.zoom": max(3, min(18, current["map.zoom"] + change)),
            "revision": revision + 1,
        }
    changes = {
        k: v
        for k, v in (relayout or {}).items()
        if k in ("map.center", "map.zoom", "map.bearing", "map.pitch")
    }
    updated = {**current, **changes}
    return updated if changes and updated != current else no_update


@app.callback(
    Output("wm-camera", "data"),
    Input("wm-map", "relayoutData"),
    Input("wm-reset", "n_clicks"),
    Input("wm-zoom-in", "n_clicks"),
    Input("wm-zoom-out", "n_clicks"),
    Input("wm-viewport", "data"),
    State("wm-camera", "data"),
    prevent_initial_call=True,
)
def remember_camera(relayout, reset, zoom_in, zoom_out, viewport, camera):
    return update_camera(camera, ctx.triggered_id, relayout, viewport)


@app.callback(
    Output("wm-map", "figure"),
    Output("wm-stations", "children"),
    Output("wm-detail", "children"),
    Output("wm-hydro", "figure"),
    Output("wm-time", "children"),
    Output("wm-lead-label", "children"),
    Output("wm-counts", "children"),
    Output("wm-dhompo", "children"),
    Output("wm-chart-title", "children"),
    Output("wm-map-popup", "children"),
    Input("wm-selected", "data"),
    Input("wm-lead", "value"),
    Input("wm-search", "value"),
    Input("wm-filter", "value"),
    Input("wm-layer-checklist", "value"),
    Input("wm-basemap", "value"),
    Input("wm-viewport", "data"),
    Input("wm-camera", "data"),
    Input("wm-popup-station", "data"),
)
def render(selected, lead, search, status, layers, basemap, viewport, camera, popup_station):
    selected = selected if selected in fixture()["stations"] else "Dhompo"
    lead = int(lead or 0)
    rows = snapshot(lead)
    row = next(r for r in rows if r["name"] == selected)
    dhompo = next(r for r in rows if r["name"] == "Dhompo")
    visible = [
        r
        for r in rows
        if (search or "").casefold() in r["name"].casefold()
        and (status == "all" or r["status"] == status)
    ]
    buttons = [
        html.Button(
            [
                html.Span(
                    [
                        html.I(style={"background": r["color"]}),
                        html.Span(
                            [
                                html.Strong(r["name"]),
                                html.Small(r["status"], style={"color": r["color"]}),
                            ]
                        ),
                    ],
                    className="wm-station-name",
                ),
                html.Span(
                    [
                        html.Strong(f"{r['value']:.2f} m"),
                        html.Small(
                            f"{'↑' if r['delta'] >= 0 else '↓'} {abs(r['delta']):.2f} m / 3j"
                        ),
                    ],
                    className="wm-station-values",
                ),
            ],
            id={"type": "wm-station", "name": r["name"]},
            n_clicks=0,
            className="wm-station" + (" is-selected" if r["name"] == selected else ""),
        )
        for r in visible
    ]
    if not buttons:
        buttons = html.Div(
            "Tidak ada stasiun yang cocok. Ubah pencarian atau pilih Semua status.",
            className="wm-empty",
        )
    peak_index = 48 + int(row["values"][48:].argmax())
    peak_time = fixture()["times"][peak_index].strftime("%H:%M")
    detail = [
        html.Div(
            [
                html.Span("STASIUN TERPILIH", className="wm-eyebrow"),
                pill(row["status"], row["color"]),
            ],
            className="wm-detail-top",
        ),
        html.H2(selected),
        html.Div([f"{row['value']:.2f}", html.Span(" m")], className="wm-level"),
        html.P(
            f"{'↑' if row['delta'] >= 0 else '↓'} {abs(row['delta']):.2f} m dalam 3 jam",
            className="wm-trend",
        ),
        html.Div(
            [
                html.Div([html.Small("WASPADA"), html.Strong(f"{row['alert']:.2f} m")]),
                html.Div([html.Small("BAHAYA"), html.Strong(f"{row['danger']:.2f} m")]),
            ],
            className="wm-thresholds",
        ),
        html.P(
            [
                "Puncak proyeksi ",
                html.Strong(f"{max(row['values'][48:]):.2f} m"),
                f" pada {peak_time} WIB",
            ],
            className="wm-peak",
        ),
        html.Small(
            "Muka air, hujan, dan ambang adalah simulasi.", className="wm-disclaimer"
        ),
    ]
    counts = [
        html.Span(
            [
                html.I(style={"background": c}),
                html.Strong(str(sum(r["status"] == s for r in rows))),
                s,
            ]
        )
        for s, c in COLORS.items()
    ]
    focus = [
        html.Div(
            [
                html.Span("TITIK FOKUS / DHOMPO", className="wm-eyebrow"),
                pill(dhompo["status"], dhompo["color"]),
            ],
            className="wm-focus-heading",
        ),
        html.Div(
            [
                html.Strong(f"{dhompo['value']:.2f}"),
                html.Span(" m"),
                html.Small(
                    f"{'↑' if dhompo['delta']>=0 else '↓'} {abs(dhompo['delta']):.2f} m / 3j"
                ),
            ],
            className="wm-focus-value",
        ),
    ]
    mobile = (viewport or 1440) <= 700
    figure = map_figure(
        rows,
        selected,
        layers or [],
        f"{(camera or {}).get('revision',0)}-{mobile}-{basemap}",
        basemap=basemap,
    )
    if mobile:
        figure.update_layout(map=dict(center=dict(lat=-7.69, lon=112.794), zoom=9.4))
    # Restore a previous session's viewport; map updates keep it via uirevision.
    if camera and ctx.triggered_id != "wm-viewport":
        for key, value in camera.items():
            prop = key.removeprefix("map.")
            if prop in ("center", "zoom", "bearing", "pitch"):
                figure.update_layout(map={prop: value})
    active = NOW + pd.Timedelta(hours=lead)
    return (
        figure,
        buttons,
        detail,
        hydro_figure(selected, lead),
        active.strftime("18 Feb 2026 · %H:%M WIB"),
        active.strftime("%H:%M WIB"),
        counts,
        focus,
        f"Muka air · {selected}",
        station_popup(
            next(
                (
                    r
                    for r in rows
                    if r["name"]
                    == (
                        popup_station
                        if popup_station is not None
                        else (selected if selected != "Dhompo" else "")
                    )
                ),
                None,
            )
        ),
    )


@app.callback(
    Output("wm-lead", "value"),
    Output("wm-tick", "disabled"),
    Output("wm-play", "children"),
    Input("wm-play", "n_clicks"),
    Input("wm-tick", "n_intervals"),
    Input("wm-lead", "value"),
    State("wm-tick", "disabled"),
    prevent_initial_call=True,
)
def playback(clicks, ticks, lead, paused):
    lead = int(lead or 0)
    if ctx.triggered_id == "wm-play":
        paused = not paused
        if not paused and lead == 5:
            lead = 0
    elif ctx.triggered_id == "wm-tick" and not paused:
        lead = min(5, lead + 1)
        paused = lead == 5
    elif lead == 5:
        paused = True
    return lead, paused, "▶ Putar" if paused else "Ⅱ Jeda"


@app.callback(
    Output("wm-nav-tab", "data"),
    Input("wm-nav-overview", "n_clicks"),
    Input("wm-nav-layers", "n_clicks"),
    Input("wm-nav-legend", "n_clicks"),
    Input("wm-nav-tools", "n_clicks"),
    Input("wm-nav-close", "n_clicks"),
    State("wm-nav-tab", "data"),
    prevent_initial_call=True,
)
def nav_tab(overview, layers, legend, tools, close, current):
    tid = ctx.triggered_id
    if tid == "wm-nav-close":
        return "" if current else no_update
    tid = str(tid).replace("wm-nav-", "")
    return tid if tid != current else no_update


@app.callback(
    Output("wm-nav-panel", "className"),
    Output("wm-nav-title", "children"),
    Output("wm-nav-page-overview", "style"),
    Output("wm-nav-page-layers", "style"),
    Output("wm-nav-page-legend", "style"),
    Output("wm-nav-page-tools", "style"),
    Output("wm-nav-overview", "className"),
    Output("wm-nav-layers", "className"),
    Output("wm-nav-legend", "className"),
    Output("wm-nav-tools", "className"),
    Input("wm-nav-tab", "data"),
)
def sync_nav_panel(tab):
    open_tab = tab if tab in NAV_TABS else None
    titles = {
        "overview": "Overview",
        "layers": "Layers",
        "legend": "Legend",
        "tools": "Tools",
    }
    panel = f"wm-nav-panel is-open is-{open_tab}" if open_tab else "wm-nav-panel"
    page_styles = {
        t: {"display": "block" if t == open_tab else "none"} for t in NAV_TABS
    }
    buttons = {
        t: "wm-nav-btn is-active" if t == open_tab else "wm-nav-btn" for t in NAV_TABS
    }
    return (
        panel,
        titles[open_tab] if open_tab else "Overview",
        page_styles["overview"],
        page_styles["layers"],
        page_styles["legend"],
        page_styles["tools"],
        buttons["overview"],
        buttons["layers"],
        buttons["legend"],
        buttons["tools"],
    )


@app.callback(
    Output("wm-app", "className"),
    Input("wm-collapse", "value"),
    Input("wm-rail-tab", "value"),
)
def panels(collapsed, tab):
    return (
        "wm-app"
        + (" chart-collapsed" if collapsed else "")
        + (" detail-tab" if tab == "Detail" else "")
    )
