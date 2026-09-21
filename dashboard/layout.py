"""Editorial atlas layout for the Dhompo watershed dashboard."""

from __future__ import annotations

import dash
from dash import dcc, html

from dashboard.data import get_demo


def _lead_marks(lead_hours: list[int]) -> dict[int, str]:
    marks = {0: "Sekarang"}
    marks.update({h: f"+{h}j" for h in lead_hours if h > 0})
    return marks


def build_layout() -> html.Div:
    demo = get_demo()
    max_lead = int(max(demo.config["data"].get("lead_hours", [5])))
    lead_hours = [int(h) for h in demo.config["data"].get("lead_hours", list(range(max_lead + 1)))]
    last_ts = demo.last_ts.strftime("%d %b %Y · %H:%M")
    eyebrow_date = demo.last_ts.strftime("%d / %m / %Y")
    station_options = [{"label": station, "value": station} for station in demo.window.columns]
    return html.Div(
        className="atlas-app",
        children=[
            dcc.Store(id="store-selected", data={"station": "Dhompo"}),
            dcc.Store(id="store-play", data={"running": False}),
            html.Header(
                className="atlas-header",
                children=[
                    html.Div(
                        className="masthead",
                        children=[
                            html.Div(f"CATATAN DAS · {eyebrow_date}", className="eyebrow"),
                            html.H1("Sungai Welang"),
                            html.Div("Prediksi kondisi muka air di Dhompo · Pasuruan, Jawa Timur", className="dek"),
                        ],
                    ),
                    html.Div(className="timestamp", children=[html.Span("Observasi terakhir"), html.Div(last_ts, id="header-time"), html.Div(id="header-model", className="model-badge")]),
                ],
            ),
            html.Section(
                className="risk-lede",
                children=[
                    html.Div("STATUS DHOMPO", className="eyebrow"),
                    html.Div("—", id="summary-status", className="risk-status"),
                    html.Div("—", id="summary-value", className="risk-value"),
                    html.Div("Memuat ringkasan…", id="summary-detail", className="risk-detail"),
                    html.Div("—", id="summary-count", className="risk-count"),
                ],
            ),
            html.Section(
                className="forecast-strip",
                children=[
                    html.Div("PREDIKSI PER JAM · DHOMPO", className="eyebrow"),
                    html.Div(
                        id="forecast-strip",
                        className="forecast-chips",
                        children="Memuat…",
                    ),
                ],
            ),
            html.Main(
                className="atlas-main",
                children=[
                    html.Section(
                        className="hero-grid",
                        children=[
                            html.Div(
                                className="map-wrap",
                                children=[
                                    html.Div(className="section-label", children=[html.Span("01"), html.H2("Lokasi stasiun")]),
                                    dcc.Graph(id="fig-geo", config={"displayModeBar": False, "responsive": True}),
                                    html.Div("Marker menunjukkan lokasi stasiun. Hubungan aliran dibaca pada diagram jaringan di bawah.", className="map-caption"),
                                ],
                            ),
                            html.Aside(
                                className="station-rail",
                                children=[
                                    html.Div(className="section-label", children=[html.Span("02"), html.H2("Stasiun")]),
                                    dcc.Dropdown(id="station-selector", options=station_options, value="Dhompo", clearable=False, searchable=True, className="station-selector"),
                                    html.Button("Kembali ke Dhompo", id="btn-dhompo", className="btn-reset", n_clicks=0),
                                    html.Div(id="detail-station", className="station-detail"),
                                    html.Div("Peta geografis menggunakan koordinat stasiun yang tersedia.", className="rail-note"),
                                ],
                            ),
                        ],
                    ),
                    html.Section(
                        className="time-ruler",
                        children=[
                            html.Div(className="time-ruler-head", children=[html.Div("03 · Waktu prediksi", className="eyebrow"), html.Button("▶ Putar", id="btn-play", className="btn-play", n_clicks=0)]),
                            dcc.Slider(id="lead-slider", min=0, max=max_lead, step=1, value=0, marks=_lead_marks(lead_hours), tooltip={"placement": "top", "always_visible": False}),
                            html.Div("Sekarang", id="lead-label", className="lead-label"),
                            dcc.Interval(id="interval-play", interval=900, disabled=True, n_intervals=0),
                        ],
                    ),
                    html.Section(
                        className="atlas-section",
                        children=[
                            html.Div(className="section-label", children=[html.Span("04"), html.H2("Riwayat dan prediksi")]),
                            html.Div("Garis biru putus-putus hanya tersedia untuk prediksi model Dhompo.", className="section-note"),
                            dcc.Graph(id="fig-hydro", config={"displayModeBar": False, "responsive": True}),
                        ],
                    ),
                    html.Section(
                        className="atlas-section flow-block",
                        children=[
                            html.Div(className="section-label", children=[html.Span("05"), html.H2("Aliran menuju Dhompo")]),
                            html.Div("Urutan hulu → hilir dan estimasi waktu tempuh.", className="section-note"),
                            dcc.Graph(id="fig-profile", config={"displayModeBar": False, "responsive": True}),
                        ],
                    ),
                    html.Details(className="atlas-details", children=[html.Summary("Tentang data dan model"), html.P("Prediksi model hanya tersedia untuk Dhompo. Nilai stasiun hulu pada timeline adalah propagasi observasi berdasarkan estimasi waktu tempuh, bukan prediksi ML."), html.P("Ambang waspada dan bahaya diturunkan dari persentil data demo. Koordinat stasiun berasal dari configs/dhompo/station_geo.csv.")]),
                ],
            ),
            html.Footer(className="atlas-footer", children="DAS Dhompo · Sungai Welang · Catatan visual berbasis data historis"),
        ],
    )


app = dash.Dash(
    __name__,
    title="Sungai Welang · Catatan DAS",
    suppress_callback_exceptions=False,
    external_stylesheets=["https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&display=swap"],
)
app.layout = build_layout()
