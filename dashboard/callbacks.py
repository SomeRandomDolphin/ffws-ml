"""Callback Dash untuk dashboard DAS Dhompo."""

from __future__ import annotations

import pandas as pd
from dash import Input, Output, State, ctx, dcc, html, no_update

from dashboard.data import get_demo
from dashboard.figures import (
    build_geo_figure,
    build_hydro_figure,
    build_profile_figure,
    build_spark_figure,
)
from dashboard.geo import load_station_geo, travel_hours
from dashboard.palette import COLOR_TEXT, STATUS_BADGES, STATUS_LABELS, status_color
from dashboard.simulation import MAX_LEAD, build_frame


def _model_tier_markup(demo) -> html.Div:
    if demo.serving_tier == "A" and not demo.degraded:
        text = "Model serving aktif · tier A"
        color = "#166534"
    else:
        n = len(demo.degraded)
        text = f"Fallback persistence · tier B (patch: {n})" if n else "Fallback persistence · tier B"
        color = "#92400e"
    return html.Div(
        text,
        id="tier-flag",
        className="model-badge-text",
        style={"borderColor": color, "color": color},
        title="; ".join(demo.degraded) or "Prediksi penuh dari model file",
    )


def _forecast_strip_markup(demo) -> list:
    th = demo.thresholds.get("Dhompo")
    chips: list[html.Div] = []
    for h, val in demo.predictions.items():
        if th is not None:
            if val >= th.danger:
                status = "bahaya"
            elif val >= th.alert:
                status = "waspada"
            else:
                status = "normal"
        else:
            status = "normal"
        chips.append(
            html.Div(
                className="forecast-chip",
                style={"borderColor": status_color(status)},
                children=[
                    html.Div(h.upper(), className="forecast-chip-hour"),
                    html.Div(f"{val:.2f} m", className="forecast-chip-value"),
                    html.Div(
                        STATUS_LABELS[status].upper(),
                        className="forecast-chip-status",
                        style={"color": status_color(status)},
                    ),
                ],
            )
        )
    return chips


def _rain_markup(demo) -> html.Div | None:
    rain = getattr(demo, "rainfall", None)
    if rain is None:
        return None
    rain_window = rain.reindex(demo.window.index).fillna(0.0)
    total = float(rain_window.sum())
    label = f"Curah hujan {len(demo.window) / 2:g} jam terakhir: {total:.1f} mm"
    return html.Div(
        (
            label
            + " · data generated 2023"
            if getattr(demo, "source", "") == "2023_generated"
            else label
        ),
        className="detail-rain",
    )


def _detail_markup(frame, station: str, demo) -> list:
    st = frame.statuses.get(station)
    if st is None:
        return [
            html.Div(station, className="detail-name"),
            html.Div("Di luar model — hanya observasi (Bd. Sentono)", className="detail-note"),
        ]

    badge_text, badge_color = STATUS_BADGES[st.status]
    value = frame.values[station]
    is_target = station == "Dhompo"
    is_upstream = station in frame.statuses and not is_target and station != "Jalan Nasional"

    notes: list[html.Div] = []
    if is_target and frame.lead >= 1:
        notes.append(html.Div(
            f"Sumber nilai: prediksi model h{frame.lead}",
            className="detail-note",
        ))
    elif is_target:
        notes.append(html.Div("Sumber nilai: observasi terakhir", className="detail-note"))
    elif is_upstream:
        notes.append(html.Div(
            f"Sumber nilai: propagasi observasi · waktu tempuh ≈ {travel_hours(station):g} jam",
            className="detail-note",
        ))

    delta = st.delta_3h
    delta_color = "#dc2626" if delta > 0 else COLOR_TEXT

    span = st.danger - st.alert
    pct = min(max((st.value - st.alert) / span, 0.0), 1.0) * 100 if span > 0 else 0.0

    return [
        html.Div(
            [
                html.Span(station, className="detail-name"),
                html.Span(badge_text, className="status-badge", style={"background": badge_color}),
            ],
            className="detail-head",
        ),
        html.Div(f"{value:.2f} m", className="detail-value"),
        html.Div(
            f"Perubahan observasi 3 jam: {delta:+.2f} m",
            className="detail-delta",
            style={"color": delta_color},
        ),
        html.Div(
            [
                html.Span(f"Waspada ≥ {st.alert:.2f} m", className="detail-th"),
                html.Span(f"Bahaya ≥ {st.danger:.2f} m", className="detail-th"),
            ],
            className="detail-thresholds",
        ),
        html.Div(
            [
                html.Div(
                    html.Div(
                        style={"width": f"{pct:.0f}%", "background": status_color(st.status)}
                    ),
                    className="risk-meter",
                ),
                html.Div(
                    f"{pct:.0f}% dari rentang waspada → bahaya",
                    className="risk-meter-caption",
                ),
            ],
            className="risk-meter-block",
        ),
        *notes,
        dcc.Graph(
            figure=build_spark_figure(demo, station),
            config={"displayModeBar": False, "responsive": True},
            className="detail-spark",
        ),
        _rain_markup(demo),
    ]


def _register_callbacks(app) -> None:
    @app.callback(
        Output("lead-label", "children"),
        Output("fig-geo", "figure"),
        Output("fig-profile", "figure"),
        Output("fig-hydro", "figure"),
        Output("summary-status", "children"),
        Output("summary-status", "style"),
        Output("summary-value", "children"),
        Output("summary-detail", "children"),
        Output("summary-count", "children"),
        Output("detail-station", "children"),
        Output("header-model", "children"),
        Output("forecast-strip", "children"),
        Input("lead-slider", "value"),
        Input("store-selected", "data"),
    )
    def update_all(lead: int, selected: dict):
        demo = get_demo()
        lead = int(lead)
        frame = build_frame(demo, lead)
        geo = load_station_geo()
        station = (selected or {}).get("station", "Dhompo")

        if lead == 0:
            label = f"Sekarang ({demo.last_ts.strftime('%d %b %H:%M')})"
        else:
            target = demo.last_ts + pd.Timedelta(hours=lead)
            label = f"+{lead} jam ({target.strftime('%d %b %H:%M')})"

        summary_state = frame.statuses["Dhompo"].status
        summary_color = status_color(summary_state)
        max_lead = int(max(demo.config["data"].get("lead_hours", [5])))

        return (
            label,
            build_geo_figure(frame, geo, station),
            build_profile_figure(demo, frame, station),
            build_hydro_figure(demo, lead, station),
            f"{STATUS_LABELS[frame.statuses['Dhompo'].status].upper()} · {frame.statuses['Dhompo'].value:.2f} m",
            {"color": summary_color},
            f"Maksimum {frame.max_predicted:.2f} m · horizon +{max_lead} jam",
            f"Horizon aktif {label} · maksimum {frame.max_predicted:.2f} m diperkirakan dalam {max_lead} jam",
            f"{frame.area_waspada} titik berstatus waspada atau bahaya pada {label}",
            _detail_markup(frame, station, demo),
            _model_tier_markup(demo),
            _forecast_strip_markup(demo),
        )

    @app.callback(
        Output("store-selected", "data"),
        Input("fig-geo", "clickData"),
        Input("station-selector", "value"),
        Input("btn-dhompo", "n_clicks"),
        State("store-selected", "data"),
    )
    def on_node_click(geo_click, station_selector, dhompo_clicks, current):
        if ctx.triggered_id == "btn-dhompo":
            return {"station": "Dhompo"}
        if ctx.triggered_id == "station-selector" and station_selector:
            return {"station": station_selector}
        if geo_click and geo_click.get("points"):
            pts = geo_click["points"][0]
            station = pts.get("customdata")
            if station:
                return {"station": station}
        return no_update

    @app.callback(Output("station-selector", "value"), Input("store-selected", "data"))
    def sync_station_selector(selected):
        return (selected or {}).get("station", "Dhompo")

    @app.callback(
        Output("store-play", "data"),
        Input("btn-play", "n_clicks"),
        State("store-play", "data"),
    )
    def toggle_play(n_clicks, play):
        if n_clicks is None or n_clicks == 0:
            return no_update
        return {"running": not (play or {}).get("running", False)}

    @app.callback(
        Output("interval-play", "disabled"),
        Output("btn-play", "children"),
        Input("store-play", "data"),
    )
    def sync_play_button(play):
        running = (play or {}).get("running", False)
        if running:
            return False, "⏸ Jeda"
        return True, "▶ Putar"

    @app.callback(
        Output("lead-slider", "value"),
        Output("store-play", "data"),
        Input("interval-play", "n_intervals"),
        State("lead-slider", "value"),
        State("store-play", "data"),
    )
    def advance_play(n_intervals, value, play):
        if not (play or {}).get("running", False):
            return no_update, no_update
        value = int(value)
        if value >= MAX_LEAD:
            return MAX_LEAD, {"running": False}
        return value + 1, {"running": True}


def register_callbacks(app) -> None:
    _register_callbacks(app)
