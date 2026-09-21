"""Pembangun figur Plotly untuk dashboard DAS Dhompo.

Semua figur adalah fungsi murni dari `LeadFrame` (lihat simulation.py)
sehingga mudah diuji dan dipanggil ulang oleh callback slider.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dhompo.data.loader import TARGET_STATION

from dashboard.geo import (
    AUX_STATION,
    EAST_BRANCH,
    WEST_BRANCH,
    short_name,
    travel_hours,
)
from dashboard.palette import (
    COLOR_AUX,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_OBSERVED,
    COLOR_PREDICTED,
    COLOR_TEXT,
    COLOR_THRESHOLD_ALERT,
    COLOR_THRESHOLD_DANGER,
    SEGMENT_WIDTH,
    STATUS_COLORS,
    STATUS_LABELS,
    STATUS_ORDER,
    hex_to_rgba,
    status_color,
)

_FONT_FAMILY = "Plus Jakarta Sans, Segoe UI, system-ui, sans-serif"
_FONT = dict(family=_FONT_FAMILY, color=COLOR_TEXT, size=12)

# Lapisan sungai: (offset lebar, alpha) untuk efek badan air
_WATER_HALO = (7, 0.15)
_WATER_BODY = (4, 0.35)


def _common_layout(title: str | None = None, height: int = 520) -> go.Layout:
    layout = go.Layout(
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=_FONT,
        height=height,
        margin=dict(l=40, r=24, t=40, b=40),
        hoverlabel=dict(font=dict(family=_FONT_FAMILY, color="white")),
    )
    if title:
        layout.title = dict(
            text=title,
            font=dict(size=14, color=COLOR_TEXT),
            x=0.01,
            xanchor="left",
        )
    return layout


def _status_legend_traces() -> list[go.Scatter]:
    return [
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(color=STATUS_COLORS[st], size=11),
            name=STATUS_LABELS[st],
            showlegend=True,
            hoverinfo="skip",
        )
        for st in STATUS_ORDER
    ]


def _node_value(frame, station: str) -> float | None:
    if station == AUX_STATION or station not in frame.values:
        return None
    v = frame.values[station]
    return v if v == v else None


# ---------------------------------------------------------------- georef ----

def build_geo_figure(frame, geo: dict, selected: str = TARGET_STATION) -> go.Figure:
    statuses = frame.statuses
    traces: list[go.Scattermap] = []

    lons, lats, sizes, colors, hovers, customs = [], [], [], [], [], []
    for station, g in geo.items():
        lons.append(g.longitude)
        lats.append(g.latitude)
        customs.append(station)
        if station == AUX_STATION:
            colors.append(COLOR_AUX)
            sizes.append(10)
            hovers.append(
                f"<b>{station.upper()}</b><br>"
                "<span style='color:#667085'>Stasiun observasi</span><br>"
                "<span style='color:#667085'>Data model belum tersedia</span>"
            )
        elif station not in statuses:
            colors.append(COLOR_MUTED)
            sizes.append(11)
            hovers.append(
                f"<b>{station.upper()}</b><br>"
                "<span style='color:#667085'>Data belum tersedia</span>"
            )
        else:
            st = statuses[station]
            colors.append(status_color(st.status))
            sizes.append(22 if station == TARGET_STATION else 13)
            value = _node_value(frame, station)
            vtxt = f"{value:.2f} m" if value is not None else "n/a"
            trend = "Naik" if st.delta_3h > 0 else "Turun" if st.delta_3h < 0 else "Stabil"
            trend_color = "#b42318" if st.delta_3h > 0 else "#027a48" if st.delta_3h < 0 else "#667085"
            status = STATUS_LABELS[st.status]
            hovers.append(
                f"<b>{station.upper()}</b><br>"
                "<span style='color:#667085'>Muka air, meter</span><br>"
                f"<b style='font-size:18px'>{vtxt}</b><br>"
                f"<span style='color:#667085'>Waktu simulasi: {'sekarang' if frame.lead == 0 else f'+{frame.lead} jam'}</span><br>"
                f"<span style='color:#667085'>Status: </span><b>{status}</b><br>"
                f"<span style='color:{trend_color}'>{trend} {abs(st.delta_3h):.2f} m dalam 3 jam</span><br>"
                f"<span style='color:#667085'>Ambang waspada: {st.alert:.2f} m</span>"
            )
    traces.append(
        go.Scattermap(
            lon=lons,
            lat=lats,
            mode="markers",
            customdata=customs,
            hovertext=hovers,
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(color="#ffffff", size=[s + 5 for s in sizes], opacity=0.9),
        )
    )
    traces.append(
        go.Scattermap(
            lon=lons,
            lat=lats,
            mode="markers",
            customdata=customs,
            hovertext=hovers,
            hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(color=colors, size=sizes),
        )
    )

    if selected in geo:
        traces.append(
            go.Scattermap(
                lon=[geo[selected].longitude],
                lat=[geo[selected].latitude],
                mode="markers",
                marker=dict(size=34, color="rgba(15,23,42,0.30)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    dhompo = geo[TARGET_STATION]
    layout = go.Layout(
        template="none",
        height=600,
        margin=dict(l=8, r=8, t=40, b=8),
        font=_FONT,
        title=dict(text="", font=dict(size=1, color="rgba(0,0,0,0)")),
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.96)",
            bordercolor="#c8d0d8",
            font=dict(family=_FONT_FAMILY, color="#1e2a32", size=12),
            align="left",
            namelength=0,
        ),
        map=dict(
            style="carto-positron-nolabels",
            center=dict(lat=dhompo.latitude, lon=dhompo.longitude),
            zoom=10.2,
            uirevision="welang-map",
        ),
    )
    return go.Figure(data=traces, layout=layout)


# ---------------------------------------------------------------- profile ---

def _profile_x(station: str, east: bool = False, east_index: int = 0) -> float:
    if station == TARGET_STATION:
        return 0.0
    if station == "Jalan Nasional":
        return 0.7
    x = -travel_hours(station)
    if east:
        x += 0.22 + 0.06 * east_index
    return x


def build_profile_figure(demo, frame, selected: str = TARGET_STATION) -> go.Figure:
    """Diagram topologi hulu → hilir yang mudah dibaca tanpa sumbu abstrak."""
    traces: list[go.Scatter] = []
    chains = [(WEST_BRANCH, "Cabang barat", -0.20), (EAST_BRANCH, "Cabang timur", 0.20)]
    for branch, name, offset in chains:
        chain = branch + [TARGET_STATION]
        xs = list(np.linspace(-1.0, 0.0, len(chain)) + offset)
        ys = [1.0 + (i % 2) * 0.06 for i in range(len(chain))]
        traces.append(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#b8c4cf", width=3), hoverinfo="skip", showlegend=False))
        values = [frame.values.get(st, float("nan")) for st in chain]
        colors = [status_color(frame.statuses[st].status) for st in chain]
        hover = [f"{st}<br>{values[i]:.2f} m<br>waktu tempuh ≈ {travel_hours(st):g} jam" for i, st in enumerate(chain)]
        traces.append(go.Scatter(x=xs, y=ys, mode="markers+text", marker=dict(size=[25 if st == TARGET_STATION else 15 for st in chain], color=colors, line=dict(color="#ffffff", width=2)), text=[short_name(st) for st in chain], textposition="top center", textfont=dict(size=10, family=_FONT_FAMILY), customdata=chain, hovertext=hover, hovertemplate="%{hovertext}<extra></extra>", name=name, showlegend=False))
        for x, y, st, value in zip(xs, ys, chain, values):
            traces.append(go.Scatter(x=[x], y=[y - 0.18], mode="text", text=[f"{value:.2f} m"], textfont=dict(size=9, color=COLOR_MUTED, family=_FONT_FAMILY), hoverinfo="skip", showlegend=False))
    traces.append(go.Scatter(x=[-0.95, -0.55, -0.15], y=[1.72, 1.72, 1.72], mode="text", text=["HULU", "→", "DHOMPO"], textfont=dict(size=10, color=COLOR_MUTED, family=_FONT_FAMILY), hoverinfo="skip", showlegend=False))
    traces.extend(_status_legend_traces())
    layout = _common_layout(None, height=250)
    layout.update(xaxis=dict(visible=False, range=[-1.35, 1.35]), yaxis=dict(visible=False, range=[0.55, 1.95]), legend=dict(orientation="h", y=-0.02, x=0, font=dict(size=10)), hovermode="closest", margin=dict(l=10, r=10, t=10, b=40))
    fig = go.Figure(data=traces, layout=layout)
    fig.add_annotation(x=0, y=1.76, text=f"Dhompo · horizon +{frame.lead} jam", showarrow=False, font=dict(size=10, color=COLOR_PREDICTED, family=_FONT_FAMILY))
    return fig


# ---------------------------------------------------------------- hydro -----

def build_hydro_figure(demo, lead: int, station: str = TARGET_STATION) -> go.Figure:
    window = demo.window
    station = station if station in window.columns else TARGET_STATION
    ts = window.index[-1]
    observed = window[station].dropna()
    th = demo.thresholds.get(station)
    y_values = list(observed.values)
    pred_hours = [int(h) for h in demo.config["data"].get("lead_hours", [1, 2, 3, 4, 5]) if int(h) > 0]
    pred_times = [ts + pd.Timedelta(hours=h) for h in pred_hours]
    pred_values = [demo.predictions[f"h{h}"] for h in pred_hours] if station == TARGET_STATION else []
    y_values.extend(pred_values)
    if th is not None:
        y_values.extend([th.alert, th.danger])
    ytop = max(y_values) * 1.06 if y_values else 1.0
    ybot = min(y_values) * 0.97 if y_values else 0.0

    traces: list[go.Scatter] = [
        go.Scatter(
            x=observed.index,
            y=observed.values,
            mode="lines",
            line=dict(color=hex_to_rgba(COLOR_PREDICTED, 0.35), width=0),
            fill="tozeroy",
            fillcolor=hex_to_rgba(COLOR_PREDICTED, 0.06),
            hoverinfo="skip",
            showlegend=False,
        ),
        go.Scatter(
            x=observed.index,
            y=observed.values,
            mode="lines+markers",
            name="Observasi",
            line=dict(color=COLOR_OBSERVED, width=2.4),
            marker=dict(size=5.5, color=COLOR_OBSERVED),
            hovertemplate="%{x|%d %b %H:%M}<br>%.2f m<extra>observasi</extra>",
        ),
    ]

    if station == TARGET_STATION:
        traces.extend([
            go.Scatter(x=[ts] + pred_times, y=[float(observed.iloc[-1])] + pred_values, mode="lines+markers", name="Prediksi model", line=dict(color=COLOR_PREDICTED, width=3.0, dash="dot"), marker=dict(size=8, color=COLOR_PREDICTED, symbol="diamond"), hovertemplate="%{x|%d %b %H:%M}<br>%.2f m<extra>prediksi</extra>"),
            go.Scatter(x=[ts], y=[float(observed.iloc[-1])], mode="markers", marker=dict(size=12, color=COLOR_PREDICTED, line=dict(color="#ffffff", width=2.5)), name="Acuan", hovertemplate="%{x|%d %b %H:%M}<br>%.2f m<extra>observasi terakhir</extra>"),
        ])
    if th is not None:
        end_time = pred_times[-1] if station == TARGET_STATION else ts
        traces.extend([
            go.Scatter(x=[observed.index[0], end_time], y=[th.alert, th.alert], mode="lines", name=f"Waspada ({th.alert:.1f} m)", line=dict(color=COLOR_THRESHOLD_ALERT, width=1.5, dash="dash"), hoverinfo="skip"),
            go.Scatter(x=[observed.index[0], end_time], y=[th.danger, th.danger], mode="lines", name=f"Bahaya ({th.danger:.1f} m)", line=dict(color=COLOR_THRESHOLD_DANGER, width=1.5, dash="dash"), hoverinfo="skip"),
        ])

    layout = _common_layout(None, height=390)
    layout.update(
        xaxis=dict(showgrid=True, gridcolor=COLOR_GRID, rangeslider=dict(visible=False)),
        yaxis=dict(
            title=dict(text="tinggi air (m)", font=dict(size=11)),
            showgrid=True,
            gridcolor=COLOR_GRID,
            range=[ybot, ytop],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        shapes=[
            dict(
                type="rect",
                x0=observed.index[0],
                x1=pred_times[-1] if station == TARGET_STATION else ts,
                y0=th.danger,
                y1=ytop,
                fillcolor=hex_to_rgba(COLOR_THRESHOLD_DANGER, 0.10),
                line=dict(width=0),
                layer="below",
            ),
            dict(
                type="rect",
                x0=observed.index[0],
                x1=pred_times[-1] if station == TARGET_STATION else ts,
                y0=th.alert,
                y1=th.danger,
                fillcolor=hex_to_rgba(COLOR_THRESHOLD_ALERT, 0.06),
                line=dict(width=0),
                layer="below",
            ),
            dict(
                type="rect",
                x0=ts,
                x1=pred_times[-1] if station == TARGET_STATION else ts,
                y0=ybot,
                y1=ytop,
                fillcolor=hex_to_rgba(COLOR_PREDICTED, 0.05),
                line=dict(width=0),
                layer="below",
            ),
        ],
    )
    fig = go.Figure(data=traces, layout=layout)
    if station == TARGET_STATION and lead >= 1:
        lead_ts = ts + pd.Timedelta(hours=lead)
        fig.add_shape(
            type="line",
            x0=lead_ts, x1=lead_ts, y0=ybot, y1=ytop,
            line=dict(color=COLOR_PREDICTED, width=1.6, dash="dash"),
        )
        fig.add_annotation(
            x=lead_ts,
            y=ytop * 0.985,
            text=f"+{lead} jam",
            showarrow=False,
            font=dict(size=11, color=COLOR_PREDICTED, family=_FONT_FAMILY),
        )
    return fig


# ---------------------------------------------------------------- spark -----

def build_spark_figure(demo, station: str) -> go.Figure:
    series = demo.window[station].dropna() if station in demo.window else None
    th = demo.thresholds.get(station)
    traces: list[go.Scatter] = []
    ymax = 1.0
    if series is not None and not series.empty:
        traces.append(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                line=dict(color=COLOR_PREDICTED, width=2.2),
                fill="tozeroy",
                fillcolor=hex_to_rgba(COLOR_PREDICTED, 0.09),
                hoverinfo="skip",
                name="riwayat",
            )
        )
        ymax = float(series.max())
    if th is not None:
        traces.append(
            go.Scatter(
                x=[0, 1],
                y=[th.danger, th.danger],
                mode="lines",
                line=dict(color=COLOR_THRESHOLD_DANGER, width=1.4, dash="dot"),
                hoverinfo="skip",
                name="bahaya",
            )
        )
        traces.append(
            go.Scatter(
                x=[0, 1],
                y=[th.alert, th.alert],
                mode="lines",
                line=dict(color=COLOR_THRESHOLD_ALERT, width=1.4, dash="dash"),
                hoverinfo="skip",
                name="waspada",
            )
        )
        ymax = max(ymax, th.danger)

    layout = _common_layout(f"Riwayat — {station}", height=240)
    layout.update(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, range=[0, ymax * 1.15 or 1.0]),
        margin=dict(l=8, r=8, t=34, b=8),
        showlegend=False,
    )
    return go.Figure(data=traces, layout=layout)
