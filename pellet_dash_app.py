from datetime import datetime
import math
import re
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output

# ------------------------------------------------------------
# CSV-Datei (anpassen!)
# ------------------------------------------------------------
CSV_PATH = "/Users/aponivi/documents/programmieren/eta-pellet/pellet_level_log.csv"

# ------------------------------------------------------------
# CSV laden und vorbereiten
# ------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",", header=None, dtype=str, comment="#")
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    mask_header = df.iloc[:, 0].str.lower().isin(
        ["ts", "time", "timestamp", "datum", "date"]
    ) if df.shape[1] > 0 else pd.Series(False, index=df.index)
    df = df.loc[~mask_header].copy()

    def parse_ts(s: str):
        s = (s or "").strip()
        if "_" in s and s.endswith("Z") and re.fullmatch(r"\d{8}_\d{4}Z", s):
            return pd.to_datetime(datetime.strptime(s, "%Y%m%d_%H%MZ"), utc=True)
        return pd.to_datetime(s, utc=True, errors="coerce")

    ts = df.iloc[:, 0].astype(str).map(parse_ts)

    pellet = (
        df.iloc[:, 1].astype(str)
          .str.replace(",", ".", regex=False)
          .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
          .astype(float)
    )
    temp = (
        df.iloc[:, 3].astype(str)
          .str.replace(",", ".", regex=False)
          .str.extract(r"(-?\d+(?:\.\d+)?)", expand=False)
          .astype(float)
    )

    out = pd.DataFrame({"timestamp": ts, "pellet_kg": pellet, "temperature": temp}).dropna()
    return out.sort_values("timestamp").reset_index(drop=True)

# ------------------------------------------------------------
# Tagesaggregation: letzter Messpunkt pro Tag (ohne Füllung)
# ------------------------------------------------------------
def to_daily_last(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genau ein Punkt/Tag: der *letzte* vorhandene Messpunkt.
    - pellet_kg: letzter Wert des Tages
    - temperature_mean: Tagesmittel (für Sterne)
    Tage ohne Messung werden NICHT aufgefüllt.
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "pellet_kg", "temperature_mean"])

    dfi = df.set_index("timestamp").sort_index()

    pellet_daily = dfi["pellet_kg"].resample("D").last()          # kein ffill
    temp_daily   = dfi["temperature"].resample("D").mean()

    daily = pd.DataFrame({
        "date": pellet_daily.index,               # tz-aware UTC
        "pellet_kg": pellet_daily.values,
        "temperature_mean": temp_daily.values
    })

    # Nur Tage behalten, an denen es einen Pelletswert gibt
    daily = daily.dropna(subset=["pellet_kg"]).reset_index(drop=True)
    return daily

# ------------------------------------------------------------
# Figure-Builder (Pellets nur noch als Tages-Endwerte)
#   - y (links)  = Temperatur (°C)  [dynamisch gesteuert via y2_range/y2_dtick Parameter]
#   - y2 (rechts)= Pellets [kg]     [fix 0–12'000]
# ------------------------------------------------------------
def make_figure(df: pd.DataFrame, y2_range=None, y2_dtick=5, x_range=None):
    df = df.copy()

    # Tagesdaten (letzter Messpunkt je Tag)
    daily = to_daily_last(df)

    fig = go.Figure()

    # -------------------- Temperatur (Originalzeitreihe) -> y (links) --------------------
    thr = 0.0
    temp_pos = df["temperature"].where(df["temperature"] >= thr)
    temp_neg = df["temperature"].where(df["temperature"] <  thr)

    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=temp_pos,
        name=f"Temp ≥ {int(thr)}°C",
        mode="lines+markers", yaxis="y",
        line=dict(color="red", width=2),
        marker=dict(color="red", size=6),
        connectgaps=False,
        hovertemplate="Temperatur: %{y:.1f} °C<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=temp_neg,
        name=f"Temp < {int(thr)}°C",
        mode="lines+markers", yaxis="y",
        line=dict(color="blue", width=2),
        marker=dict(color="blue", size=6),
        connectgaps=False,
        hovertemplate="Temperatur: %{y:.1f} °C<extra></extra>",
    ))

    # Tagesmittel-Temperatur als Sterne (aus daily.temperature_mean) -> y
    if not daily.empty:
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["temperature_mean"],
            name="Tagesmittel (°C)",
            mode="markers", yaxis="y",
            marker=dict(symbol="star", size=12, color="yellow",
                        line=dict(width=1, color="black")),
            hovertemplate="Tagesmittel: %{y:.1f} °C<extra></extra>",
        ))

    # -------------------- Pellets: Tages-Endwerte -> y2 (rechts) --------------------
    if not daily.empty:
        fig.add_trace(go.Bar(
            x=daily["date"],
            y=daily["pellet_kg"],
            name="Pelletstand",
            yaxis="y2",
            marker=dict(color="saddlebrown", opacity=0.6),
            width=24 * 60 * 60 * 1000 * 0.35,  # 35% Tagesbreite als Balken
            hovertemplate="Pelletstand: %{y:,.0f} kg<extra></extra>",
        ))
        x_min, x_max = daily["date"].min(), daily["date"].max()

        # --- Wert im letzten Balken anzeigen (90° nach links gedreht) ---
        last_date = daily["date"].iloc[-1]
        last_value = float(daily["pellet_kg"].iloc[-1])

        # Zahl mit Schweizer Apostroph (12'345) + Einheit
        label_text = f"{int(round(last_value)):,} kg".replace(",", "'")

        # Annotation mittig im letzten Balken, gedreht
        fig.add_annotation(
            x=last_date,
            y=last_value / 2.0,
            xref="x", yref="y2",
            text=label_text,
            showarrow=False,
            yanchor="middle",
            xanchor="center",
            textangle=-90,
            font=dict(color="black", size=14, family="Arial"),
            bgcolor="rgba(0,0,0,0)"
        )

    else:
        # Fallback, falls daily leer (z. B. Datenfenster ohne Punkte)
        x_min, x_max = df["timestamp"].min(), df["timestamp"].max()

    # Pellets-Schwellenlinien -> y2
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], y=[2000, 2000],
        name="Schwelle 2 000 kg",
        mode="lines", yaxis="y2",
        line=dict(color="orange", width=2, dash="dash"),
        hoverinfo="skip", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], y=[1000, 1000],
        name="Schwelle 1 000 kg",
        mode="lines", yaxis="y2",
        line=dict(color="red", width=2, dash="dash"),
        hoverinfo="skip", showlegend=True,
    ))

    # -------------------- Achsen-Ticks --------------------
    MONTH_ABBR_DE = {1:"Jan",2:"Feb",3:"Mär",4:"Apr",5:"Mai",6:"Jun",
                     7:"Jul",8:"Aug",9:"Sep",10:"Okt",11:"Nov",12:"Dez"}
    # Für die X-Achse nutzen wir den Pellet-Zeitraum
    day_range = pd.date_range(start=x_min.normalize(), end=x_max.normalize(), freq="D", tz="UTC")
    tickvals = list(day_range)
    ticktext = [f"{MONTH_ABBR_DE[d.month]}{d.year%100:02d}" if d.day==1 else str(d.day) for d in day_range]

    # -------------------- y (Temperatur) Layout & x-Range --------------------
    # HINWEIS: y2_range/y2_dtick steuern weiterhin die Temperatur-Skala (jetzt y).
    y_left_layout = (dict(title="Temperatur (°C)",
                          autorange=True, dtick=y2_dtick, tick0=0,
                          showgrid=False, zeroline=False)
                     if y2_range is None else
                     dict(title="Temperatur (°C)",
                          autorange=False, range=y2_range, dtick=y2_dtick, tick0=0,
                          showgrid=False, zeroline=False))

    if x_range is None:
        x_range = [x_min, x_max]

    fig.update_layout(
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        # hält Zoom/Slider-Interaktionen stabil
        uirevision="keep-my-zoom",
        xaxis=dict(
            title="Zeit (UTC)",
            range=x_range,
            tickmode="array", tickvals=tickvals, ticktext=ticktext,
            showgrid=True, gridcolor="lightgray", gridwidth=1,
            rangeslider=dict(visible=True)
        ),
        # y = Temperatur (links, nicht overlayed)
        yaxis=y_left_layout,
        # y2 = Pellets (rechts, overlayed auf y)
        yaxis2=dict(
            title="Pellet [kg]",
            overlaying="y", side="right",
            range=[0, 12000], dtick=2000,
            showgrid=True, gridcolor="lightgray", gridwidth=1,
            zeroline=False
        ),
        shapes=[dict(type="rect", xref="paper", yref="paper",
                     x0=0, y0=0, x1=1, y1=1,
                     line=dict(color="black", width=1),
                     fillcolor="rgba(0,0,0,0)",
                     layer="below")],
        title=dict(text="Temperatur (°C) & Pelletstand [kg] (Pellets: letzter Wert je Tag)",
                   x=0.5, y=0.95),
        legend=dict(
            orientation="h",          # horizontal
            yanchor="bottom",
            y=1.12,                   # etwas über dem Plot
            xanchor="center",
            x=0.5,                    # mittig
            bordercolor="lightgray",
            borderwidth=1,
        ),
        margin=dict(l=70, r=70, t=100, b=60)
    )

    # Komma → Apostroph im Pellet-Hover
    for tr in fig.data:
        if tr.name.startswith("Pellet"):
            tr.hovertemplate = tr.hovertemplate.replace(",", "&#x27;")

    return fig

# ------------------------------------------------------------
# Statistik erstellen (auf Rohdaten)
# ------------------------------------------------------------
def make_stats(dfs: pd.DataFrame):
    return [
        {"metric": "Pellet Min [t]", "value": round(dfs['pellet_kg'].min()/1000.0, 3)},
        {"metric": "Pellet Max [t]", "value": round(dfs['pellet_kg'].max()/1000.0, 3)},
        {"metric": "Pellet Δ [t]",   "value": round((dfs['pellet_kg'].iloc[-1] - dfs['pellet_kg'].iloc[0])/1000.0, 3)},
        {"metric": "Temp Min (°C)",  "value": dfs['temperature'].min()},
        {"metric": "Temp Max (°C)",  "value": dfs['temperature'].max()},
        {"metric": "Temp Mittel (°C)","value": round(dfs['temperature'].mean(), 3)},
        {"metric": "Messpunkte",     "value": int(len(dfs))}
    ]

# ------------------------------------------------------------
# Sichtbaren Bereich bestimmen
# ------------------------------------------------------------
def extract_visible_range(relayoutData, df):
    start = df['timestamp'].min()
    end   = df['timestamp'].max()

    if relayoutData:
        if 'xaxis.range[0]' in relayoutData and 'xaxis.range[1]' in relayoutData:
            start = pd.to_datetime(relayoutData['xaxis.range[0]'], utc=True, errors='coerce')
            end   = pd.to_datetime(relayoutData['xaxis.range[1]'], utc=True, errors='coerce')
        elif 'xaxis.range' in relayoutData and isinstance(relayoutData['xaxis.range'], list):
            r = relayoutData['xaxis.range']
            if len(r) == 2:
                start = pd.to_datetime(r[0], utc=True, errors='coerce')
                end   = pd.to_datetime(r[1], utc=True, errors='coerce')
        elif 'xaxis' in relayoutData and isinstance(relayoutData['xaxis'], dict) and 'range' in relayoutData['xaxis']:
            r = relayoutData['xaxis']['range']
            if isinstance(r, list) and len(r) == 2:
                start = pd.to_datetime(r[0], utc=True, errors='coerce')
                end   = pd.to_datetime(r[1], utc=True, errors='coerce')

    return start, end

# ------------------------------------------------------------
# Dash App
# ------------------------------------------------------------
app = Dash(__name__)
app.title = "Pellet & Temperatur Dashboard"

df = load_data(CSV_PATH)
initial_fig = make_figure(df)

app.layout = html.Div([
    dcc.Graph(id='pellet-temp-graph', figure=initial_fig),
    html.Div([
        html.H3("Statistiken (Rohdaten im sichtbaren Bereich)"),
        dash_table.DataTable(
            id='stats-table',
            columns=[{"name": "Metrik", "id": "metric"}, {"name": "Wert", "id": "value"}],
            data=make_stats(df),
            style_table={"maxWidth": "520px"},
            style_cell={"padding": "6px"},
        )
    ], style={"marginTop": "20px"})
])

# ------------------------------------------------------------
# Callback: Stats auf Rohdaten; y (Temperatur) dynamisch; X-Bereich stabil
# ------------------------------------------------------------
@app.callback(
    Output('stats-table', 'data'),
    Output('pellet-temp-graph', 'figure'),
    Input('pellet-temp-graph', 'relayoutData')
)
def update_view(relayoutData):
    start, end = extract_visible_range(relayoutData, df)
    dff = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

    if dff.empty:
        # auch wenn leer, aktuellen x_range respektieren
        return make_stats(df), make_figure(df, x_range=[start, end])

    # Stats basieren weiterhin auf ROHDATEN (z. B. 10-min)
    stats = make_stats(dff)

    # y (Temperatur) dynamisch
    t_min, t_max = float(dff['temperature'].min()), float(dff['temperature'].max())
    amplitude = t_max - t_min
    if amplitude > 36:
        dt = 8
    elif amplitude > 24:
        dt = 4
    elif amplitude < 12:
        dt = 2
    else:
        dt = 5

    lo = math.floor(t_min / dt) * dt
    hi = math.ceil(t_max / dt) * dt
    if lo == hi:
        lo -= dt
        hi += dt

    # entscheidend: x_range=[start, end], damit nichts springt
    fig = make_figure(df, y2_range=[lo, hi], y2_dtick=dt, x_range=[start, end])
    return stats, fig

# ------------------------------------------------------------
# Start
# ------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
