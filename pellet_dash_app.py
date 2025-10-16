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
# Figure-Builder
#  - Weißer Hintergrund
#  - Rahmen
#  - Einheitliches horizontales Gitter (von y1)
#  - y1: 0–12000 kg in 2000er Schritten
#  - y2: dynamisch (Range + dtick)
#  - Temperaturfarben: rot > 0°, blau < 0°
# ------------------------------------------------------------
def make_figure(df: pd.DataFrame, y2_range=None, y2_dtick=5):
    df = df.copy()

    # Tagesmitteltemperatur (pro Kalendertag)
    df['date'] = df['timestamp'].dt.floor('D')
    daily_temp = df.groupby('date', as_index=False)['temperature'].mean()

    fig = go.Figure()

    # --- X-Bereich & Ticks vorbereiten ---
    x_min, x_max = df['timestamp'].min(), df['timestamp'].max()
    MONTH_ABBR_DE = {1:"Jan", 2:"Feb", 3:"Mär", 4:"Apr", 5:"Mai", 6:"Jun",
                     7:"Jul", 8:"Aug", 9:"Sep", 10:"Okt", 11:"Nov", 12:"Dez"}
    day_range = pd.date_range(start=x_min.normalize(), end=x_max.normalize(), freq="D")
    tickvals = [d.tz_localize("UTC") if d.tzinfo is None else d.tz_convert("UTC") for d in day_range]
    ticktext = [f"{MONTH_ABBR_DE[d.month]}{d.year % 100:02d}" if d.day == 1 else str(d.day) for d in day_range]

    # --- Layout-Grundgerüst (Hintergrund & Rahmen ganz unten) ---
    fig.update_layout(
        title="Pelletstand [kg] & Temperatur (°C) über Zeit",
        hovermode='x unified',
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(
            title='Zeit (UTC)',
            range=[x_min, x_max],
            rangeslider=dict(visible=True, range=[x_min, x_max]),
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=0,
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1
        ),
        yaxis=dict(
            title='Pellet [kg]',
            range=[0, 12000],
            dtick=2000,
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            zeroline=False
        ),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        margin=dict(l=70, r=70, t=60, b=60),
        shapes=[
            # Rahmen (unter allen Traces)
            dict(type='rect', xref='paper', yref='paper',
                 x0=0, y0=0, x1=1, y1=1,
                 line=dict(color='black', width=1),
                 fillcolor='rgba(0,0,0,0)',
                 layer='below')
        ]
    )

    # ----------------- ZEICHENREIHENFOLGE -----------------
    # 1) Pelletkurve (soll unter den Schwellen/Temperatur liegen)
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['pellet_kg'],
        name='Pellet [kg]',
        mode='lines+markers',
        yaxis='y1',
        line=dict(color='gray', width=2),
        marker=dict(color='gray', size=6),
        hovertemplate="Pellets: %{y:,.0f} kg<extra></extra>",
    ))

    # 2) Pellet-Schwellen ALS TRACES (zwischen Pellet und Temperatur)
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], y=[2000, 2000],
        name="Schwelle 2'000 kg",
        mode='lines',
        yaxis='y1',
        line=dict(color='orange', width=2, dash='dash'),
        hoverinfo='skip',
        showlegend=True
    ))
    fig.add_trace(go.Scatter(
        x=[x_min, x_max], y=[1000, 1000],
        name="Schwelle 1'000 kg",
        mode='lines',
        yaxis='y1',
        line=dict(color='red', width=2, dash='dash'),
        hoverinfo='skip',
        showlegend=True
    ))

    # 3) Temperatur-Traces (ganz oben)
    #    Farbwechsel bei 5°C, ohne durchgezogene Linien über Lücken
    thr = 5.0
    temp_pos = df['temperature'].where(df['temperature'] >= thr)  # rot ab 5°C
    temp_neg = df['temperature'].where(df['temperature'] <  thr)  # blau unter 5°C

    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=temp_pos,
        name=f'Temp ≥ {int(thr)}°C',
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='red', width=2),
        marker=dict(color='red', size=6),
        connectgaps=False,
        hovertemplate="Temperatur: %{y:.1f} °C<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=temp_neg,
        name=f'Temp < {int(thr)}°C',
        mode='lines+markers',
        yaxis='y2',
        line=dict(color='blue', width=2),
        marker=dict(color='blue', size=6),
        connectgaps=False,
        hovertemplate="Temperatur: %{y:.1f} °C<extra></extra>"
    ))

    # Tagesmittel (Sterne) – ebenfalls oben
    fig.add_trace(go.Scatter(
        x=daily_temp['date'], y=daily_temp['temperature'],
        name='Tagesmittel (°C)',
        mode='markers',
        yaxis='y2',
        marker=dict(symbol='star', size=12, color='yellow',
                    line=dict(width=1, color='black')),
        hovertemplate="Tagesmittel: %{y:.1f} °C<extra></extra>"
    ))
    # ------------------------------------------------------

    # Rechte Achse (dynamisch/übergeben)
    if y2_range is None:
        y2_layout = dict(
            title='Temperatur (°C)', overlaying='y', side='right',
            autorange=True, dtick=y2_dtick, tick0=0,
            showgrid=False, zeroline=False
        )
    else:
        y2_layout = dict(
            title='Temperatur (°C)', overlaying='y', side='right',
            autorange=False, range=y2_range, dtick=y2_dtick, tick0=0,
            showgrid=False, zeroline=False
        )
    fig.update_layout(yaxis2=y2_layout)

    # Optional: 10,192 → 10'192 im Pellet-Hover
    for tr in fig.data:
        if tr.name.startswith('Pellet'):
            tr.hovertemplate = tr.hovertemplate.replace(",", "&#x27;")

    return fig




# ------------------------------------------------------------
# Statistik erstellen
# ------------------------------------------------------------
def make_stats(dfs: pd.DataFrame):
    return [
        {"metric": "Pellet Min [kg]", "value": round(dfs['pellet_kg'].min()/1000.0, 3)},
        {"metric": "Pellet Max [kg]", "value": round(dfs['pellet_kg'].max()/1000.0, 3)},
        {"metric": "Pellet Δ [kg]", "value": round((dfs['pellet_kg'].iloc[-1] - dfs['pellet_kg'].iloc[0])/1000.0, 3)},
        {"metric": "Temp Min (°C)", "value": dfs['temperature'].min()},
        {"metric": "Temp Max (°C)", "value": dfs['temperature'].max()},
        {"metric": "Temp Mittel (°C)", "value": round(dfs['temperature'].mean(), 3)},
        {"metric": "Messpunkte", "value": int(len(dfs))}
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
        html.H3("Statistiken"),
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
# Callback: dynamische y2-Achse (Amplitude-abhängig) + Farbplot
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
        return make_stats(df), make_figure(df)

    stats = make_stats(dff)

    t_min, t_max = float(dff['temperature'].min()), float(dff['temperature'].max())
    amplitude = t_max - t_min

    # Dynamische Tickregel
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

    fig = make_figure(df, y2_range=[lo, hi], y2_dtick=dt)
    return stats, fig

# ------------------------------------------------------------
# Start
# ------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
