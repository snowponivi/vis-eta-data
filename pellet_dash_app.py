from datetime import datetime
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
# Plot erstellen – inkl. Tagesmitteltemperatur (gelbe Sterne)
# + Weißer Hintergrund + einheitliches horizontales Gitter
# ------------------------------------------------------------
def make_figure(df: pd.DataFrame):
    df = df.copy()
    df['pellet_t'] = df['pellet_kg']

    # Tagesmitteltemperatur (pro Kalendertag)
    df['date'] = df['timestamp'].dt.floor('D')
    daily_temp = df.groupby('date', as_index=False)['temperature'].mean()

    fig = go.Figure()

    # Pellets (linke Achse)
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['pellet_t'],
        name='Pellet [kg]', mode='lines+markers', yaxis='y1',
        hovertemplate="Pellets: %{y:,.0f} kg<extra></extra>",
    ))

    # Temperatur Einzelmessungen (rechte Achse, halbtransparent)
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['temperature'],
        name='Temperatur (°C)', mode='lines+markers', yaxis='y2',
        opacity=0.4,
        hovertemplate="Temperatur: %{y:.1f} Grad<extra></extra>"
    ))

    # Tagesmitteltemperatur (rechte Achse, gelbe Sterne)
    fig.add_trace(go.Scatter(
        x=daily_temp['date'], y=daily_temp['temperature'],
        name='Tagesmittel (°C)',
        mode='markers',
        yaxis='y2',
        marker=dict(
            symbol='star',
            size=12,
            color='yellow',
            line=dict(width=1, color='black')
        ),
        hovertemplate="Tagesmittel: %{y:.1f} Grad<extra></extra>"
    ))

    x_min, x_max = df['timestamp'].min(), df['timestamp'].max()

    # --- Tägliche Tickwerte + Sonderlabel am Monatsersten ---
    MONTH_ABBR_DE = {1:"Jan", 2:"Feb", 3:"Mär", 4:"Apr", 5:"Mai", 6:"Jun",
                     7:"Jul", 8:"Aug", 9:"Sep", 10:"Okt", 11:"Nov", 12:"Dez"}

    day_range = pd.date_range(start=x_min.normalize(), end=x_max.normalize(), freq="D")
    tickvals = [d.tz_localize("UTC") if d.tzinfo is None else d.tz_convert("UTC") for d in day_range]

    def label_for(d: pd.Timestamp) -> str:
        if d.day == 1:
            return f"{MONTH_ABBR_DE[d.month]}{d.year % 100:02d}"
        return str(d.day)

    ticktext = [label_for(d) for d in day_range]

    # ------------------------------
    # Layout: Weißer Hintergrund + einheitliches horizontales Gitter
    # ------------------------------
    fig.update_layout(
        title="Pelletstand [kg] & Temperatur (°C) über Zeit",
        hovermode='x unified',

        # Weißer Hintergrund
        plot_bgcolor="white",
        paper_bgcolor="white",

        # Achsen
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
        # y1 zeichnet das horizontale Gitter für beide
        yaxis=dict(
            title='Pellet [kg]',
            range=[0, 12000],
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
            zeroline=False
        ),
        # y2 ohne eigenes Gitter (nutzt optisch das von y1)
        yaxis2=dict(
            title='Temperatur (°C)',
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False
        ),

        shapes=[
            dict(type='line', xref='paper', x0=0, x1=1, yref='y', y0=2, y1=2,
                 line=dict(color='orange', width=2, dash='dash')),
            dict(type='line', xref='paper', x0=0, x1=1, yref='y', y0=1, y1=1,
                 line=dict(color='red', width=2, dash='dash'))
        ],
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )

    # Komma → Apostroph im Pellet-Hover (10,192 → 10'192)
    for trace in fig.data:
        if 'Pellet' in trace.name:
            trace.hovertemplate = trace.hovertemplate.replace(",", "&#x27;")

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
# Dash App
# ------------------------------------------------------------
app = Dash(__name__)
app.title = "Pellet & Temperatur Dashboard"

df = load_data(CSV_PATH)

app.layout = html.Div([
    dcc.Graph(id='pellet-temp-graph', figure=make_figure(df)),
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
# Callback: Statistik an sichtbaren Bereich (Range-Slider/Zoom) anpassen
# ------------------------------------------------------------
@app.callback(
    Output('stats-table', 'data'),
    Input('pellet-temp-graph', 'relayoutData')
)
def update_stats(relayoutData):
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

    dff = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

    if dff.empty:
        return [
            {"metric": "Pellet Min [kg]", "value": "-"},
            {"metric": "Pellet Max [kg]", "value": "-"},
            {"metric": "Pellet Δ [kg]", "value": "-"},
            {"metric": "Temp Min (°C)", "value": "-"},
            {"metric": "Temp Max (°C)", "value": "-"},
            {"metric": "Temp Mittel (°C)", "value": "-"},
            {"metric": "Messpunkte", "value": 0},
        ]

    return make_stats(dff)

# ------------------------------------------------------------
# Start
# ------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
