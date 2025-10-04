# dashboard_app.py
import os
import re
import glob
import itertools
import collections
import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import dash_cytoscape as cyto
import networkx as nx
from prophet import Prophet
import datetime

# -----------------------------
# Config
# -----------------------------
CSV_FOLDER = os.getenv("CSV_FOLDER", "datos")  # carpeta donde estén los CSV generados por la función
TOP_WORDS = 25   # nodos máximos del grafo
FORECAST_HOURS = 8  # horizonte de forecast en horas
RESAMPLE_FREQ = "1H"  # frecuencia de resampleo para Prophet

# -----------------------------
# Stopwords español (ajusta si quieres añadir más)
# -----------------------------
stopwords_es = {
    "a","ante","bajo","cabe","con","contra","de","del","desde","durante","en","entre",
    "hacia","hasta","mediante","para","por","según","sin","so","sobre","tras","versus","vía",
    "el","la","los","las","un","una","unos","unas","lo","al","su","sus","mi","mis","tu","tus",
    "nuestro","nuestra","nuestros","nuestras","vosotros","vosotras","vuestro","vuestra","vuestros",
    "vuestras","ellos","ellas","nosotros","nosotras","yo","tú","usted","ustedes","él","ella",
    "me","te","se","nos","os","les","le","y","o","que","qué","como","cómo","para","porque","pero",
    "si","ya","tan","muy","más","menos","también","cuando","donde","dónde","ser","estar","haber"
}

# helper: limpiar token
_punct_re = re.compile(r'^[\W_]+|[\W_]+$')

def clean_token(tok: str) -> str:
    t = _punct_re.sub("", tok.lower())
    return t

# -----------------------------
# Cargar último CSV disponible
# -----------------------------
def load_latest_csv(folder: str):
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    if not csv_files:
        print("⚠️ No hay CSV en", folder)
        return pd.DataFrame(), None
    latest = max(csv_files, key=os.path.getmtime)
    try:
        df = pd.read_csv(latest)
        print("✅ CSV cargado correctamente:", latest)
        return df, latest
    except Exception as e:
        print("⚠️ Error cargando CSV:", e)
        return pd.DataFrame(), None

df, csv_path = load_latest_csv(CSV_FOLDER)

# -----------------------------
# Función: generar grafo (nodes+edges para cytoscape)
# -----------------------------
def generar_grafo_palabras(df: pd.DataFrame, top_n: int = TOP_WORDS):
    if df.empty or "Post" not in df.columns:
        return []

    # contar palabras y acumular sentimientos por palabra
    word_counts = collections.Counter()
    word_sent_map = collections.defaultdict(list)
    posts_tokens = []

    for _, row in df.iterrows():
        text = str(row.get("Post", ""))
        # separar por espacios, limpiar, filtrar stopwords y tokens cortos
        tokens = [clean_token(t) for t in re.split(r"\s+", text) if t and len(t) > 0]
        tokens = [t for t in tokens if t and t not in stopwords_es and len(t) > 2]
        unique_tokens = list(dict.fromkeys(tokens))  # mantener orden pero únicos por post
        posts_tokens.append(unique_tokens)
        for t in unique_tokens:
            word_counts[t] += 1
            word_sent_map[t].append(str(row.get("Sentimiento", "")).lower())

    if not word_counts:
        return []

    top_words = [w for w, _ in word_counts.most_common(top_n)]

    # crear grafo donde nodos = top_words, aristas = coocurrencia en el mismo post (combinaciones)
    G = nx.Graph()
    color_map = {"positivo":"#2ca02c", "positivo.":"#2ca02c", "positive":"#2ca02c",
                 "negativo":"#d62728", "negative":"#d62728",
                 "neutro":"#7f7f7f", "neutral":"#7f7f7f"}

    for w in top_words:
        freq = word_counts[w]
        # determinar sentimiento predominante para la palabra (mode)
        sents = [s for s in word_sent_map.get(w, []) if s]
        sent_mode = None
        if sents:
            try:
                sent_mode = collections.Counter(sents).most_common(1)[0][0]
            except Exception:
                sent_mode = None
        color = color_map.get(sent_mode, "#7f7f7f")
        size = max(20, 8 + freq * 7)  # escala visual
        G.add_node(w, size=size, color=color, freq=int(freq), sentiment=sent_mode)

    # edges by co-occurrence within same post (combinations of words present and in top_words)
    edge_counts = collections.Counter()
    for tokens in posts_tokens:
        present = [t for t in tokens if t in top_words]
        for a, b in itertools.combinations(sorted(set(present)), 2):
            edge_counts[(a,b)] += 1

    for (a,b), w in edge_counts.items():
        if a in G.nodes and b in G.nodes:
            G.add_edge(a, b, weight=int(w))

    # convertir a elementos para cytoscape
    elements = []
    for node, attrs in G.nodes(data=True):
        elements.append({
            "data": {"id": node, "label": node, "freq": attrs.get("freq", 1), "sentiment": attrs.get("sentiment")},
            "style": {"width": attrs["size"], "height": attrs["size"], "background-color": attrs["color"]}
        })
    for source, target, attrs in G.edges(data=True):
        elements.append({
            "data": {"id": f"e-{source}-{target}", "source": source, "target": target, "weight": attrs.get("weight", 1)}
        })

    return elements

# -----------------------------
# Función: preparar series horarias y forecast con Prophet
# -----------------------------
def build_forecast_figure(df: pd.DataFrame, hours_ahead: int = FORECAST_HOURS, resample_freq: str = RESAMPLE_FREQ):
    # figura vacía por defecto
    fig = go.Figure()
    fig.update_layout(title="Sin datos para pronóstico")

    if df.empty or "Fecha" not in df.columns or "Sentimiento" not in df.columns:
        return fig

    # Normalizar columna Fecha: quitar timezone (si existe) de forma robusta
    def strip_tz_str(s):
        if pd.isna(s):
            return s
        s = str(s).strip()
        # eliminar sufijos tipo +0000, +00:00, Z, -0500, etc.
        s = re.sub(r'([+-]\d{2}:?\d{2}|Z|[+-]\d{4})$', '', s).strip()
        return s

    # crear copia
    dfc = df.copy()
    dfc["Fecha_clean"] = dfc["Fecha"].apply(strip_tz_str)
    dfc["Fecha_parsed"] = pd.to_datetime(dfc["Fecha_clean"], errors="coerce")

    dfc = dfc.dropna(subset=["Fecha_parsed", "Sentimiento"]).copy()
    if dfc.empty:
        return fig

    # mapear sentimiento a numerico
    sent_map = {"Positivo": 1, "Negativo": -1, "Neutro": 0,
                "positivo": 1, "negativo": -1, "neutro": 0,
                "positive": 1, "negative": -1, "neutral": 0}
    dfc["sent_score"] = dfc["Sentimiento"].map(sent_map).fillna(0).astype(float)

    # resampleo por hora (o la frecuencia que configures)
    dfc = dfc.set_index("Fecha_parsed").sort_index()
    # promedio por intervalo
    df_hour = dfc["sent_score"].resample(resample_freq).mean().to_frame()
    # rellenar huecos con 0 (neutro) — si prefieres interpolación, cambia esta línea
    df_hour["y"] = df_hour["sent_score"].fillna(0)
    df_hour = df_hour.reset_index().rename(columns={"Fecha_parsed": "ds"})

    if df_hour.empty or df_hour["y"].nunique() <= 1:
        fig.update_layout(title="No hay suficiente variación de datos para pronóstico")
        return fig

    # preparar para Prophet
    df_prophet = df_hour[["ds", "y"]].copy()
    # asegurar que ds es naive datetime (sin tz)
    if pd.api.types.is_datetime64_any_dtype(df_prophet["ds"]):
        # convert to naive by replacing tz-aware if any (should be naive after strip)
        df_prophet["ds"] = pd.to_datetime(df_prophet["ds"]).dt.tz_localize(None)

    try:
        model = Prophet()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=hours_ahead, freq=resample_freq)
        forecast = model.predict(future)
    except Exception as e:
        fig.update_layout(title=f"Error entrenando Prophet: {e}")
        return fig

    # construir figura: histórico (puntos) + forecast (línea) + banda de confianza
    fig = go.Figure()
    # histórico (puntos)
    fig.add_trace(go.Scatter(
        x=df_prophet["ds"],
        y=df_prophet["y"],
        mode="markers",
        name="Histórico",
        marker=dict(color="blue", size=6)
    ))
    # pronóstico completo (línea)
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode="lines",
        name=f"Pronóstico ({hours_ahead}h)",
        line=dict(color="orange")
    ))
    # banda de confianza (yhat_upper / yhat_lower) — dibujamos como relleno
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(255,165,0,0.2)",
        name="Confianza 95%"
    ))

    fig.update_layout(
        title=f"Pronóstico de sentimiento ({hours_ahead}h)",
        xaxis_title="Hora",
        yaxis_title="Sentimiento promedio",
        legend=dict(orientation="v")
    )

    return fig

# -----------------------------
# Crear app Dash
# -----------------------------
app = dash.Dash(__name__)
app.title = "Dashboard Análisis de Sentimientos"

app.layout = html.Div([
    html.H1("📊 Dashboard de Opiniones", style={"textAlign": "center"}),
    html.Div([
        dash_table.DataTable(
            id="tabla-posts",
            columns=[{"name": i, "id": i} for i in df.columns] if not df.empty else [],
            data=df.to_dict("records"),
            page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "whiteSpace": "normal"}
        )
    ], style={"margin": "10px"}),
    html.Div([
        dcc.Graph(id="grafico-sentimientos"),
        dcc.Graph(id="forecast-sentimiento"),
    ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "padding": "10px"}),
    html.Div([
        html.H2("🔗 Grafo de palabras"),
        cyto.Cytoscape(
            id="grafo-palabras",
            layout={"name": "cose"},
            style={"width": "100%", "height": "520px"},
            elements=[]
        )
    ], style={"padding": "10px"}),
    html.Div(id="last-update", style={"marginTop": "10px", "textAlign": "center"})
])

# -----------------------------
# Callback principal
# -----------------------------
@app.callback(
    [
        Output("tabla-posts", "columns"),
        Output("tabla-posts", "data"),
        Output("grafico-sentimientos", "figure"),
        Output("forecast-sentimiento", "figure"),
        Output("grafo-palabras", "elements"),
        Output("last-update", "children")
    ],
    [Input("tabla-posts", "id")]
)
def update_dashboard(_):
    global df
    # recargar último CSV por si la función lo actualizó
    df_new, path = load_latest_csv(CSV_FOLDER)
    if not df_new.empty:
        df = df_new

    if df.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Sin datos")
        return [], [], empty_fig, empty_fig, [], "Sin datos (ningún CSV disponible)"

    # columnas y data para la tabla
    columns = [{"name": i, "id": i} for i in df.columns]
    data = df.to_dict("records")

    # histograma de sentimientos
    try:
        fig_sent = px.histogram(df, x="Sentimiento", color="Sentimiento", title="Distribución de sentimientos")
    except Exception:
        fig_sent = go.Figure()
        fig_sent.update_layout(title="No es posible mostrar histograma")

    # forecast con Prophet
    try:
        fig_forecast = build_forecast_figure(df, hours_ahead=FORECAST_HOURS, resample_freq=RESAMPLE_FREQ)
    except Exception as e:
        fig_forecast = go.Figure()
        fig_forecast.update_layout(title=f"Error generando forecast: {e}")

    # grafo de palabras
    try:
        elements = generar_grafo_palabras(df, top_n=TOP_WORDS)
    except Exception as e:
        elements = []
        print("Error generando grafo:", e)

    last_update_text = f"Última actualización local: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    if csv_path:
        last_update_text += f"  ·  CSV: {os.path.basename(csv_path)}"

    return columns, data, fig_sent, fig_forecast, elements, last_update_text

# -----------------------------
# Ejecutar servidor
# -----------------------------
if __name__ == "__main__":
    # usar app.run en vez de run_server (compatibilidad con versiones nuevas)
    app.run(debug=True)

