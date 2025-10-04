# dashboard_fblogin.py
"""
OAuth manual para Facebook Business Config (config_id).
- Rutas:
  /facebook/login   -> redirige al diálogo de Facebook (incluye config_id)
  /facebook/callback -> recibe 'code' y hace intercambio por access_token
  /logout           -> borra sesión
- Dash usa session['fb_token'] para solicitar /me
- RECUERDA: en Facebook Developers debes añadir EXACTAMENTE:
  https://<TU_DOMINIO>/facebook/callback
  (por ejemplo https://predictions-dashboard.onrender.com/facebook/callback)
"""

import os
import secrets
import urllib.parse
import requests
from flask import Flask, redirect, url_for, request, session, jsonify
import dash
from dash import html, dcc, dash_table, Input, Output
import pandas as pd
import plotly.express as px

# -----------------------------
# Configuración básica y envvars
# -----------------------------
FB_APP_ID = os.environ.get("FACEBOOK_OAUTH_CLIENT_ID")
FB_APP_SECRET = os.environ.get("FACEBOOK_OAUTH_CLIENT_SECRET")
FB_BUSINESS_CONFIG_ID = os.environ.get("FACEBOOK_BUSINESS_CONFIG_ID")  # 1161725706017833
FLASK_SECRET = os.environ.get("FLASK_SECRET_KEY", secrets.token_urlsafe(32))

if not FB_APP_ID or not FB_APP_SECRET:
    # no abort here to keep script importable; logs will show later when used
    pass

server = Flask(__name__)
server.secret_key = FLASK_SECRET
# Opcional: en producción podrías ajustar cookie secure/httponly, etc.
server.config.update(
    SESSION_COOKIE_SAMESITE="Lax"
)

# -----------------------------
# Helper OAuth
# -----------------------------
def get_redirect_uri():
    # crea la URI de callback tal como Facebook requiere (debe coincidir exactamente)
    # request.url_root incluye el trailing slash, quitamos para evitar '//' 
    root = request.url_root.rstrip('/')
    return f"{root}/facebook/callback"

def build_facebook_auth_url():
    """
    Genera la URL de autorización de Facebook para Business Config (config_id).
    Usamos response_type=code (SUAT) y pasamos config_id.
    """
    redirect_uri = get_redirect_uri()
    state = secrets.token_urlsafe(16)
    session['oauth_state'] = state

    params = {
        "client_id": FB_APP_ID,
        "redirect_uri": redirect_uri,
        "config_id": FB_BUSINESS_CONFIG_ID,  # recomendado por Meta para Business Login
        "response_type": "code",
        # Aunque config_id reemplaza scope, mantener scope básico no hace daño:
        # "scope": "email,public_profile",
        "state": state,
    }
    # construir URL
    base = "https://www.facebook.com/v23.0/dialog/oauth"
    url = base + "?" + urllib.parse.urlencode(params)
    return url

def exchange_code_for_token(code):
    """
    Intercambia el código por access_token (servidor a servidor).
    Retorna dict con la respuesta JSON.
    """
    redirect_uri = session.get('last_redirect_uri') or None
    # el redirect_uri usado aquí debe ser el mismo que el enviado en la autorización
    # preferimos generar en tiempo de request para evitar inconsistencias
    if not redirect_uri:
        # recomponer a partir de request si no está en sesión
        redirect_uri = request.url_root.rstrip('/') + "/facebook/callback"

    token_endpoint = "https://graph.facebook.com/v23.0/oauth/access_token"
    params = {
        "client_id": FB_APP_ID,
        "redirect_uri": redirect_uri,
        "client_secret": FB_APP_SECRET,
        "code": code
    }
    try:
        r = requests.get(token_endpoint, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "raw": getattr(r, "text", None)}

def fb_get_me(access_token):
    """Pide /me?fields=name,email con el token dado"""
    try:
        r = requests.get("https://graph.facebook.com/me", params={
            "access_token": access_token,
            "fields": "name,email"
        }, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# Rutas de login manual
# -----------------------------
@server.route("/facebook/login")
def facebook_login():
    """Inicia el flujo de login redirigiendo a Facebook."""
    if not FB_APP_ID or not FB_BUSINESS_CONFIG_ID:
        return ("Facebook app config missing. Set FACEBOOK_OAUTH_CLIENT_ID "
                "and FACEBOOK_BUSINESS_CONFIG_ID in environment."), 500

    # guardamos la redirect_uri usada para poder usarla luego al intercambiar código
    session['last_redirect_uri'] = get_redirect_uri()
    auth_url = build_facebook_auth_url()
    return redirect(auth_url)

@server.route("/facebook/callback")
def facebook_callback():
    """Recibimos el 'code' de Facebook y hacemos el intercambio por access_token."""
    error = request.args.get("error")
    if error:
        desc = request.args.get("error_description") or ""
        return f"Facebook OAuth error: {error} {desc}", 400

    state = request.args.get("state")
    if not state or session.get("oauth_state") != state:
        return "Invalid OAuth state (possible CSRF).", 400

    code = request.args.get("code")
    if not code:
        return "No code provided by Facebook.", 400

    # Intercambiar código por token
    token_resp = exchange_code_for_token(code)
    if not token_resp or token_resp.get("error"):
        return f"Token exchange error: {token_resp}", 400

    access_token = token_resp.get("access_token")
    if not access_token:
        return f"Token not found in response: {token_resp}", 400

    # Guardar token en sesión (para demo). En producción, preferir storage server-side.
    session['fb_token'] = access_token
    # borrar oauth_state ya que fue consumido
    session.pop('oauth_state', None)

    # redirigir de regreso al dashboard (ruta raíz del Dash)
    return redirect("/")

@server.route("/logout")
def logout():
    session.pop("fb_token", None)
    return redirect("/")

# -----------------------------
# Configuración de Dash
# -----------------------------
app = dash.Dash(__name__, server=server, url_base_pathname='/', suppress_callback_exceptions=True)
app.title = "Dashboard de Opiniones"

# datos de ejemplo (puedes reemplazarlos por tu CSV/Blob)
df_sentimientos = pd.DataFrame({
    "Fecha": pd.date_range(start="2025-01-01", periods=30),
    "Sentimiento": [0.1, 0.3, -0.2, 0.5, -0.1, 0.0, 0.2, -0.3, 0.4, 0.1,
                    0.2, 0.1, -0.2, 0.3, 0.0, 0.2, -0.1, 0.1, 0.3, -0.2,
                    0.2, 0.0, -0.1, 0.3, 0.4, 0.2, -0.3, 0.1, 0.0, 0.2]
})

df_tabla = pd.DataFrame({
    "Publicación": [f"Post {i}" for i in range(1, 11)],
    "Likes": [10, 25, 15, 30, 20, 12, 22, 18, 35, 28],
    "Comentarios": [1, 5, 2, 4, 3, 2, 1, 0, 5, 3]
})

app.layout = html.Div([
    html.H1("📊 Dashboard de Opiniones con Facebook Login (Business OAuth)"),
    html.Div(id="login-area"),
    dcc.Tabs([
        dcc.Tab(label="Gráficos", children=[
            dcc.Graph(
                id="grafico-sentimientos",
                figure=px.line(df_sentimientos, x="Fecha", y="Sentimiento", title="Evolución del Sentimiento")
            )
        ]),
        dcc.Tab(label="Tabla", children=[
            dash_table.DataTable(
                id="tabla-publicaciones",
                columns=[{"name": i, "id": i} for i in df_tabla.columns],
                data=df_tabla.to_dict("records"),
                page_size=5,
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center"}
            )
        ]),
        dcc.Tab(label="Resumen", children=[
            html.Div([
                html.P(f"Total Posts: {len(df_tabla)}"),
                html.P(f"Likes Totales: {df_tabla['Likes'].sum()}"),
                html.P(f"Comentarios Totales: {df_tabla['Comentarios'].sum()}")
            ], style={"margin": "20px", "fontSize": "18px"})
        ])
    ])
])

# -----------------------------
# Callback que actualiza el área de login
# -----------------------------
@app.callback(
    Output("login-area", "children"),
    Input("login-area", "id")  # dummy input para forzar ejecución
)
def render_login_area(_):
    token = session.get("fb_token")
    if not token:
        # mostrar botón que redirige a /facebook/login
        return html.Div([
            html.P("No estás logueado en Facebook."),
            html.A("Inicia sesión con Facebook (Business Login)", href="/facebook/login")
        ])
    # con token -> pedir /me
    me = fb_get_me(token)
    if me.get("error"):
        # si falla, forzar logout para poder reintentar el login
        return html.Div([
            html.P(f"Error al obtener usuario: {me.get('error')}"),
            html.A("Volver a iniciar sesión", href="/facebook/login"),
            html.Br(),
            html.A("Cerrar sesión local", href="/logout")
        ])
    # mostrar nombre y email
    name = me.get("name", "Usuario")
    email = me.get("email", "No disponible")
    return html.Div([
        html.P(f"¡Bienvenido, {name}!"),
        html.P(f"Correo: {email}"),
        html.A("Cerrar sesión", href="/logout")
    ])

# -----------------------------
# Ejecutar app en Render (local fallback)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)



