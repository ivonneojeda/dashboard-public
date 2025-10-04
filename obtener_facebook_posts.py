# obtener_facebook_posts.py
import os
import requests
import pandas as pd
from datetime import datetime
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# -------------------------
# Variables de entorno
# -------------------------
access_token = os.getenv("FACEBOOK_ACCESS_TOKEN")
page_id = os.getenv("META_PAGE_ID")
azure_key = os.getenv("AZURE_TEXT_KEY")
azure_endpoint = os.getenv("AZURE_TEXT_ENDPOINT")

# Validar que existan las variables
if not all([access_token, page_id, azure_key, azure_endpoint]):
    print("❌ Faltan variables de entorno. Asegúrate de definir FACEBOOK_ACCESS_TOKEN, META_PAGE_ID, AZURE_TEXT_KEY y AZURE_TEXT_ENDPOINT")
    exit()

print("✅ Variables de entorno cargadas correctamente")

# -------------------------
# Configurar cliente Azure
# -------------------------
client = TextAnalyticsClient(
    endpoint=azure_endpoint,
    credential=AzureKeyCredential(azure_key)
)

# -------------------------
# Obtener posts de Facebook
# -------------------------
url = f"https://graph.facebook.com/v17.0/{page_id}/posts"
params = {
    "fields": "message,created_time,likes.summary(true)",
    "limit": 50,
    "access_token": access_token
}

try:
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json().get("data", [])
    if not data:
        print("⚠️ No se encontraron posts")
except requests.exceptions.RequestException as e:
    print(f"⚠️ Error obteniendo posts de Meta: {e}")
    data = []

# -------------------------
# Preparar DataFrame
# -------------------------
posts_list = []
for post in data:
    text = post.get("message", "")
    created_time = post.get("created_time")
    likes = post.get("likes", {}).get("summary", {}).get("total_count", 0)

    # Analizar sentimiento con Azure
    if text.strip():
        try:
            sentiment_result = client.analyze_sentiment([text])[0]
            if sentiment_result.sentiment == "positive":
                sentiment = "Positivo"
            elif sentiment_result.sentiment == "negative":
                sentiment = "Negativo"
            else:
                sentiment = "Neutro"
        except Exception as e:
            sentiment = "Neutro"
            print(f"⚠️ Error analizando sentimiento: {e}")
    else:
        sentiment = "Neutro"

    posts_list.append({
        "Fecha": created_time,
        "Post": text,
        "Likes": likes,
        "Sentimiento": sentiment
    })

# -------------------------
# Guardar CSV
# -------------------------
if posts_list:
    df = pd.DataFrame(posts_list)
    df.to_csv("facebook_posts.csv", index=False)
    print(f"✅ CSV generado en facebook_posts.csv con {len(df)} posts")
else:
    print("⚠️ Sin datos para generar CSV")
