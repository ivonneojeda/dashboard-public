import logging
import azure.functions as func
import os
import requests
import pandas as pd
from sentiment_utils import read_latest_blob, save_dataframe_to_blob
from datetime import datetime
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

app = func.FunctionApp()

def authenticate_client():
    """Autentica el cliente de Azure Cognitive Services"""
    key = os.environ.get("AZURE_TEXT_KEY")
    endpoint = os.environ.get("AZURE_TEXT_ENDPOINT")
    if not key or not endpoint:
        logging.error("Faltan variables de entorno de Azure Cognitive Services.")
        return None
    credential = AzureKeyCredential(key)
    client = TextAnalyticsClient(endpoint=endpoint, credential=credential)
    return client

@app.timer_trigger(
    schedule="0 */5 * * * *",  # Cada 5 minutos
    arg_name="myTimer",
    run_on_startup=False,
    use_monitor=False
)
def timer_trigger(myTimer: func.TimerRequest) -> None:
    ACCESS_TOKEN = os.environ.get("FACEBOOK_ACCESS_TOKEN")
    PAGE_ID = os.environ.get("META_PAGE_ID", "100578801707401")
    CONTAINER_NAME = os.environ.get("AZURE_CONTAINER_NAME", "datos-facebook")

    if not ACCESS_TOKEN:
        logging.error("Falta FACEBOOK_ACCESS_TOKEN en las variables de entorno.")
        return

    client = authenticate_client()
    if client is None:
        return

    try:
        # Llamada a la API de Facebook
        response = requests.get(
            f"https://graph.facebook.com/v19.0/{PAGE_ID}/feed",
            params={
                "fields": "message,likes.summary(true),created_time",
                "access_token": ACCESS_TOKEN
            }
        )
        response.raise_for_status()
        posts = response.json().get("data", [])

        if not posts:
            logging.info("No se encontraron posts.")
            return

        # Filtrar posts con mensaje
        documents = [post["message"] for post in posts if "message" in post]

        if not documents:
            logging.info("No hay mensajes para analizar.")
            return

        # Análisis de sentimiento real con Azure
        sentiment_results = client.analyze_sentiment(documents)

        # Construir DataFrame final
        results = []
        for post, sentiment in zip(posts, sentiment_results):
            results.append({
                "Post": post.get("message", "N/A"),
                "Likes": post.get("likes", {}).get("summary", {}).get("total_count", 0),
                "Sentimiento": sentiment.sentiment if sentiment else "N/A"
            })

        df = pd.DataFrame(results)
        save_dataframe_to_blob(df, CONTAINER_NAME)
        logging.info(f"Se procesaron {len(df)} posts correctamente.")

    except Exception as e:
        logging.error(f"Error en timer trigger: {e}")



