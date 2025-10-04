import logging
import azure.functions as func
import os
import requests
import pandas as pd
from sentiment_utils import read_latest_blob, save_dataframe_to_blob
from datetime import datetime

app = func.FunctionApp()

@app.timer_trigger(
    schedule="0 */5 * * * *",  # Cada 5 minutos
    arg_name="myTimer",
    run_on_startup=False,
    use_monitor=False
)
def timer_trigger(myTimer: func.TimerRequest) -> None:
    ACCESS_TOKEN = os.environ.get("FACEBOOK_ACCESS_TOKEN")
    PAGE_ID = "100578801707401"
    AZURE_TEXT_KEY = os.environ.get("AZURE_TEXT_KEY")
    AZURE_TEXT_ENDPOINT = os.environ.get("AZURE_TEXT_ENDPOINT")
    CONTAINER_NAME = os.environ.get("AZURE_CONTAINER_NAME", "datos-facebook")

    if not all([ACCESS_TOKEN, AZURE_TEXT_KEY, AZURE_TEXT_ENDPOINT]):
        logging.error("Faltan variables de entorno.")
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

        # Preparar documentos para análisis de sentimiento
        documents = [
            {"id": str(i), "text": post["message"]}
            for i, post in enumerate(posts) if "message" in post
        ]

        # Llamada a Azure Cognitive Services (simulada)
        # Aquí puedes reemplazar por tu función real de análisis de sentimiento
        all_sentiment_data = [{"sentiment": "neutral"}] * len(documents)

        # Construir DataFrame final
        results = [
            {
                "Post": post.get("message", "N/A"),
                "Likes": post.get("likes", {}).get("summary", {}).get("total_count", 0),
                "Sentimiento": sentiment.get("sentiment", "N/A")
            }
            for post, sentiment in zip(posts, all_sentiment_data)
        ]

        df = pd.DataFrame(results)
        save_dataframe_to_blob(df, CONTAINER_NAME)
        logging.info(f"Se procesaron {len(df)} posts correctamente.")

    except Exception as e:
        logging.error(f"Error en timer trigger: {e}")


