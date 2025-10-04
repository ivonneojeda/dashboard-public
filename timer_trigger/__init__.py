import logging
import os
import io
from datetime import datetime
import pandas as pd
import requests
import json
import azure.functions as func
from azure.storage.blob import BlobServiceClient


def save_dataframe_to_blob(df_to_save, container_name: str):
    """Guarda un DataFrame como CSV en Azure Blob Storage."""
    try:
        connect_str = os.environ.get("AzureWebJobsStorage")
        if not connect_str:
            logging.error("La variable AzureWebJobsStorage no está configurada.")
            return

        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        container_client = blob_service_client.get_container_client(container_name)

        if not container_client.exists():
            container_client.create_container()

        now = datetime.now()
        file_name = f"sentimiento_{now.strftime('%Y-%m-%d_%H-%M-%S')}.csv"

        output = io.StringIO()
        df_to_save.to_csv(output, index=False)
        data = output.getvalue()

        blob_client = container_client.get_blob_client(blob=file_name)
        blob_client.upload_blob(data, overwrite=True)
        logging.info(f"✅ DataFrame guardado en {container_name}/{file_name}")

    except Exception as e:
        logging.error(f"⚠️ Error al guardar datos en Blob Storage: {e}")


def main(myTimer: func.TimerRequest) -> None:
    """Función ejecutada por Timer Trigger"""
    logging.info("⏰ Timer trigger ejecutado")

    ACCESS_TOKEN = os.environ.get("FACEBOOK_ACCESS_TOKEN")
    PAGE_ID = os.environ.get("META_PAGE_ID")
    AZURE_API_KEY = os.environ.get("AZURE_TEXT_KEY")
    AZURE_ENDPOINT = os.environ.get("AZURE_TEXT_ENDPOINT")

    if not all([ACCESS_TOKEN, PAGE_ID, AZURE_API_KEY, AZURE_ENDPOINT]):
        logging.error("⚠️ Variables de entorno no configuradas correctamente.")
        return

    logging.info(f"✅ Usando Page ID: {PAGE_ID}")

    url_facebook = f"https://graph.facebook.com/v19.0/{PAGE_ID}/feed"
    params_facebook = {
        "fields": "message,likes.summary(true),created_time",
        "access_token": ACCESS_TOKEN
    }

    try:
        response_facebook = requests.get(url_facebook, params=params_facebook)
        
        # Logging detallado para depurar errores
        if response_facebook.status_code != 200:
            try:
                error_info = response_facebook.json().get("error", {})
                logging.error(f"⚠️ Error de Graph API: {error_info}")
            except json.JSONDecodeError:
                logging.error(f"⚠️ Error de Graph API, status {response_facebook.status_code}")
            return
        
        posts = response_facebook.json().get("data", [])

        if not posts:
            logging.info("No se encontraron posts en la página.")
            return

        documents = [
            {"id": str(i), "text": post["message"]}
            for i, post in enumerate(posts) if "message" in post
        ]

        batch_size = 10
        document_batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

        all_sentiment_data = []
        for batch in document_batches:
            url_azure = f"{AZURE_ENDPOINT}/text/analytics/v3.0/sentiment"
            headers_azure = {
                "Ocp-Apim-Subscription-Key": AZURE_API_KEY,
                "Content-Type": "application/json"
            }
            body_azure = {"documents": batch}
            response_azure = requests.post(url_azure, headers=headers_azure, data=json.dumps(body_azure))
            response_azure.raise_for_status()
            all_sentiment_data.extend(response_azure.json().get("documents", []))

        results = []
        for post, sentiment in zip(posts, all_sentiment_data):
            results.append({
                "Fecha": post.get("created_time", "N/A"),
                "Post": post.get("message", "N/A"),
                "Likes": post.get("likes", {}).get("summary", {}).get("total_count", 0),
                "Sentimiento": sentiment.get("sentiment", "N/A")
            })

        df = pd.DataFrame(results)
        logging.info("📂 Guardando los datos en Azure Blob Storage...")
        save_dataframe_to_blob(df, "datos-facebook")

    except requests.exceptions.RequestException as req_ex:
        logging.error(f"⚠️ Error de solicitud HTTP: {req_ex}")
    except Exception as ex:
        logging.error(f"⚠️ Ocurrió un error al procesar posts: {ex}")

    if myTimer.past_due:
        logging.warning("⏱️ El timer está retrasado.")

    logging.info("✅ Python timer trigger function ejecutada correctamente.")
