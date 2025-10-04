# sentiment_utils.py
import os
import io
import logging
import pandas as pd
from azure.storage.blob import BlobServiceClient

def read_latest_blob(container_name: str, return_last_modified: bool = False):
    """
    Lee el archivo más reciente de un contenedor de Azure Blob Storage y devuelve un DataFrame.
    Si return_last_modified=True, también devuelve la fecha de última modificación del blob.
    """
    try:
        # Buscar la cadena de conexión en variables de entorno
        connect_str = (
            os.getenv("AzureWebJobsStorage")
            or os.getenv("AZUREWEBJOBSSTORAGE")
            or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        )

        if not connect_str:
            logging.error("❌ No se encontró ninguna cadena de conexión en las variables de entorno.")
            return (pd.DataFrame(), None) if return_last_modified else pd.DataFrame()

        logging.info("✅ Usando cadena de conexión para Blob Storage.")

        # Cliente de Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        container_client = blob_service_client.get_container_client(container_name)

        # Listar blobs
        blobs = list(container_client.list_blobs())
        if not blobs:
            logging.info(f"No se encontraron blobs en {container_name}.")
            return (pd.DataFrame(), None) if return_last_modified else pd.DataFrame()

        # Blob más reciente (por fecha de última modificación)
        latest_blob = max(blobs, key=lambda b: b.last_modified)
        blob_client = container_client.get_blob_client(latest_blob.name)
        stream = blob_client.download_blob().readall()

        # Leer CSV
        df = pd.read_csv(io.BytesIO(stream))
        logging.info(f"📂 Archivo cargado: {latest_blob.name}")
        logging.info(f"📊 Columnas detectadas: {df.columns.tolist()}")
        print("📂 Columnas en el CSV:", df.columns.tolist())

        # Devolver resultados
        if return_last_modified:
            return df, latest_blob.last_modified
        else:
            return df

    except Exception as e:
        logging.error(f"Error al leer datos desde Blob Storage: {e}")
        return (pd.DataFrame(), None) if return_last_modified else pd.DataFrame()
