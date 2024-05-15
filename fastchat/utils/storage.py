"""
Storage utilities for handling image files etc.
"""
from io import BytesIO
import base64
import os


def load_image(image_file):
    from PIL import Image
    import requests

    image = None

    if image_file.startswith("http://") or image_file.startswith("https://"):
        timeout = int(os.getenv("REQUEST_TIMEOUT", "3"))
        response = requests.get(image_file, timeout=timeout)
        image = Image.open(BytesIO(response.content))
    elif image_file.lower().endswith(("png", "jpg", "jpeg", "webp", "gif")):
        image = Image.open(image_file)
    elif image_file.startswith("data:"):
        image_file = image_file.split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_file)))
    else:
        image = Image.open(BytesIO(base64.b64decode(image_file)))

    return image


def upload_image_file_to_gcs(image, filename):
    from google.cloud import storage
    import io

    storage_client = storage.Client()
    # upload file to GCS
    bucket = storage_client.get_bucket("arena_user_content")

    blob = bucket.blob(f"{filename}")
    if not blob.exists():
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        blob.upload_from_file(buffer, content_type="image/png")

    return blob.public_url


def get_image_file_from_gcs(filename):
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.get_bucket("arena_user_content")
    blob = bucket.blob(f"{filename}")
    contents = blob.download_as_bytes()

    return contents
