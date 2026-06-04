import os
import tempfile

import requests
from fastapi import UploadFile

from app.core.config import settings


ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


def download_image_to_temp_file(image_url: str, prefix: str) -> str:
    try:
        response = requests.get(
            image_url,
            stream=True,
            timeout=settings.REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as error:
        raise ValueError(f"Could not download image: {str(error)}")

    if response.status_code != 200:
        raise ValueError(f"Image download failed with status code {response.status_code}")

    content_type = response.headers.get("content-type", "").split(";")[0].lower()

    if content_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError(f"Invalid image type: {content_type}")

    suffix = ALLOWED_IMAGE_TYPES[content_type]

    max_bytes = settings.MAX_IMAGE_MB * 1024 * 1024
    total_bytes = 0

    file_descriptor, temp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix)

    try:
        with os.fdopen(file_descriptor, "wb") as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue

                total_bytes += len(chunk)

                if total_bytes > max_bytes:
                    raise ValueError(
                        f"Image size is too large. Max allowed size is {settings.MAX_IMAGE_MB} MB"
                    )

                temp_file.write(chunk)

        return temp_path

    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def delete_temp_file(file_path: str | None) -> None:
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

async def save_uploaded_image_to_temp_file(upload_file: UploadFile, prefix: str) -> str:
    content_type = upload_file.content_type.lower() if upload_file.content_type else ""

    if content_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError(f"Invalid image type: {content_type}")

    suffix = ALLOWED_IMAGE_TYPES[content_type]

    max_bytes = settings.MAX_IMAGE_MB * 1024 * 1024
    total_bytes = 0

    file_descriptor, temp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix)

    try:
        with os.fdopen(file_descriptor, "wb") as temp_file:
            while True:
                chunk = await upload_file.read(8192)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > max_bytes:
                    raise ValueError(
                        f"Image size is too large. Max allowed size is {settings.MAX_IMAGE_MB} MB"
                    )

                temp_file.write(chunk)

        return temp_path

    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise

    finally:
        await upload_file.close()        