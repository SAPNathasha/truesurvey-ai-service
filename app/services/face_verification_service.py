import os

# Helps avoid some TensorFlow/Keras compatibility issues.
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from deepface import DeepFace

from app.core.config import settings


def compare_document_face_with_selfie(document_image_path: str, selfie_image_path: str) -> dict:
    try:
        result = DeepFace.verify(
            img1_path=document_image_path,
            img2_path=selfie_image_path,
            model_name=settings.MODEL_NAME,
            detector_backend=settings.DETECTOR_BACKEND,
            align=True,
            enforce_detection=True,
        )

        verified = bool(result.get("verified", False))
        distance = float(result.get("distance", 0))
        threshold = float(result.get("threshold", 0))

        if verified:
            verification_status = "VERIFIED"
            reason = "Selfie face matches the document face."
        else:
            verification_status = "REJECTED"
            reason = "Selfie face does not match the document face."

        return {
            "verificationStatus": verification_status,
            "faceMatched": verified,
            "distance": distance,
            "threshold": threshold,
            "reason": reason,
        }

    except ValueError as error:
        return {
            "verificationStatus": "MANUAL_REVIEW",
            "faceMatched": False,
            "distance": None,
            "threshold": None,
            "reason": f"Face could not be detected clearly: {str(error)}",
        }

    except Exception as error:
        return {
            "verificationStatus": "ERROR",
            "faceMatched": False,
            "distance": None,
            "threshold": None,
            "reason": f"Verification failed: {str(error)}",
        }