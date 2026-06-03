from fastapi import Depends, FastAPI
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.schemas.verification_schema import (
    FaceVerificationRequest,
    FaceVerificationResponse,
)
from app.security.api_key import verify_api_key
from app.services.face_verification_service import compare_document_face_with_selfie
from app.services.image_service import delete_temp_file, download_image_to_temp_file


app = FastAPI(
    title="TrueSurvey AI Service",
    description="FastAPI service for NIC/licence and selfie verification",
    version="1.0.0",
)


@app.get("/")
def health_check():
    return {
        "message": "TrueSurvey AI Service is running",
        "service": "face-verification",
    }


@app.post(
    "/api/v1/verification/face-match",
    response_model=FaceVerificationResponse,
    dependencies=[Depends(verify_api_key)],
)
async def verify_face(payload: FaceVerificationRequest):
    document_temp_path = None
    selfie_temp_path = None

    try:
        document_temp_path = download_image_to_temp_file(
            image_url=str(payload.documentImageUrl),
            prefix="document_",
        )

        selfie_temp_path = download_image_to_temp_file(
            image_url=str(payload.selfieImageUrl),
            prefix="selfie_",
        )

        comparison_result = await run_in_threadpool(
            compare_document_face_with_selfie,
            document_temp_path,
            selfie_temp_path,
        )

        return FaceVerificationResponse(
            userId=payload.userId,
            nicNumber=payload.nicNumber,
            verificationStatus=comparison_result["verificationStatus"],
            faceMatched=comparison_result["faceMatched"],
            distance=comparison_result["distance"],
            threshold=comparison_result["threshold"],
            modelName=settings.MODEL_NAME,
            detectorBackend=settings.DETECTOR_BACKEND,
            reason=comparison_result["reason"],
        )

    except Exception as error:
        return FaceVerificationResponse(
            userId=payload.userId,
            nicNumber=payload.nicNumber,
            verificationStatus="ERROR",
            faceMatched=False,
            distance=None,
            threshold=None,
            modelName=settings.MODEL_NAME,
            detectorBackend=settings.DETECTOR_BACKEND,
            reason=str(error),
        )

    finally:
        delete_temp_file(document_temp_path)
        delete_temp_file(selfie_temp_path)