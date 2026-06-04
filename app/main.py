import os
from dotenv import load_dotenv

# Load .env before importing settings/services that read environment variables
load_dotenv()

from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.schemas.verification_schema import (
    FaceVerificationRequest,
    FaceVerificationResponse,
)
from app.security.api_key import verify_api_key
from app.services.face_verification_service import compare_document_face_with_selfie
from app.services.image_service import (
    delete_temp_file,
    download_image_to_temp_file,
    save_uploaded_image_to_temp_file,
)
from app.routers.ai_questions import router as ai_questions_router


app = FastAPI(
    title="TrueSurvey AI Service",
    description="FastAPI service for NIC/licence face verification and AI survey question generation",
    version="1.0.0",
)


# -----------------------------
# CORS CONFIGURATION
# -----------------------------
frontend_url = settings.FRONTEND_URL

allowed_origins = [
    origin.strip()
    for origin in frontend_url.split(",")
    if origin.strip()
]

if "http://localhost:3000" not in allowed_origins:
    allowed_origins.append("http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# ROUTERS
# -----------------------------
# This adds:
# POST /api/v1/ai/generate-questions
#
# I added API key protection here too, because later Nest.js should call this service securely.
# If you want to test AI generation without API key temporarily, remove:
# dependencies=[Depends(verify_api_key)]
app.include_router(
    ai_questions_router,
    dependencies=[Depends(verify_api_key)],
)


# -----------------------------
# HEALTH CHECK
# -----------------------------
@app.get("/")
def health_check():
    return {
        "message": "TrueSurvey AI Service is running",
        "status": "OK",
        "services": [
            "face-verification",
            "ai-question-generation",
        ],
    }


# -----------------------------
# FACE VERIFICATION ROUTE
# -----------------------------
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

@app.post(
    "/api/v1/verification/face-match-files",
    response_model=FaceVerificationResponse,
    dependencies=[Depends(verify_api_key)],
)
async def verify_face_with_uploaded_files(
    userId: str = Form(...),
    nicNumber: str = Form(...),
    documentImage: UploadFile = File(...),
    selfieImage: UploadFile = File(...),
):
    document_temp_path = None
    selfie_temp_path = None

    try:
        document_temp_path = await save_uploaded_image_to_temp_file(
            upload_file=documentImage,
            prefix="document_",
        )

        selfie_temp_path = await save_uploaded_image_to_temp_file(
            upload_file=selfieImage,
            prefix="selfie_",
        )

        comparison_result = await run_in_threadpool(
            compare_document_face_with_selfie,
            document_temp_path,
            selfie_temp_path,
        )

        return FaceVerificationResponse(
            userId=userId,
            nicNumber=nicNumber,
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
            userId=userId,
            nicNumber=nicNumber,
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