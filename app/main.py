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
from app.services.nic_ocr_service import verify_nic_number_from_document
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
# POST /api/ai/questions/generate
app.include_router(
    ai_questions_router,
    prefix="/api/ai/questions",
    tags=["AI Questions"],
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
# FACE VERIFICATION ROUTE - URL IMAGES
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

        face_result = await run_in_threadpool(
            compare_document_face_with_selfie,
            document_temp_path,
            selfie_temp_path,
        )

        nic_result = await run_in_threadpool(
            verify_nic_number_from_document,
            document_temp_path,
            payload.nicNumber,
        )

        face_matched = face_result["faceMatched"]
        nic_matched = nic_result["nicMatched"]

        if face_result["verificationStatus"] == "ERROR":
            verification_status = "ERROR"
            reason = face_result["reason"]

        elif face_result["verificationStatus"] == "MANUAL_REVIEW":
            verification_status = "MANUAL_REVIEW"
            reason = face_result["reason"]

        elif not face_matched:
            verification_status = "REJECTED"
            reason = "Selfie face does not match the document face."

        elif not nic_matched:
            verification_status = "REJECTED"
            reason = nic_result["reason"]

        else:
            verification_status = "VERIFIED"
            reason = "Face matched and submitted NIC number matches the document NIC number."

        return FaceVerificationResponse(
            userId=payload.userId,
            nicNumber=payload.nicNumber,
            verificationStatus=verification_status,
            faceMatched=face_matched,
            nicMatched=nic_matched,
            extractedNicNumber=nic_result["extractedNicNumber"],
            distance=face_result["distance"],
            threshold=face_result["threshold"],
            modelName=settings.MODEL_NAME,
            detectorBackend=settings.DETECTOR_BACKEND,
            reason=reason,
        )

    except Exception as error:
        return FaceVerificationResponse(
            userId=payload.userId,
            nicNumber=payload.nicNumber,
            verificationStatus="ERROR",
            faceMatched=False,
            nicMatched=False,
            extractedNicNumber=None,
            distance=None,
            threshold=None,
            modelName=settings.MODEL_NAME,
            detectorBackend=settings.DETECTOR_BACKEND,
            reason=str(error),
        )

    finally:
        delete_temp_file(document_temp_path)
        delete_temp_file(selfie_temp_path)


# -----------------------------
# FACE VERIFICATION ROUTE - UPLOADED FILES
# -----------------------------
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

        face_result = await run_in_threadpool(
            compare_document_face_with_selfie,
            document_temp_path,
            selfie_temp_path,
        )

        nic_result = await run_in_threadpool(
            verify_nic_number_from_document,
            document_temp_path,
            nicNumber,
        )

        face_matched = face_result["faceMatched"]
        nic_matched = nic_result["nicMatched"]

        if face_result["verificationStatus"] == "ERROR":
            verification_status = "ERROR"
            reason = face_result["reason"]

        elif face_result["verificationStatus"] == "MANUAL_REVIEW":
            verification_status = "MANUAL_REVIEW"
            reason = face_result["reason"]

        elif not face_matched:
            verification_status = "REJECTED"
            reason = "Selfie face does not match the document face."

        elif not nic_matched:
            verification_status = "REJECTED"
            reason = nic_result["reason"]

        else:
            verification_status = "VERIFIED"
            reason = "Face matched and submitted NIC number matches the document NIC number."

        return FaceVerificationResponse(
            userId=userId,
            nicNumber=nicNumber,
            verificationStatus=verification_status,
            faceMatched=face_matched,
            nicMatched=nic_matched,
            extractedNicNumber=nic_result["extractedNicNumber"],
            distance=face_result["distance"],
            threshold=face_result["threshold"],
            modelName=settings.MODEL_NAME,
            detectorBackend=settings.DETECTOR_BACKEND,
            reason=reason,
        )

    except Exception as error:
        return FaceVerificationResponse(
            userId=userId,
            nicNumber=nicNumber,
            verificationStatus="ERROR",
            faceMatched=False,
            nicMatched=False,
            extractedNicNumber=None,
            distance=None,
            threshold=None,
            modelName=settings.MODEL_NAME,
            detectorBackend=settings.DETECTOR_BACKEND,
            reason=str(error),
        )

    finally:
        delete_temp_file(document_temp_path)
        delete_temp_file(selfie_temp_path)