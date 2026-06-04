from typing import Literal, Optional

from pydantic import AnyUrl, BaseModel, Field


class FaceVerificationRequest(BaseModel):
    userId: str = Field(..., example="user_uuid_here")
    nicNumber: str = Field(..., example="200212345678")
    documentImageUrl: AnyUrl = Field(
        ...,
        example="https://your-bucket/nic-image.jpg",
    )
    selfieImageUrl: AnyUrl = Field(
        ...,
        example="https://your-bucket/selfie-image.jpg",
    )


class FaceVerificationResponse(BaseModel):
    userId: str
    nicNumber: str

    verificationStatus: Literal["VERIFIED", "REJECTED", "MANUAL_REVIEW", "ERROR"]

    faceMatched: bool
    nicMatched: bool

    extractedNicNumber: Optional[str] = None

    distance: Optional[float] = None
    threshold: Optional[float] = None

    modelName: str
    detectorBackend: str

    reason: Optional[str] = None