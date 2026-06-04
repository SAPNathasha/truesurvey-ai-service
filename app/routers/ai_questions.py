from fastapi import APIRouter, Depends

from app.schemas.ai_question_schema import (
    AIQuestionGenerationRequest,
    AIQuestionGenerationResponse,
)
from app.security.api_key import verify_api_key
from app.services.ai_question_service import generate_ai_questions


router = APIRouter()


@router.post(
    "/generate",
    response_model=AIQuestionGenerationResponse,
    dependencies=[Depends(verify_api_key)],
)
async def generate_questions(request: AIQuestionGenerationRequest):
    return generate_ai_questions(request)