from fastapi import APIRouter, HTTPException

from app.schemas.ai_questions_schema import (
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
)
from app.services.ai_question_service import generate_survey_questions

router = APIRouter(
    prefix="/api/v1/ai",
    tags=["AI Question Generation"],
)


@router.post("/generate-questions", response_model=GenerateQuestionsResponse)
def generate_questions(request: GenerateQuestionsRequest):
    try:
        return generate_survey_questions(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate survey questions. Please try again.",
        )