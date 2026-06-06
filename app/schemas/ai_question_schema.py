from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class SurveyQuestionType(str, Enum):
    MULTIPLE_CHOICE = "MULTIPLE_CHOICE"
    SINGLE_SELECT = "SINGLE_SELECT"
    RATING_SCALE = "RATING_SCALE"
    SHORT_ANSWER = "SHORT_ANSWER"
    LONG_ANSWER = "LONG_ANSWER"
    YES_NO = "YES_NO"


class AIQuestionGenerationRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=150)
    description: str = Field(..., min_length=10, max_length=1500)

    # This is now the maximum number, not exact number
    maxNumberOfQuestions: int = Field(..., ge=1, le=20)


class GeneratedQuestion(BaseModel):
    order: int
    questionText: str
    type: SurveyQuestionType
    options: List[str] = Field(default_factory=list)
    isRequired: bool = True
    helpText: str


class AIQuestionGenerationResponse(BaseModel):
    surveyTitle: str
    questions: List[GeneratedQuestion]