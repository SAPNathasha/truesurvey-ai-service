from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    single_choice = "single_choice"
    multiple_choice = "multiple_choice"
    short_answer = "short_answer"
    rating_scale = "rating_scale"
    yes_no = "yes_no"


class AIQuestionGenerationRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=150)
    description: str = Field(..., min_length=10, max_length=1500)
    numberOfQuestions: int = Field(..., ge=1, le=20)
    questionType: QuestionType


class GeneratedQuestion(BaseModel):
    order: int
    questionText: str
    type: QuestionType
    options: List[str]
    isRequired: bool
    helpText: str


class AIQuestionGenerationResponse(BaseModel):
    surveyTitle: str
    questionType: QuestionType
    questions: List[GeneratedQuestion]