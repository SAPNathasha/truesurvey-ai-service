from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field


class QuestionType(str, Enum):
    SINGLE_CHOICE = "SINGLE_CHOICE"
    MULTIPLE_CHOICE = "MULTIPLE_CHOICE"
    RATING = "RATING"
    TEXT = "TEXT"
    YES_NO = "YES_NO"


class GenerateQuestionsRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=120)
    description: str = Field(..., min_length=10, max_length=1500)
    numberOfQuestions: int = Field(..., ge=1, le=20)
    language: Literal["en", "si", "ta"] = "en"
    targetAudience: str | None = Field(default=None, max_length=300)
    preferredQuestionTypes: list[QuestionType] | None = None


class GeneratedQuestion(BaseModel):
    order: int
    questionText: str
    questionType: QuestionType
    options: list[str] = Field(default_factory=list)
    isRequired: bool = True
    helperText: str | None = None


class GenerateQuestionsResponse(BaseModel):
    surveyTitle: str
    surveyDescription: str
    questions: list[GeneratedQuestion]
    notes: str | None = None