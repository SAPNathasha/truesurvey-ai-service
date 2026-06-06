import json
import os
import re
from typing import Any, Dict, List

from fastapi import HTTPException
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.schemas.ai_question_schema import (
    AIQuestionGenerationRequest,
    AIQuestionGenerationResponse,
    SurveyQuestionType,
)


def build_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert survey question generator for TrueSurvey,
an AI-based survey system for Sri Lanka.

You must return ONLY valid JSON.
Do not use markdown.
Do not use ```json.
Do not include explanations outside the JSON.

Allowed question types:
SINGLE_SELECT
MULTIPLE_CHOICE
RATING_SCALE
SHORT_ANSWER
LONG_ANSWER
YES_NO
""",
            ),
            (
                "human",
                """
Generate survey questions using the details below.

Survey title:
{title}

Survey description:
{description}

Maximum number of questions:
{max_number_of_questions}

Important:
The maximum number is only a limit.
You can generate fewer questions if enough.

Return JSON in this exact structure:

{{
  "surveyTitle": "{title}",
  "questions": [
    {{
      "order": 1,
      "questionText": "Question text here",
      "type": "SINGLE_SELECT",
      "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
      "isRequired": true,
      "helpText": "Helpful instruction for the participant"
    }}
  ]
}}

Rules:
1. Generate at least 3 questions if possible.
2. Do not generate more than {max_number_of_questions} questions.
3. Decide the best question type for each question.
4. Use mixed question types when suitable.
5. Do not ask for NIC number, password, OTP, bank card number, exact home address, or private financial data.
6. For SINGLE_SELECT, provide 4 or 5 options.
7. For MULTIPLE_CHOICE, provide 4 or 5 options.
8. For RATING_SCALE, options must be exactly ["1", "2", "3", "4", "5"].
9. For YES_NO, options must be exactly ["Yes", "No"].
10. For SHORT_ANSWER and LONG_ANSWER, options must be [].
11. Every question must have isRequired as true.
12. Every type must be one of:
SINGLE_SELECT, MULTIPLE_CHOICE, RATING_SCALE, SHORT_ANSWER, LONG_ANSWER, YES_NO
""",
            ),
        ]
    )


def extract_json_from_text(text: str) -> Dict[str, Any]:
    cleaned = text.strip()

    cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)

    if not match:
        raise HTTPException(
            status_code=500,
            detail=(
                "Local AI model did not return JSON. "
                "Try again with fewer maximum questions or use a better Ollama model."
            ),
        )

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=(
                "Local AI model returned invalid JSON. "
                "Try again with fewer maximum questions or use a better Ollama model."
            ),
        )


def get_message_content(response: Any) -> str:
    if isinstance(response, AIMessage):
        return str(response.content)

    if hasattr(response, "content"):
        return str(response.content)

    return str(response)


def get_fallback_question_type(index: int) -> SurveyQuestionType:
    fallback_types = [
        SurveyQuestionType.SINGLE_SELECT,
        SurveyQuestionType.MULTIPLE_CHOICE,
        SurveyQuestionType.RATING_SCALE,
        SurveyQuestionType.YES_NO,
        SurveyQuestionType.SHORT_ANSWER,
        SurveyQuestionType.LONG_ANSWER,
    ]

    return fallback_types[(index - 1) % len(fallback_types)]


def normalize_question_type(
    value: Any,
    fallback_type: SurveyQuestionType,
) -> SurveyQuestionType:
    raw_value = str(value or "").strip().upper()
    raw_value = raw_value.replace(" ", "_").replace("-", "_")

    aliases = {
        "SINGLE_SELECT": "SINGLE_SELECT",
        "SINGLE_CHOICE": "SINGLE_SELECT",
        "SINGLE": "SINGLE_SELECT",
        "RADIO": "SINGLE_SELECT",

        "MULTIPLE_CHOICE": "MULTIPLE_CHOICE",
        "MULTIPLE_SELECT": "MULTIPLE_CHOICE",
        "CHECKBOX": "MULTIPLE_CHOICE",
        "CHECKBOXES": "MULTIPLE_CHOICE",

        "RATING_SCALE": "RATING_SCALE",
        "RATING": "RATING_SCALE",
        "SCALE": "RATING_SCALE",

        "SHORT_ANSWER": "SHORT_ANSWER",
        "SHORT_TEXT": "SHORT_ANSWER",

        "LONG_ANSWER": "LONG_ANSWER",
        "LONG_TEXT": "LONG_ANSWER",
        "PARAGRAPH": "LONG_ANSWER",

        "YES_NO": "YES_NO",
        "YES_OR_NO": "YES_NO",
        "BOOLEAN": "YES_NO",
    }

    mapped_value = aliases.get(raw_value)

    if mapped_value in SurveyQuestionType._value2member_map_:
        return SurveyQuestionType(mapped_value)

    return fallback_type


def clean_options(options: Any) -> List[str]:
    if not isinstance(options, list):
        return []

    cleaned_options: List[str] = []

    for option in options:
        option_text = str(option).strip()

        if option_text and option_text not in cleaned_options:
            cleaned_options.append(option_text)

    return cleaned_options


def create_fallback_questions(
    request: AIQuestionGenerationRequest,
) -> Dict[str, Any]:
    count = min(request.maxNumberOfQuestions, 6)

    base_questions = [
        {
            "questionText": "How satisfied are you with your overall experience related to this survey topic?",
            "type": "RATING_SCALE",
            "options": ["1", "2", "3", "4", "5"],
            "helpText": "Rate your satisfaction from 1 to 5.",
        },
        {
            "questionText": "Which factors are most important to you regarding this topic?",
            "type": "MULTIPLE_CHOICE",
            "options": ["Quality", "Price", "Convenience", "Customer support", "Speed"],
            "helpText": "Select all options that apply.",
        },
        {
            "questionText": "Would you recommend this service or experience to others?",
            "type": "YES_NO",
            "options": ["Yes", "No"],
            "helpText": "Select Yes or No.",
        },
        {
            "questionText": "What is the main improvement you would suggest?",
            "type": "SHORT_ANSWER",
            "options": [],
            "helpText": "Write a short suggestion.",
        },
        {
            "questionText": "How often do you interact with this service or experience?",
            "type": "SINGLE_SELECT",
            "options": ["Daily", "Weekly", "Monthly", "Rarely", "Never"],
            "helpText": "Select the closest option.",
        },
        {
            "questionText": "Please describe your overall opinion in detail.",
            "type": "LONG_ANSWER",
            "options": [],
            "helpText": "Write your detailed feedback.",
        },
    ]

    questions = []

    for index, question in enumerate(base_questions[:count], start=1):
        questions.append(
            {
                "order": index,
                "questionText": question["questionText"],
                "type": question["type"],
                "options": question["options"],
                "isRequired": True,
                "helpText": question["helpText"],
            }
        )

    return {
        "surveyTitle": request.title,
        "questions": questions,
    }


def normalize_question_result(
    result: Dict[str, Any],
    request: AIQuestionGenerationRequest,
) -> Dict[str, Any]:
    result["surveyTitle"] = request.title

    questions = result.get("questions")

    if not isinstance(questions, list) or len(questions) == 0:
        return create_fallback_questions(request)

    if len(questions) > request.maxNumberOfQuestions:
        questions = questions[: request.maxNumberOfQuestions]

    normalized_questions = []

    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            continue

        fallback_type = get_fallback_question_type(index)

        question_type = normalize_question_type(
            question.get("type"),
            fallback_type,
        )

        question_text = str(question.get("questionText") or "").strip()
        help_text = str(question.get("helpText") or "").strip()

        if not question_text:
            question_text = f"Question {index}"

        if not help_text:
            help_text = "Please answer this question based on your experience."

        options = clean_options(question.get("options"))

        if question_type == SurveyQuestionType.YES_NO:
            options = ["Yes", "No"]

        elif question_type == SurveyQuestionType.RATING_SCALE:
            options = ["1", "2", "3", "4", "5"]

        elif question_type in [
            SurveyQuestionType.SHORT_ANSWER,
            SurveyQuestionType.LONG_ANSWER,
        ]:
            options = []

        elif question_type in [
            SurveyQuestionType.SINGLE_SELECT,
            SurveyQuestionType.MULTIPLE_CHOICE,
        ]:
            if len(options) < 2:
                options = [
                    "Very satisfied",
                    "Satisfied",
                    "Neutral",
                    "Dissatisfied",
                    "Very dissatisfied",
                ]

            options = options[:6]

        normalized_questions.append(
            {
                "order": len(normalized_questions) + 1,
                "questionText": question_text,
                "type": question_type.value,
                "options": options,
                "isRequired": True,
                "helpText": help_text,
            }
        )

    if len(normalized_questions) == 0:
        return create_fallback_questions(request)

    result["questions"] = normalized_questions

    return result


def generate_ai_questions(
    request: AIQuestionGenerationRequest,
) -> AIQuestionGenerationResponse:
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

    try:
        llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.1,
            format="json",
        )

        prompt = build_prompt_template()

        chain = prompt | llm

        response = chain.invoke(
            {
                "title": request.title,
                "description": request.description,
                "max_number_of_questions": request.maxNumberOfQuestions,
            }
        )

        raw_text = get_message_content(response)
        result = extract_json_from_text(raw_text)
        result = normalize_question_result(result, request)

        return AIQuestionGenerationResponse(**result)

    except HTTPException:
        raise

    except Exception as error:
        # Final safe fallback, useful for demos
        try:
            fallback_result = create_fallback_questions(request)
            return AIQuestionGenerationResponse(**fallback_result)
        except Exception:
            raise HTTPException(
                status_code=500,
                detail=f"LangChain Ollama question generation failed: {str(error)}",
            )