import json
import os
import re
from typing import Any, Dict

import requests
from fastapi import HTTPException

from app.schemas.ai_question_schema import (
    AIQuestionGenerationRequest,
    AIQuestionGenerationResponse,
)


def build_prompt(request: AIQuestionGenerationRequest) -> str:
    return f"""
You are generating survey questions for TrueSurvey, an AI-based survey system for Sri Lanka.

Return ONLY valid JSON. Do not include markdown. Do not include explanation.

The JSON must have this exact structure:

{{
  "surveyTitle": "string",
  "questionType": "{request.questionType.value}",
  "questions": [
    {{
      "order": 1,
      "questionText": "string",
      "type": "{request.questionType.value}",
      "options": ["string"],
      "isRequired": true,
      "helpText": "string"
    }}
  ]
}}

Survey title:
{request.title}

Survey description:
{request.description}

Number of questions:
{request.numberOfQuestions}

Question type:
{request.questionType.value}

Rules:
1. Generate exactly {request.numberOfQuestions} questions.
2. Every question type must be "{request.questionType.value}".
3. Use simple and clear language.
4. Avoid duplicate questions.
5. Do not ask for sensitive information such as NIC number, password, bank card number, OTP, exact home address, or private financial data.
6. If questionType is "single_choice", provide 4 or 5 options.
7. If questionType is "multiple_choice", provide 4 or 5 options.
8. If questionType is "yes_no", options must be exactly ["Yes", "No"].
9. If questionType is "rating_scale", options must be exactly ["1", "2", "3", "4", "5"].
10. If questionType is "short_answer", options must be [].
11. isRequired must be true for every question.
"""


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()

    cleaned = re.sub(r"^```json", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise HTTPException(
                status_code=500,
                detail="Local AI model did not return valid JSON.",
            )

        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail="Local AI model returned invalid JSON format.",
            )


def normalize_question_result(
    result: Dict[str, Any],
    request: AIQuestionGenerationRequest,
) -> Dict[str, Any]:
    result["surveyTitle"] = request.title
    result["questionType"] = request.questionType.value

    questions = result.get("questions")

    if not isinstance(questions, list):
        raise HTTPException(
            status_code=500,
            detail="Local AI model response does not contain a valid questions array.",
        )

    if len(questions) != request.numberOfQuestions:
        raise HTTPException(
            status_code=500,
            detail=f"Local AI model generated {len(questions)} questions instead of {request.numberOfQuestions}. Please try again.",
        )

    for index, question in enumerate(questions, start=1):
        question["order"] = index
        question["type"] = request.questionType.value
        question["isRequired"] = True

        if not question.get("questionText"):
            question["questionText"] = f"Question {index}"

        if not question.get("helpText"):
            question["helpText"] = "Please answer this question based on your experience."

        if request.questionType.value == "yes_no":
            question["options"] = ["Yes", "No"]

        elif request.questionType.value == "rating_scale":
            question["options"] = ["1", "2", "3", "4", "5"]

        elif request.questionType.value == "short_answer":
            question["options"] = []

        elif request.questionType.value in ["single_choice", "multiple_choice"]:
            options = question.get("options")

            if not isinstance(options, list) or len(options) < 2:
                question["options"] = [
                    "Strongly agree",
                    "Agree",
                    "Neutral",
                    "Disagree",
                    "Strongly disagree",
                ]

    return result


def generate_ai_questions(
    request: AIQuestionGenerationRequest,
) -> AIQuestionGenerationResponse:
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

    url = f"{ollama_base_url}/api/generate"

    payload = {
        "model": ollama_model,
        "system": (
            "You are an expert survey question generator. "
            "You must always return valid JSON only."
        ),
        "prompt": build_prompt(request),
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 2500,
        },
    }

    try:
        response = requests.post(url, json=payload, timeout=180)

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Ollama request failed: {response.text}",
            )

        data = response.json()
        ai_text = data.get("response")

        if not ai_text:
            raise HTTPException(
                status_code=500,
                detail="Ollama returned an empty response.",
            )

        result = extract_json_object(ai_text)
        result = normalize_question_result(result, request)

        return AIQuestionGenerationResponse(**result)

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=500,
            detail=(
                "Cannot connect to Ollama. Make sure Ollama is installed and running on http://localhost:11434."
            ),
        )

    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=500,
            detail="Ollama took too long to generate questions. Try a smaller model or fewer questions.",
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Local AI question generation failed: {str(error)}",
        )