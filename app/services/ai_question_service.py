import os
from typing import Any, Dict

from fastapi import HTTPException
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from app.schemas.ai_question_schema import (
    AIQuestionGenerationRequest,
    AIQuestionGenerationResponse,
)


def model_to_dict(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()

    if hasattr(model, "dict"):
        return model.dict()

    return dict(model)


def build_prompt_template(parser: PydanticOutputParser) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert survey question generator for TrueSurvey,
an AI-based survey system for Sri Lanka.

You must generate clear, unbiased, practical survey questions.
You must not ask for sensitive information such as NIC number,
password, OTP, bank card number, exact home address, or private financial data.
Return only the structured output requested by the format instructions.
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

Number of questions:
{number_of_questions}

Question type:
{question_type}

Rules:
1. Generate exactly {number_of_questions} questions.
2. Every question must use this type: {question_type}.
3. Use simple language suitable for Sri Lankan users.
4. Do not create duplicate questions.
5. If questionType is single_choice, provide 4 or 5 options.
6. If questionType is multiple_choice, provide 4 or 5 options.
7. If questionType is yes_no, options must be exactly ["Yes", "No"].
8. If questionType is rating_scale, options must be exactly ["1", "2", "3", "4", "5"].
9. If questionType is short_answer, options must be [].
10. isRequired must be true for every question.

{format_instructions}
""",
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())


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
            detail=(
                f"Local AI model generated {len(questions)} questions instead of "
                f"{request.numberOfQuestions}. Please try again."
            ),
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
                    "Very satisfied",
                    "Satisfied",
                    "Neutral",
                    "Dissatisfied",
                    "Very dissatisfied",
                ]

    return result


def generate_ai_questions(
    request: AIQuestionGenerationRequest,
) -> AIQuestionGenerationResponse:
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

    try:
        parser = PydanticOutputParser(
            pydantic_object=AIQuestionGenerationResponse
        )

        llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.2,
            format="json",
        )

        prompt = build_prompt_template(parser)

        chain = prompt | llm | parser

        generated_response = chain.invoke(
            {
                "title": request.title,
                "description": request.description,
                "number_of_questions": request.numberOfQuestions,
                "question_type": request.questionType.value,
            }
        )

        result = model_to_dict(generated_response)
        result = normalize_question_result(result, request)

        return AIQuestionGenerationResponse(**result)

    except OutputParserException:
        raise HTTPException(
            status_code=500,
            detail=(
                "Local AI model returned invalid JSON. "
                "Try again, reduce the number of questions, or use a better Ollama model."
            ),
        )

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"LangChain Ollama question generation failed: {str(error)}",
        )