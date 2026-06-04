import os
from openai import OpenAI

from app.schemas.ai_questions_schema import (
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


SYSTEM_PROMPT = """
You are an expert survey question generator for TrueSurvey, an AI-based survey system.

Your job:
- Generate clear, unbiased, useful survey questions.
- Avoid leading questions.
- Avoid duplicate questions.
- Use simple wording.
- Return exactly the requested number of questions.
- Use appropriate question types.
- For SINGLE_CHOICE and MULTIPLE_CHOICE questions, provide 3 to 5 options.
- For RATING questions, provide options like ["1", "2", "3", "4", "5"].
- For TEXT questions, keep options as an empty list.
- For YES_NO questions, use ["Yes", "No"].
- Do not ask for private sensitive data such as NIC number, phone number, address, or bank details.
"""


def generate_survey_questions(
    request: GenerateQuestionsRequest,
) -> GenerateQuestionsResponse:
    model_name = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

    preferred_types = (
        ", ".join([q.value for q in request.preferredQuestionTypes])
        if request.preferredQuestionTypes
        else "Choose the best mix from SINGLE_CHOICE, MULTIPLE_CHOICE, RATING, TEXT, YES_NO"
    )

    user_prompt = f"""
Generate survey questions using the following details.

Survey title:
{request.title}

Survey description:
{request.description}

Number of questions:
{request.numberOfQuestions}

Language:
{request.language}

Target audience:
{request.targetAudience or "General audience"}

Preferred question types:
{preferred_types}

Return exactly {request.numberOfQuestions} questions.
"""

    response = client.responses.parse(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        text_format=GenerateQuestionsResponse,
    )

    result = response.output_parsed

    if result is None:
        raise ValueError("AI model did not return a valid structured response.")

    if len(result.questions) != request.numberOfQuestions:
        raise ValueError(
            f"Expected {request.numberOfQuestions} questions, but got {len(result.questions)}."
        )

    for index, question in enumerate(result.questions, start=1):
        question.order = index

    return result