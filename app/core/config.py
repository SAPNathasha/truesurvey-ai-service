from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # -----------------------------
    # TRUE SURVEY FASTAPI SECURITY
    # -----------------------------
    FASTAPI_API_KEY: str

    # -----------------------------
    # FACE VERIFICATION SETTINGS
    # -----------------------------
    MODEL_NAME: str = "SFace"
    DETECTOR_BACKEND: str = "opencv"

    REQUEST_TIMEOUT_SECONDS: int = 20
    MAX_IMAGE_MB: int = 8

    # -----------------------------
    # OPENAI AI QUESTION GENERATION
    # -----------------------------
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-5.4-mini"

    # -----------------------------
    # FRONTEND URL
    # -----------------------------
    FRONTEND_URL: str = "http://localhost:3000"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()