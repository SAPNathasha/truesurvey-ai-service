from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    FASTAPI_API_KEY: str

    MODEL_NAME: str = "SFace"
    DETECTOR_BACKEND: str = "opencv"

    REQUEST_TIMEOUT_SECONDS: int = 20
    MAX_IMAGE_MB: int = 8

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()