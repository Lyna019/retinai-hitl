"""Configuration loaded from environment variables."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_name: str = "RetinAI HITL"
    environment: str = "development"
    secret_key: str = "change-me-in-prod"
    jwt_algorithm: str = "HS256"
    jwt_expires_minutes: int = 60 * 8

    # Database (PostgreSQL)
    postgres_user: str = "retinai"
    postgres_password: str = "retinai_dev"
    postgres_db: str = "retinai"
    postgres_host: str = "db"
    postgres_port: int = 5432

    # Model service (PyTorch inference)
    model_service_url: str = "http://model-service:9000"

    # RetiZero service (zero-shot CLIP-based fundus classifier)
    retizero_service_url: str = "http://retizero-service:9001"

    # VLM service (vLLM / Gemma on HPC) — kept as optional fallback
    vlm_service_url: str | None = None
    vlm_api_key: str | None = None
    vlm_kubeflow_cookie: str | None = None

    # Hospital clinical DB write-back (CONSULTATION.DIAGNOSIS)
    # Set HOSPITAL_DB_TYPE to: oracle | mssql | postgresql | mysql
    hospital_db_type: str | None = None
    hospital_db_url: str | None = None      # full SQLAlchemy URL e.g. oracle+cx_oracle://user:pw@host/db

    # Whisper
    whisper_model: str = "small"
    whisper_language: str = "fr"

    # Paths
    images_root: str = "/app/data/images"
    audio_root:  str = "/app/data/audio"
    dicom_root:  str = "/app/data/dicom"
    gradcam_root: str = "/app/data/gradcam"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
