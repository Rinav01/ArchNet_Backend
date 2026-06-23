from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator


class Settings(BaseSettings):
    SECRET_KEY: str = "change_me_in_production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # DB URL: local developer SQLite fallback
    DATABASE_URL: str = "sqlite:///./mlbuilder_local.db"

    # Redis URL: local developer Redis fallback
    REDIS_URL: str = "redis://127.0.0.1:6379/0"

    # AWS Storage Configuration (Optional in Phase 1)
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_BUCKET_NAME: str = ""

    # GCP Storage & Vertex AI Configuration (Optional)
    GCP_PROJECT_ID: str = ""
    GCP_BUCKET_NAME: str = ""
    GCP_CREDENTIALS_JSON: str = ""

    # LLM API Keys
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""

    GRAPHQL_DEBUG: bool = True

    # Runtime environment: "development" | "staging" | "production"
    # Controls dev-only behaviours (auto dev-user, OTel fallback, etc.)
    ENVIRONMENT: str = "development"

    # CORS: comma-separated list of allowed origins.
    # Override in production via CORS_ORIGINS env var.
    # e.g. CORS_ORIGINS=https://archnet.io,https://www.archnet.io
    CORS_ORIGINS: str = "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"

    # Shared secret for GCP Vertex AI training webhook callbacks.
    # Must be set to a strong random value in non-development environments.
    # Generate with: python -c "import secrets; print(secrets.token_hex(32))"
    WEBHOOK_SECRET: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS_ORIGINS env var into a list of allowed origins."""
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @model_validator(mode="after")
    def validate_production_secrets(self) -> "Settings":
        """Raise a hard error at startup if weak placeholder secrets are used in production."""
        if self.ENVIRONMENT != "development":
            if self.SECRET_KEY in ("change_me_in_production", ""):
                raise ValueError(
                    "SECRET_KEY must be set to a strong random value in non-development "
                    "environments. Generate one with: "
                    "python -c \"import secrets; print(secrets.token_hex(32))\""
                )
            if self.WEBHOOK_SECRET == "":
                raise ValueError(
                    "WEBHOOK_SECRET must be set in non-development environments. "
                    "Generate one with: "
                    "python -c \"import secrets; print(secrets.token_hex(32))\""
                )
        return self

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
