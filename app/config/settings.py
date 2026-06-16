from pydantic_settings import BaseSettings, SettingsConfigDict

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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
