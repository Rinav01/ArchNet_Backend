from pydantic_settings import BaseSettings
from pydantic import ConfigDict

class Settings(BaseSettings):
    SECRET_KEY: str = "prod_ready_secret_key_mlbuilder_2026_super_secure"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # DB URL: docker-compose overrides this
    DATABASE_URL: str = "postgresql://mlbuilder:password@localhost:5432/mlbuilder_db"
    
    # Redis URL
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # AWS Storage Configuration (Optional in Phase 1)
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_BUCKET_NAME: str = ""
    
    # GCP Storage & Vertex AI Configuration (Optional)
    GCP_PROJECT_ID: str = ""
    GCP_BUCKET_NAME: str = ""
    GCP_CREDENTIALS_JSON: str = ""
    
    GRAPHQL_DEBUG: bool = True


    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
